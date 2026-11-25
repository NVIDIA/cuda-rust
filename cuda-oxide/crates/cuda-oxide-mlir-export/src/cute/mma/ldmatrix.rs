/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Matrix-load address plans and ldmatrix instruction emission.

use crate::cute::copy::emit::{Emitter, integer, require_input};
use crate::{
    MlirAttribute, MlirOperation, MlirType, OperationInput, OperationTranslation,
    TranslationSession,
};
use cute_layout::{ComposedLayout, IntTuple, OffsetUnit};
use dialect_cute::{attributes::CuteMatrixRoleAttr, ops::CuteLdmatrixOp};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Emit lane-specific ldmatrix addresses and the verified transpose/register-count variant.
pub(in crate::cute) struct MatrixTranslation;

/// Bound every intermediate in the byte-offset calculation, not just its final
/// value: negative strides/offsets could otherwise cancel an overflowing term.
/// The verified layout and the unsafe matrix-load API promise a hardware lane
/// (0..32) and an in-bounds selected matrix window. Thus the window coordinates,
/// their row/column expansions, and every nested-layout digit are bounded by
/// the positive shape products. This is not a check of dynamic caller inputs.
pub(in crate::cute) fn matrix_offset_fits_i32(layout: &ComposedLayout) -> bool {
    let limit = i64::from(i32::MAX);
    if layout.unit() != OffsetUnit::Bytes
        || layout.outer().modes().len() != 2
        || !layout
            .outer()
            .checked_size()
            .is_some_and(|size| size <= limit)
        || !(0..=limit).contains(&layout.offset())
    {
        return false;
    }
    let mut maximum = layout.offset();
    for (&extent, &stride) in layout
        .outer()
        .shape
        .flatten()
        .iter()
        .zip(&layout.outer().stride.flatten())
    {
        if !(0..=limit).contains(&stride) {
            return false;
        }
        let Some(next) = (extent - 1)
            .checked_mul(stride)
            .and_then(|term| maximum.checked_add(term))
        else {
            return false;
        };
        if next > limit {
            return false;
        }
        maximum = next;
    }
    // Keeping both fields below the sign bit bounds the mask, shifted bits,
    // and final XOR even when swizzling raises the maximum physical offset.
    let swizzle = layout.inner();
    swizzle.is_valid()
        && (0..=limit).contains(&swizzle.y_mask())
        && (0..=limit).contains(&swizzle.z_mask())
}

/// A proved split between each window base and its lane-dependent byte offsets.
pub(in crate::cute) struct MatrixAddressPlan {
    /// Low address bits up to the highest bit touched by the swizzle.
    low_mask: i64,
    /// Remaining nonnegative i32 address bits kept in the uniform window base.
    high_mask: i64,
}

/// Prove a carry-free split between window-only and lane-only offsets. Nested
/// layout indexing need not distribute over coordinate addition, and a fixed
/// offset may overlap lane bits, so compare the factored result with the full
/// checked map for every legal window/lane. The existing static-validation cap
/// bounds this proof to at most one lane address per four tile elements.
pub(in crate::cute) fn matrix_address_plan(
    layout: &ComposedLayout,
    role: CuteMatrixRoleAttr,
) -> Option<MatrixAddressPlan> {
    if !matrix_offset_fits_i32(layout)
        || layout.outer().checked_size()? > cute_layout::validation::MAX_STATIC_VALIDATION_ELEMENTS
    {
        return None;
    }
    let modes = layout.outer().modes();
    let rows = modes[0].checked_size()?;
    let columns = modes[1].checked_size()?;
    let window_columns = if role == CuteMatrixRoleAttr::B { 8 } else { 16 };
    if rows < 16 || columns < window_columns {
        return None;
    }
    let index = |mode: usize, coordinate| modes[mode].checked_call(&IntTuple::Leaf(coordinate));
    let swizzle = layout.inner();
    let active = swizzle.y_mask() | swizzle.z_mask();
    let low_mask = (1i64 << (i64::BITS - active.leading_zeros())) - 1;
    let high_mask = i64::from(i32::MAX) ^ low_mask;
    for window_row in 0..rows / 16 {
        for window_column in 0..columns / window_columns {
            let tile_row = window_row * 16;
            let tile_column = window_column * window_columns;
            let uniform = layout.offset() + index(0, tile_row)? + index(1, tile_column)?;
            for lane in 0..32 {
                let (local_row, local_column) = if role == CuteMatrixRoleAttr::BPairKMajor {
                    (lane % 8 + lane / 16 * 8, lane / 8 % 2 * 8)
                } else {
                    (
                        lane % 16,
                        if role == CuteMatrixRoleAttr::B {
                            0
                        } else {
                            lane / 16 * 8
                        },
                    )
                };
                let local = index(0, local_row)? + index(1, local_column)?;
                if uniform & local != 0 {
                    return None;
                }
                let high = uniform & high_mask;
                let low = swizzle.apply(uniform & low_mask) ^ swizzle.apply(local);
                let coordinate = tile_row + local_row + rows * (tile_column + local_column);
                let original = layout.checked_call(&IntTuple::Leaf(coordinate))?;
                if high & low != 0 || high.checked_add(low)? != original {
                    return None;
                }
            }
        }
    }
    Some(MatrixAddressPlan {
        low_mask,
        high_mask,
    })
}

impl OperationTranslation for MatrixTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let load = CuteLdmatrixOp::wrap(source);
        load.verify(ctx).map_err(|e| e.to_string())?;
        input.attributes.remove("ldmatrix_elem");
        require_input(&input, 5)?;
        let role = *load.get_attr_matrix_role(ctx).unwrap();
        let (count, columns, transpose) = match role {
            CuteMatrixRoleAttr::A | CuteMatrixRoleAttr::BPairKMajor => (4, 16, false),
            CuteMatrixRoleAttr::B => (2, 8, true),
            CuteMatrixRoleAttr::BPair => (4, 16, true),
        };
        let layout = load
            .get_attr_ldmatrix_smem_layout(ctx)
            .unwrap()
            .0
            .to_byte_offsets(2)
            .map_err(|e| e.to_string())?;
        let mut e = Emitter::new(session, input.location);
        let shared = e.pointer(input.operands[0].clone(), 3)?;
        let width = if matrix_offset_fits_i32(&layout) {
            32
        } else {
            64
        };
        let (window_row, window_column, lane) = if width == 32 {
            (
                e.value(
                    "arith.trunci",
                    MlirType::Integer(32),
                    vec![input.operands[1].clone()],
                )?,
                e.value(
                    "arith.trunci",
                    MlirType::Integer(32),
                    vec![input.operands[2].clone()],
                )?,
                input.operands[3].clone(),
            )
        } else {
            (
                input.operands[1].clone(),
                input.operands[2].clone(),
                e.value(
                    "arith.extui",
                    MlirType::Integer(64),
                    vec![input.operands[3].clone()],
                )?,
            )
        };
        let eight = e.constant(8, width)?;
        let two = e.constant(2, width)?;
        let sixteen = e.constant(16, width)?;
        let column_extent = e.constant(columns, width)?;
        let submatrix = e.binary("arith.divui", lane.clone(), eight.clone())?;
        let lane_row = e.binary("arith.remui", lane, eight.clone())?;
        // Each group of eight lanes supplies one 8x8 matrix to ldmatrix.x4.
        // K-major B pairs need [top-left, top-right, bottom-left, bottom-right]
        // so adjacent result registers form an MMA B operand. A fragments
        // and transposed B pairs retain their original column-first ordering.
        let (row_half_op, column_half_op) = if role == CuteMatrixRoleAttr::BPairKMajor {
            ("arith.divui", "arith.remui")
        } else {
            ("arith.remui", "arith.divui")
        };
        let row_half = e.binary(row_half_op, submatrix.clone(), two.clone())?;
        let row_half = e.binary("arith.muli", row_half, eight.clone())?;
        let local_row = e.binary("arith.addi", row_half, lane_row)?;
        let window_row = e.binary("arith.muli", window_row, sixteen)?;
        let window_column = e.binary("arith.muli", window_column, column_extent)?;
        let local_column = if count == 4 {
            let half = e.binary(column_half_op, submatrix, two)?;
            e.binary("arith.muli", half, eight)?
        } else {
            e.constant(0, width)?
        };
        let source = if let Some(plan) = matrix_address_plan(&layout, role) {
            let uniform = e.linear_pair(layout.outer(), window_row, window_column)?;
            let offset = e.constant(layout.offset(), 32)?;
            let uniform = e.binary("arith.addi", uniform, offset)?;
            let local = e.linear_pair(layout.outer(), local_row, local_column)?;
            let high_mask = e.constant(plan.high_mask, 32)?;
            let high = e.binary("arith.andi", uniform.clone(), high_mask)?;
            let low_mask = e.constant(plan.low_mask, 32)?;
            let low = e.binary("arith.andi", uniform, low_mask)?;
            let low = e.swizzle(layout.inner(), low)?;
            let local = e.swizzle(layout.inner(), local)?;
            let low = e.binary("arith.xori", low, local)?;
            // Keep the high window offset additive so constant matrix-row
            // displacements can remain GEP/immediate offsets after lowering.
            // The allocation pointer itself never enters the swizzle.
            let high = e.value("arith.extui", MlirType::Integer(64), vec![high])?;
            let low = e.value("arith.extui", MlirType::Integer(64), vec![low])?;
            let base = e.gep(shared, high, MlirType::Integer(8))?;
            e.gep(base, low, MlirType::Integer(8))?
        } else {
            let row = e.binary("arith.addi", window_row, local_row)?;
            let column = e.binary("arith.addi", window_column, local_column)?;
            let offset = e.composed(&layout, row, column)?;
            // Preserve 64-bit pointer arithmetic in both general paths.
            let offset = if width == 32 {
                e.value("arith.extui", MlirType::Integer(64), vec![offset])?
            } else {
                offset
            };
            e.gep(shared, offset, MlirType::Integer(8))?
        };
        let aggregate_type =
            MlirType::dialect(format!("!llvm.struct<({})>", vec!["i32"; count].join(", ")))?;
        let (result, aggregate) = e.fresh(aggregate_type);
        let mut load = e.operation("nvvm.ldmatrix", vec![source])?;
        load.results.push(result);
        load.properties
            .insert("num".into(), integer(count as i128, 32));
        load.properties.insert(
            "layout".into(),
            MlirAttribute::dialect(if transpose {
                "#nvvm.mma_layout<col>"
            } else {
                "#nvvm.mma_layout<row>"
            })?,
        );
        e.operations.push(load);
        for index in 0..count {
            let (result, register) = e.fresh(MlirType::Integer(32));
            let mut extract = e.operation("llvm.extractvalue", vec![aggregate.clone()])?;
            extract.results.push(result);
            extract.properties.insert(
                "position".into(),
                MlirAttribute::DenseI64Array(vec![index as i64]),
            );
            e.operations.push(extract);
            let index = e.c64(index as i64)?;
            let slot = e.gep(input.operands[4].clone(), index, MlirType::Integer(32))?;
            let mut store = e.operation("llvm.store", vec![register, slot])?;
            store.properties.insert("alignment".into(), integer(4, 64));
            e.operations.push(store);
        }
        Ok(e.operations)
    }
}

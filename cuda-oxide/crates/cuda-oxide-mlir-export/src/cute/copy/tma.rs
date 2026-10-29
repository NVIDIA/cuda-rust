/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-dimensional TMA view and transfer recipes shared by GEMM capabilities.
//!
//! Source coordinates count tiles in `(row, column)` order. NVVM takes element
//! coordinates in `(column, row)` order. The verified source view supplies tile
//! extents even though its translated type is just a pointer. For example:
//!
//! ```
//! let (rows, columns) = (128, 64);
//! let (tile_row, tile_column) = (2, 3);
//! let nvvm_coordinates = (tile_column * columns, tile_row * rows);
//! assert_eq!(nvvm_coordinates, (192, 256));
//! ```

use crate::cute::common::{
    emit_binary, emit_i64_constant, expect_shape, fresh, generic_pointer_type, operation,
    shared_pointer_type, stringify,
};
use crate::{
    MlirAttribute, MlirLocation, MlirOperation, MlirType, MlirValueUse, OperationInput,
    OperationTranslation, TranslationRegistry, TranslationSession, TypeTranslation,
};
use dialect_cute::{
    attributes::CuteTensorAddressSpaceAttr, epilogue_ops::CuteTmaStore2dSemanticOp,
    gemm_tma_ops::CuteTmaCopy2dOp, types::CuteTmaViewType,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Erase a verified TMA view to a generic or CTA-shared pointer according to its space.
pub(in crate::cute) struct TmaViewTypeTranslation;

impl TypeTranslation for TmaViewTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let view = source_ref
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| "expected cute.tma_view".to_owned())?;
        view.verify(ctx).map_err(stringify)?;
        let tensor = view
            .tensor_view(ctx)
            .ok_or_else(|| "cute.tma_view lost its tensor facts".to_owned())?;
        match tensor.space {
            CuteTensorAddressSpaceAttr::Gmem => generic_pointer_type(),
            CuteTensorAddressSpaceAttr::Smem => shared_pointer_type(),
        }
    }
}

/// Emit a two-dimensional TMA load with tile coordinates converted to element coordinates.
pub(in crate::cute) struct TmaLoadTranslation;

impl OperationTranslation for TmaLoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTmaCopy2dOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.tma_copy_2d", &input, 0, 5)?;
        let (rows, cols) = tma_tile_extents(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        let mut operations = vec![];
        let column = tma_coordinate(
            session,
            &mut operations,
            input.operands[3].clone(),
            cols,
            &input.location,
        )?;
        let row = tma_coordinate(
            session,
            &mut operations,
            input.operands[2].clone(),
            rows,
            &input.location,
        )?;
        let mut copy = operation(
            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
            vec![],
            vec![
                input.operands[1].clone(),
                input.operands[0].clone(),
                column,
                row,
                input.operands[4].clone(),
            ],
            &input.location,
        )?;
        copy.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 0, 0, 0]),
        );
        copy.attributes
            .insert("isCTAOnly".into(), MlirAttribute::Bool(true));
        operations.push(copy);
        Ok(operations)
    }
}

/// Emit a two-dimensional TMA store with NVVM's column-before-row coordinate order.
pub(in crate::cute) struct TmaStoreTranslation;

impl OperationTranslation for TmaStoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTmaStore2dSemanticOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.tma_store_2d", &input, 0, 4)?;
        let (rows, cols) = tma_tile_extents(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        let mut operations = vec![];
        let column = tma_coordinate(
            session,
            &mut operations,
            input.operands[3].clone(),
            cols,
            &input.location,
        )?;
        let row = tma_coordinate(
            session,
            &mut operations,
            input.operands[2].clone(),
            rows,
            &input.location,
        )?;
        let mut copy = operation(
            "nvvm.cp.async.bulk.tensor.global.shared.cta",
            vec![],
            vec![
                input.operands[1].clone(),
                input.operands[0].clone(),
                column,
                row,
            ],
            &input.location,
        )?;
        copy.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 0, 0]),
        );
        operations.push(copy);
        Ok(operations)
    }
}

/// Read positive row and column element counts from the source view's shared layout.
fn tma_tile_extents(ctx: &Context, ty: TypeHandle) -> Result<(u64, u64), String> {
    let ty_ref = ty.deref(ctx);
    let view = ty_ref
        .downcast_ref::<CuteTmaViewType>()
        .ok_or_else(|| "expected a cute.tma_view operand".to_owned())?;
    let modes = view.smem_layout.0.outer().modes();
    if modes.len() != 2 {
        return Err(format!("TMA mapping needs two modes, got {}", modes.len()));
    }
    let rows = modes[0]
        .checked_size()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "TMA row extent is invalid".to_owned())?;
    let cols = modes[1]
        .checked_size()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "TMA column extent is invalid".to_owned())?;
    Ok((rows, cols))
}

/// Multiply a tile index by its extent and produce NVVM's i32 element coordinate.
fn tma_coordinate(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    tile: MlirValueUse,
    extent: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let extent = emit_i64_constant(session, operations, extent, location)?;
    let elements = emit_binary(session, operations, "arith.muli", tile, extent, location)?;
    let (result, use_) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "arith.trunci",
        vec![result],
        vec![elements],
        location,
    )?);
    Ok(use_)
}

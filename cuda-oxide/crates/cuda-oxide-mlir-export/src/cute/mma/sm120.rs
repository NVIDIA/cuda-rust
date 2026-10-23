/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM120 block-scaled MMA carriers, shared loads, and instruction emission.

use crate::cute::common::{
    cast_int, emit_binary, emit_i32_constant, emit_i64_constant, expect_shape, fresh, gep_value,
    operation, stringify,
};
use crate::{
    MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse, OperationInput,
    OperationTranslation, TranslationSession,
};
use dialect_cute::smem_mma_ops::{
    CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp, CuteTiledGemmOp,
};
use dialect_mir::types::{MirArrayType, MirStructType, MirTupleType};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Byte-address bits selected by the verified FP16 shared-layout swizzle.
const LDMATRIX_BYTE_SWIZZLE_MASK: u64 = 0x180;
/// Right shift mapping the selected byte-address bits onto their XOR destinations.
const LDMATRIX_BYTE_SWIZZLE_SHIFT: u64 = 3;
/// Recursive physical LLVM carrier layout retaining the logical scalar register order.
#[derive(Clone)]
pub(in crate::cute) struct CarrierLayout {
    /// LLVM type used to transport this carrier node.
    pub(in crate::cute) target: MlirType,
    /// Whether this node is a scalar leaf or a list of physical aggregate slots.
    kind: CarrierKind,
}

/// Distinguish scalar registers from recursively nested aggregate storage.
#[derive(Clone)]
enum CarrierKind {
    /// One scalar register with no nested fields.
    Leaf,
    /// Physical field indices paired with their recursively verified carrier layouts.
    Aggregate(
        #[doc = "Physical field indices paired with child carrier layouts in logical register order."]
         Vec<(i64, CarrierLayout)>,
    ),
}

impl CarrierLayout {
    /// Build a carrier tree from MIR aggregate layout, rejecting unsupported shapes or excessive nesting.
    pub(in crate::cute) fn from_source(
        ctx: &Context,
        source: TypeHandle,
        session: &TranslationSession<'_>,
        depth: usize,
    ) -> Result<Self, String> {
        if depth > 64 {
            return Err("MMA carrier aggregate is unexpectedly deep".into());
        }
        let target = session.translate_type(source).map_err(stringify)?;
        let source_ref = source.deref(ctx);
        if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
            let count = usize::try_from(array.size())
                .map_err(|_| "MMA carrier array is too large".to_owned())?;
            let child = Self::from_source(ctx, array.element_type(), session, depth + 1)?;
            return Ok(Self {
                target,
                kind: CarrierKind::Aggregate(
                    (0..count)
                        .map(|index| (index as i64, child.clone()))
                        .collect(),
                ),
            });
        }

        let fields = if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
            Some(tuple.get_types().to_vec())
        } else {
            source_ref
                .downcast_ref::<MirStructType>()
                .map(|structure| structure.field_types.clone())
        };
        let Some(fields) = fields else {
            return Ok(Self {
                target,
                kind: CarrierKind::Leaf,
            });
        };
        drop(source_ref);

        let plan = crate::mir::memory::aggregate_plan(ctx, source, |ty| {
            session.translate_type(ty).map_err(stringify)
        })?;
        if plan.ty != target {
            return Err("MMA carrier aggregate disagrees with MIR type translation".into());
        }
        let mut children = Vec::new();
        for (declaration, field) in fields.into_iter().enumerate() {
            let Some(slot) = plan.decl_to_slot[declaration] else {
                continue;
            };
            children.push((
                slot as i64,
                Self::from_source(ctx, field, session, depth + 1)?,
            ));
        }
        Ok(Self {
            target,
            kind: CarrierKind::Aggregate(children),
        })
    }

    /// Count scalar register leaves while ignoring aggregate nesting.
    pub(in crate::cute) fn leaf_count(&self) -> usize {
        match &self.kind {
            CarrierKind::Leaf => 1,
            CarrierKind::Aggregate(children) => {
                children.iter().map(|(_, child)| child.leaf_count()).sum()
            }
        }
    }
}

/// Append LLVM extracts in logical register order through physical aggregate slots.
pub(in crate::cute) fn extract_carrier(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    layout: &CarrierLayout,
    leaves: &mut Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<(), String> {
    match &layout.kind {
        CarrierKind::Leaf => {
            leaves.push(value);
            Ok(())
        }
        CarrierKind::Aggregate(children) => {
            for (slot, child) in children {
                let (result, extracted) = fresh(session, child.target.clone());
                let mut extract = operation(
                    "llvm.extractvalue",
                    vec![result],
                    vec![value.clone()],
                    location,
                )?;
                extract
                    .properties
                    .insert("position".into(), MlirAttribute::DenseI64Array(vec![*slot]));
                operations.push(extract);
                extract_carrier(session, operations, extracted, child, leaves, location)?;
            }
            Ok(())
        }
    }
}

/// Reconstruct the original physical aggregate from ordered scalar register results.
fn rebuild_carrier(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    layout: &CarrierLayout,
    leaves: &[MlirValueUse],
    final_result: MlirResult,
    location: &MlirLocation,
) -> Result<(), String> {
    let mut cursor = 0;
    build_carrier_node(
        session,
        operations,
        layout,
        leaves,
        &mut cursor,
        Some(final_result),
        location,
    )?;
    if cursor != leaves.len() {
        return Err(format!(
            "MMA carrier consumed {cursor} leaves but received {}",
            leaves.len()
        ));
    }
    Ok(())
}

/// Recursively insert scalar leaves into one physical carrier node.
fn build_carrier_node(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    layout: &CarrierLayout,
    leaves: &[MlirValueUse],
    cursor: &mut usize,
    final_result: Option<MlirResult>,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    if matches!(layout.kind, CarrierKind::Leaf) {
        let leaf = leaves
            .get(*cursor)
            .cloned()
            .ok_or_else(|| "MMA carrier did not receive enough leaves".to_owned())?;
        *cursor += 1;
        if let Some(result) = final_result {
            let use_ = MlirValueUse {
                id: result.id,
                ty: result.ty.clone(),
            };
            operations.push(operation(
                "builtin.unrealized_conversion_cast",
                vec![result],
                vec![leaf],
                location,
            )?);
            return Ok(use_);
        }
        return Ok(leaf);
    }

    let CarrierKind::Aggregate(children) = &layout.kind else {
        unreachable!()
    };
    let mut built_children = Vec::with_capacity(children.len());
    for (slot, child) in children {
        built_children.push((
            *slot,
            build_carrier_node(session, operations, child, leaves, cursor, None, location)?,
        ));
    }

    let poison_result = if built_children.is_empty() {
        final_result
            .clone()
            .unwrap_or_else(|| fresh(session, layout.target.clone()).0)
    } else {
        fresh(session, layout.target.clone()).0
    };
    let mut aggregate = MlirValueUse {
        id: poison_result.id,
        ty: poison_result.ty.clone(),
    };
    operations.push(operation(
        "llvm.mlir.poison",
        vec![poison_result],
        vec![],
        location,
    )?);
    let child_count = built_children.len();
    for (index, (slot, child)) in built_children.into_iter().enumerate() {
        let result = if index + 1 == child_count {
            final_result
                .clone()
                .unwrap_or_else(|| fresh(session, layout.target.clone()).0)
        } else {
            fresh(session, layout.target.clone()).0
        };
        let next = MlirValueUse {
            id: result.id,
            ty: result.ty.clone(),
        };
        let mut insert = operation(
            "llvm.insertvalue",
            vec![result],
            vec![aggregate, child],
            location,
        )?;
        insert
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![slot]));
        operations.push(insert);
        aggregate = next;
    }
    Ok(aggregate)
}

/// Fill every scalar leaf of a verified accumulator carrier with the source value.
pub(in crate::cute) struct FragmentFillTranslation;

impl OperationTranslation for FragmentFillTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteFragmentFillOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.fragment_fill", &input, 1, 1)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 64 {
            return Err("cute.fragment_fill needs exactly 64 f32 leaves".into());
        }
        let leaves = vec![input.operands[0].clone(); 64];
        let mut operations = vec![];
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &leaves,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

/// Load and arrange scale words for the verified SM120 block-scaled MMA plan.
pub(in crate::cute) struct MmaLoadScalesTranslation;

impl OperationTranslation for MmaLoadScalesTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteMmaLoadScalesOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("mma_load_scales_a_view");
        input.attributes.remove("mma_load_scales_b_view");
        expect_shape("cute.mma_load_scales", &input, 1, 7)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 10 {
            return Err("cute.mma_load_scales needs exactly ten u32 leaves".into());
        }

        let mut operations = vec![];
        let lane = cast_int(
            session,
            &mut operations,
            input.operands[0].clone(),
            MlirType::Integer(64),
            "arith.extui",
            &input.location,
        )?;
        let four = emit_i64_constant(session, &mut operations, 4, &input.location)?;
        let q = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            lane.clone(),
            four,
            &input.location,
        )?;
        let three = emit_i64_constant(session, &mut operations, 3, &input.location)?;
        let r = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane,
            three,
            &input.location,
        )?;
        let one = emit_i64_constant(session, &mut operations, 1, &input.location)?;
        let parity = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            r,
            one,
            &input.location,
        )?;
        let eight = emit_i64_constant(session, &mut operations, 8, &input.location)?;
        let parity_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            parity,
            eight,
            &input.location,
        )?;
        let a_provider = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            q.clone(),
            parity_band,
            &input.location,
        )?;
        let sixteen = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let a_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[5].clone(),
            sixteen.clone(),
            &input.location,
        )?;
        let a_row0 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            a_band,
            a_provider,
            &input.location,
        )?;
        let sixty_four = emit_i64_constant(session, &mut operations, 64, &input.location)?;
        let a_row1 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            a_row0.clone(),
            sixty_four,
            &input.location,
        )?;
        let b_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[6].clone(),
            sixteen,
            &input.location,
        )?;
        let b_row0 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            b_band,
            q,
            &input.location,
        )?;
        let mut rows = vec![a_row0, a_row1, b_row0.clone()];
        for delta in [8_u64, 32, 40, 64, 72, 96, 104] {
            let delta = emit_i64_constant(session, &mut operations, delta, &input.location)?;
            rows.push(emit_binary(
                session,
                &mut operations,
                "arith.addi",
                b_row0.clone(),
                delta,
                &input.location,
            )?);
        }
        let mut words = Vec::with_capacity(10);
        for (index, row) in rows.into_iter().enumerate() {
            words.push(load_scale_word(
                session,
                &mut operations,
                if index < 2 {
                    input.operands[1].clone()
                } else {
                    input.operands[3].clone()
                },
                row,
                &input.location,
            )?);
        }
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &words,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

/// Select a K slice and unpack its packed register fragments for the next MMA step.
pub(in crate::cute) struct FragmentSliceKTranslation;

impl OperationTranslation for FragmentSliceKTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteFragmentSliceKOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.fragment_slice_k", &input, 1, 2)?;
        let source_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_operand(0).get_type(ctx),
            session,
            0,
        )?;
        let result_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        let mut operations = vec![];
        let mut words = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[0].clone(),
            &source_layout,
            &mut words,
            &input.location,
        )?;
        if words.len() != 10 || result_layout.leaf_count() != 10 {
            return Err("cute.fragment_slice_k needs ten packed scale words".into());
        }
        let sixteen = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let shift64 = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[1].clone(),
            sixteen,
            &input.location,
        )?;
        let shift = cast_int(
            session,
            &mut operations,
            shift64,
            MlirType::Integer(32),
            "arith.trunci",
            &input.location,
        )?;
        let mask = emit_i32_constant(session, &mut operations, 0xffff, &input.location)?;
        let mut selected = Vec::with_capacity(10);
        for word in words {
            let shifted = emit_binary(
                session,
                &mut operations,
                "arith.shrui",
                word,
                shift.clone(),
                &input.location,
            )?;
            selected.push(emit_binary(
                session,
                &mut operations,
                "arith.andi",
                shifted,
                mask.clone(),
                &input.location,
            )?);
        }
        rebuild_carrier(
            session,
            &mut operations,
            &result_layout,
            &selected,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

/// Load the A fragment using the verified shared layout and ldmatrix lane assignment.
pub(in crate::cute) struct MmaLoadATranslation;

impl OperationTranslation for MmaLoadATranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteMmaLoadAOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("mma_load_a_view");
        expect_shape("cute.mma_load_a", &input, 1, 5)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 8 {
            return Err("cute.mma_load_a needs exactly eight u32 leaves".into());
        }
        let mut operations = vec![];
        let first = emit_ldmatrix_x4(
            session,
            &mut operations,
            input.operands[1].clone(),
            input.operands[3].clone(),
            input.operands[4].clone(),
            input.operands[0].clone(),
            &input.location,
        )?;
        let four = emit_i64_constant(session, &mut operations, 4, &input.location)?;
        let second_row = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            input.operands[3].clone(),
            four,
            &input.location,
        )?;
        let second = emit_ldmatrix_x4(
            session,
            &mut operations,
            input.operands[1].clone(),
            second_row,
            input.operands[4].clone(),
            input.operands[0].clone(),
            &input.location,
        )?;
        let mut registers = first.to_vec();
        registers.extend(second);
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &registers,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

/// Compute swizzled FP16 shared addresses and emit a four-register ldmatrix load.
fn emit_ldmatrix_x4(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    warp_tile_row: MlirValueUse,
    warp_tile_column: MlirValueUse,
    lane: MlirValueUse,
    location: &MlirLocation,
) -> Result<[MlirValueUse; 4], String> {
    let lane = cast_int(
        session,
        operations,
        lane,
        MlirType::Integer(64),
        "arith.extui",
        location,
    )?;
    let eight = emit_i64_constant(session, operations, 8, location)?;
    let submatrix = emit_binary(
        session,
        operations,
        "arith.divui",
        lane.clone(),
        eight.clone(),
        location,
    )?;
    let row_in = emit_binary(
        session,
        operations,
        "arith.remui",
        lane,
        eight.clone(),
        location,
    )?;
    let two = emit_i64_constant(session, operations, 2, location)?;
    let parity = emit_binary(
        session,
        operations,
        "arith.remui",
        submatrix.clone(),
        two.clone(),
        location,
    )?;
    let row_offset = emit_binary(
        session,
        operations,
        "arith.muli",
        parity,
        eight.clone(),
        location,
    )?;
    let sixteen = emit_i64_constant(session, operations, 16, location)?;
    let row = emit_binary(
        session,
        operations,
        "arith.muli",
        warp_tile_row,
        sixteen.clone(),
        location,
    )?;
    let row = emit_binary(session, operations, "arith.addi", row, row_offset, location)?;
    let row = emit_binary(session, operations, "arith.addi", row, row_in, location)?;

    let column = emit_binary(
        session,
        operations,
        "arith.muli",
        warp_tile_column,
        sixteen,
        location,
    )?;
    let sub_half = emit_binary(
        session,
        operations,
        "arith.divui",
        submatrix,
        two.clone(),
        location,
    )?;
    let column_offset = emit_binary(session, operations, "arith.muli", sub_half, eight, location)?;
    let column = emit_binary(
        session,
        operations,
        "arith.addi",
        column,
        column_offset,
        location,
    )?;

    // The verified shared layout is f16 (128,32):(32,1), converted to byte
    // strides (64,2), followed by S<2,3,3>.
    let sixty_four = emit_i64_constant(session, operations, 64, location)?;
    let row_bytes = emit_binary(session, operations, "arith.muli", row, sixty_four, location)?;
    let column_bytes = emit_binary(session, operations, "arith.muli", column, two, location)?;
    let plain = emit_binary(
        session,
        operations,
        "arith.addi",
        row_bytes,
        column_bytes,
        location,
    )?;
    // The source layout's element-unit S<2,3,3> becomes S<2,4,3>
    // after converting f16 elements to byte offsets.  Mask bits 8..7 and
    // shift them onto bits 5..4, preserving the 16-byte ldmatrix segment.
    let mask = emit_i64_constant(session, operations, LDMATRIX_BYTE_SWIZZLE_MASK, location)?;
    let source_bits = emit_binary(
        session,
        operations,
        "arith.andi",
        plain.clone(),
        mask,
        location,
    )?;
    let three = emit_i64_constant(session, operations, LDMATRIX_BYTE_SWIZZLE_SHIFT, location)?;
    let moved = emit_binary(
        session,
        operations,
        "arith.shrui",
        source_bits,
        three,
        location,
    )?;
    let byte_offset = emit_binary(session, operations, "arith.xori", plain, moved, location)?;
    let pointer = gep_value(
        session,
        operations,
        base,
        byte_offset,
        MlirType::Integer(8),
        location,
    )?;

    let result_type = MlirType::dialect("!llvm.struct<(i32, i32, i32, i32)>")?;
    let (matrix_result, matrix) = fresh(session, result_type);
    let mut load = operation(
        "nvvm.ldmatrix",
        vec![matrix_result],
        vec![pointer],
        location,
    )?;
    load.properties.insert(
        "layout".into(),
        MlirAttribute::dialect("#nvvm.mma_layout<row>")?,
    );
    load.properties.insert(
        "num".into(),
        MlirAttribute::Integer {
            value: 4,
            ty: MlirType::Integer(32),
        },
    );
    operations.push(load);
    let mut registers = Vec::with_capacity(4);
    for index in 0..4 {
        let (result, use_) = fresh(session, MlirType::Integer(32));
        let mut extract = operation(
            "llvm.extractvalue",
            vec![result],
            vec![matrix.clone()],
            location,
        )?;
        extract
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![index]));
        operations.push(extract);
        registers.push(use_);
    }
    registers
        .try_into()
        .map_err(|_| "ldmatrix did not produce four registers".to_owned())
}

/// Emit the verified block-scaled MMA sequence and rebuild its accumulator carrier.
pub(in crate::cute) struct TiledGemmTranslation;

impl OperationTranslation for TiledGemmTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let gemm = CuteTiledGemmOp::wrap(source);
        gemm.verify(ctx).map_err(stringify)?;
        let plan = gemm
            .plan(ctx)
            .ok_or_else(|| "cute.tiled_gemm missing plan".to_owned())?;
        let is_nvf4 =
            plan.atom.scale_format == dialect_cute::attributes::CuteTensorFormatAttr::E4M3;
        input.attributes.remove("tiled_gemm_b_view");
        expect_shape("cute.tiled_gemm", &input, 1, 8)?;
        let operation_ref = source.deref(ctx);
        let a_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(1).get_type(ctx),
            session,
            0,
        )?;
        let scales_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(6).get_type(ctx),
            session,
            0,
        )?;
        let accumulator_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(7).get_type(ctx),
            session,
            0,
        )?;
        let result_layout =
            CarrierLayout::from_source(ctx, operation_ref.get_result(0).get_type(ctx), session, 0)?;
        drop(operation_ref);

        let mut operations = vec![];
        let mut a = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[1].clone(),
            &a_layout,
            &mut a,
            &input.location,
        )?;
        let mut scales = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[6].clone(),
            &scales_layout,
            &mut scales,
            &input.location,
        )?;
        let mut accumulator = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[7].clone(),
            &accumulator_layout,
            &mut accumulator,
            &input.location,
        )?;
        if a.len() != 8
            || scales.len() != 10
            || accumulator.len() != 64
            || result_layout.leaf_count() != 64
        {
            return Err("cute.tiled_gemm carrier shape changed after verification".into());
        }

        for pair in 0..4_usize {
            let pair_offset =
                emit_i64_constant(session, &mut operations, (pair * 2) as u64, &input.location)?;
            let b_row = emit_binary(
                session,
                &mut operations,
                "arith.addi",
                input.operands[4].clone(),
                pair_offset,
                &input.location,
            )?;
            let loaded = emit_ldmatrix_x4(
                session,
                &mut operations,
                input.operands[2].clone(),
                b_row,
                input.operands[5].clone(),
                input.operands[0].clone(),
                &input.location,
            )?;
            let b_fragments = [
                [loaded[0].clone(), loaded[2].clone()],
                [loaded[1].clone(), loaded[3].clone()],
            ];
            for m in 0..2_usize {
                for (n_in_pair, b_fragment) in b_fragments.iter().enumerate() {
                    let n = pair * 2 + n_in_pair;
                    let cell = m * 8 + n;
                    let updated = emit_mxf4_mma(
                        session,
                        &mut operations,
                        &a[m * 4..m * 4 + 4],
                        b_fragment,
                        &accumulator[cell * 4..cell * 4 + 4],
                        scales[m].clone(),
                        scales[2 + n].clone(),
                        is_nvf4,
                        &input.location,
                    )?;
                    accumulator[cell * 4..cell * 4 + 4].clone_from_slice(&updated);
                }
            }
        }
        rebuild_carrier(
            session,
            &mut operations,
            &result_layout,
            &accumulator,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

/// Emit one MXF4/NVF4 register MMA with A, B, accumulator, and scale operands in NVVM order.
#[allow(clippy::too_many_arguments)]
fn emit_mxf4_mma(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    a: &[MlirValueUse],
    b: &[MlirValueUse],
    accumulator: &[MlirValueUse],
    scale_a: MlirValueUse,
    scale_b: MlirValueUse,
    is_nvf4: bool,
    location: &MlirLocation,
) -> Result<[MlirValueUse; 4], String> {
    if a.len() != 4 || b.len() != 2 || accumulator.len() != 4 {
        return Err("SM120 MXF4 MMA received the wrong register count".into());
    }
    let scale_a = if is_nvf4 {
        scale_a
    } else {
        cast_int(
            session,
            operations,
            scale_a,
            MlirType::Integer(16),
            "arith.trunci",
            location,
        )?
    };
    let scale_b = if is_nvf4 {
        scale_b
    } else {
        cast_int(
            session,
            operations,
            scale_b,
            MlirType::Integer(16),
            "arith.trunci",
            location,
        )?
    };
    let mut operands = Vec::with_capacity(12);
    operands.extend_from_slice(a);
    operands.extend_from_slice(b);
    operands.extend_from_slice(accumulator);
    operands.extend([scale_a, scale_b]);

    let result_type = MlirType::Float(crate::MlirFloatType::F32);
    let mut results = Vec::with_capacity(4);
    let mut uses = Vec::with_capacity(4);
    for _ in 0..4 {
        let (result, use_) = fresh(session, result_type.clone());
        results.push(result);
        uses.push(use_);
    }
    let mut operation_ = operation(
        "cute_nvgpu.arch.mma.SM120.block_scaled",
        results,
        operands,
        location,
    )?;
    operation_.properties.insert(
        "a_type".into(),
        MlirAttribute::Type(MlirType::Float(crate::MlirFloatType::Other(
            "f4E2M1FN".into(),
        ))),
    );
    operation_.properties.insert(
        "b_type".into(),
        MlirAttribute::Type(MlirType::Float(crate::MlirFloatType::Other(
            "f4E2M1FN".into(),
        ))),
    );
    operation_.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![4, 2, 4, 1, 1, 0, 0]),
    );
    operation_.properties.insert(
        "sf_type".into(),
        MlirAttribute::Type(MlirType::Float(crate::MlirFloatType::Other(
            if is_nvf4 { "f8E4M3FN" } else { "f8E8M0FNU" }.into(),
        ))),
    );
    operation_.properties.insert(
        "shape_MNK".into(),
        MlirAttribute::dialect("#cute.shape<\"(16,8,64)\">")?,
    );
    operation_.properties.insert(
        "thread_id_a".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(16),
        },
    );
    operation_.properties.insert(
        "thread_id_b".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(16),
        },
    );
    operation_.properties.insert(
        "vec_size".into(),
        MlirAttribute::Integer {
            value: if is_nvf4 { 16 } else { 32 },
            ty: MlirType::Integer(32),
        },
    );
    operations.push(operation_);
    uses.try_into()
        .map_err(|_| "SM120 MXF4 MMA did not produce four accumulators".to_owned())
}

/// Select a packed scale word using the source lane and scale-column assignment.
fn load_scale_word(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    row: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let mask31 = emit_i64_constant(session, operations, 31, location)?;
    let low = emit_binary(
        session,
        operations,
        "arith.andi",
        row.clone(),
        mask31,
        location,
    )?;
    let four = emit_i64_constant(session, operations, 4, location)?;
    let low = emit_binary(session, operations, "arith.muli", low, four, location)?;
    let five = emit_i64_constant(session, operations, 5, location)?;
    let quadrant = emit_binary(session, operations, "arith.shrui", row, five, location)?;
    let three = emit_i64_constant(session, operations, 3, location)?;
    let quadrant = emit_binary(session, operations, "arith.andi", quadrant, three, location)?;
    let offset = emit_binary(session, operations, "arith.addi", low, quadrant, location)?;
    let pointer = gep_value(
        session,
        operations,
        base,
        offset,
        MlirType::Integer(32),
        location,
    )?;
    let (result, use_) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "llvm.load",
        vec![result],
        vec![pointer],
        location,
    )?);
    Ok(use_)
}

/// Apply the verified byte-address XOR permutation while preserving 16-byte segments.
#[cfg(test)]
pub(in crate::cute) const fn ldmatrix_swizzled_byte_offset(plain: u64) -> u64 {
    plain ^ ((plain & LDMATRIX_BYTE_SWIZZLE_MASK) >> LDMATRIX_BYTE_SWIZZLE_SHIFT)
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVFP4 GEMV mappings for the official CUTLASS full-CuTe MLIR profile.
//!
//! Source views become LLVM aggregates holding runtime pointers and shapes.
//! Keeping these fields in SSA preserves the dependencies between operations:
//!
//! ```text
//! tensor_make_2d -> scaled_view_make -> row -> KTile<64>
//!                                               |
//!                                               v
//!             f4/UE8M0 CuTe views <- selected byte addresses
//!                                               |
//!                                               v
//!                  load_vec -> cvt_fpext -> ordered f32 dot
//! ```
//!
//! The mapping translates CuTe operations directly into operations accepted
//! by CUTLASS.

use crate::cute::{TensorViewTypeTranslation, register_cute_elementwise_mappings};

use dialect_cute::{
    attributes::{
        CuteScaledLayoutAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr,
        CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
    },
    gemv_ops::{
        CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
        CuteScaledViewRowOp, CuteTensorMake2DOp, GEMV_K_TILE_WIDTH,
    },
    types::{CuteFragmentType, CuteScaledViewType, CuteTensorViewType},
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};
use pliron_mlir_export::{
    MlirAttribute, MlirFloatType, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};

const VALUE_BYTES_PER_TILE: u64 = 32;
const VALUE_BYTES_PER_LOAD: u64 = 16;
const VALUE_LANES_PER_LOAD: u64 = 32;
const SCALE_BYTES_PER_TILE: u64 = 4;
const SCALE_ATOM_ROWS: u64 = 128;
const SCALE_ATOM_K: u64 = 64;
const SCALE_ATOM_BYTES: u64 = 512;

/// Register the composed elementwise + NVFP4 GEMV mapping pack.
///
/// `cute.tensor_view` has one registry owner. Its translator dispatches from
/// the verified source type and fails closed for every view outside the two
/// reviewed mapping families.
pub(crate) fn register_cute_cutlass_profile_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_type::<CuteTensorViewType>(CombinedTensorViewTypeTranslation)?;
    register_cute_elementwise_mappings(registry)?;
    register_gemv_mappings(registry)
}

fn register_gemv_mappings(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_type::<CuteScaledViewType>(ScaledViewTypeTranslation)?;
    registry.register_type::<CuteFragmentType>(FragmentTypeTranslation)?;

    registry.register_operation::<CuteTensorMake2DOp>(TensorMake2DTranslation)?;
    registry.register_operation::<CuteScaledViewMakeOp>(ScaledViewMakeTranslation)?;
    registry.register_operation::<CuteScaledViewRowOp>(ScaledViewRowTranslation)?;
    registry.register_operation::<CuteScaledViewKTileOp>(ScaledViewKTileTranslation)?;
    registry.register_operation::<CuteScaledViewLoadOp>(ScaledViewLoadTranslation)?;
    registry.register_operation::<CuteDotOp>(DotTranslation)?;
    Ok(())
}

struct CombinedTensorViewTypeTranslation;

impl TypeTranslation for CombinedTensorViewTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let view = source_ref
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| "expected cute.tensor_view".to_owned())?;
        if require_gemv_tensor_view(ctx, view).is_ok() {
            return tensor_carrier_type();
        }
        drop(source_ref);
        TensorViewTypeTranslation.translate(ctx, source, registry)
    }
}

struct ScaledViewTypeTranslation;

impl TypeTranslation for ScaledViewTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let view = source_ref
            .downcast_ref::<CuteScaledViewType>()
            .ok_or_else(|| "expected cute.scaled_view".to_owned())?;
        view.verify(ctx).map_err(stringify)?;
        scaled_carrier_type()
    }
}

struct FragmentTypeTranslation;

impl TypeTranslation for FragmentTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let fragment = source_ref
            .downcast_ref::<CuteFragmentType>()
            .ok_or_else(|| "expected cute.fragment".to_owned())?;
        fragment.verify(ctx).map_err(stringify)?;
        fragment_carrier_type()
    }
}

struct TensorMake2DTranslation;

impl OperationTranslation for TensorMake2DTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_make_2d", &input, 1, 4)?;
        CuteTensorMake2DOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        let result_view = tensor_view(ctx, source.deref(ctx).get_result(0))?;
        require_gemv_tensor_view(ctx, &result_view)?;
        build_aggregate(
            session,
            input.results[0].clone(),
            input.operands,
            &input.location,
        )
    }
}

struct ScaledViewMakeTranslation;

impl OperationTranslation for ScaledViewMakeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.scaled_view_make", &input, 1, 2)?;
        CuteScaledViewMakeOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        let result = scaled_view(ctx, source.deref(ctx).get_result(0))?;
        if result.layout != CuteScaledLayoutAttr::Full {
            return Err("cute.scaled_view_make must produce Full layout".into());
        }

        let i64_type = MlirType::Integer(64);
        let (zero_result, zero_use) = fresh(session, i64_type);
        let zero = integer_constant(zero_result, 0, &input.location)?;
        let mut operations = vec![zero];
        operations.extend(build_aggregate(
            session,
            input.results[0].clone(),
            vec![
                input.operands[0].clone(),
                input.operands[1].clone(),
                zero_use.clone(),
                zero_use.clone(),
                zero_use,
            ],
            &input.location,
        )?);
        Ok(operations)
    }
}

struct ScaledViewRowTranslation;

impl OperationTranslation for ScaledViewRowTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.scaled_view_row", &input, 1, 3)?;
        CuteScaledViewRowOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        let result = scaled_view(ctx, source.deref(ctx).get_result(0))?;
        if result.layout != CuteScaledLayoutAttr::Row {
            return Err("cute.scaled_view_row must produce Row layout".into());
        }

        let (partial_result, partial_use) = fresh(session, input.results[0].ty.clone());
        let batch = insert_value(
            partial_result,
            input.operands[0].clone(),
            input.operands[1].clone(),
            2,
            &input.location,
        )?;
        let row = insert_value(
            input.results[0].clone(),
            partial_use,
            input.operands[2].clone(),
            3,
            &input.location,
        )?;
        Ok(vec![batch, row])
    }
}

struct ScaledViewKTileTranslation;

impl OperationTranslation for ScaledViewKTileTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.scaled_view_k_tile", &input, 1, 2)?;
        CuteScaledViewKTileOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        let result = scaled_view(ctx, source.deref(ctx).get_result(0))?;
        if result.layout != CuteScaledLayoutAttr::KTile(GEMV_K_TILE_WIDTH) {
            return Err("cute.scaled_view_k_tile must produce KTile<64>".into());
        }
        Ok(vec![insert_value(
            input.results[0].clone(),
            input.operands[0].clone(),
            input.operands[1].clone(),
            4,
            &input.location,
        )?])
    }
}

struct ScaledViewLoadTranslation;

impl OperationTranslation for ScaledViewLoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.scaled_view_load", &input, 1, 1)?;
        let source_op = CuteScaledViewLoadOp::wrap(source);
        source_op.verify(ctx).map_err(stringify)?;
        let value_alignment = source_op
            .promised_value_alignment(ctx)
            .ok_or_else(|| "cute.scaled_view_load has no value alignment promise".to_owned())?;
        let scale_alignment = source_op
            .promised_scale_alignment(ctx)
            .ok_or_else(|| "cute.scaled_view_load has no scale alignment promise".to_owned())?;
        if value_alignment < VALUE_BYTES_PER_LOAD || scale_alignment < SCALE_BYTES_PER_TILE {
            return Err("K=64 GEMV load needs value alignment >=16 and scale alignment >=4".into());
        }

        let source_fragment = fragment(ctx, source.deref(ctx).get_result(0))?;
        let source_view = source_fragment
            .source_view(ctx)
            .ok_or_else(|| "GEMV fragment lost its scaled-view provenance".to_owned())?;
        if source_view.layout != CuteScaledLayoutAttr::KTile(GEMV_K_TILE_WIDTH) {
            return Err("GEMV fragment must come from KTile<64>".into());
        }

        emit_scaled_k64_load(
            session,
            input.operands[0].clone(),
            input.results[0].clone(),
            value_alignment,
            scale_alignment,
            &input.location,
        )
    }
}

struct DotTranslation;

impl OperationTranslation for DotTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.dot", &input, 1, 3)?;
        CuteDotOp::wrap(source).verify(ctx).map_err(stringify)?;

        let source_ref = source.deref(ctx);
        let matrix = fragment(ctx, source_ref.get_operand(0))?
            .source_view(ctx)
            .ok_or_else(|| "cute.dot matrix fragment lost provenance".to_owned())?;
        let vector = fragment(ctx, source_ref.get_operand(1))?
            .source_view(ctx)
            .ok_or_else(|| "cute.dot vector fragment lost provenance".to_owned())?;
        if matrix.role != CuteTensorRoleAttr::Mkl || vector.role != CuteTensorRoleAttr::Nkl {
            return Err("cute.dot requires Mkl on the left and Nkl on the right".into());
        }
        drop(source_ref);

        emit_ordered_k64_dot(
            session,
            input.operands[0].clone(),
            input.operands[1].clone(),
            input.operands[2].clone(),
            input.results[0].clone(),
            &input.location,
        )
    }
}

fn emit_scaled_k64_load(
    session: &mut TranslationSession<'_>,
    scaled: MlirValueUse,
    final_result: MlirResult,
    value_alignment: u64,
    scale_alignment: u64,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    let tensor_type = tensor_carrier_type()?;
    let pointer_type = llvm_pointer_type(0)?;
    let i64_type = MlirType::Integer(64);
    let mut operations = vec![];

    let (values_result, values) = fresh(session, tensor_type.clone());
    operations.push(extract_value(values_result, scaled.clone(), 0, location)?);
    let (scales_result, scales) = fresh(session, tensor_type);
    operations.push(extract_value(scales_result, scaled.clone(), 1, location)?);
    let (batch_result, batch) = fresh(session, i64_type.clone());
    operations.push(extract_value(batch_result, scaled.clone(), 2, location)?);
    let (row_result, row) = fresh(session, i64_type.clone());
    operations.push(extract_value(row_result, scaled.clone(), 3, location)?);
    let (tile_result, tile) = fresh(session, i64_type.clone());
    operations.push(extract_value(tile_result, scaled, 4, location)?);

    let (value_pointer_result, value_pointer) = fresh(session, pointer_type.clone());
    operations.push(extract_value(
        value_pointer_result,
        values.clone(),
        0,
        location,
    )?);
    let (value_rows_result, value_rows) = fresh(session, i64_type.clone());
    operations.push(extract_value(
        value_rows_result,
        values.clone(),
        2,
        location,
    )?);
    let (value_k_result, value_k) = fresh(session, i64_type.clone());
    operations.push(extract_value(value_k_result, values, 3, location)?);

    let (scale_pointer_result, scale_pointer) = fresh(session, pointer_type);
    operations.push(extract_value(
        scale_pointer_result,
        scales.clone(),
        0,
        location,
    )?);
    let (scale_rows_result, scale_rows) = fresh(session, i64_type.clone());
    operations.push(extract_value(
        scale_rows_result,
        scales.clone(),
        2,
        location,
    )?);
    let (scale_k_result, scale_k) = fresh(session, i64_type.clone());
    operations.push(extract_value(scale_k_result, scales, 3, location)?);

    let two = emit_i64_constant(session, &mut operations, 2, location)?;
    let packed_k = emit_binary(
        session,
        &mut operations,
        "arith.divui",
        value_k,
        two,
        location,
    )?;
    let batch_rows = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        batch.clone(),
        value_rows,
        location,
    )?;
    let linear_row = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        batch_rows,
        row.clone(),
        location,
    )?;
    let value_row_base = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        linear_row,
        packed_k,
        location,
    )?;
    let value_stride = emit_i64_constant(session, &mut operations, VALUE_BYTES_PER_TILE, location)?;
    let value_tile_offset = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        tile.clone(),
        value_stride,
        location,
    )?;
    let value_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        value_row_base,
        value_tile_offset,
        location,
    )?;

    let rest_m = emit_ceil_div(
        session,
        &mut operations,
        scale_rows,
        SCALE_ATOM_ROWS,
        location,
    )?;
    let rest_k = emit_ceil_div(session, &mut operations, scale_k, SCALE_ATOM_K, location)?;
    let block_bytes = emit_i64_constant(session, &mut operations, SCALE_ATOM_BYTES, location)?;
    let batch_blocks = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        batch,
        rest_m,
        location,
    )?;
    let batch_blocks = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        batch_blocks,
        rest_k.clone(),
        location,
    )?;
    let batch_bytes = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        batch_blocks,
        block_bytes.clone(),
        location,
    )?;

    let shift_seven = emit_i64_constant(session, &mut operations, 7, location)?;
    let row_block = emit_binary(
        session,
        &mut operations,
        "arith.shrui",
        row.clone(),
        shift_seven,
        location,
    )?;
    let row_blocks = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        row_block,
        rest_k,
        location,
    )?;
    let row_block_bytes = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        row_blocks,
        block_bytes,
        location,
    )?;

    let mask_31 = emit_i64_constant(session, &mut operations, 31, location)?;
    let row_in_quadrant = emit_binary(
        session,
        &mut operations,
        "arith.andi",
        row.clone(),
        mask_31,
        location,
    )?;
    let shift_four = emit_i64_constant(session, &mut operations, 4, location)?;
    let row_in_quadrant_bytes = emit_binary(
        session,
        &mut operations,
        "arith.shli",
        row_in_quadrant,
        shift_four,
        location,
    )?;

    let shift_five = emit_i64_constant(session, &mut operations, 5, location)?;
    let quadrant = emit_binary(
        session,
        &mut operations,
        "arith.shrui",
        row,
        shift_five,
        location,
    )?;
    let mask_three = emit_i64_constant(session, &mut operations, 3, location)?;
    let quadrant = emit_binary(
        session,
        &mut operations,
        "arith.andi",
        quadrant,
        mask_three,
        location,
    )?;
    let shift_two = emit_i64_constant(session, &mut operations, 2, location)?;
    let quadrant_bytes = emit_binary(
        session,
        &mut operations,
        "arith.shli",
        quadrant,
        shift_two,
        location,
    )?;

    let scale_row_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        batch_bytes,
        row_block_bytes,
        location,
    )?;
    let scale_row_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        scale_row_base,
        row_in_quadrant_bytes,
        location,
    )?;
    let scale_row_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        scale_row_base,
        quadrant_bytes,
        location,
    )?;
    let scale_stride = emit_i64_constant(session, &mut operations, SCALE_ATOM_BYTES, location)?;
    let scale_tile_offset = emit_binary(
        session,
        &mut operations,
        "arith.muli",
        tile,
        scale_stride,
        location,
    )?;
    let scale_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        scale_row_base,
        scale_tile_offset,
        location,
    )?;

    let value_lo = emit_narrow_load(
        session,
        &mut operations,
        value_pointer.clone(),
        value_base.clone(),
        NarrowLoad::E2M1,
        value_alignment,
        location,
    )?;
    let sixteen = emit_i64_constant(session, &mut operations, VALUE_BYTES_PER_LOAD, location)?;
    let value_high_base = emit_binary(
        session,
        &mut operations,
        "arith.addi",
        value_base,
        sixteen,
        location,
    )?;
    let value_hi = emit_narrow_load(
        session,
        &mut operations,
        value_pointer,
        value_high_base,
        NarrowLoad::E2M1,
        value_alignment,
        location,
    )?;
    let scale_values = emit_narrow_load(
        session,
        &mut operations,
        scale_pointer,
        scale_base,
        NarrowLoad::UE8M0,
        scale_alignment,
        location,
    )?;

    operations.extend(build_aggregate(
        session,
        final_result,
        vec![value_lo, value_hi, scale_values],
        location,
    )?);
    Ok(operations)
}

#[derive(Clone, Copy)]
enum NarrowLoad {
    E2M1,
    UE8M0,
}

fn emit_narrow_load(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base_pointer: MlirValueUse,
    byte_offset: MlirValueUse,
    kind: NarrowLoad,
    alignment: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let pointer_type = llvm_pointer_type(0)?;
    let (offset_result, offset_pointer) = fresh(session, pointer_type);
    let mut gep = operation(
        "llvm.getelementptr",
        vec![offset_result],
        vec![base_pointer, byte_offset],
        location,
    )?;
    gep.properties.insert(
        "elem_type".into(),
        MlirAttribute::Type(MlirType::Integer(8)),
    );
    gep.properties.insert(
        "noWrapFlags".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(32),
        },
    );
    gep.properties.insert(
        "rawConstantIndices".into(),
        MlirAttribute::DenseI32Array(vec![i32::MIN]),
    );
    operations.push(gep);

    let global_pointer_type = llvm_pointer_type(1)?;
    let (global_result, global_pointer) = fresh(session, global_pointer_type);
    operations.push(operation(
        "llvm.addrspacecast",
        vec![global_result],
        vec![offset_pointer],
        location,
    )?);

    let (element, lanes, intermediate) = match kind {
        NarrowLoad::E2M1 => ("f4E2M1FN", VALUE_LANES_PER_LOAD, "f16"),
        NarrowLoad::UE8M0 => ("f8E8M0FNU", SCALE_BYTES_PER_TILE, "bf16"),
    };
    // ptrtoint intentionally accepts only the canonical byte-aligned pointer
    // form. State align<1> before strengthening the selected address below.
    let loose_pointer_type = MlirType::dialect(format!("!cute.ptr<{element}, gmem, align<1>>"))?;
    let (loose_result, loose_pointer) = fresh(session, loose_pointer_type);
    operations.push(operation(
        "builtin.unrealized_conversion_cast",
        vec![loose_result],
        vec![global_pointer],
        location,
    )?);

    let (address_result, address) = fresh(session, MlirType::Integer(64));
    operations.push(operation(
        "cute.ptrtoint",
        vec![address_result],
        vec![loose_pointer],
        location,
    )?);
    let constrained_type = MlirType::dialect(format!("!cute.i64<divby {alignment}>"))?;
    let (constrained_result, constrained) = fresh(session, constrained_type);
    operations.push(operation(
        "cute.assume",
        vec![constrained_result],
        vec![address],
        location,
    )?);
    let aligned_pointer_type =
        MlirType::dialect(format!("!cute.ptr<{element}, gmem, align<{alignment}>>"))?;
    let (aligned_result, aligned_pointer) = fresh(session, aligned_pointer_type);
    operations.push(operation(
        "cute.inttoptr",
        vec![aligned_result],
        vec![constrained],
        location,
    )?);

    let memref_type = MlirType::dialect(format!(
        "!cute.memref<{element}, gmem, align<{alignment}>, \"({lanes}):(1)\">"
    ))?;
    let (view_result, view) = fresh(session, memref_type);
    operations.push(operation(
        "cute.make_view",
        vec![view_result],
        vec![aligned_pointer],
        location,
    )?);

    let narrow_type = narrow_vector_type(kind);
    let (load_result, loaded) = fresh(session, narrow_type);
    let mut load = operation(
        "cute.memref.load_vec",
        vec![load_result],
        vec![view],
        location,
    )?;
    load.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![1, 0, 0]),
    );
    operations.push(load);

    let intermediate_type = vector_type(lanes, narrow_float(intermediate));
    let (convert_result, converted) = fresh(session, intermediate_type);
    operations.push(operation(
        "nvgpu.cvt_fpext",
        vec![convert_result],
        vec![loaded],
        location,
    )?);

    let f32_type = vector_type(lanes, MlirType::Float(MlirFloatType::F32));
    let (extended_result, extended) = fresh(session, f32_type);
    operations.push(operation(
        "arith.extf",
        vec![extended_result],
        vec![converted],
        location,
    )?);
    Ok(extended)
}

fn emit_ordered_k64_dot(
    session: &mut TranslationSession<'_>,
    matrix: MlirValueUse,
    vector: MlirValueUse,
    mut accumulator: MlirValueUse,
    final_result: MlirResult,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    let value_vector_type = vector_type(VALUE_LANES_PER_LOAD, MlirType::Float(MlirFloatType::F32));
    let scale_vector_type = vector_type(SCALE_BYTES_PER_TILE, MlirType::Float(MlirFloatType::F32));
    let mut operations = vec![];

    let mut matrix_values = Vec::with_capacity(2);
    let mut vector_values = Vec::with_capacity(2);
    for position in 0..2 {
        let (result, value) = fresh(session, value_vector_type.clone());
        operations.push(extract_value(result, matrix.clone(), position, location)?);
        matrix_values.push(value);
        let (result, value) = fresh(session, value_vector_type.clone());
        operations.push(extract_value(result, vector.clone(), position, location)?);
        vector_values.push(value);
    }
    let (matrix_scales_result, matrix_scales) = fresh(session, scale_vector_type.clone());
    operations.push(extract_value(matrix_scales_result, matrix, 2, location)?);
    let (vector_scales_result, vector_scales) = fresh(session, scale_vector_type);
    operations.push(extract_value(vector_scales_result, vector, 2, location)?);

    for group in 0..SCALE_BYTES_PER_TILE {
        let scale_a = emit_vector_extract(
            session,
            &mut operations,
            matrix_scales.clone(),
            group,
            location,
        )?;
        let scale_b = emit_vector_extract(
            session,
            &mut operations,
            vector_scales.clone(),
            group,
            location,
        )?;
        for lane_in_group in 0..16 {
            let lane = group * 16 + lane_in_group;
            let chunk = usize::try_from(lane / VALUE_LANES_PER_LOAD)
                .map_err(|_| "GEMV vector chunk does not fit usize")?;
            let chunk_lane = lane % VALUE_LANES_PER_LOAD;
            let a = emit_vector_extract(
                session,
                &mut operations,
                matrix_values[chunk].clone(),
                chunk_lane,
                location,
            )?;
            let b = emit_vector_extract(
                session,
                &mut operations,
                vector_values[chunk].clone(),
                chunk_lane,
                location,
            )?;
            let scaled_a = emit_binary(
                session,
                &mut operations,
                "arith.mulf",
                a,
                scale_a.clone(),
                location,
            )?;
            let product = emit_binary(
                session,
                &mut operations,
                "arith.mulf",
                scaled_a,
                b,
                location,
            )?;
            let scaled_b = emit_binary(
                session,
                &mut operations,
                "arith.mulf",
                product,
                scale_b.clone(),
                location,
            )?;
            let is_last = group + 1 == SCALE_BYTES_PER_TILE && lane_in_group == 15;
            let result = if is_last {
                final_result.clone()
            } else {
                fresh(session, MlirType::Float(MlirFloatType::F32)).0
            };
            let result_use = MlirValueUse {
                id: result.id,
                ty: result.ty.clone(),
            };
            operations.push(binary(
                "arith.addf",
                result,
                accumulator,
                scaled_b,
                location,
            )?);
            accumulator = result_use;
        }
    }
    Ok(operations)
}

fn emit_ceil_div(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    divisor: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let divisor = emit_i64_constant(session, operations, divisor, location)?;
    let quotient = emit_binary(
        session,
        operations,
        "arith.divui",
        value.clone(),
        divisor.clone(),
        location,
    )?;
    let remainder = emit_binary(session, operations, "arith.remui", value, divisor, location)?;
    let zero = emit_i64_constant(session, operations, 0, location)?;
    let (nonzero_result, nonzero) = fresh(session, MlirType::Integer(1));
    let mut compare = operation(
        "arith.cmpi",
        vec![nonzero_result],
        vec![remainder, zero],
        location,
    )?;
    compare.properties.insert(
        "predicate".into(),
        MlirAttribute::Integer {
            value: 1,
            ty: MlirType::Integer(64),
        },
    );
    operations.push(compare);
    let (round_up_result, round_up) = fresh(session, MlirType::Integer(64));
    operations.push(operation(
        "arith.extui",
        vec![round_up_result],
        vec![nonzero],
        location,
    )?);
    emit_binary(
        session,
        operations,
        "arith.addi",
        quotient,
        round_up,
        location,
    )
}

fn emit_i64_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, MlirType::Integer(64));
    operations.push(integer_constant(result, u64_bits(value), location)?);
    Ok(use_)
}

fn emit_binary(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    name: &str,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    if left.ty != right.ty {
        return Err(format!(
            "{name} operands have different types: {:?} and {:?}",
            left.ty, right.ty
        ));
    }
    let (result, use_) = fresh(session, left.ty.clone());
    operations.push(binary(name, result, left, right, location)?);
    Ok(use_)
}

fn emit_vector_extract(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    vector: MlirValueUse,
    position: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let MlirType::Vector { shape, element } = &vector.ty else {
        return Err(format!(
            "vector.extract expected a vector, got {:?}",
            vector.ty
        ));
    };
    if shape.as_slice() != [shape[0]] || position >= shape[0] {
        return Err(format!(
            "vector.extract position {position} is outside {:?}",
            shape
        ));
    }
    let (result, use_) = fresh(session, (**element).clone());
    let mut extract = operation("vector.extract", vec![result], vec![vector], location)?;
    extract.properties.insert(
        "static_position".into(),
        MlirAttribute::DenseI64Array(vec![position as i64]),
    );
    operations.push(extract);
    Ok(use_)
}

fn build_aggregate(
    session: &mut TranslationSession<'_>,
    final_result: MlirResult,
    fields: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    if fields.is_empty() {
        return Err("cannot build an empty GEMV aggregate".into());
    }
    let (poison_result, mut aggregate) = fresh(session, final_result.ty.clone());
    let poison = operation("llvm.mlir.poison", vec![poison_result], vec![], location)?;
    let mut operations = vec![poison];
    let field_count = fields.len();
    for (position, field) in fields.into_iter().enumerate() {
        let result = if position + 1 == field_count {
            final_result.clone()
        } else {
            fresh(session, final_result.ty.clone()).0
        };
        let next = MlirValueUse {
            id: result.id,
            ty: result.ty.clone(),
        };
        operations.push(insert_value(
            result,
            aggregate,
            field,
            position as i64,
            location,
        )?);
        aggregate = next;
    }
    Ok(operations)
}

fn extract_value(
    result: MlirResult,
    aggregate: MlirValueUse,
    position: i64,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("llvm.extractvalue", vec![result], vec![aggregate], location)?;
    operation.properties.insert(
        "position".into(),
        MlirAttribute::DenseI64Array(vec![position]),
    );
    Ok(operation)
}

fn insert_value(
    result: MlirResult,
    aggregate: MlirValueUse,
    field: MlirValueUse,
    position: i64,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation(
        "llvm.insertvalue",
        vec![result],
        vec![aggregate, field],
        location,
    )?;
    operation.properties.insert(
        "position".into(),
        MlirAttribute::DenseI64Array(vec![position]),
    );
    Ok(operation)
}

fn tensor_carrier_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.struct<(!llvm.ptr, i64, i64, i64)>")
}

fn scaled_carrier_type() -> Result<MlirType, String> {
    MlirType::dialect(
        "!llvm.struct<(!llvm.struct<(!llvm.ptr, i64, i64, i64)>, !llvm.struct<(!llvm.ptr, i64, i64, i64)>, i64, i64, i64)>",
    )
}

fn fragment_carrier_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.struct<(vector<32xf32>, vector<32xf32>, vector<4xf32>)>")
}

fn narrow_vector_type(kind: NarrowLoad) -> MlirType {
    match kind {
        NarrowLoad::E2M1 => vector_type(
            VALUE_LANES_PER_LOAD,
            MlirType::Float(MlirFloatType::Other("f4E2M1FN".into())),
        ),
        NarrowLoad::UE8M0 => vector_type(
            SCALE_BYTES_PER_TILE,
            MlirType::Float(MlirFloatType::Other("f8E8M0FNU".into())),
        ),
    }
}

fn narrow_float(spelling: &str) -> MlirType {
    match spelling {
        "f16" => MlirType::Float(MlirFloatType::F16),
        "bf16" => MlirType::Float(MlirFloatType::BF16),
        "f4E2M1FN" | "f8E8M0FNU" => MlirType::Float(MlirFloatType::Other(spelling.into())),
        _ => unreachable!("fixed GEMV conversion intermediate"),
    }
}

fn vector_type(size: u64, element: MlirType) -> MlirType {
    MlirType::Vector {
        shape: vec![size],
        element: Box::new(element),
    }
}

fn tensor_view(ctx: &Context, value: pliron::value::Value) -> Result<CuteTensorViewType, String> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<CuteTensorViewType>()
        .cloned()
        .ok_or_else(|| "expected cute.tensor_view value".to_owned())
}

fn scaled_view(ctx: &Context, value: pliron::value::Value) -> Result<CuteScaledViewType, String> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<CuteScaledViewType>()
        .cloned()
        .ok_or_else(|| "expected cute.scaled_view value".to_owned())
}

fn fragment(ctx: &Context, value: pliron::value::Value) -> Result<CuteFragmentType, String> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<CuteFragmentType>()
        .cloned()
        .ok_or_else(|| "expected cute.fragment value".to_owned())
}

fn require_gemv_tensor_view(ctx: &Context, view: &CuteTensorViewType) -> Result<(), String> {
    view.verify(ctx).map_err(stringify)?;
    if view.space != CuteTensorAddressSpaceAttr::Gmem
        || view.access != CuteTensorAccessAttr::ReadOnly
        || view.alignment.0 != 1
        || !matches!(view.role, CuteTensorRoleAttr::Mkl | CuteTensorRoleAttr::Nkl)
    {
        return Err(format!(
            "GEMV tensor must be a byte-aligned read-only Mkl/Nkl gmem view, got {:?}",
            view
        ));
    }
    match (view.format, view.layout) {
        (CuteTensorFormatAttr::E2M1, CuteTensorLayoutAttr::KMajor)
        | (CuteTensorFormatAttr::UE8M0, CuteTensorLayoutAttr::BlockScaleKMajor(16)) => Ok(()),
        other => Err(format!(
            "GEMV tensor needs E2M1/KMajor or UE8M0/BlockScaleKMajor<16>, got {other:?}"
        )),
    }
}

fn expect_shape(
    name: &str,
    input: &OperationInput,
    results: usize,
    operands: usize,
) -> Result<(), String> {
    if input.results.len() != results || input.operands.len() != operands {
        return Err(format!(
            "{name} expected {results} results and {operands} operands, got {} and {}",
            input.results.len(),
            input.operands.len()
        ));
    }
    if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty() {
        return Err(format!(
            "{name} unexpectedly retained successors, regions, or source-only attributes"
        ));
    }
    Ok(())
}

fn operation(
    name: &str,
    results: Vec<MlirResult>,
    operands: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.results = results;
    operation.operands = operands;
    operation.location = location.clone();
    Ok(operation)
}

fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

fn integer_constant(
    result: MlirResult,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("arith.constant", vec![result.clone()], vec![], location)?;
    operation.properties.insert(
        "value".into(),
        MlirAttribute::Integer {
            value,
            ty: result.ty,
        },
    );
    Ok(operation)
}

fn binary(
    name: &str,
    result: MlirResult,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(name, vec![result], vec![left, right], location)
}

fn llvm_pointer_type(address_space: u32) -> Result<MlirType, String> {
    if address_space == 0 {
        MlirType::dialect("!llvm.ptr")
    } else {
        MlirType::dialect(format!("!llvm.ptr<{address_space}>"))
    }
}

fn u64_bits(value: u64) -> i128 {
    i64::from_ne_bytes(value.to_ne_bytes()) as i128
}

fn stringify(error: impl std::fmt::Display) -> String {
    error.to_string()
}

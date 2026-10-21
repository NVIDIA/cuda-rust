/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use dialect_mir::ops::MirUndefOp;
use dialect_mir::types::MirFP16Type;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Typed placeholder operands shared by the operation-family verifier tests.
struct Inputs {
    /// Global pointer to packed matrix values.
    matrix_values: Value,
    /// Global pointer to matrix block scales.
    matrix_scales: Value,
    /// Global pointer to packed vector values.
    vector_values: Value,
    /// Global pointer to vector block scales.
    vector_scales: Value,
    /// Unsigned runtime extent/index value used by the fixture.
    size: Value,
    /// FP32 seed accumulator used by the dot-product fixture.
    acc: Value,
}

/// Create matrix/vector value and scale pointers plus the FP32 dot seed.
fn setup() -> (Context, Inputs) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let global_u8_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u8_ty, false, address_space::GLOBAL).into();
    let matrix_values = undef(&mut ctx, global_u8_ty);
    let matrix_scales = undef(&mut ctx, global_u8_ty);
    let vector_values = undef(&mut ctx, global_u8_ty);
    let vector_scales = undef(&mut ctx, global_u8_ty);
    let size = undef(&mut ctx, u64_ty);
    let acc = undef(&mut ctx, f32_ty);
    (
        ctx,
        Inputs {
            matrix_values,
            matrix_scales,
            vector_values,
            vector_scales,
            size,
            acc,
        },
    )
}

/// Return result zero from the fixture operation for connecting a semantic chain.
fn result(ctx: &Context, op: Ptr<Operation>) -> Value {
    op.deref(ctx).get_result(0)
}

/// Bind compatible fixture value and scale tensors for a requested matrix/vector coordinate role.
fn make_scaled(
    ctx: &mut Context,
    values: Value,
    scales: Value,
    size: Value,
    role: CuteTensorRoleAttr,
) -> (CuteScaledViewMakeOp, Value) {
    let values = CuteTensorMake2DOp::new_e2m1(ctx, values, size, size, size, role, 1);
    assert!(values.verify(ctx).is_ok());
    let values = result(ctx, values.get_operation());
    let scales = CuteTensorMake2DOp::new_ue8m0(ctx, scales, size, size, size, role, 1, 16);
    assert!(scales.verify(ctx).is_ok());
    let scales = result(ctx, scales.get_operation());
    let scaled = CuteScaledViewMakeOp::new(ctx, values, scales);
    let scaled_value = result(ctx, scaled.get_operation());
    (scaled, scaled_value)
}

/// Select a fixture row and K tile, then load its block-scaled register fragment.
fn load_fragment(ctx: &mut Context, scaled: Value, index: Value) -> (CuteScaledViewLoadOp, Value) {
    let row = CuteScaledViewRowOp::new(ctx, scaled, index, index);
    assert!(row.verify(ctx).is_ok());
    let row = result(ctx, row.get_operation());
    let tile = CuteScaledViewKTileOp::new(ctx, row, index);
    assert!(tile.verify(ctx).is_ok());
    let tile = result(ctx, tile.get_operation());
    let load = CuteScaledViewLoadOp::new(ctx, tile, 16, 4);
    let fragment = result(ctx, load.get_operation());
    (load, fragment)
}

/// The packed-value chain remains separate tensor/view/select/load/dot operations rather than a fused kernel call.
#[test]
fn gemv_story_stays_visible_as_six_small_operations() {
    let (mut ctx, inputs) = setup();
    let (matrix, matrix_value) = make_scaled(
        &mut ctx,
        inputs.matrix_values,
        inputs.matrix_scales,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
    );
    assert!(matrix.verify(&ctx).is_ok());
    let (vector, vector_value) = make_scaled(
        &mut ctx,
        inputs.vector_values,
        inputs.vector_scales,
        inputs.size,
        CuteTensorRoleAttr::Nkl,
    );
    assert!(vector.verify(&ctx).is_ok());

    let (matrix_load, matrix_fragment) = load_fragment(&mut ctx, matrix_value, inputs.size);
    let (vector_load, vector_fragment) = load_fragment(&mut ctx, vector_value, inputs.size);
    assert!(matrix_load.verify(&ctx).is_ok());
    assert!(vector_load.verify(&ctx).is_ok());

    let dot = CuteDotOp::new(&mut ctx, matrix_fragment, vector_fragment, inputs.acc);
    assert!(dot.verify(&ctx).is_ok());
    assert_eq!(
        dot.get_operation().deref(&ctx).get_result(0).get_type(&ctx),
        inputs.acc.get_type(&ctx)
    );
}

/// Values and their scales must address the same matrix or vector row coordinate.
#[test]
fn scaled_view_rejects_a_scale_tensor_with_the_wrong_role() {
    let (mut ctx, inputs) = setup();
    let values = CuteTensorMake2DOp::new_e2m1(
        &mut ctx,
        inputs.matrix_values,
        inputs.size,
        inputs.size,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
        1,
    );
    let values = result(&ctx, values.get_operation());
    let scales = CuteTensorMake2DOp::new_ue8m0(
        &mut ctx,
        inputs.matrix_scales,
        inputs.size,
        inputs.size,
        inputs.size,
        CuteTensorRoleAttr::Nkl,
        1,
        16,
    );
    let scales = result(&ctx, scales.get_operation());
    let scaled = CuteScaledViewMakeOp::new(&mut ctx, values, scales);

    assert!(
        scaled
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("share one Mkl or Nkl role")
    );
}

/// A dot pairs matrix M and vector N fragments; two matrix-role inputs are invalid.
#[test]
fn dot_rejects_two_mkl_fragments() {
    let (mut ctx, inputs) = setup();
    let (_, left) = make_scaled(
        &mut ctx,
        inputs.matrix_values,
        inputs.matrix_scales,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
    );
    let (_, right) = make_scaled(
        &mut ctx,
        inputs.vector_values,
        inputs.vector_scales,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
    );
    let (_, left) = load_fragment(&mut ctx, left, inputs.size);
    let (_, right) = load_fragment(&mut ctx, right, inputs.size);
    let dot = CuteDotOp::new(&mut ctx, left, right, inputs.acc);

    assert!(
        dot.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("Mkl and Nkl")
    );
}

/// The supported GEMV fragment requires the expected number of logical values per scale.
#[test]
fn scaled_view_rejects_the_wrong_scale_group_width() {
    let (mut ctx, inputs) = setup();
    let values = CuteTensorMake2DOp::new_e2m1(
        &mut ctx,
        inputs.matrix_values,
        inputs.size,
        inputs.size,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
        1,
    );
    let values = result(&ctx, values.get_operation());
    let scales = CuteTensorMake2DOp::new_ue8m0(
        &mut ctx,
        inputs.matrix_scales,
        inputs.size,
        inputs.size,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
        1,
        32,
    );
    let scales = result(&ctx, scales.get_operation());
    let scaled = CuteScaledViewMakeOp::new(&mut ctx, values, scales);

    assert!(
        scaled
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("BlockScaleKMajor<16>")
    );
}

/// Equal static tensor types do not excuse differing runtime value and scale extents.
#[test]
fn scaled_view_rejects_different_runtime_shapes() {
    let (mut ctx, inputs) = setup();
    let u64_ty = inputs.size.get_type(&ctx);
    let other_rows = undef(&mut ctx, u64_ty);
    let values = CuteTensorMake2DOp::new_e2m1(
        &mut ctx,
        inputs.matrix_values,
        inputs.size,
        inputs.size,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
        1,
    );
    let values = result(&ctx, values.get_operation());
    let scales = CuteTensorMake2DOp::new_ue8m0(
        &mut ctx,
        inputs.matrix_scales,
        inputs.size,
        other_rows,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
        1,
        16,
    );
    let scales = result(&ctx, scales.get_operation());
    let scaled = CuteScaledViewMakeOp::new(&mut ctx, values, scales);

    assert!(
        scaled
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("same logical rows and K")
    );
}

/// A selected tile load may carry stronger alignment than the original tensor base promises.
#[test]
fn load_owns_the_selected_address_alignment_promises() {
    let (mut ctx, inputs) = setup();
    let (_, scaled) = make_scaled(
        &mut ctx,
        inputs.matrix_values,
        inputs.matrix_scales,
        inputs.size,
        CuteTensorRoleAttr::Mkl,
    );
    let row = CuteScaledViewRowOp::new(&mut ctx, scaled, inputs.size, inputs.size);
    let row = result(&ctx, row.get_operation());
    let tile = CuteScaledViewKTileOp::new(&mut ctx, row, inputs.size);
    let tile = result(&ctx, tile.get_operation());

    let weak_values = CuteScaledViewLoadOp::new(&mut ctx, tile, 8, 4);
    assert!(
        weak_values
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("at least 16 bytes")
    );
    let weak_scales = CuteScaledViewLoadOp::new(&mut ctx, tile, 16, 2);
    assert!(
        weak_scales
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("at least 4 bytes")
    );
    let valid = CuteScaledViewLoadOp::new(&mut ctx, tile, 16, 4);
    assert_eq!(valid.promised_value_alignment(&ctx), Some(16));
    assert_eq!(valid.promised_scale_alignment(&ctx), Some(4));
    assert!(valid.verify(&ctx).is_ok());
}

/// A tensor constructor cannot claim shared storage for an ordinary global source pointer.
#[test]
fn tensor_make_2d_rejects_shared_result_facts_over_a_global_pointer() {
    let (mut ctx, inputs) = setup();
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
    let forged: TypeHandle = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        f16_ty,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        2,
        CuteTensorFormatAttr::E2M1,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::KMajor,
    )
    .into();
    let raw = Operation::new(
        &mut ctx,
        CuteTensorMake2DOp::get_concrete_op_info(),
        vec![forged],
        vec![inputs.matrix_values, inputs.size, inputs.size, inputs.size],
        vec![],
        0,
    );
    let make = CuteTensorMake2DOp::wrap(raw);

    let error = make.verify(&ctx).unwrap_err();
    assert!(
        error.to_string().contains("read-only global"),
        "unexpected verifier error: {error}"
    );
}

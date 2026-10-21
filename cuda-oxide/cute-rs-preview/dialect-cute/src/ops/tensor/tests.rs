/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use dialect_mir::ops::MirUndefOp;
use pliron::builtin::types::FP32Type;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Create a context and typed global/register pointers for the elementwise tensor-flow tests.
fn setup() -> (Context, TypeHandle, Value, Value, Value, Value) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let global_ro_ty: TypeHandle =
        MirPtrType::get(&mut ctx, f32_ty, false, address_space::GLOBAL).into();
    let global_rw_ty: TypeHandle =
        MirPtrType::get(&mut ctx, f32_ty, true, address_space::GLOBAL).into();
    let local_rw_ty: TypeHandle =
        MirPtrType::get(&mut ctx, f32_ty, true, address_space::LOCAL).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let global_ro = undef(&mut ctx, global_ro_ty);
    let global_rw = undef(&mut ctx, global_rw_ty);
    let local = undef(&mut ctx, local_rw_ty);
    let u64_value = undef(&mut ctx, u64_ty);
    (ctx, f32_ty, global_ro, global_rw, local, u64_value)
}

/// The supported make/divide/select/load/store/tail operations compose while preserving tensor metadata.
#[test]
fn all_eight_ops_verify_as_one_visible_tensor_flow() {
    let (mut ctx, f32_ty, global_ro, global_rw, local, index) = setup();

    let read = CuteTensorMakeOp::new(
        &mut ctx,
        global_ro,
        index,
        f32_ty,
        f32_ty,
        CuteTensorAccessAttr::ReadOnly,
        4,
    );
    assert!(read.verify(&ctx).is_ok());
    let read_contiguous = read.get_operation().deref(&ctx).get_result(0);
    let read_zipped = CuteTensorZippedDivideOp::new(&mut ctx, read_contiguous, 4);
    assert!(read_zipped.verify(&ctx).is_ok());
    let read_zipped_value = read_zipped.get_operation().deref(&ctx).get_result(0);
    let read_tile = CuteTensorSliceOp::new(&mut ctx, read_zipped_value, index);
    assert!(read_tile.verify(&ctx).is_ok());
    let read_tile_value = read_tile.get_operation().deref(&ctx).get_result(0);

    assert!(
        CuteTensorIsFullOp::new(&mut ctx, read_tile_value)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        CuteTensorBaseOp::new(&mut ctx, read_tile_value)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        CuteTensorLoadIntoOp::new(&mut ctx, read_tile_value, local, 16)
            .verify(&ctx)
            .is_ok()
    );

    let write = CuteTensorMakeOp::new(
        &mut ctx,
        global_rw,
        index,
        f32_ty,
        f32_ty,
        CuteTensorAccessAttr::ReadWrite,
        4,
    );
    let write_contiguous = write.get_operation().deref(&ctx).get_result(0);
    let write_zipped = CuteTensorZippedDivideOp::new(&mut ctx, write_contiguous, 4);
    let write_zipped_value = write_zipped.get_operation().deref(&ctx).get_result(0);
    let write_tile = CuteTensorSliceOp::new(&mut ctx, write_zipped_value, index);
    let write_tile_value = write_tile.get_operation().deref(&ctx).get_result(0);

    assert!(
        CuteTensorStoreFromOp::new(&mut ctx, local, write_tile_value, 16)
            .verify(&ctx)
            .is_ok()
    );
    let scalar = undef(&mut ctx, f32_ty);
    assert!(
        CuteTensorStoreElementAbsOp::new(&mut ctx, write_tile_value, index, scalar)
            .verify(&ctx)
            .is_ok()
    );
}

/// A transfer cannot strengthen view access or accept an invalid alignment promise.
#[test]
fn verifiers_reject_access_and_alignment_drift() {
    let (mut ctx, f32_ty, global_ro, _global_rw, local, index) = setup();
    let read = CuteTensorMakeOp::new(
        &mut ctx,
        global_ro,
        index,
        f32_ty,
        f32_ty,
        CuteTensorAccessAttr::ReadOnly,
        4,
    );
    let contiguous = read.get_operation().deref(&ctx).get_result(0);
    let zipped = CuteTensorZippedDivideOp::new(&mut ctx, contiguous, 4);
    let zipped_value = zipped.get_operation().deref(&ctx).get_result(0);
    let tile = CuteTensorSliceOp::new(&mut ctx, zipped_value, index);
    let tile_value = tile.get_operation().deref(&ctx).get_result(0);

    let weak_alignment = CuteTensorLoadIntoOp::new(&mut ctx, tile_value, local, 8);
    assert!(
        weak_alignment
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("sufficient power-of-two alignment")
    );

    let wrong_access = CuteTensorStoreFromOp::new(&mut ctx, local, tile_value, 16);
    assert!(
        wrong_access
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("must be writable")
    );
}

/// Dividing a view may change its selection layout but must preserve its storage facts.
#[test]
fn zipped_divide_rejects_result_fact_drift() {
    let (mut ctx, f32_ty, global_ro, _global_rw, _local, index) = setup();
    let read = CuteTensorMakeOp::new(
        &mut ctx,
        global_ro,
        index,
        f32_ty,
        f32_ty,
        CuteTensorAccessAttr::ReadOnly,
        4,
    );
    let input = read.get_operation().deref(&ctx).get_result(0);
    let wrong_result: TypeHandle = CuteTensorViewType::get(
        &mut ctx,
        f32_ty,
        f32_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadWrite,
        4,
        CuteTensorLayoutAttr::Zipped1D(4),
    )
    .into();
    let raw = Operation::new(
        &mut ctx,
        CuteTensorZippedDivideOp::get_concrete_op_info(),
        vec![wrong_result],
        vec![input],
        vec![],
        0,
    );
    let divide = CuteTensorZippedDivideOp::wrap(raw);
    divide.set_attr_zipped_layout(&ctx, CuteTensorLayoutAttr::Zipped1D(4));

    assert!(
        divide
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("may change only")
    );
}

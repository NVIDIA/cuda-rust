/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use crate::layout::{Layout, OffsetUnit, Swizzle};
use dialect_mir::ops::MirUndefOp;
use pliron::builtin::types::Signedness;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Return result zero from the fixture operation for connecting a semantic chain.
fn result(ctx: &Context, op: Ptr<Operation>) -> Value {
    op.deref(ctx).get_result(0)
}

/// Build the canonical packed-value transport tile used by the four-copy stage fixture.
fn byte_layout() -> ComposedLayout {
    let outer: Layout = "(128,64):(64,1)".parse().unwrap();
    ComposedLayout::new(Swizzle::new(2, 4, 3), 0, outer, OffsetUnit::Elements).unwrap()
}

/// Build the canonical scale transport tile used by the four-copy stage fixture.
fn scale_layout() -> ComposedLayout {
    let outer: Layout = "(1,256):(256,1)".parse().unwrap();
    ComposedLayout::new(Swizzle::IDENTITY, 0, outer, OffsetUnit::Elements).unwrap()
}

/// Typed placeholder operands shared by the operation-family verifier tests.
struct Inputs {
    /// Immutable global tensor-map pointer used by the fixture.
    descriptor: Value,
    /// Shared byte pointer carrying packed FP4 input data.
    byte_base: Value,
    /// Shared pointer carrying scale transport elements.
    scale_base: Value,
    /// Unsigned runtime extent/index value used by the fixture.
    size: Value,
    /// Shared completion-barrier pointer used by the copy fixture.
    barrier: Value,
    /// Unsigned byte storage type.
    u8_ty: TypeHandle,
    /// Unsigned 16-bit transport-carrier type.
    u16_ty: TypeHandle,
}

/// Create descriptor, shared byte/scale pointers, extents, and barriers for transport tests.
fn setup() -> (Context, Inputs) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
    let byte_base_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
    let scale_base_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u16_ty, true, address_space::SHARED).into();
    let barrier_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
    let descriptor = undef(&mut ctx, descriptor_ty);
    let byte_base = undef(&mut ctx, byte_base_ty);
    let scale_base = undef(&mut ctx, scale_base_ty);
    let size = undef(&mut ctx, u64_ty);
    let barrier = undef(&mut ctx, barrier_ty);
    (
        ctx,
        Inputs {
            descriptor,
            byte_base,
            scale_base,
            size,
            barrier,
            u8_ty,
            u16_ty,
        },
    )
}

/// Build a detached transport copy from locally verified source and destination views.
fn copy(
    ctx: &mut Context,
    descriptor: Value,
    base: Value,
    size: Value,
    barrier: Value,
    view: (TypeHandle, ComposedLayout, u64),
) -> CuteTmaCopy2dOp {
    let (element, layout, alignment) = view;
    let source = CuteTmaGmemViewOp::new(ctx, descriptor, element, layout.clone());
    assert!(source.verify(ctx).is_ok());
    let source = result(ctx, source.get_operation());
    let destination = CuteTmaSmemViewOp::new(ctx, base, size, element, layout, alignment);
    assert!(destination.verify(ctx).is_ok());
    let destination = result(ctx, destination.get_operation());
    CuteTmaCopy2dOp::new(ctx, source, destination, size, size, barrier)
}

/// A/B values and A/B scales remain four explicit copies with individually typed view pairs.
#[test]
fn four_stage_copies_stay_four_composable_operations() {
    let (mut ctx, inputs) = setup();
    let a = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.byte_base,
        inputs.size,
        inputs.barrier,
        (inputs.u8_ty, byte_layout(), 1024),
    );
    let b = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.byte_base,
        inputs.size,
        inputs.barrier,
        (inputs.u8_ty, byte_layout(), 1024),
    );
    let sfa = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.scale_base,
        inputs.size,
        inputs.barrier,
        (inputs.u16_ty, scale_layout(), 512),
    );
    let sfb = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.scale_base,
        inputs.size,
        inputs.barrier,
        (inputs.u16_ty, scale_layout(), 512),
    );

    for operation in [&a, &b, &sfa, &sfb] {
        assert!(operation.verify(&ctx).is_ok());
    }
}

/// TMA transports storage carriers without assigning packed-number decoding or matrix roles.
#[test]
fn transport_view_keeps_bytes_plain_and_role_free() {
    let (mut ctx, inputs) = setup();
    let source = CuteTmaGmemViewOp::new(&mut ctx, inputs.descriptor, inputs.u8_ty, byte_layout());
    assert!(source.verify(&ctx).is_ok());
    let source = result(&ctx, source.get_operation());
    let view = tma_view_of(&ctx, source).unwrap();
    let tensor = view.tensor_view(&ctx).unwrap();

    assert_eq!(tensor.format, CuteTensorFormatAttr::Plain);
    assert_eq!(tensor.role, CuteTensorRoleAttr::Generic);
    assert_eq!(tensor.space, CuteTensorAddressSpaceAttr::Gmem);
    assert_eq!(view.tile_elements(), Some(128 * 64));
    assert_eq!(view.tile_bytes(&ctx), Some(128 * 64));
}

/// Source and destination transport views must describe the same physical shared tile.
#[test]
fn copy_rejects_a_different_destination_layout() {
    let (mut ctx, inputs) = setup();
    let source = CuteTmaGmemViewOp::new(&mut ctx, inputs.descriptor, inputs.u8_ty, byte_layout());
    let source = result(&ctx, source.get_operation());
    let destination = CuteTmaSmemViewOp::new(
        &mut ctx,
        inputs.byte_base,
        inputs.size,
        inputs.u8_ty,
        ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            "(128,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap(),
        1024,
    );
    let destination = result(&ctx, destination.get_operation());
    let copy = CuteTmaCopy2dOp::new(
        &mut ctx,
        source,
        destination,
        inputs.size,
        inputs.size,
        inputs.barrier,
    );

    assert!(
        copy.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("same carrier and tile layout")
    );
}

/// Shared transport metadata cannot manufacture an address-space proof for a generic pointer.
#[test]
fn shared_view_rejects_a_generic_pointer() {
    let (mut ctx, inputs) = setup();
    let generic_base_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, inputs.u8_ty, true).into();
    let generic_base = undef(&mut ctx, generic_base_ty);
    let destination = CuteTmaSmemViewOp::new(
        &mut ctx,
        generic_base,
        inputs.size,
        inputs.u8_ty,
        byte_layout(),
        1024,
    );

    assert!(
        destination
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("CTA-shared")
    );
}

/// The destination base must satisfy the full swizzle-period alignment promise.
#[test]
fn shared_view_rejects_alignment_smaller_than_the_swizzle_phase() {
    let (mut ctx, inputs) = setup();
    let destination = CuteTmaSmemViewOp::new(
        &mut ctx,
        inputs.byte_base,
        inputs.size,
        inputs.u8_ty,
        byte_layout(),
        16,
    );

    assert!(
        destination
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("needs 512 bytes")
    );
}

/// Descriptor-backed transport preserves the immutable tensor-map pointer contract.
#[test]
fn global_view_rejects_a_mutable_descriptor_pointer() {
    let (mut ctx, inputs) = setup();
    let mutable_descriptor_ty: TypeHandle =
        MirPtrType::get_generic(&mut ctx, inputs.u8_ty, true).into();
    let mutable_descriptor = undef(&mut ctx, mutable_descriptor_ty);
    let source = CuteTmaGmemViewOp::new(&mut ctx, mutable_descriptor, inputs.u8_ty, byte_layout());

    assert!(
        source
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("immutable generic pointer")
    );
}

/// Completion requires mutable shared u64 barrier storage, not merely a pointer-shaped carrier.
#[test]
fn copy_rejects_generic_or_wrong_pointee_barriers() {
    let (mut ctx, inputs) = setup();
    let u64_ty = inputs.size.get_type(&ctx);
    let generic_barrier_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, true).into();
    let generic_barrier = undef(&mut ctx, generic_barrier_ty);
    let generic = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.byte_base,
        inputs.size,
        generic_barrier,
        (inputs.u8_ty, byte_layout(), 1024),
    );
    assert!(
        generic
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("canonical unsigned u64 Barrier")
    );

    let wrong_barrier_ty: TypeHandle =
        MirPtrType::get(&mut ctx, inputs.u8_ty, true, address_space::SHARED).into();
    let wrong_barrier = undef(&mut ctx, wrong_barrier_ty);
    let wrong = copy(
        &mut ctx,
        inputs.descriptor,
        inputs.byte_base,
        inputs.size,
        wrong_barrier,
        (inputs.u8_ty, byte_layout(), 1024),
    );
    assert!(
        wrong
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("canonical unsigned u64 Barrier")
    );
}

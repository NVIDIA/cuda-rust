/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use crate::attributes::{CuteTensorAccessAttr, CuteTensorAddressSpaceAttr};
use crate::types::CuteTensorViewType;
use dialect_mir::ops::MirUndefOp;
use pliron::builtin::types::Signedness;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Build the canonical swizzled packed-data shared layout for the MMA fixture.
fn data_placement() -> ComposedLayout {
    let outer: Layout = "(128,32):(32,1)".parse().unwrap();
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, outer, OffsetUnit::Elements).unwrap()
}

/// Storage interpretation and placement used to construct shared-view test types.
struct SmemTypeFacts {
    /// Logical interpretation of the bits in each physical storage element.
    format: CuteTensorFormatAttr,
    /// Logical M/N coordinate role assigned to the tensor.
    role: CuteTensorRoleAttr,
    /// Promised storage-base alignment, in bytes.
    alignment: u64,
    /// Static coordinate layout used to interpret the storage.
    layout: CuteTensorLayoutAttr,
    /// Mapping from logical coordinates to physical shared storage offsets.
    placement: ComposedLayout,
}

/// Construct a typed shared view from independently configurable storage facts for positive and negative cases.
fn smem_type(
    ctx: &mut Context,
    logical: TypeHandle,
    storage: TypeHandle,
    facts: SmemTypeFacts,
) -> TypeHandle {
    let tensor: TypeHandle = CuteTensorViewType::get_with_facts(
        ctx,
        logical,
        storage,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        facts.alignment,
        facts.format,
        facts.role,
        facts.layout,
    )
    .into();
    CuteSmemTensorType::get(ctx, tensor, facts.placement).into()
}

/// A mutually compatible set of shared views, register carriers, and one MMA plan.
struct TestTypes {
    /// Unsigned 32-bit index/register type.
    u32_ty: TypeHandle,
    /// Unsigned 64-bit extent/barrier type.
    u64_ty: TypeHandle,
    /// FP32 accumulator element type.
    f32_ty: TypeHandle,
    /// Pointer type for A's shared packed-value storage.
    a_ptr_ty: TypeHandle,
    /// Pointer type for B's shared packed-value storage.
    b_ptr_ty: TypeHandle,
    /// Pointer type for A's shared scale carriers.
    scale_a_ptr_ty: TypeHandle,
    /// Pointer type for B's shared scale carriers.
    scale_b_ptr_ty: TypeHandle,
    /// Semantic shared-view type for the A operand.
    a_view_ty: TypeHandle,
    /// Semantic shared-view type for the B operand.
    b_view_ty: TypeHandle,
    /// Semantic shared-view type for A scales.
    scale_a_view_ty: TypeHandle,
    /// Semantic shared-view type for B scales.
    scale_b_view_ty: TypeHandle,
    /// Ordinary aggregate type holding packed A registers.
    a_carrier_ty: TypeHandle,
    /// Ordinary aggregate type holding a stage of scale registers.
    scale_carrier_ty: TypeHandle,
    /// Ordinary aggregate type holding the FP32 accumulator registers.
    accumulator_ty: TypeHandle,
    /// Static layout and instruction contract used by this fixture or type.
    plan: CuteTiledMmaPlanAttr,
}

/// Build the compatible shared view types, register shapes, and tiled plan for compute tests.
fn setup() -> (Context, TestTypes) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
    let a_ptr_ty: TypeHandle =
        MirPtrType::get(&mut ctx, f16_ty, true, address_space::SHARED).into();
    let b_ptr_ty = a_ptr_ty;
    let scale_a_ptr_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u32_ty, true, address_space::SHARED).into();
    let scale_b_ptr_ty = scale_a_ptr_ty;
    let plan = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_placement());
    let a_view_ty = smem_type(
        &mut ctx,
        f32_ty,
        f16_ty,
        SmemTypeFacts {
            format: CuteTensorFormatAttr::E2M1,
            role: CuteTensorRoleAttr::Mkl,
            alignment: 16,
            layout: CuteTensorLayoutAttr::KMajor,
            placement: data_placement(),
        },
    );
    let b_view_ty = smem_type(
        &mut ctx,
        f32_ty,
        f16_ty,
        SmemTypeFacts {
            format: CuteTensorFormatAttr::E2M1,
            role: CuteTensorRoleAttr::Nkl,
            alignment: 16,
            layout: CuteTensorLayoutAttr::KMajor,
            placement: data_placement(),
        },
    );
    let scale_a_view_ty = smem_type(
        &mut ctx,
        f32_ty,
        u32_ty,
        SmemTypeFacts {
            format: CuteTensorFormatAttr::UE8M0,
            role: CuteTensorRoleAttr::Mkl,
            alignment: 4,
            layout: CuteTensorLayoutAttr::BlockScaleKMajor(32),
            placement: canonical_scale_placement(),
        },
    );
    let scale_b_view_ty = smem_type(
        &mut ctx,
        f32_ty,
        u32_ty,
        SmemTypeFacts {
            format: CuteTensorFormatAttr::UE8M0,
            role: CuteTensorRoleAttr::Nkl,
            alignment: 4,
            layout: CuteTensorLayoutAttr::BlockScaleKMajor(32),
            placement: canonical_scale_placement(),
        },
    );
    let a_carrier_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(&mut ctx, u32_ty, 8).into();
    let scale_carrier_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(&mut ctx, u32_ty, 10).into();
    let accumulator_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(&mut ctx, f32_ty, 64).into();
    (
        ctx,
        TestTypes {
            u32_ty,
            u64_ty,
            f32_ty,
            a_ptr_ty,
            b_ptr_ty,
            scale_a_ptr_ty,
            scale_b_ptr_ty,
            a_view_ty,
            b_view_ty,
            scale_a_view_ty,
            scale_b_view_ty,
            a_carrier_ty,
            scale_carrier_ty,
            accumulator_ty,
            plan,
        },
    )
}

/// Attach semantic shared-tensor metadata to fixture pointer/capacity carriers.
fn overlay(
    ctx: &mut Context,
    base_ty: TypeHandle,
    capacity_ty: TypeHandle,
    view_ty: TypeHandle,
) -> CuteSmemTensorOverlayOp {
    let base = undef(ctx, base_ty);
    let capacity = undef(ctx, capacity_ty);
    CuteSmemTensorOverlayOp::new(ctx, base, capacity, view_ty)
}

/// Compute operations retain Rust-shaped carriers and defer B loads until the tiled GEMM consumes them.
#[test]
fn visual_chain_keeps_carriers_ordinary_and_b_lazy() {
    let (mut ctx, types) = setup();
    assert!(types.plan.verify(&ctx).is_ok());
    assert_eq!(types.plan.compute_warps(), Some(8));
    assert_eq!(types.plan.scale_words_per_lane(), Some(10));
    assert_eq!(types.plan.a_registers_per_lane(), Some(8));
    assert_eq!(types.plan.accumulator_registers_per_lane(), Some(64));

    let a = overlay(&mut ctx, types.a_ptr_ty, types.u64_ty, types.a_view_ty);
    let b = overlay(&mut ctx, types.b_ptr_ty, types.u64_ty, types.b_view_ty);
    let scale_a = overlay(
        &mut ctx,
        types.scale_a_ptr_ty,
        types.u64_ty,
        types.scale_a_view_ty,
    );
    let scale_b = overlay(
        &mut ctx,
        types.scale_b_ptr_ty,
        types.u64_ty,
        types.scale_b_view_ty,
    );
    for operation in [&a, &b, &scale_a, &scale_b] {
        assert!(operation.verify(&ctx).is_ok());
    }

    let lane = undef(&mut ctx, types.u32_ty);
    let warp_m = undef(&mut ctx, types.u64_ty);
    let warp_n = undef(&mut ctx, types.u64_ty);
    let k_half = undef(&mut ctx, types.u64_ty);
    let mma = CuteTiledMmaSliceOp::new(&mut ctx, lane, types.plan.clone());
    assert!(mma.verify(&ctx).is_ok());

    let zero = undef(&mut ctx, types.f32_ty);
    let fill = CuteFragmentFillOp::new(
        &mut ctx,
        zero,
        types.accumulator_ty,
        types.plan.clone(),
        CuteMmaCarrierKindAttr::Accumulator,
    );
    assert!(fill.verify(&ctx).is_ok());

    let sliced_lane = mma.sliced_lane(&ctx);
    let scale_a_base = scale_a.base(&ctx);
    let scale_a_capacity = scale_a.capacity(&ctx);
    let scale_b_base = scale_b.base(&ctx);
    let scale_b_capacity = scale_b.capacity(&ctx);
    let load_scales = CuteMmaLoadScalesOp::new(
        &mut ctx,
        sliced_lane,
        scale_a_base,
        scale_a_capacity,
        scale_b_base,
        scale_b_capacity,
        warp_m,
        warp_n,
        types.scale_a_view_ty,
        types.scale_b_view_ty,
        types.scale_carrier_ty,
        types.plan.clone(),
    );
    assert!(load_scales.verify(&ctx).is_ok());
    let stage_scales = load_scales.scales(&ctx);
    let slice_scales = CuteFragmentSliceKOp::new(
        &mut ctx,
        stage_scales,
        k_half,
        types.scale_carrier_ty,
        types.plan.clone(),
    );
    assert!(slice_scales.verify(&ctx).is_ok());

    let a_base = a.base(&ctx);
    let a_capacity = a.capacity(&ctx);
    let load_a = CuteMmaLoadAOp::new(
        &mut ctx,
        sliced_lane,
        a_base,
        a_capacity,
        warp_m,
        k_half,
        types.a_view_ty,
        types.a_carrier_ty,
        types.plan.clone(),
    );
    assert!(load_a.verify(&ctx).is_ok());

    let b_base = b.base(&ctx);
    let b_capacity = b.capacity(&ctx);
    let partition_b = CuteMmaPartitionBOp::new(
        &mut ctx,
        b_base,
        b_capacity,
        warp_n,
        k_half,
        types.b_view_ty,
        types.plan.clone(),
    );
    assert!(partition_b.verify(&ctx).is_ok());
    {
        let partition_operation = partition_b.get_operation().deref(&ctx);
        assert_eq!(partition_operation.get_num_results(), 4);
        assert!((0..4).all(|index| {
            let value = partition_operation.get_result(index);
            mma_carrier_shape(&ctx, value.get_type(&ctx)).is_none()
        }));
    }

    let a_fragment = load_a.fragment(&ctx);
    let partition_base = partition_b.base(&ctx);
    let partition_capacity = partition_b.capacity(&ctx);
    let partition_warp_n = partition_b.warp_n(&ctx);
    let partition_k_half = partition_b.k_half(&ctx);
    let selected_scales = slice_scales.scales(&ctx);
    let initial_accumulator = fill.fragment(&ctx);
    let gemm = CuteTiledGemmOp::new(
        &mut ctx,
        sliced_lane,
        a_fragment,
        partition_base,
        partition_capacity,
        partition_warp_n,
        partition_k_half,
        selected_scales,
        initial_accumulator,
        types.b_view_ty,
        types.plan,
    );
    assert!(gemm.verify(&ctx).is_ok());
    assert_eq!(
        mma_carrier_shape(&ctx, gemm.accumulator(&ctx).get_type(&ctx)),
        Some(MmaCarrierShape::f32(64))
    );
}

/// The compute ABI requires a typed lazy B partition, preventing an eager register aggregate from replacing it.
#[test]
fn tiled_gemm_rejects_an_eager_or_untyped_b_fragment() {
    let (mut ctx, types) = setup();
    let lane = undef(&mut ctx, types.u32_ty);
    let a = undef(&mut ctx, types.a_carrier_ty);
    let eager_b = undef(&mut ctx, types.a_carrier_ty);
    let capacity = undef(&mut ctx, types.u64_ty);
    let warp_n = undef(&mut ctx, types.u64_ty);
    let k_half = undef(&mut ctx, types.u64_ty);
    let scales = undef(&mut ctx, types.scale_carrier_ty);
    let accumulator = undef(&mut ctx, types.accumulator_ty);
    let gemm = CuteTiledGemmOp::new(
        &mut ctx,
        lane,
        a,
        eager_b,
        capacity,
        warp_n,
        k_half,
        scales,
        accumulator,
        types.b_view_ty,
        types.plan,
    );
    assert!(
        gemm.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("lazy B")
    );
}

/// Static view metadata lets flattened pointer/capacity carriers retain their meaning after ordinary aggregate forwarding.
#[test]
fn typed_consumers_accept_flattened_carriers_after_aggregate_forwarding() {
    let (mut ctx, types) = setup();
    let lane = undef(&mut ctx, types.u32_ty);
    let warp_m = undef(&mut ctx, types.u64_ty);
    let warp_n = undef(&mut ctx, types.u64_ty);
    let k_half = undef(&mut ctx, types.u64_ty);
    let a_base = undef(&mut ctx, types.a_ptr_ty);
    let a_capacity = undef(&mut ctx, types.u64_ty);
    let load_a = CuteMmaLoadAOp::new(
        &mut ctx,
        lane,
        a_base,
        a_capacity,
        warp_m,
        k_half,
        types.a_view_ty,
        types.a_carrier_ty,
        types.plan.clone(),
    );
    assert!(load_a.verify(&ctx).is_ok());

    let a_fragment = load_a.fragment(&ctx);
    let b_base = undef(&mut ctx, types.b_ptr_ty);
    let b_capacity = undef(&mut ctx, types.u64_ty);
    let scales = undef(&mut ctx, types.scale_carrier_ty);
    let accumulator = undef(&mut ctx, types.accumulator_ty);
    let gemm = CuteTiledGemmOp::new(
        &mut ctx,
        lane,
        a_fragment,
        b_base,
        b_capacity,
        warp_n,
        k_half,
        scales,
        accumulator,
        types.b_view_ty,
        types.plan,
    );
    assert!(gemm.verify(&ctx).is_ok());
}

/// A/B scale views cannot exchange coordinate roles even when their storage shapes match.
#[test]
fn scale_load_rejects_swapped_m_and_n_roles() {
    let (mut ctx, types) = setup();
    let scale_a_wrong = overlay(
        &mut ctx,
        types.scale_a_ptr_ty,
        types.u64_ty,
        types.scale_b_view_ty,
    );
    let scale_b = overlay(
        &mut ctx,
        types.scale_b_ptr_ty,
        types.u64_ty,
        types.scale_b_view_ty,
    );
    let lane = undef(&mut ctx, types.u32_ty);
    let warp_m = undef(&mut ctx, types.u64_ty);
    let warp_n = undef(&mut ctx, types.u64_ty);
    let scale_a_base = scale_a_wrong.base(&ctx);
    let scale_a_capacity = scale_a_wrong.capacity(&ctx);
    let scale_b_base = scale_b.base(&ctx);
    let scale_b_capacity = scale_b.capacity(&ctx);
    let load = CuteMmaLoadScalesOp::new(
        &mut ctx,
        lane,
        scale_a_base,
        scale_a_capacity,
        scale_b_base,
        scale_b_capacity,
        warp_m,
        warp_n,
        types.scale_b_view_ty,
        types.scale_b_view_ty,
        types.scale_carrier_ty,
        types.plan,
    );
    assert!(
        load.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("A must be")
    );
}

/// Unsupported tiled plans and incorrect register aggregate sizes fail before backend selection.
#[test]
fn plan_and_carrier_verifiers_fail_closed() {
    let (mut ctx, types) = setup();
    let mut bad_plan = types.plan.clone();
    bad_plan.b_load_group = 3;
    assert!(
        bad_plan
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("B load group")
    );

    let mut wrong_ownership = types.plan.clone();
    wrong_ownership.m_ownership = crate::attributes::CuteLayoutAttr(
        "(4,2):(2,1)".parse().expect("dense alternate map is valid"),
    );
    assert!(wrong_ownership.verify(&ctx).is_ok());
    let lane = undef(&mut ctx, types.u32_ty);
    let slice = CuteTiledMmaSliceOp::new(&mut ctx, lane, wrong_ownership);
    assert!(
        slice
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("ownership maps")
    );

    let wrong_accumulator_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(&mut ctx, types.f32_ty, 63).into();
    let zero = undef(&mut ctx, types.f32_ty);
    let fill = CuteFragmentFillOp::new(
        &mut ctx,
        zero,
        wrong_accumulator_ty,
        types.plan,
        CuteMmaCarrierKindAttr::Accumulator,
    );
    assert!(
        fill.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("carrier shape")
    );
}

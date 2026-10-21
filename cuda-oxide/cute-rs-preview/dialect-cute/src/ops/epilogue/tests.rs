/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use crate::attributes::{CuteEpiloguePlanAttr, CuteTiledMmaPlanAttr};
use crate::ops::gemm_tma::{CuteTmaGmemViewOp, CuteTmaSmemViewOp};
use crate::types::CuteEpilogueTileType;
use dialect_mir::ops::MirUndefOp;
use dialect_mir::types::{MirArrayType, MirStructType};
use pliron::builtin::types::FP32Type;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Build the canonical shared placement consumed by the output-tile fixture.
fn data_layout() -> ComposedLayout {
    let inner: Layout = "(128,32):(32,1)".parse().unwrap();
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements).unwrap()
}

/// Ordinary runtime carriers and static types for a two-half epilogue test.
struct Fixture {
    /// Shared storage pointer used by this fixture.
    base: Value,
    /// Warp index used by the output-tile fixture.
    warp: Value,
    /// Lane index within one 32-thread warp.
    lane: Value,
    /// FP32 accumulator carrier used by the compute or output operation.
    accumulator: Value,
    /// Storage capacity expressed as an unsigned element count.
    capacity: Value,
    /// Runtime tile-row coordinate supplied to the copy.
    row: Value,
    /// Runtime tile-column coordinate supplied to the copy.
    column: Value,
    /// Immutable global tensor-map pointer used by the fixture.
    descriptor: Value,
    /// FP16 shared-storage type.
    f16_ty: TypeHandle,
    /// Static type describing the supported shared output tile.
    tile_ty: TypeHandle,
}

/// Build shared output, accumulator, coordinates, and static tile metadata for output tests.
fn setup() -> (Context, Fixture) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
    let base_ty: TypeHandle = MirPtrType::get(&mut ctx, f16_ty, true, address_space::SHARED).into();
    let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
    let marker: TypeHandle =
        MirStructType::get(&mut ctx, "PhantomData".into(), vec![], vec![]).into();
    let registers: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 64).into();
    let accumulator_ty: TypeHandle = MirStructType::get(
        &mut ctx,
        "Mxf4AccumulatorTile2x8".into(),
        vec![],
        vec![registers, marker],
    )
    .into();
    let mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_layout());
    let plan = CuteEpiloguePlanAttr::sm120_mxf4_128x128(mma);
    let tile_ty: TypeHandle = CuteEpilogueTileType::get(&mut ctx, f16_ty, plan).into();

    let base = undef(&mut ctx, base_ty);
    let warp = undef(&mut ctx, u64_ty);
    let lane = undef(&mut ctx, u32_ty);
    let accumulator = undef(&mut ctx, accumulator_ty);
    let capacity = undef(&mut ctx, u64_ty);
    let row = undef(&mut ctx, u64_ty);
    let column = undef(&mut ctx, u64_ty);
    let descriptor = undef(&mut ctx, descriptor_ty);
    (
        ctx,
        Fixture {
            base,
            warp,
            lane,
            accumulator,
            capacity,
            row,
            column,
            descriptor,
            f16_ty,
            tile_ty,
        },
    )
}

/// Output overlay, warp selection, synchronization, and half selection preserve ordinary runtime carriers.
#[test]
fn visual_epilogue_chain_keeps_only_ordinary_runtime_carriers() {
    let (mut ctx, fixture) = setup();
    let overlay = CuteEpilogueSmemOverlayOp::new(&mut ctx, fixture.base, fixture.tile_ty);
    assert!(overlay.verify(&ctx).is_ok());
    let overlay_base = overlay.base(&ctx);
    let slice = CuteEpilogueWarpSliceOp::new(
        &mut ctx,
        overlay_base,
        fixture.warp,
        fixture.lane,
        fixture.tile_ty,
    );
    assert!(slice.verify(&ctx).is_ok());
    let slice_base = slice.base(&ctx);
    let slice_warp = slice.warp_id(&ctx);
    let slice_lane = slice.lane(&ctx);
    let store = CuteEpilogueStoreFragmentOp::new(
        &mut ctx,
        slice_base,
        slice_warp,
        slice_lane,
        fixture.accumulator,
        fixture.tile_ty,
    );
    assert!(store.verify(&ctx).is_ok());

    let barrier = CuteCountedCtaBarrierAttr::new(2, 0, 8, 9, 32);
    let reusable = CuteEpilogueSyncOp::new(
        &mut ctx,
        overlay_base,
        fixture.tile_ty,
        barrier,
        CuteEpilogueSyncPhaseAttr::Reusable,
    );
    let ready = CuteEpilogueSyncOp::new(
        &mut ctx,
        overlay_base,
        fixture.tile_ty,
        barrier,
        CuteEpilogueSyncPhaseAttr::ReadyForTma,
    );
    assert!(reusable.verify(&ctx).is_ok());
    assert!(ready.verify(&ctx).is_ok());
    assert_eq!(barrier.participant_threads(), Some(256));
    assert_eq!(barrier.excluded_warps(), Some(1));

    let left = CuteEpilogueHalfOp::new(
        &mut ctx,
        overlay_base,
        fixture.tile_ty,
        CuteEpilogueHalfAttr(0),
    );
    let right = CuteEpilogueHalfOp::new(
        &mut ctx,
        overlay_base,
        fixture.tile_ty,
        CuteEpilogueHalfAttr(1),
    );
    assert!(left.verify(&ctx).is_ok());
    assert!(right.verify(&ctx).is_ok());
    assert_eq!(
        left.capacity(&ctx).get_type(&ctx),
        fixture.capacity.get_type(&ctx)
    );
}

/// Acquire/commit/tail represent issuer-local hardware state with no fabricated SSA queue object.
#[test]
fn store_pipeline_has_no_runtime_handle() {
    let (mut ctx, _) = setup();
    let pipeline = CuteTmaStorePipelineAttr::new(1);
    let acquire = CuteTmaStoreAcquireOp::new(&mut ctx, pipeline);
    let commit = CuteTmaStoreCommitOp::new(&mut ctx, pipeline);
    let tail = CuteTmaStoreTailOp::new(&mut ctx, pipeline);
    assert!(acquire.verify(&ctx).is_ok());
    assert!(commit.verify(&ctx).is_ok());
    assert!(tail.verify(&ctx).is_ok());
    for operation in [
        acquire.get_operation(),
        commit.get_operation(),
        tail.get_operation(),
    ] {
        assert_eq!(operation.deref(&ctx).get_num_operands(), 0);
        assert_eq!(operation.deref(&ctx).get_num_results(), 0);
    }

    assert!(
        CuteTmaStoreAcquireOp::new(&mut ctx, CuteTmaStorePipelineAttr::new(2))
            .verify(&ctx)
            .is_err()
    );
}

/// Shared output halves and writable global transport views must agree on storage and layout.
#[test]
fn semantic_s2g_store_requires_matching_writable_views() {
    let (mut ctx, fixture) = setup();
    let source = CuteTmaSmemViewOp::new(
        &mut ctx,
        fixture.base,
        fixture.capacity,
        fixture.f16_ty,
        current_half_layout(),
        1024,
    );
    let destination = CuteTmaGmemViewOp::new_destination(
        &mut ctx,
        fixture.descriptor,
        fixture.f16_ty,
        current_half_layout(),
    );
    assert!(source.verify(&ctx).is_ok());
    assert!(destination.verify(&ctx).is_ok());
    let source_view = source.get_operation().deref(&ctx).get_result(0);
    let destination_view = destination.get_operation().deref(&ctx).get_result(0);
    let store = CuteTmaStore2dSemanticOp::new(
        &mut ctx,
        source_view,
        destination_view,
        fixture.row,
        fixture.column,
    );
    assert!(store.verify(&ctx).is_ok());

    let read_only = CuteTmaGmemViewOp::new(
        &mut ctx,
        fixture.descriptor,
        fixture.f16_ty,
        current_half_layout(),
    );
    let read_only_view = read_only.get_operation().deref(&ctx).get_result(0);
    let wrong_direction = CuteTmaStore2dSemanticOp::new(
        &mut ctx,
        source_view,
        read_only_view,
        fixture.row,
        fixture.column,
    );
    assert!(wrong_direction.verify(&ctx).is_err());
}

/// Epilogue checks reject mismatched barrier participation, accumulator shape, and out-of-range half selection.
#[test]
fn verifiers_reject_wrong_barrier_fragment_and_half() {
    let (mut ctx, fixture) = setup();
    let wrong_barrier = CuteEpilogueSyncOp::new(
        &mut ctx,
        fixture.base,
        fixture.tile_ty,
        CuteCountedCtaBarrierAttr::new(2, 0, 9, 9, 32),
        CuteEpilogueSyncPhaseAttr::Reusable,
    );
    assert!(wrong_barrier.verify(&ctx).is_err());

    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let short_accumulator_ty: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 63).into();
    let short_accumulator = undef(&mut ctx, short_accumulator_ty);
    let wrong_fragment = CuteEpilogueStoreFragmentOp::new(
        &mut ctx,
        fixture.base,
        fixture.warp,
        fixture.lane,
        short_accumulator,
        fixture.tile_ty,
    );
    assert!(wrong_fragment.verify(&ctx).is_err());

    let wrong_half = CuteEpilogueHalfOp::new(
        &mut ctx,
        fixture.base,
        fixture.tile_ty,
        CuteEpilogueHalfAttr(2),
    );
    assert!(wrong_half.verify(&ctx).is_err());
}

/// Unsupported output-plan geometry or storage interpretation is rejected independently of backend choice.
#[test]
fn static_plan_and_tile_fail_closed_before_backend_selection() {
    let (mut ctx, fixture) = setup();
    let mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_layout());
    let valid = CuteEpiloguePlanAttr::sm120_mxf4_128x128(mma.clone());
    assert!(valid.verify(&ctx).is_ok());

    let mut one_half = valid.clone();
    one_half.halves = 1;
    assert!(one_half.verify(&ctx).is_err());

    let alternate_data =
        ComposedLayout::from_layout("(128,32):(32,1)".parse().unwrap(), OffsetUnit::Elements);
    let alternate_mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(alternate_data);
    assert!(
        CuteEpiloguePlanAttr::sm120_mxf4_128x128(alternate_mma)
            .verify(&ctx)
            .is_err()
    );

    let wrong_storage: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let wrong_tile = CuteEpilogueTileType::get(
        &mut ctx,
        wrong_storage,
        CuteEpiloguePlanAttr::sm120_mxf4_128x128(mma),
    );
    assert!(wrong_tile.deref(&ctx).verify(&ctx).is_err());
    assert!(fixture.tile_ty.deref(&ctx).verify(&ctx).is_ok());

    assert!(CuteTmaStorePipelineAttr::new(0).verify(&ctx).is_err());
    assert!(
        CuteCountedCtaBarrierAttr::new(16, 0, 8, 9, 32)
            .verify(&ctx)
            .is_err()
    );
}

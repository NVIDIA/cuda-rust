/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared placement and asynchronous output for the SM120 GEMM epilogue.
//!
//! Runtime values stay ordinary MIR values. Static attributes explain the
//! shared tile and the store protocol:
//!
//! ```text
//! 64xf32 accumulator
//!         │
//!         ▼
//! epilogue_store_fragment ──► two 128x64 shared f16 halves
//!         │                              │
//!         ▼                              ▼
//! proxy fence + counted sync       two tma_store_2d
//!                                          │
//!                                          ▼
//!                                  one committed group
//! ```
//!
//! The TMA store queue is hardware state owned by the issuing thread. Its
//! acquire, commit, and tail operations therefore have no fake SSA handle or
//! loop-carried counter. Their order in the CFG is the state.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::builtin::{
    attributes::TypeAttr,
    op_interfaces::{NOpdsInterface, NResultsInterface},
    types::{IntegerType, Signedness},
};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::{
    CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
    CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTmaStorePipelineAttr,
};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use crate::types::{CuteEpilogueTileType, CuteTmaViewType, MmaCarrierShape, mma_carrier_shape};

fn is_u32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned())
}

fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

fn epilogue_tile_of_type(ctx: &Context, ty: TypeHandle) -> Option<CuteEpilogueTileType> {
    ty.deref(ctx)
        .downcast_ref::<CuteEpilogueTileType>()
        .cloned()
}

fn tma_view_of(ctx: &Context, value: Value) -> Option<CuteTmaViewType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTmaViewType>()
        .cloned()
}

fn shared_pointer_matches(ctx: &Context, value: Value, storage: TypeHandle) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some_and(|pointer| {
            pointer.pointee == storage
                && pointer.is_mutable
                && pointer.address_space == address_space::SHARED
        })
}

fn checked_tile(
    tile: Option<CuteEpilogueTileType>,
    ctx: &Context,
) -> Result<CuteEpilogueTileType, String> {
    let tile = tile.ok_or_else(|| "must carry a cute.epilogue_tile TypeAttr".to_owned())?;
    tile.verify(ctx)
        .map_err(|error| format!("carries an invalid epilogue tile: {error}"))?;
    Ok(tile)
}

fn current_half_layout() -> ComposedLayout {
    let inner: Layout = "(128,64):(64,1)"
        .parse()
        .expect("fixed epilogue half layout is valid");
    ComposedLayout::new(Swizzle::new(3, 3, 3), 0, inner, OffsetUnit::Elements)
        .expect("fixed epilogue half swizzle is valid")
}

/// Attach result-tile meaning to one shared pointer without changing it.
#[pliron_op(
    name = "cute.epilogue_smem_overlay",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (epilogue_overlay_tile: TypeAttr)
)]
pub struct CuteEpilogueSmemOverlayOp;

impl CuteEpilogueSmemOverlayOp {
    pub fn new(ctx: &mut Context, base: Value, tile: TypeHandle) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![base.get_type(ctx)],
                vec![base],
                vec![],
                0,
            ),
        };
        operation.set_attr_epilogue_overlay_tile(ctx, TypeAttr::new(tile));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn input_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_overlay_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }
}

impl Verify for CuteEpilogueSmemOverlayOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.epilogue_smem_overlay needs 1 operand and 1 result"
            );
        }
        let tile = match checked_tile(self.tile(ctx), ctx) {
            Ok(tile) => tile,
            Err(message) => return verify_err!(op.loc(), "cute.epilogue_smem_overlay {message}"),
        };
        if !shared_pointer_matches(ctx, self.input_base(ctx), tile.storage) {
            return verify_err!(
                op.loc(),
                "cute.epilogue_smem_overlay base must be a mutable CTA-shared f16 pointer"
            );
        }
        if self.base(ctx).get_type(ctx) != self.input_base(ctx).get_type(ctx) {
            return verify_err!(
                op.loc(),
                "cute.epilogue_smem_overlay must preserve the base pointer type"
            );
        }
        Ok(())
    }
}

/// Keep the pointer, warp, and lane selected by `get_slice` visible.
///
/// All three results are ordinary values and may be forwarded through the
/// Rust `Sm120EpilogueWarp128x128` aggregate.
#[pliron_op(
    name = "cute.epilogue_warp_slice",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<3>],
    attributes = (epilogue_warp_slice_tile: TypeAttr)
)]
pub struct CuteEpilogueWarpSliceOp;

impl CuteEpilogueWarpSliceOp {
    pub fn new(
        ctx: &mut Context,
        base: Value,
        warp_id: Value,
        lane: Value,
        tile: TypeHandle,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![
                    base.get_type(ctx),
                    warp_id.get_type(ctx),
                    lane.get_type(ctx),
                ],
                vec![base, warp_id, lane],
                vec![],
                0,
            ),
        };
        operation.set_attr_epilogue_warp_slice_tile(ctx, TypeAttr::new(tile));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn input_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn input_warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn input_lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(2)
    }

    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_warp_slice_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }
}

impl Verify for CuteEpilogueWarpSliceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 3 {
            return verify_err!(
                op.loc(),
                "cute.epilogue_warp_slice needs 3 operands and 3 results"
            );
        }
        let tile = match checked_tile(self.tile(ctx), ctx) {
            Ok(tile) => tile,
            Err(message) => return verify_err!(op.loc(), "cute.epilogue_warp_slice {message}"),
        };
        if !shared_pointer_matches(ctx, self.input_base(ctx), tile.storage)
            || !is_u64(ctx, self.input_warp_id(ctx))
            || !is_u32(ctx, self.input_lane(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.epilogue_warp_slice needs a shared f16 base, u64 warp ID, and u32 lane"
            );
        }
        if (0..3)
            .any(|index| op.get_result(index).get_type(ctx) != op.get_operand(index).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.epilogue_warp_slice must preserve all three carrier types"
            );
        }
        Ok(())
    }
}

/// Convert and place one warp's complete 2x8 accumulator tile in shared memory.
#[pliron_op(
    name = "cute.epilogue_store_fragment",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
    attributes = (epilogue_store_fragment_tile: TypeAttr)
)]
pub struct CuteEpilogueStoreFragmentOp;

impl CuteEpilogueStoreFragmentOp {
    pub fn new(
        ctx: &mut Context,
        base: Value,
        warp_id: Value,
        lane: Value,
        accumulator: Value,
        tile: TypeHandle,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![base, warp_id, lane, accumulator],
                vec![],
                0,
            ),
        };
        operation.set_attr_epilogue_store_fragment_tile(ctx, TypeAttr::new(tile));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    #[must_use]
    pub fn accumulator(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }

    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_store_fragment_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }
}

impl Verify for CuteEpilogueStoreFragmentOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.epilogue_store_fragment needs 4 operands and 0 results"
            );
        }
        let tile = match checked_tile(self.tile(ctx), ctx) {
            Ok(tile) => tile,
            Err(message) => {
                return verify_err!(op.loc(), "cute.epilogue_store_fragment {message}");
            }
        };
        if !shared_pointer_matches(ctx, self.base(ctx), tile.storage)
            || !is_u64(ctx, self.warp_id(ctx))
            || !is_u32(ctx, self.lane(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.epilogue_store_fragment needs a shared f16 base, u64 warp ID, and u32 lane"
            );
        }
        let expected = MmaCarrierShape::f32(
            tile.plan
                .tiled_mma
                .accumulator_registers_per_lane()
                .expect("verified epilogue plan has an accumulator size"),
        );
        if mma_carrier_shape(ctx, self.accumulator(ctx).get_type(ctx)) != Some(expected) {
            return verify_err!(
                op.loc(),
                "cute.epilogue_store_fragment accumulator must be an ordinary 64xf32 MIR carrier"
            );
        }
        Ok(())
    }
}

/// Meet before shared reuse or after every writer has published its result.
#[pliron_op(
    name = "cute.epilogue_sync",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
    attributes = (
        epilogue_sync_tile: TypeAttr,
        epilogue_sync_barrier: CuteCountedCtaBarrierAttr,
        epilogue_sync_phase: CuteEpilogueSyncPhaseAttr
    )
)]
pub struct CuteEpilogueSyncOp;

impl CuteEpilogueSyncOp {
    /// Build an epilogue synchronization boundary.
    ///
    /// `ReadyForTma` semantically includes publication from the generic
    /// shared-memory proxy before the counted hand-off. Backends must lower
    /// that phase to both their proxy fence and barrier sequence.
    pub fn new(
        ctx: &mut Context,
        base: Value,
        tile: TypeHandle,
        barrier: CuteCountedCtaBarrierAttr,
        phase: CuteEpilogueSyncPhaseAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![base],
                vec![],
                0,
            ),
        };
        operation.set_attr_epilogue_sync_tile(ctx, TypeAttr::new(tile));
        operation.set_attr_epilogue_sync_barrier(ctx, barrier);
        operation.set_attr_epilogue_sync_phase(ctx, phase);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_sync_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn barrier(&self, ctx: &Context) -> Option<CuteCountedCtaBarrierAttr> {
        self.get_attr_epilogue_sync_barrier(ctx)
            .map(|barrier| *barrier)
    }

    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Option<CuteEpilogueSyncPhaseAttr> {
        self.get_attr_epilogue_sync_phase(ctx).map(|phase| *phase)
    }
}

impl Verify for CuteEpilogueSyncOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "cute.epilogue_sync needs 1 operand and 0 results");
        }
        let tile = match checked_tile(self.tile(ctx), ctx) {
            Ok(tile) => tile,
            Err(message) => return verify_err!(op.loc(), "cute.epilogue_sync {message}"),
        };
        if !shared_pointer_matches(ctx, self.base(ctx), tile.storage) {
            return verify_err!(
                op.loc(),
                "cute.epilogue_sync base must be a mutable CTA-shared f16 pointer"
            );
        }
        let Some(barrier) = self.barrier(ctx) else {
            return verify_err!(op.loc(), "cute.epilogue_sync must carry a counted barrier");
        };
        barrier.verify(ctx)?;
        let compute_warps = u32::try_from(
            tile.plan
                .tiled_mma
                .compute_warps()
                .expect("verified epilogue plan has a warp count"),
        )
        .expect("v0 compute-warp count fits u32");
        if barrier.barrier_id != 2
            || barrier.first_warp != 0
            || barrier.warp_count != compute_warps
            || barrier.cta_warps != compute_warps + 1
            || barrier.lanes_per_warp != 32
            || barrier.participant_threads() != Some(256)
            || barrier.excluded_warps() != Some(1)
        {
            return verify_err!(
                op.loc(),
                "cute.epilogue_sync v0 needs barrier 2 for 8 compute warps, with one producer warp excluded"
            );
        }
        if self.phase(ctx).is_none() {
            return verify_err!(op.loc(), "cute.epilogue_sync must name its protocol phase");
        }
        Ok(())
    }
}

/// Select one ordinary pointer/capacity pair from the full shared result tile.
#[pliron_op(
    name = "cute.epilogue_half",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<2>],
    attributes = (
        epilogue_half_tile: TypeAttr,
        epilogue_half_index: CuteEpilogueHalfAttr
    )
)]
pub struct CuteEpilogueHalfOp;

impl CuteEpilogueHalfOp {
    pub fn new(
        ctx: &mut Context,
        base: Value,
        tile: TypeHandle,
        half: CuteEpilogueHalfAttr,
    ) -> Self {
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![base.get_type(ctx), u64_ty],
                vec![base],
                vec![],
                0,
            ),
        };
        operation.set_attr_epilogue_half_tile(ctx, TypeAttr::new(tile));
        operation.set_attr_epilogue_half_index(ctx, half);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn full_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn half_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_half_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn half(&self, ctx: &Context) -> Option<CuteEpilogueHalfAttr> {
        self.get_attr_epilogue_half_index(ctx).map(|half| *half)
    }
}

impl Verify for CuteEpilogueHalfOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 2 {
            return verify_err!(op.loc(), "cute.epilogue_half needs 1 operand and 2 results");
        }
        let tile = match checked_tile(self.tile(ctx), ctx) {
            Ok(tile) => tile,
            Err(message) => return verify_err!(op.loc(), "cute.epilogue_half {message}"),
        };
        if !shared_pointer_matches(ctx, self.full_base(ctx), tile.storage)
            || self.half_base(ctx).get_type(ctx) != self.full_base(ctx).get_type(ctx)
            || !is_u64(ctx, self.capacity(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.epilogue_half must return a shared f16 pointer and u64 capacity"
            );
        }
        let Some(half) = self.half(ctx) else {
            return verify_err!(op.loc(), "cute.epilogue_half must name half 0 or 1");
        };
        half.verify(ctx)?;
        if half.0 >= tile.plan.halves {
            return verify_err!(
                op.loc(),
                "cute.epilogue_half index is outside the full tile"
            );
        }
        Ok(())
    }
}

fn check_store_pipeline(
    pipeline: Option<CuteTmaStorePipelineAttr>,
    ctx: &Context,
) -> Result<(), String> {
    let pipeline = pipeline.ok_or_else(|| "must carry a store-pipeline config".to_owned())?;
    pipeline
        .verify(ctx)
        .map_err(|error| format!("has an invalid store-pipeline config: {error}"))?;
    if pipeline.stages != 1 {
        return Err("v0 supports the one-buffer store pipeline".to_owned());
    }
    Ok(())
}

/// Wait until the shared result buffer may be overwritten.
#[pliron_op(
    name = "cute.tma_store_acquire",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_acquire_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreAcquireOp;

impl CuteTmaStoreAcquireOp {
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_acquire_config(ctx, pipeline);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_acquire_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreAcquireOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_acquire needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_acquire {message}");
        }
        Ok(())
    }
}

/// Close this thread's pending TMA stores as one hardware copy group.
#[pliron_op(
    name = "cute.tma_store_commit",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_commit_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreCommitOp;

impl CuteTmaStoreCommitOp {
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_commit_config(ctx, pipeline);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_commit_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreCommitOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_commit needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_commit {message}");
        }
        Ok(())
    }
}

/// Before exit, wait until no TMA store still reads shared memory.
#[pliron_op(
    name = "cute.tma_store_tail",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_tail_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreTailOp;

impl CuteTmaStoreTailOp {
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_tail_config(ctx, pipeline);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_tail_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreTailOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_tail needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_tail {message}");
        }
        Ok(())
    }
}

/// Copy one typed shared tile to a descriptor-backed global destination.
///
/// Both operands are short-lived direct TMA-view ghosts. The source view owns
/// the ordinary shared pointer/capacity pair; the destination view owns the
/// immutable descriptor pointer.
#[pliron_op(
    name = "cute.tma_store_2d",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>]
)]
pub struct CuteTmaStore2dSemanticOp;

impl CuteTmaStore2dSemanticOp {
    pub fn new(
        ctx: &mut Context,
        source: Value,
        destination: Value,
        tile_row: Value,
        tile_column: Value,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![source, destination, tile_row, tile_column],
                vec![],
                0,
            ),
        }
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn tile_row(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    #[must_use]
    pub fn tile_column(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }
}

impl Verify for CuteTmaStore2dSemanticOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "cute.tma_store_2d needs 4 operands and 0 results");
        }
        let Some(source) = tma_view_of(ctx, self.source(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_store_2d source must be a TMA view");
        };
        let Some(destination) = tma_view_of(ctx, self.destination(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_store_2d destination must be a TMA view");
        };
        source.verify(ctx)?;
        destination.verify(ctx)?;
        let source_tensor = source.tensor_view(ctx).expect("verified source tensor");
        let destination_tensor = destination
            .tensor_view(ctx)
            .expect("verified destination tensor");
        if source_tensor.space != CuteTensorAddressSpaceAttr::Smem
            || source_tensor.access != CuteTensorAccessAttr::ReadWrite
            || destination_tensor.space != CuteTensorAddressSpaceAttr::Gmem
            || destination_tensor.access != CuteTensorAccessAttr::ReadWrite
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d direction must be writable Smem to writable Gmem"
            );
        }
        if source_tensor.storage != destination_tensor.storage
            || source.smem_layout != destination.smem_layout
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d source and destination need the same carrier and tile layout"
            );
        }
        if source_tensor
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
            || source.smem_layout.0 != current_half_layout()
            || source.tile_elements() != Some(8192)
            || source.tile_bytes(ctx) != Some(16 * 1024)
            || source_tensor.alignment.0 != 1024
            || destination_tensor.alignment.0 != 2
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d v0 needs one 1024-byte-aligned B128 128x64 f16 half"
            );
        }
        if !is_u64(ctx, self.tile_row(ctx)) || !is_u64(ctx, self.tile_column(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d row and column must be unsigned 64-bit integers"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::{CuteEpiloguePlanAttr, CuteTiledMmaPlanAttr};
    use crate::gemm_tma_ops::{CuteTmaGmemViewOp, CuteTmaSmemViewOp};
    use crate::types::CuteEpilogueTileType;
    use dialect_mir::ops::MirUndefOp;
    use dialect_mir::types::{MirArrayType, MirStructType};
    use pliron::builtin::types::FP32Type;

    fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
        MirUndefOp::new(ctx, ty)
            .get_operation()
            .deref(ctx)
            .get_result(0)
    }

    fn data_layout() -> ComposedLayout {
        let inner: Layout = "(128,32):(32,1)".parse().unwrap();
        ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements).unwrap()
    }

    struct Fixture {
        base: Value,
        warp: Value,
        lane: Value,
        accumulator: Value,
        capacity: Value,
        row: Value,
        column: Value,
        descriptor: Value,
        f16_ty: TypeHandle,
        tile_ty: TypeHandle,
    }

    fn setup() -> (Context, Fixture) {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
        let base_ty: TypeHandle =
            MirPtrType::get(&mut ctx, f16_ty, true, address_space::SHARED).into();
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
}

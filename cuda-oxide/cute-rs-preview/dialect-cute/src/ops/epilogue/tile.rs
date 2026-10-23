/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared output tiles, warp placement, fragment stores, and synchronization.

use super::*;

/// Attach result-tile meaning to one shared pointer without changing it.
#[pliron_op(
    name = "cute.epilogue_smem_overlay",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (epilogue_overlay_tile: TypeAttr)
)]
pub struct CuteEpilogueSmemOverlayOp;

impl CuteEpilogueSmemOverlayOp {
    /// Build a detached overlay retaining the shared pointer and attaching the output-tile type.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the original shared pointer supplied to the semantic overlay.
    #[must_use]
    pub fn input_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the shared storage pointer retained by this semantic operation.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Read the static shared output-tile type attached to the operation.
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
    /// Build a detached output-slice operation forwarding the pointer, warp, and lane carriers.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the original shared pointer supplied to the semantic overlay.
    #[must_use]
    pub fn input_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the warp identifier supplied when selecting the output slice.
    #[must_use]
    pub fn input_warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the lane identifier supplied when selecting the output slice.
    #[must_use]
    pub fn input_lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Return the shared storage pointer retained by this semantic operation.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Return the ordinary warp-index carrier for output placement.
    #[must_use]
    pub fn warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Return the lane-index carrier identifying register ownership within a warp.
    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(2)
    }

    /// Read the static shared output-tile type attached to the operation.
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
    /// Build a detached operation placing one warp's complete accumulator into shared output storage.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the shared storage pointer retained by this semantic operation.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the ordinary warp-index carrier for output placement.
    #[must_use]
    pub fn warp_id(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the lane-index carrier identifying register ownership within a warp.
    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Return the ordinary FP32 accumulator carrier.
    #[must_use]
    pub fn accumulator(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }

    /// Read the static shared output-tile type attached to the operation.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the shared storage pointer retained by this semantic operation.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Read the static shared output-tile type attached to the operation.
    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_sync_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }

    /// Clone the counted CTA-barrier participation contract, if present.
    #[must_use]
    pub fn barrier(&self, ctx: &Context) -> Option<CuteCountedCtaBarrierAttr> {
        self.get_attr_epilogue_sync_barrier(ctx)
            .map(|barrier| *barrier)
    }

    /// Return the reusable/ready hand-off phase encoded by the synchronization operation.
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
    /// Build a detached selector producing one half's shared pointer and element capacity.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the shared base pointer of the complete output tile.
    #[must_use]
    pub fn full_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the shared pointer result selecting one output half.
    #[must_use]
    pub fn half_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Return the ordinary unsigned storage-capacity carrier, in elements.
    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Read the static shared output-tile type attached to the operation.
    #[must_use]
    pub fn tile(&self, ctx: &Context) -> Option<CuteEpilogueTileType> {
        let attr = self.get_attr_epilogue_half_tile(ctx)?;
        epilogue_tile_of_type(ctx, attr.get_type(ctx))
    }

    /// Return the static left/right output-half selection.
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

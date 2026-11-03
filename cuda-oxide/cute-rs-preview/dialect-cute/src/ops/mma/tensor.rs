/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared tensor overlays and per-lane tiled MMA selection.

use super::*;

/// Give a shared pointer/capacity pair typed tensor meaning without changing
/// either runtime value.
///
/// The `smem_overlay_view` attribute is a `!cute.smem_tensor` type. Results
/// have exactly the operand types, so backend lowering may erase this
/// operation to two identity replacements.
#[pliron_op(
    name = "cute.smem_tensor_overlay",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<2>],
    attributes = (smem_overlay_view: TypeAttr)
)]
pub struct CuteSmemTensorOverlayOp;

impl CuteSmemTensorOverlayOp {
    /// Build a detached overlay annotating ordinary shared pointer/capacity carriers.
    pub fn new(ctx: &mut Context, base: Value, capacity: Value, view: TypeHandle) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![base.get_type(ctx), capacity.get_type(ctx)],
                vec![base, capacity],
                vec![],
                0,
            ),
        };
        operation.set_attr_smem_overlay_view(ctx, TypeAttr::new(view));
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

    /// Return the original shared element-capacity operand.
    #[must_use]
    pub fn input_capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the shared storage pointer retained by this semantic operation.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Return the ordinary unsigned storage-capacity carrier, in elements.
    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Static logical/storage/role/layout facts attached to this pair.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_smem_overlay_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }
}

impl Verify for CuteSmemTensorOverlayOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay needs 2 operands and 2 results"
            );
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay must carry a cute.smem_tensor type"
            );
        };
        view.verify(ctx)?;
        let tensor = view.tensor_view(ctx).expect("verified shared tensor");
        if !shared_pointer_matches(ctx, self.input_base(ctx), tensor.storage) {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay base must be a mutable CTA-shared storage pointer"
            );
        }
        if !is_u64(ctx, self.input_capacity(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay capacity must be an unsigned 64-bit integer"
            );
        }
        if self.base(ctx).get_type(ctx) != self.input_base(ctx).get_type(ctx)
            || self.capacity(ctx).get_type(ctx) != self.input_capacity(ctx).get_type(ctx)
        {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay results must preserve the pointer and capacity types"
            );
        }
        Ok(())
    }
}

/// Attach one tiled-MMA plan to the calling lane number.
///
/// The result remains an ordinary `u32`; it may safely cross loop edges.
#[pliron_op(
    name = "cute.tiled_mma_slice",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (tiled_mma_slice_plan: CuteTiledMmaPlanAttr)
)]
pub struct CuteTiledMmaSliceOp;

impl CuteTiledMmaSliceOp {
    /// Build a detached lane selection carrying the static tiled-MMA ownership plan.
    pub fn new(ctx: &mut Context, lane: Value, plan: CuteTiledMmaPlanAttr) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![lane.get_type(ctx)],
                vec![lane],
                vec![],
                0,
            ),
        };
        operation.set_attr_tiled_mma_slice_plan(ctx, plan);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the lane-index carrier identifying register ownership within a warp.
    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the forwarded lane result annotated by the tiled-MMA plan.
    #[must_use]
    pub fn sliced_lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_tiled_mma_slice_plan(ctx))
    }
}

impl Verify for CuteTiledMmaSliceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tiled_mma_slice needs 1 operand and 1 result"
            );
        }
        if !is_u32(ctx, self.lane(ctx))
            || self.sliced_lane(ctx).get_type(ctx) != self.lane(ctx).get_type(ctx)
        {
            return verify_err!(
                op.loc(),
                "cute.tiled_mma_slice must preserve one unsigned u32 lane"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.tiled_mma_slice must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.tiled_mma_slice {message}");
        }
        Ok(())
    }
}

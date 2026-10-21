/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared scale and A loads with lazy B selection.

use super::*;

/// Load the ten packed scale words used by one warp for a K=128 stage.
#[pliron_op(
    name = "cute.mma_load_scales",
    format,
    interfaces = [NOpdsInterface<7>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mma_load_scales_plan: CuteTiledMmaPlanAttr,
        mma_load_scales_a_view: TypeAttr,
        mma_load_scales_b_view: TypeAttr
    )
)]
pub struct CuteMmaLoadScalesOp;

impl CuteMmaLoadScalesOp {
    /// Build a detached stage-scale load retaining both operand views and lane selection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        scale_a_base: Value,
        scale_a_capacity: Value,
        scale_b_base: Value,
        scale_b_capacity: Value,
        warp_m: Value,
        warp_n: Value,
        scale_a_view: TypeHandle,
        scale_b_view: TypeHandle,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![
                    lane,
                    scale_a_base,
                    scale_a_capacity,
                    scale_b_base,
                    scale_b_capacity,
                    warp_m,
                    warp_n,
                ],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_load_scales_plan(ctx, plan);
        operation.set_attr_mma_load_scales_a_view(ctx, TypeAttr::new(scale_a_view));
        operation.set_attr_mma_load_scales_b_view(ctx, TypeAttr::new(scale_b_view));
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_load_scales_plan(ctx))
    }

    /// Static M-role scale tensor facts for operands 1 and 2.
    #[must_use]
    pub fn scale_a_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_scales_a_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    /// Static N-role scale tensor facts for operands 3 and 4.
    #[must_use]
    pub fn scale_b_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_scales_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    /// Return the ordinary scale-register aggregate for this K selection.
    #[must_use]
    pub fn scales(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteMmaLoadScalesOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 7 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales needs 7 operands and 1 result"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_load_scales must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_load_scales {message}");
        }
        if !is_u32(ctx, op.get_operand(0))
            || !is_u64(ctx, op.get_operand(5))
            || !is_u64(ctx, op.get_operand(6))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales needs a u32 lane and u64 warp-M/warp-N positions"
            );
        }
        let Some(scale_a) = self.scale_a_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales must carry an M-role shared scale view"
            );
        };
        let Some(scale_b) = self.scale_b_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales must carry an N-role shared scale view"
            );
        };
        if let Err(message) = check_scale_view(ctx, &scale_a, &plan, CuteTensorRoleAttr::Mkl) {
            return verify_err!(op.loc(), "cute.mma_load_scales A {message}");
        }
        if let Err(message) = check_scale_view(ctx, &scale_b, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.mma_load_scales B {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(1), op.get_operand(2), &scale_a)
            || !shared_carriers_match_view(ctx, op.get_operand(3), op.get_operand(4), &scale_b)
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales pointer/capacity carriers must match their typed scale views"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, self.scales(ctx), expected) {
            return verify_err!(op.loc(), "cute.mma_load_scales result {message}");
        }
        Ok(())
    }
}
/// Load the two A fragments owned by one warp for one K=64 step.
#[pliron_op(
    name = "cute.mma_load_a",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mma_load_a_plan: CuteTiledMmaPlanAttr,
        mma_load_a_view: TypeAttr
    )
)]
pub struct CuteMmaLoadAOp;

impl CuteMmaLoadAOp {
    /// Build a detached packed A-register load for the selected lane, warp-M, and K position.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        base: Value,
        capacity: Value,
        warp_m: Value,
        k_half: Value,
        view: TypeHandle,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![lane, base, capacity, warp_m, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_load_a_plan(ctx, plan);
        operation.set_attr_mma_load_a_view(ctx, TypeAttr::new(view));
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_load_a_plan(ctx))
    }

    /// Static M-role shared tensor facts for the pointer/capacity operands.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_a_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    /// Return the ordinary register aggregate produced by this operation.
    #[must_use]
    pub fn fragment(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteMmaLoadAOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.mma_load_a needs 5 operands and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_load_a must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_load_a {message}");
        }
        if !is_u32(ctx, op.get_operand(0))
            || !is_u64(ctx, op.get_operand(3))
            || !is_u64(ctx, op.get_operand(4))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a needs a u32 lane and u64 warp-M/K-half positions"
            );
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a must carry an M-role shared value view"
            );
        };
        if let Err(message) = check_data_view(ctx, &view, &plan, CuteTensorRoleAttr::Mkl) {
            return verify_err!(op.loc(), "cute.mma_load_a input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(1), op.get_operand(2), &view) {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a pointer/capacity carriers must match its typed A view"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.a_registers_per_lane()
                .expect("verified plan has an A carrier size"),
        );
        if let Err(message) = check_carrier(ctx, self.fragment(ctx), expected) {
            return verify_err!(op.loc(), "cute.mma_load_a result {message}");
        }
        Ok(())
    }
}

/// Keep B as a lazy shared-memory selection.
///
/// All four results preserve ordinary operand types. In particular, this op
/// has no register-fragment result and performs no `ldmatrix` load.
#[pliron_op(
    name = "cute.mma_partition_b",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<4>],
    attributes = (
        mma_partition_b_plan: CuteTiledMmaPlanAttr,
        mma_partition_b_view: TypeAttr
    )
)]
pub struct CuteMmaPartitionBOp;

impl CuteMmaPartitionBOp {
    /// Build a detached lazy B view retaining its base, capacity, warp-N, and K-half carriers.
    pub fn new(
        ctx: &mut Context,
        base: Value,
        capacity: Value,
        warp_n: Value,
        k_half: Value,
        view: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![
                    base.get_type(ctx),
                    capacity.get_type(ctx),
                    warp_n.get_type(ctx),
                    k_half.get_type(ctx),
                ],
                vec![base, capacity, warp_n, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_partition_b_plan(ctx, plan);
        operation.set_attr_mma_partition_b_view(ctx, TypeAttr::new(view));
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_partition_b_plan(ctx))
    }

    /// Static N-role shared tensor facts for the lazy B selection.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_partition_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
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

    /// Return the lazy B partition's warp-N coordinate carrier.
    #[must_use]
    pub fn warp_n(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(2)
    }

    /// Return the lazy B partition's K-half coordinate carrier.
    #[must_use]
    pub fn k_half(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(3)
    }
}

impl Verify for CuteMmaPartitionBOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 4 {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b needs 4 operands and exactly 4 scalar results"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_partition_b must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_partition_b {message}");
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b must carry an N-role shared value view"
            );
        };
        if let Err(message) = check_data_view(ctx, &view, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.mma_partition_b input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(0), op.get_operand(1), &view) {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b pointer/capacity carriers must match its typed B view"
            );
        }
        if !is_u64(ctx, op.get_operand(2)) || !is_u64(ctx, op.get_operand(3)) {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b warp-N and K-half must be unsigned u64 values"
            );
        }
        if (0..4)
            .any(|index| op.get_result(index).get_type(ctx) != op.get_operand(index).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b must preserve pointer, capacity, warp-N, and K-half types"
            );
        }
        Ok(())
    }
}

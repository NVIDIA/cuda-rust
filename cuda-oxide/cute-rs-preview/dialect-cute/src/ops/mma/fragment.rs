/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Accumulator initialization and scale-fragment K selection.

use super::*;

/// Fill one physical register carrier while retaining its fragment meaning.
///
/// V0 uses this for the 64-FP32 accumulator carrier only.
#[pliron_op(
    name = "cute.fragment_fill",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        fragment_fill_plan: CuteTiledMmaPlanAttr,
        fragment_fill_kind: CuteMmaCarrierKindAttr
    )
)]
pub struct CuteFragmentFillOp;

impl CuteFragmentFillOp {
    /// Build a detached ordinary register aggregate initialized from the supplied FP32 scalar.
    pub fn new(
        ctx: &mut Context,
        fill: Value,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
        kind: CuteMmaCarrierKindAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![fill],
                vec![],
                0,
            ),
        };
        operation.set_attr_fragment_fill_plan(ctx, plan);
        operation.set_attr_fragment_fill_kind(ctx, kind);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the FP32 value used to initialize every accumulator register.
    #[must_use]
    pub fn fill(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the ordinary register aggregate produced by this operation.
    #[must_use]
    pub fn fragment(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_fragment_fill_plan(ctx))
    }

    /// Return the semantic role assigned to the ordinary register aggregate.
    #[must_use]
    pub fn kind(&self, ctx: &Context) -> Option<CuteMmaCarrierKindAttr> {
        self.get_attr_fragment_fill_kind(ctx).map(|kind| *kind)
    }
}

impl Verify for CuteFragmentFillOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.fragment_fill needs 1 operand and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.fragment_fill must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.fragment_fill {message}");
        }
        if self.kind(ctx) != Some(CuteMmaCarrierKindAttr::Accumulator)
            || !is_f32(ctx, self.fill(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.fragment_fill v0 fills an accumulator from one f32 value"
            );
        }
        let expected = MmaCarrierShape::f32(
            plan.accumulator_registers_per_lane()
                .expect("verified plan has an accumulator size"),
        );
        if let Err(message) = check_carrier(ctx, self.fragment(ctx), expected) {
            return verify_err!(op.loc(), "cute.fragment_fill result {message}");
        }
        Ok(())
    }
}
/// Select the low or high K=64 scale pair from every packed K=128 word.
#[pliron_op(
    name = "cute.fragment_slice_k",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (fragment_slice_k_plan: CuteTiledMmaPlanAttr)
)]
pub struct CuteFragmentSliceKOp;

impl CuteFragmentSliceKOp {
    /// Build a detached K=64 selection from a loaded stage-scale aggregate.
    pub fn new(
        ctx: &mut Context,
        stage_scales: Value,
        k_half: Value,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![stage_scales, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_fragment_slice_k_plan(ctx, plan);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_fragment_slice_k_plan(ctx))
    }

    /// Return the ordinary scale-register aggregate for this K selection.
    #[must_use]
    pub fn scales(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteFragmentSliceKOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.fragment_slice_k needs 2 operands and 1 result"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.fragment_slice_k must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.fragment_slice_k {message}");
        }
        if plan.cta_k != plan.atom.k * 2 || !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.fragment_slice_k v0 selects one u64-indexed K=64 half of K=128"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(0), expected) {
            return verify_err!(op.loc(), "cute.fragment_slice_k input {message}");
        }
        if let Err(message) = check_carrier(ctx, self.scales(ctx), expected) {
            return verify_err!(op.loc(), "cute.fragment_slice_k result {message}");
        }
        Ok(())
    }
}

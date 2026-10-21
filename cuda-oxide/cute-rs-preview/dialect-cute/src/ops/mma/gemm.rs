/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tiled GEMM over ordinary register and shared-memory carriers.

use super::*;

/// Multiply one A K=64 bundle by one lazy B selection and add it to C.
///
/// B arrives as four scalar carriers, never as eight eager fragments. Backend
/// A must honor `plan.b_load_group`: load two B fragments, issue their four
/// MMA atoms, then move to the next pair.
#[pliron_op(
    name = "cute.tiled_gemm",
    format,
    interfaces = [NOpdsInterface<8>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        tiled_gemm_plan: CuteTiledMmaPlanAttr,
        tiled_gemm_b_view: TypeAttr
    )
)]
pub struct CuteTiledGemmOp;

impl CuteTiledGemmOp {
    /// Build a detached accumulator update from typed A/scales and lazy shared B carriers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        a: Value,
        b_base: Value,
        b_capacity: Value,
        b_warp_n: Value,
        b_k_half: Value,
        scales: Value,
        accumulator: Value,
        b_view: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![accumulator.get_type(ctx)],
                vec![
                    lane,
                    a,
                    b_base,
                    b_capacity,
                    b_warp_n,
                    b_k_half,
                    scales,
                    accumulator,
                ],
                vec![],
                0,
            ),
        };
        operation.set_attr_tiled_gemm_plan(ctx, plan);
        operation.set_attr_tiled_gemm_b_view(ctx, TypeAttr::new(b_view));
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the operation's typed tiled-MMA plan, if present.
    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_tiled_gemm_plan(ctx))
    }

    /// Static N-role shared tensor facts carried by the four lazy B scalars.
    #[must_use]
    pub fn b_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_tiled_gemm_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    /// Return the ordinary FP32 accumulator carrier.
    #[must_use]
    pub fn accumulator(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteTiledGemmOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 8 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tiled_gemm needs 8 operands and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.tiled_gemm must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.tiled_gemm {message}");
        }
        if !is_u32(ctx, op.get_operand(0)) {
            return verify_err!(op.loc(), "cute.tiled_gemm lane must be an unsigned u32");
        }
        let expected_a = MmaCarrierShape::u32(
            plan.a_registers_per_lane()
                .expect("verified plan has an A carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(1), expected_a) {
            return verify_err!(op.loc(), "cute.tiled_gemm A input {message}");
        }
        let Some(b_view) = self.b_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm must carry an N-role shared view for lazy B"
            );
        };
        if let Err(message) = check_data_view(ctx, &b_view, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.tiled_gemm B input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(2), op.get_operand(3), &b_view)
            || !is_u64(ctx, op.get_operand(4))
            || !is_u64(ctx, op.get_operand(5))
        {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm lazy B must be pointer, capacity, warp-N, and K-half scalars matching its typed view"
            );
        }
        let expected_scales = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(6), expected_scales) {
            return verify_err!(op.loc(), "cute.tiled_gemm scale input {message}");
        }
        let expected_accumulator = MmaCarrierShape::f32(
            plan.accumulator_registers_per_lane()
                .expect("verified plan has an accumulator size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(7), expected_accumulator) {
            return verify_err!(op.loc(), "cute.tiled_gemm accumulator input {message}");
        }
        if self.accumulator(ctx).get_type(ctx) != op.get_operand(7).get_type(ctx) {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm result must preserve the accumulator carrier type"
            );
        }
        if let Err(message) = check_carrier(ctx, self.accumulator(ctx), expected_accumulator) {
            return verify_err!(op.loc(), "cute.tiled_gemm result {message}");
        }
        Ok(())
    }
}

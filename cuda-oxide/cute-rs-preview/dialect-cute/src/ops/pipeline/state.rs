/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ordinary scalar slot and phase state for a pipeline ring.

use super::*;

/// Create the initial ordinary `(slot, phase)` pair for one pipeline side.
#[pliron_op(
    name = "cute.pipeline_state_new",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<2>],
    attributes = (pipeline_state_new_config: CutePipelineStateAttr)
)]
pub struct CutePipelineStateNewOp;

impl CutePipelineStateNewOp {
    /// Build detached initial slot/phase results for the selected pipeline role.
    pub fn new(ctx: &mut Context, state: CutePipelineStateAttr) -> Self {
        let ty = u32_type(ctx);
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty, ty],
                vec![],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_state_new_config(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the ordinary u32 ring-slot carrier used by this operation.
    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Return the ordinary u32 barrier-parity carrier used by this operation.
    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Clone the static producer/consumer role and ring-size attribute, if present.
    #[must_use]
    pub fn state(&self, ctx: &Context) -> Option<CutePipelineStateAttr> {
        self.get_attr_pipeline_state_new_config(ctx)
            .map(|state| *state)
    }
}

impl Verify for CutePipelineStateNewOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_new needs 0 operands and 2 results"
            );
        }
        if !is_u32(ctx, self.slot(ctx)) || !is_u32(ctx, self.phase(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_new slot and phase must be unsigned 32-bit integers"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(op.loc(), "cute.pipeline_state_new must carry state facts");
        };
        state.verify(ctx)
    }
}

/// Widen a `u32` slot to the `u64` index used for shared-memory addressing.
#[pliron_op(
    name = "cute.pipeline_state_slot",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (pipeline_state_slot_config: CutePipelineStateAttr)
)]
pub struct CutePipelineStateSlotOp;

impl CutePipelineStateSlotOp {
    /// Build a detached conversion from a u32 cursor slot to a u64 addressing index.
    pub fn new(ctx: &mut Context, slot: Value, state: CutePipelineStateAttr) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![u64_type(ctx)],
                vec![slot],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_state_slot_config(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the ordinary u32 ring-slot carrier used by this operation.
    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Slot widened for pointer offset arithmetic.
    #[must_use]
    pub fn index(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Clone the static producer/consumer role and ring-size attribute, if present.
    #[must_use]
    pub fn state(&self, ctx: &Context) -> Option<CutePipelineStateAttr> {
        self.get_attr_pipeline_state_slot_config(ctx)
            .map(|state| *state)
    }
}

impl Verify for CutePipelineStateSlotOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_slot needs 1 operand and 1 result"
            );
        }
        if !is_u32(ctx, self.slot(ctx)) || !is_u64(ctx, self.index(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_slot needs a u32 slot and returns a u64 index"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(op.loc(), "cute.pipeline_state_slot must carry state facts");
        };
        state.verify(ctx)
    }
}

/// Move one ordinary `(slot, phase)` pair to the next ring position.
#[pliron_op(
    name = "cute.pipeline_state_advance",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<2>],
    attributes = (pipeline_state_advance_config: CutePipelineStateAttr)
)]
pub struct CutePipelineStateAdvanceOp;

impl CutePipelineStateAdvanceOp {
    /// Build a detached ring-cursor advance preserving the producer/consumer role.
    pub fn new(ctx: &mut Context, slot: Value, phase: Value, state: CutePipelineStateAttr) -> Self {
        let ty = u32_type(ctx);
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty, ty],
                vec![slot, phase],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_state_advance_config(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the ordinary u32 ring-slot carrier used by this operation.
    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the ordinary u32 barrier-parity carrier used by this operation.
    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the advanced ring-slot result, wrapping at the static stage count.
    #[must_use]
    pub fn next_slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Return the advanced parity result, toggled when the ring wraps.
    #[must_use]
    pub fn next_phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Clone the static producer/consumer role and ring-size attribute, if present.
    #[must_use]
    pub fn state(&self, ctx: &Context) -> Option<CutePipelineStateAttr> {
        self.get_attr_pipeline_state_advance_config(ctx)
            .map(|state| *state)
    }
}

impl Verify for CutePipelineStateAdvanceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_advance needs 2 operands and 2 results"
            );
        }
        if ![
            self.slot(ctx),
            self.phase(ctx),
            self.next_slot(ctx),
            self.next_phase(ctx),
        ]
        .into_iter()
        .all(|value| is_u32(ctx, value))
        {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_advance slot and phase values must be unsigned 32-bit integers"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_state_advance must carry state facts"
            );
        };
        state.verify(ctx)
    }
}

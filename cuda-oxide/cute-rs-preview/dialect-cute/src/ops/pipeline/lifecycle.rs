/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Producer and consumer barrier-ring lifecycle operations.

use super::*;

/// Generate the common handle, slot, and static cursor-state accessors.
macro_rules! pipeline_state_accessors {
    ($attr_getter:ident) => {
        /// Return the direct compiler-only load-pipeline handle operand.
        #[must_use]
        pub fn pipeline(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(0)
        }

        /// Return the ordinary u32 slot selecting the current barrier stage.
        #[must_use]
        pub fn slot(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(1)
        }

        /// Clone the static producer/consumer role and ring size, if present.
        #[must_use]
        pub fn state(&self, ctx: &Context) -> Option<CutePipelineStateAttr> {
            self.$attr_getter(ctx).map(|state| *state)
        }
    };
}

/// Wait until the producer may reuse its current empty stage.
#[pliron_op(
    name = "cute.pipeline_producer_acquire",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    attributes = (pipeline_producer_acquire_state: CutePipelineStateAttr)
)]
pub struct CutePipelineProducerAcquireOp;

impl CutePipelineProducerAcquireOp {
    /// Build a detached wait for the current empty stage and producer parity.
    pub fn new(
        ctx: &mut Context,
        pipeline: Value,
        slot: Value,
        phase: Value,
        state: CutePipelineStateAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, slot, phase],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_producer_acquire_state(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_producer_acquire_state);

    /// Return the ordinary u32 barrier-parity carrier used by this operation.
    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

impl Verify for CutePipelineProducerAcquireOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_acquire needs 3 operands and 0 results"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_acquire must carry producer state facts"
            );
        };
        if let Err(message) = checked_pipeline_state(
            ctx,
            self.pipeline(ctx),
            state,
            CutePipelineRoleAttr::Producer,
        ) {
            return verify_err!(op.loc(), "cute.pipeline_producer_acquire {message}");
        }
        if !is_u32(ctx, self.slot(ctx)) || !is_u32(ctx, self.phase(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_acquire slot and phase must be unsigned 32-bit integers"
            );
        }
        Ok(())
    }
}

/// Set the expected byte count and return this stage's full barrier pointer.
///
/// The returned pointer is real SSA. The four existing TMA-copy operations
/// consume it directly, so their ABI does not change.
#[pliron_op(
    name = "cute.pipeline_producer_expect_tx",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (pipeline_producer_expect_state: CutePipelineStateAttr)
)]
pub struct CutePipelineProducerExpectTxOp;

impl CutePipelineProducerExpectTxOp {
    /// Build a detached byte-expectation operation returning the stage's full barrier.
    pub fn new(
        ctx: &mut Context,
        pipeline: Value,
        slot: Value,
        state: CutePipelineStateAttr,
    ) -> Self {
        let make = pipeline_make_of(ctx, pipeline)
            .expect("cute.pipeline_producer_expect_tx builder needs a direct pipeline make");
        let barrier_pointer = make.base(ctx).get_type(ctx);
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![barrier_pointer],
                vec![pipeline, slot],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_producer_expect_state(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_producer_expect_state);

    /// Mutable shared pointer consumed by the existing four TMA copies.
    #[must_use]
    pub fn completion_barrier(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CutePipelineProducerExpectTxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_expect_tx needs 2 operands and 1 result"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_expect_tx must carry producer state facts"
            );
        };
        let (_, make) = match checked_pipeline_state(
            ctx,
            self.pipeline(ctx),
            state,
            CutePipelineRoleAttr::Producer,
        ) {
            Ok(pair) => pair,
            Err(message) => {
                return verify_err!(op.loc(), "cute.pipeline_producer_expect_tx {message}");
            }
        };
        if !is_u32(ctx, self.slot(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_expect_tx slot must be an unsigned 32-bit integer"
            );
        }
        if self.completion_barrier(ctx).get_type(ctx) != make.base(ctx).get_type(ctx) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_expect_tx result must match the pipeline's shared Barrier pointer type"
            );
        }
        Ok(())
    }
}

/// Wait until all expected TMA bytes have reached the current full stage.
#[pliron_op(
    name = "cute.pipeline_consumer_wait",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    attributes = (pipeline_consumer_wait_state: CutePipelineStateAttr)
)]
pub struct CutePipelineConsumerWaitOp;

impl CutePipelineConsumerWaitOp {
    /// Build a detached wait for all expected input bytes at the consumer cursor.
    pub fn new(
        ctx: &mut Context,
        pipeline: Value,
        slot: Value,
        phase: Value,
        state: CutePipelineStateAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, slot, phase],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_consumer_wait_state(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_consumer_wait_state);

    /// Return the ordinary u32 barrier-parity carrier used by this operation.
    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

impl Verify for CutePipelineConsumerWaitOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_wait needs 3 operands and 0 results"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_wait must carry consumer state facts"
            );
        };
        if let Err(message) = checked_pipeline_state(
            ctx,
            self.pipeline(ctx),
            state,
            CutePipelineRoleAttr::Consumer,
        ) {
            return verify_err!(op.loc(), "cute.pipeline_consumer_wait {message}");
        }
        if !is_u32(ctx, self.slot(ctx)) || !is_u32(ctx, self.phase(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_wait slot and phase must be unsigned 32-bit integers"
            );
        }
        Ok(())
    }
}

/// Mark the current stage empty after all consumer warps finish reading it.
#[pliron_op(
    name = "cute.pipeline_consumer_release",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (pipeline_consumer_release_state: CutePipelineStateAttr)
)]
pub struct CutePipelineConsumerReleaseOp;

impl CutePipelineConsumerReleaseOp {
    /// Build a detached empty-stage arrival after consumer reads finish.
    pub fn new(
        ctx: &mut Context,
        pipeline: Value,
        slot: Value,
        state: CutePipelineStateAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, slot],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_consumer_release_state(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_consumer_release_state);
}

impl Verify for CutePipelineConsumerReleaseOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_release needs 2 operands and 0 results"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_release must carry consumer state facts"
            );
        };
        if let Err(message) = checked_pipeline_state(
            ctx,
            self.pipeline(ctx),
            state,
            CutePipelineRoleAttr::Consumer,
        ) {
            return verify_err!(op.loc(), "cute.pipeline_consumer_release {message}");
        }
        if !is_u32(ctx, self.slot(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_consumer_release slot must be an unsigned 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Before the producer exits, wait until every stage is empty again.
#[pliron_op(
    name = "cute.pipeline_producer_tail",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    attributes = (pipeline_producer_tail_state: CutePipelineStateAttr)
)]
pub struct CutePipelineProducerTailOp;

impl CutePipelineProducerTailOp {
    /// Build a detached exit wait draining the producer's reusable-stage ring.
    pub fn new(
        ctx: &mut Context,
        pipeline: Value,
        slot: Value,
        phase: Value,
        state: CutePipelineStateAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, slot, phase],
                vec![],
                0,
            ),
        };
        operation.set_attr_pipeline_producer_tail_state(ctx, state);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_producer_tail_state);

    /// Return the ordinary u32 barrier-parity carrier used by this operation.
    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

impl Verify for CutePipelineProducerTailOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_tail needs 3 operands and 0 results"
            );
        }
        let Some(state) = self.state(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_tail must carry producer state facts"
            );
        };
        if let Err(message) = checked_pipeline_state(
            ctx,
            self.pipeline(ctx),
            state,
            CutePipelineRoleAttr::Producer,
        ) {
            return verify_err!(op.loc(), "cute.pipeline_producer_tail {message}");
        }
        if !is_u32(ctx, self.slot(ctx)) || !is_u32(ctx, self.phase(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.pipeline_producer_tail slot and phase must be unsigned 32-bit integers"
            );
        }
        Ok(())
    }
}

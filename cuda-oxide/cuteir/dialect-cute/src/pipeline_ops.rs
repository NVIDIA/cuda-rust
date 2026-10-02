/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A small semantic layer for the GEMM TMA load pipeline.
//!
//! The pipeline owns two hardware-barrier rings:
//!
//! ```text
//!                    one slot
//! producer:  wait empty ──► expect bytes ──► four TMA copies
//!                                                  │
//! consumer:  release empty ◄── read tile ◄── wait full
//! ```
//!
//! Only [`CuteTmaLoadPipelineType`] is a compiler-only handle. The loop still
//! carries the same ordinary scalar state as Rust:
//!
//! ```text
//! slot: u32, phase: u32 ── advance<3> ──► next_slot, next_phase
//! ```
//!
//! A backend continuation maps these operations to its barrier, transaction,
//! and publication primitives. It does not change the four TMA copies or add
//! a runtime pipeline object.
//!
//! A pipeline handle must come directly from `tma_load_pipeline_make`. It may
//! only feed the lifecycle operations in this module. It cannot enter a
//! function signature, pointer, aggregate, load/store, or block argument.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::builtin::{
    op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
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

use crate::attributes::{CuteAlignmentAttr, CutePipelineRoleAttr, CutePipelineStateAttr};
use crate::types::CuteTmaLoadPipelineType;

const BARRIER_BYTES: u64 = 8;

fn u32_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Unsigned).into()
}

fn u64_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Unsigned).into()
}

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

fn pipeline_type_of(ctx: &Context, value: Value) -> Option<CuteTmaLoadPipelineType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTmaLoadPipelineType>()
        .cloned()
}

fn pipeline_make_of(ctx: &Context, value: Value) -> Option<CuteTmaLoadPipelineMakeOp> {
    let producer = value.defining_op()?;
    if Operation::get_opid(producer, ctx) != CuteTmaLoadPipelineMakeOp::get_opid_static() {
        return None;
    }
    let make = CuteTmaLoadPipelineMakeOp::wrap(producer);
    (make.pipeline(ctx) == value).then_some(make)
}

fn checked_pipeline(
    ctx: &Context,
    value: Value,
) -> Result<(CuteTmaLoadPipelineType, CuteTmaLoadPipelineMakeOp), String> {
    let pipeline = pipeline_type_of(ctx, value)
        .ok_or_else(|| "pipeline operand must be a TMA load-pipeline handle".to_owned())?;
    pipeline
        .verify(ctx)
        .map_err(|error| format!("pipeline type is invalid: {error}"))?;
    let make = pipeline_make_of(ctx, value).ok_or_else(|| {
        "pipeline handle must come directly from cute.tma_load_pipeline_make".to_owned()
    })?;
    Ok((pipeline, make))
}

fn checked_pipeline_state(
    ctx: &Context,
    pipeline_value: Value,
    state: CutePipelineStateAttr,
    required_role: CutePipelineRoleAttr,
) -> Result<(CuteTmaLoadPipelineType, CuteTmaLoadPipelineMakeOp), String> {
    state
        .verify(ctx)
        .map_err(|error| format!("pipeline state is invalid: {error}"))?;
    if state.role != required_role {
        return Err(format!(
            "pipeline state must have {required_role:?} role, got {:?}",
            state.role
        ));
    }
    let (pipeline, make) = checked_pipeline(ctx, pipeline_value)?;
    if state.stages != pipeline.stages {
        return Err(format!(
            "pipeline state has {} stages, but its pipeline has {}",
            state.stages, pipeline.stages
        ));
    }
    Ok((pipeline, make))
}

fn barrier_pointer_of(ctx: &Context, value: Value) -> Option<MirPtrType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .cloned()
}

fn valid_shared_barrier_pointer(ctx: &Context, value: Value) -> bool {
    let Some(pointer) = barrier_pointer_of(ctx, value) else {
        return false;
    };
    if !pointer.is_mutable || pointer.address_space != address_space::SHARED {
        return false;
    }
    pointer
        .pointee
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|barrier| barrier.width() == 64 && barrier.is_unsigned())
}

fn allowed_pipeline_user(ctx: &Context, user: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(user, ctx);
    [
        CuteTmaLoadPipelineInitOp::get_opid_static(),
        CutePipelineProducerAcquireOp::get_opid_static(),
        CutePipelineProducerExpectTxOp::get_opid_static(),
        CutePipelineConsumerWaitOp::get_opid_static(),
        CutePipelineConsumerReleaseOp::get_opid_static(),
        CutePipelineProducerTailOp::get_opid_static(),
    ]
    .contains(&opid)
}

/// Bind one shared-memory barrier allocation to a TMA load-pipeline config.
///
/// The base points at this exact physical order:
///
/// ```text
/// [full 0 ... full S-1][empty 0 ... empty S-1]
/// ```
///
/// MIR represents CUDA's 8-byte `Barrier` storage as one unsigned `u64`.
#[pliron_op(
    name = "cute.tma_load_pipeline_make",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (pipeline_base_alignment_bytes: CuteAlignmentAttr)
)]
pub struct CuteTmaLoadPipelineMakeOp;

impl CuteTmaLoadPipelineMakeOp {
    /// Create a compiler-only view over an existing shared barrier ring.
    pub fn new(
        ctx: &mut Context,
        base: Value,
        stages: u64,
        consumer_warps: u32,
        transaction_bytes: u32,
        base_alignment_bytes: u64,
    ) -> Self {
        let pipeline: TypeHandle =
            CuteTmaLoadPipelineType::get(ctx, stages, consumer_warps, transaction_bytes).into();
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![pipeline],
                vec![base],
                vec![],
                0,
            ),
        };
        operation
            .set_attr_pipeline_base_alignment_bytes(ctx, CuteAlignmentAttr(base_alignment_bytes));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Shared base of the first full barrier.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Compiler-only pipeline handle.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Alignment promised for the start of the whole barrier ring.
    #[must_use]
    pub fn promised_base_alignment(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_pipeline_base_alignment_bytes(ctx)
            .map(|alignment| alignment.0)
    }

    /// Static pipeline configuration carried by the result type.
    #[must_use]
    pub fn pipeline_type(&self, ctx: &Context) -> Option<CuteTmaLoadPipelineType> {
        pipeline_type_of(ctx, self.pipeline(ctx))
    }
}

impl Verify for CuteTmaLoadPipelineMakeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make needs 1 operand and 1 result"
            );
        }
        if !valid_shared_barrier_pointer(ctx, self.base(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make base must be a mutable shared pointer to the canonical unsigned u64 Barrier storage"
            );
        }
        let Some(pipeline) = self.pipeline_type(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make result must be a TMA load-pipeline handle"
            );
        };
        pipeline.verify(ctx)?;
        let Some(alignment) = self.promised_base_alignment(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make must carry a base-alignment promise"
            );
        };
        CuteAlignmentAttr(alignment).verify(ctx)?;
        if alignment < BARRIER_BYTES {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make base alignment must be at least 8 bytes"
            );
        }
        if self
            .pipeline(ctx)
            .uses(ctx)
            .into_iter()
            .any(|r#use| !allowed_pipeline_user(ctx, r#use.user_op()))
        {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline handle may only feed load-pipeline lifecycle operations"
            );
        }
        Ok(())
    }
}

/// Initialize all full/empty barriers once, then publish them to the CTA.
#[pliron_op(
    name = "cute.tma_load_pipeline_init",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>]
)]
pub struct CuteTmaLoadPipelineInitOp;

impl CuteTmaLoadPipelineInitOp {
    pub fn new(ctx: &mut Context, pipeline: Value, init_thread: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, init_thread],
                vec![],
                0,
            ),
        }
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Thread index that initializes every barrier. All threads synchronize.
    #[must_use]
    pub fn init_thread(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for CuteTmaLoadPipelineInitOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_init needs 2 operands and 0 results"
            );
        }
        if let Err(message) = checked_pipeline(ctx, self.pipeline(ctx)) {
            return verify_err!(op.loc(), "cute.tma_load_pipeline_init {message}");
        }
        if !is_u32(ctx, self.init_thread(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_init thread index must be an unsigned 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Create the initial ordinary `(slot, phase)` pair for one pipeline side.
#[pliron_op(
    name = "cute.pipeline_state_new",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<2>],
    attributes = (pipeline_state_new_config: CutePipelineStateAttr)
)]
pub struct CutePipelineStateNewOp;

impl CutePipelineStateNewOp {
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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Slot widened for pointer offset arithmetic.
    #[must_use]
    pub fn index(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn next_slot(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn next_phase(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

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

macro_rules! pipeline_state_accessors {
    ($attr_getter:ident) => {
        #[must_use]
        pub fn pipeline(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(0)
        }

        #[must_use]
        pub fn slot(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(1)
        }

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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_producer_acquire_state);

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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_consumer_wait_state);

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

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pipeline_state_accessors!(get_attr_pipeline_producer_tail_state);

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

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirUndefOp;

    const STAGES: u64 = 3;
    const CONSUMER_WARPS: u32 = 8;
    const TRANSACTION_BYTES: u32 = 17_408;

    fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
        MirUndefOp::new(ctx, ty)
            .get_operation()
            .deref(ctx)
            .get_result(0)
    }

    fn shared_pointer(ctx: &mut Context, pointee: TypeHandle, mutable: bool) -> Value {
        let pointer: TypeHandle =
            MirPtrType::get(ctx, pointee, mutable, address_space::SHARED).into();
        undef(ctx, pointer)
    }

    struct Setup {
        ctx: Context,
        base: Value,
        u32_value: Value,
        u64_value: Value,
    }

    fn setup() -> Setup {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let barrier = u64_type(&ctx);
        let base = shared_pointer(&mut ctx, barrier, true);
        let u32_ty = u32_type(&ctx);
        let u64_ty = u64_type(&ctx);
        let u32_value = undef(&mut ctx, u32_ty);
        let u64_value = undef(&mut ctx, u64_ty);
        Setup {
            ctx,
            base,
            u32_value,
            u64_value,
        }
    }

    #[test]
    fn load_pipeline_stays_a_composable_ring_protocol() {
        let Setup {
            mut ctx,
            base,
            u32_value,
            ..
        } = setup();
        let make = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            base,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        let pipeline = make.pipeline(&ctx);

        let init = CuteTmaLoadPipelineInitOp::new(&mut ctx, pipeline, u32_value);

        let producer_state = CutePipelineStateAttr::producer(STAGES);
        let producer = CutePipelineStateNewOp::new(&mut ctx, producer_state);
        let producer_slot = producer.slot(&ctx);
        let producer_phase = producer.phase(&ctx);
        let producer_index = CutePipelineStateSlotOp::new(&mut ctx, producer_slot, producer_state);
        let acquire = CutePipelineProducerAcquireOp::new(
            &mut ctx,
            pipeline,
            producer_slot,
            producer_phase,
            producer_state,
        );
        let expect =
            CutePipelineProducerExpectTxOp::new(&mut ctx, pipeline, producer_slot, producer_state);
        let producer_advance = CutePipelineStateAdvanceOp::new(
            &mut ctx,
            producer_slot,
            producer_phase,
            producer_state,
        );
        let producer_next_slot = producer_advance.next_slot(&ctx);
        let producer_next_phase = producer_advance.next_phase(&ctx);
        let tail = CutePipelineProducerTailOp::new(
            &mut ctx,
            pipeline,
            producer_next_slot,
            producer_next_phase,
            producer_state,
        );

        let consumer_state = CutePipelineStateAttr::consumer(STAGES);
        let consumer = CutePipelineStateNewOp::new(&mut ctx, consumer_state);
        let consumer_slot = consumer.slot(&ctx);
        let consumer_phase = consumer.phase(&ctx);
        let consumer_index = CutePipelineStateSlotOp::new(&mut ctx, consumer_slot, consumer_state);
        let wait = CutePipelineConsumerWaitOp::new(
            &mut ctx,
            pipeline,
            consumer_slot,
            consumer_phase,
            consumer_state,
        );
        let release =
            CutePipelineConsumerReleaseOp::new(&mut ctx, pipeline, consumer_slot, consumer_state);
        let consumer_advance = CutePipelineStateAdvanceOp::new(
            &mut ctx,
            consumer_slot,
            consumer_phase,
            consumer_state,
        );

        assert!(make.verify(&ctx).is_ok());
        assert!(init.verify(&ctx).is_ok());
        assert!(producer.verify(&ctx).is_ok());
        assert!(producer_index.verify(&ctx).is_ok());
        assert!(acquire.verify(&ctx).is_ok());
        assert!(expect.verify(&ctx).is_ok());
        assert!(producer_advance.verify(&ctx).is_ok());
        assert!(tail.verify(&ctx).is_ok());
        assert!(consumer.verify(&ctx).is_ok());
        assert!(consumer_index.verify(&ctx).is_ok());
        assert!(wait.verify(&ctx).is_ok());
        assert!(release.verify(&ctx).is_ok());
        assert!(consumer_advance.verify(&ctx).is_ok());
        assert_eq!(make.pipeline_type(&ctx).unwrap().storage_bytes(), Some(48));
        assert_eq!(
            expect.completion_barrier(&ctx).get_type(&ctx),
            base.get_type(&ctx)
        );
    }

    #[test]
    fn pipeline_config_rejects_impossible_rings() {
        let ctx = Context::new();
        assert!(
            CuteTmaLoadPipelineType {
                stages: 0,
                consumer_warps: 8,
                transaction_bytes: 1,
            }
            .verify(&ctx)
            .is_err()
        );
        assert!(
            CuteTmaLoadPipelineType {
                stages: 3,
                consumer_warps: 33,
                transaction_bytes: 1,
            }
            .verify(&ctx)
            .is_err()
        );
        assert!(
            CuteTmaLoadPipelineType {
                stages: 3,
                consumer_warps: 8,
                transaction_bytes: 0,
            }
            .verify(&ctx)
            .is_err()
        );
        let overflow = CuteTmaLoadPipelineType {
            stages: u64::MAX,
            consumer_warps: 8,
            transaction_bytes: 1,
        };
        assert_eq!(overflow.storage_bytes(), None);
        assert!(overflow.verify(&ctx).is_err());
        assert!(CutePipelineStateAttr::producer(0).verify(&ctx).is_err());
    }

    #[test]
    fn make_rejects_wrong_storage_and_weak_promises() {
        let Setup {
            mut ctx,
            base,
            u32_value,
            ..
        } = setup();
        let weak = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            base,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            4,
        );
        assert!(
            weak.verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("at least 8")
        );

        let wrong_pointee_ty = u32_value.get_type(&ctx);
        let wrong_pointee = shared_pointer(&mut ctx, wrong_pointee_ty, true);
        let wrong = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            wrong_pointee,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        assert!(
            wrong
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("unsigned u64 Barrier")
        );

        let barrier = u64_type(&ctx);
        let immutable = shared_pointer(&mut ctx, barrier, false);
        let immutable = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            immutable,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        assert!(immutable.verify(&ctx).is_err());

        let barrier = u64_type(&ctx);
        let generic_pointer_ty: TypeHandle =
            MirPtrType::get_generic(&mut ctx, barrier, true).into();
        let generic_pointer = undef(&mut ctx, generic_pointer_ty);
        let generic = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            generic_pointer,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        assert!(generic.verify(&ctx).is_err());

        let signed_u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let signed_storage = shared_pointer(&mut ctx, signed_u64, true);
        let signed_storage = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            signed_storage,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        assert!(signed_storage.verify(&ctx).is_err());
    }

    #[test]
    fn effects_reject_role_stage_and_scalar_drift() {
        let Setup {
            mut ctx,
            base,
            u32_value,
            u64_value,
        } = setup();
        let make = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            base,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        let pipeline = make.pipeline(&ctx);

        let wrong_role = CutePipelineProducerAcquireOp::new(
            &mut ctx,
            pipeline,
            u32_value,
            u32_value,
            CutePipelineStateAttr::consumer(STAGES),
        );
        assert!(
            wrong_role
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("Producer")
        );

        let wrong_stages = CutePipelineConsumerWaitOp::new(
            &mut ctx,
            pipeline,
            u32_value,
            u32_value,
            CutePipelineStateAttr::consumer(2),
        );
        assert!(
            wrong_stages
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("3")
        );

        let wrong_scalar = CutePipelineConsumerReleaseOp::new(
            &mut ctx,
            pipeline,
            u64_value,
            CutePipelineStateAttr::consumer(STAGES),
        );
        assert!(
            wrong_scalar
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("32-bit")
        );
    }

    #[test]
    fn expect_tx_rejects_barrier_pointer_type_drift() {
        let Setup {
            mut ctx,
            base,
            u32_value,
            ..
        } = setup();
        let make = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            base,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        let pipeline = make.pipeline(&ctx);
        let base_pointer_ty = base.get_type(&ctx);
        let wrong_result: TypeHandle =
            MirPtrType::get_generic(&mut ctx, base_pointer_ty, true).into();
        let raw = Operation::new(
            &mut ctx,
            CutePipelineProducerExpectTxOp::get_concrete_op_info(),
            vec![wrong_result],
            vec![pipeline, u32_value],
            vec![],
            0,
        );
        let expect = CutePipelineProducerExpectTxOp::wrap(raw);
        expect
            .set_attr_pipeline_producer_expect_state(&ctx, CutePipelineStateAttr::producer(STAGES));

        assert!(
            expect
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("must match")
        );
    }

    #[test]
    fn ghost_pipeline_rejects_an_unrelated_consumer() {
        let Setup { mut ctx, base, .. } = setup();
        let make = CuteTmaLoadPipelineMakeOp::new(
            &mut ctx,
            base,
            STAGES,
            CONSUMER_WARPS,
            TRANSACTION_BYTES,
            8,
        );
        let pipeline = make.pipeline(&ctx);
        let _unrelated = CutePipelineStateSlotOp::new(
            &mut ctx,
            pipeline,
            CutePipelineStateAttr::producer(STAGES),
        );

        assert!(
            make.verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("lifecycle operations")
        );
    }
}

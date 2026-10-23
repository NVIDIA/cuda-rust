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
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::{CuteAlignmentAttr, CutePipelineRoleAttr, CutePipelineStateAttr};
use crate::types::CuteTmaLoadPipelineType;

/// Shared storage occupied by one hardware barrier, in bytes.
const BARRIER_BYTES: u64 = 8;

/// Intern the unsigned 32-bit scalar type used for cursor state.
fn u32_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Unsigned).into()
}

/// Intern the unsigned 64-bit scalar type used for extents and indices.
fn u64_type(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Unsigned).into()
}

/// Check that the SSA value has an unsigned 32-bit integer type.
fn is_u32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned())
}

/// Check that the SSA value has an unsigned 64-bit integer type.
fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

/// Read the ring configuration from a compiler-only pipeline handle's type.
fn pipeline_type_of(ctx: &Context, value: Value) -> Option<CuteTmaLoadPipelineType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTmaLoadPipelineType>()
        .cloned()
}

/// Resolve a handle only when defined directly by a load-pipeline constructor.
fn pipeline_make_of(ctx: &Context, value: Value) -> Option<CuteTmaLoadPipelineMakeOp> {
    let producer = value.defining_op()?;
    if Operation::get_opid(producer, ctx) != CuteTmaLoadPipelineMakeOp::get_opid_static() {
        return None;
    }
    let make = CuteTmaLoadPipelineMakeOp::wrap(producer);
    (make.pipeline(ctx) == value).then_some(make)
}

/// Require a valid direct pipeline handle before checking a lifecycle operation.
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

/// Require a valid cursor role and stage count consistent with the pipeline.
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

/// Downcast an SSA value to its MIR pointer type for barrier checks.
fn barrier_pointer_of(ctx: &Context, value: Value) -> Option<MirPtrType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .cloned()
}

/// Require mutable shared storage containing unsigned 64-bit barrier words.
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

/// Recognize lifecycle operations allowed to consume a direct pipeline handle.
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

mod construction;
mod lifecycle;
mod state;

pub use construction::{CuteTmaLoadPipelineInitOp, CuteTmaLoadPipelineMakeOp};
pub use lifecycle::{
    CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
    CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp,
};
pub use state::{CutePipelineStateAdvanceOp, CutePipelineStateNewOp, CutePipelineStateSlotOp};

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;

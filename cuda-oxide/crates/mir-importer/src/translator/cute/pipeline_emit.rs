/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit the semantic TMA load-pipeline protocol.
//!
//! Only the pipeline handle is compiler-only. Cursor state stays as the same
//! two Rust scalars used by the source loop:
//!
//! ```text
//! make -> init
//! state_new -> slot -> acquire/expect or wait/release -> advance -> tail
//! ```

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::terminator::helpers::emit_goto;
use crate::translator::types::translate_type;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::{CutePipelineRoleAttr, CutePipelineStateAttr};
use dialect_cute::pipeline_ops::{
    CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
    CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
    CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
    CuteTmaLoadPipelineMakeOp,
};
use dialect_cute::types::CuteTmaLoadPipelineType;
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr};
use dialect_mir::ops::{
    MirCastOp, MirExtractFieldOp, MirFieldAddrOp, MirInsertFieldOp, MirLoadOp, MirStoreOp,
    MirUndefOp,
};
use dialect_mir::types::{MirPtrType, MirStructType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::IntegerType;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

use super::pipeline::{
    TmaLoadPipelineRust, decode_pipeline_state, decode_pipeline_state_receiver,
    decode_tma_load_pipeline,
};

const BARRIER_BYTES: u64 = 8;

fn insert(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
) {
    if let Some(prev) = prev {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block, ctx);
    }
}

fn finish(
    ctx: &mut Context,
    effect: Ptr<Operation>,
    target: &Option<usize>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    what: &str,
) -> TranslationResult<Ptr<Operation>> {
    let Some(target) = target else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!("{what} call without target is not supported"))
        );
    };
    Ok(emit_goto(ctx, *target, effect, block_map, loc))
}

#[allow(clippy::too_many_arguments)]
fn finish_value(
    ctx: &mut Context,
    producer: Ptr<Operation>,
    value: Value,
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    what: &str,
) -> TranslationResult<Ptr<Operation>> {
    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{what} destination with projections is not supported"
            ))
        );
    }
    let after_store = value_map
        .store_local(ctx, destination.local, value, block, Some(producer))
        .unwrap_or(producer);
    finish(ctx, after_store, target, block_map, loc, what)
}

fn destination_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    what: &str,
) -> TranslationResult<TypeHandle> {
    let ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} destination type: {error:?}"
        )))
    })?;
    translate_type(ctx, &ty).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot translate {what} destination type: {}",
            error.disp(ctx)
        )))
    })
}

fn build_state(
    ctx: &mut Context,
    ty: TypeHandle,
    slot: Value,
    phase: Value,
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let undef = MirUndefOp::new(ctx, ty).get_operation();
    undef.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, undef, block, Some(prev));
    let mut value = undef.deref(ctx).get_result(0);
    let mut anchor = undef;
    for (index, field) in [slot, phase].into_iter().enumerate() {
        let op = Operation::new(
            ctx,
            MirInsertFieldOp::get_concrete_op_info(),
            vec![ty],
            vec![value, field],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        MirInsertFieldOp::new(op).set_attr_insert_index(ctx, FieldIndexAttr(index as u32));
        insert(ctx, op, block, Some(anchor));
        value = op.deref(ctx).get_result(0);
        anchor = op;
    }
    (value, anchor)
}

fn pipeline_facts_from_type(
    ty: &mir::Operand,
    body: &mir::Body,
    what: &str,
) -> TranslationResult<TmaLoadPipelineRust> {
    let ty = ty.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} pipeline type: {error:?}"
        )))
    })?;
    decode_tma_load_pipeline(&ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what} pipeline: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} needs a by-value TmaLoadPipeline receiver"
            )))
        })
}

fn state_facts_from_operand(
    operand: &mir::Operand,
    body: &mir::Body,
    what: &str,
) -> TranslationResult<CutePipelineStateAttr> {
    let ty = operand.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read {what} state type: {error:?}"
        )))
    })?;
    decode_pipeline_state_receiver(&ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid {what} state: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "{what} needs a PipelineState receiver"
            )))
        })
}

fn require_state(
    state: CutePipelineStateAttr,
    stages: u64,
    role: Option<CutePipelineRoleAttr>,
    what: &str,
) -> TranslationResult<()> {
    if state.stages != stages {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            format!(
                "{what} state has {} stages, but its pipeline has {stages}",
                state.stages
            )
        )));
    }
    if let Some(role) = role
        && state.role != role
    {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            format!("{what} needs {role:?} state, got {:?}", state.role)
        )));
    }
    Ok(())
}

fn recover_shared_barrier_base(
    ctx: &mut Context,
    base: Value,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let pointer = {
        let ty = base.get_type(ctx);
        ty.deref(ctx).downcast_ref::<MirPtrType>().cloned()
    }
    .ok_or_else(|| {
        pliron::input_error_noloc!(TranslationErr::unsupported(
            "TMA load-pipeline base must be a pointer".to_owned()
        ))
    })?;
    if !pointer.is_mutable {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            "TMA load-pipeline base must be mutable".to_owned()
        )));
    }
    let valid_barrier = {
        let pointee = pointer.pointee.deref(ctx);
        pointee
            .downcast_ref::<IntegerType>()
            .is_some_and(|barrier| barrier.width() == 64 && barrier.is_unsigned())
    };
    if !valid_barrier {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            "TMA load-pipeline base must point to the canonical u64 Barrier storage".to_owned()
        )));
    }
    if pointer.address_space == address_space::SHARED {
        return Ok((base, prev));
    }
    if pointer.address_space != address_space::GENERIC {
        return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
            format!(
                "TMA load-pipeline base must be generic or CTA-shared memory, found address space {}",
                pointer.address_space
            )
        )));
    }

    let shared: TypeHandle =
        MirPtrType::get(ctx, pointer.pointee, true, address_space::SHARED).into();
    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared],
        vec![base],
        vec![],
        0,
    );
    cast.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    insert(ctx, cast, block, prev);
    Ok((cast.deref(ctx).get_result(0), Some(cast)))
}

fn field_addr(
    ctx: &mut Context,
    pointer: Value,
    index: usize,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
    what: &str,
) -> TranslationResult<(Value, TypeHandle, Ptr<Operation>)> {
    let field = {
        let ty = pointer.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
            return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
                format!("{what} must be a pointer to PipelineState")
            )));
        };
        let pointee = pointer.pointee.deref(ctx);
        let Some(state) = pointee.downcast_ref::<MirStructType>() else {
            return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
                format!("{what} must point to PipelineState")
            )));
        };
        let Some(field) = state.field_types.get(index).copied() else {
            return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
                format!("{what} PipelineState has no field {index}")
            )));
        };
        (field, pointer.is_mutable, pointer.address_space)
    };
    let field_pointer: TypeHandle = MirPtrType::get(ctx, field.0, field.1, field.2).into();
    // Go through the one construction door so `aggregate_ty` is stamped;
    // the verifier rejects a `mir.field_addr` without it.
    let op = MirFieldAddrOp::build(ctx, pointer, field_pointer, index as u32)?;
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, prev);
    Ok((op.deref(ctx).get_result(0), field.0, op))
}

fn load(
    ctx: &mut Context,
    pointer: Value,
    ty: TypeHandle,
    block: Ptr<BasicBlock>,
    prev: Ptr<Operation>,
    loc: &Location,
) -> (Value, Ptr<Operation>) {
    let op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![ty],
        vec![pointer],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    MirLoadOp::new(op).set_volatile(ctx, false);
    insert(ctx, op, block, Some(prev));
    (op.deref(ctx).get_result(0), op)
}

struct StateValues {
    slot: Value,
    phase: Value,
    slot_pointer: Option<Value>,
    phase_pointer: Option<Value>,
    anchor: Ptr<Operation>,
}

fn state_values(
    ctx: &mut Context,
    value: Value,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    loc: &Location,
    what: &str,
) -> TranslationResult<StateValues> {
    let ty = value.get_type(ctx);
    if ty.deref(ctx).downcast_ref::<MirPtrType>().is_some() {
        let (slot_pointer, slot_ty, slot_addr) = field_addr(ctx, value, 0, block, prev, loc, what)?;
        let (slot, slot_load) = load(ctx, slot_pointer, slot_ty, block, slot_addr, loc);
        let (phase_pointer, phase_ty, phase_addr) =
            field_addr(ctx, value, 1, block, Some(slot_load), loc, what)?;
        let (phase, phase_load) = load(ctx, phase_pointer, phase_ty, block, phase_addr, loc);
        return Ok(StateValues {
            slot,
            phase,
            slot_pointer: Some(slot_pointer),
            phase_pointer: Some(phase_pointer),
            anchor: phase_load,
        });
    }

    let fields = {
        let ty = ty.deref(ctx);
        let Some(state) = ty.downcast_ref::<MirStructType>() else {
            return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
                format!("{what} must be PipelineState or a reference to it")
            )));
        };
        if state.field_types.len() < 2 {
            return Err(pliron::input_error_noloc!(TranslationErr::unsupported(
                format!("{what} PipelineState must contain slot and phase")
            )));
        }
        [state.field_types[0], state.field_types[1]]
    };
    let mut values = Vec::with_capacity(2);
    let mut anchor = prev;
    for (index, field) in fields.into_iter().enumerate() {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![field],
            vec![value],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
        insert(ctx, op, block, anchor);
        values.push(op.deref(ctx).get_result(0));
        anchor = Some(op);
    }
    Ok(StateValues {
        slot: values[0],
        phase: values[1],
        slot_pointer: None,
        phase_pointer: None,
        anchor: anchor.expect("two state extracts always produce an anchor"),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_make(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 || !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "TMA load-pipeline construction expects one base and a local destination"
                    .to_owned()
            )
        );
    }
    let rust_ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read TMA load-pipeline result type: {error:?}"
        )))
    })?;
    let facts = decode_tma_load_pipeline(&rust_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid TMA load-pipeline result: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "TMA load-pipeline construction must return TmaLoadPipeline".to_owned()
            ))
        })?;
    let (base, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (base, after) = recover_shared_barrier_base(ctx, base, block, after.or(prev), &loc)?;
    let make = CuteTmaLoadPipelineMakeOp::new(
        ctx,
        base,
        facts.stages,
        facts.consumer_warps,
        facts.transaction_bytes,
        BARRIER_BYTES,
    );
    let op = make.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after);
    value_map.set_direct_value(destination.local, make.pipeline(ctx));
    finish(ctx, op, target, block_map, loc, "tma_load_pipeline_make")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_init(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "TMA load-pipeline init expects pipeline and thread".to_owned()
            )
        );
    }
    let _facts = pipeline_facts_from_type(&args[0], body, "pipeline init")?;
    let (pipeline, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let (thread, after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        after.or(prev),
        loc.clone(),
    )?;
    let init = CuteTmaLoadPipelineInitOp::new(ctx, pipeline, thread);
    let op = init.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, after.or(prev));
    finish(ctx, op, target, block_map, loc, "tma_load_pipeline_init")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_state_new(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "pipeline state construction expects no arguments".to_owned()
            )
        );
    }
    let rust_ty = destination.ty(body.locals()).map_err(|error| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "cannot read pipeline state result type: {error:?}"
        )))
    })?;
    let state = decode_pipeline_state(&rust_ty)
        .map_err(|error| {
            pliron::input_error_noloc!(TranslationErr::unsupported(format!(
                "invalid pipeline state result: {error}"
            )))
        })?
        .ok_or_else(|| {
            pliron::input_error_noloc!(TranslationErr::unsupported(
                "pipeline state construction must return PipelineState".to_owned()
            ))
        })?;
    let operation = CutePipelineStateNewOp::new(ctx, state);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, prev);
    let ty = destination_type(ctx, body, destination, "pipeline state")?;
    let (value, anchor) = build_state(
        ctx,
        ty,
        operation.slot(ctx),
        operation.phase(ctx),
        block,
        op,
        &loc,
    );
    finish_value(
        ctx,
        anchor,
        value,
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "pipeline_state_new",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_state_slot(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("pipeline state slot expects one receiver".to_owned())
        );
    }
    let state = state_facts_from_operand(&args[0], body, "pipeline state slot")?;
    let (value, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let values = state_values(
        ctx,
        value,
        block,
        after.or(prev),
        &loc,
        "pipeline state slot",
    )?;
    let operation = CutePipelineStateSlotOp::new(ctx, values.slot, state);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(values.anchor));
    finish_value(
        ctx,
        op,
        operation.index(ctx),
        destination,
        target,
        block,
        value_map,
        block_map,
        loc,
        "pipeline_state_slot",
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_state_advance(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported("pipeline state advance expects one receiver".to_owned())
        );
    }
    let state = state_facts_from_operand(&args[0], body, "pipeline state advance")?;
    let (value, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let values = state_values(
        ctx,
        value,
        block,
        after.or(prev),
        &loc,
        "pipeline state advance",
    )?;
    let Some(slot_pointer) = values.slot_pointer else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "pipeline state advance receiver must be mutable".to_owned()
            )
        );
    };
    let Some(phase_pointer) = values.phase_pointer else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "pipeline state advance receiver must be mutable".to_owned()
            )
        );
    };
    let operation = CutePipelineStateAdvanceOp::new(ctx, values.slot, values.phase, state);
    let op = operation.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(values.anchor));
    let slot_store = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![slot_pointer, operation.next_slot(ctx)],
        vec![],
        0,
    );
    slot_store.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, slot_store, block, Some(op));
    let phase_store = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![phase_pointer, operation.next_phase(ctx)],
        vec![],
        0,
    );
    phase_store.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, phase_store, block, Some(slot_store));
    finish(
        ctx,
        phase_store,
        target,
        block_map,
        loc,
        "pipeline_state_advance",
    )
}

enum Lifecycle {
    ProducerAcquire,
    ProducerExpectTx,
    ConsumerWait,
    ConsumerRelease,
    ProducerTail,
}

#[allow(clippy::too_many_arguments)]
fn emit_lifecycle(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: Option<&mir::Place>,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    lifecycle: Lifecycle,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "pipeline lifecycle call expects pipeline and state".to_owned()
            )
        );
    }
    let facts = pipeline_facts_from_type(&args[0], body, "pipeline lifecycle")?;
    let state = state_facts_from_operand(&args[1], body, "pipeline lifecycle")?;
    let required_role = match lifecycle {
        Lifecycle::ConsumerWait | Lifecycle::ConsumerRelease => CutePipelineRoleAttr::Consumer,
        Lifecycle::ProducerAcquire | Lifecycle::ProducerExpectTx | Lifecycle::ProducerTail => {
            CutePipelineRoleAttr::Producer
        }
    };
    require_state(
        state,
        facts.stages,
        Some(required_role),
        "pipeline lifecycle",
    )?;
    let (pipeline, after) =
        translate_operand(ctx, body, &args[0], value_map, block, prev, loc.clone())?;
    let pipeline_type = pipeline.get_type(ctx);
    let matches_pipeline = pipeline_type
        .deref(ctx)
        .downcast_ref::<CuteTmaLoadPipelineType>()
        .is_some_and(|ty| {
            ty.stages == facts.stages
                && ty.consumer_warps == facts.consumer_warps
                && ty.transaction_bytes == facts.transaction_bytes
        });
    if !matches_pipeline {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "pipeline SSA value does not match its Rust type".to_owned()
            )
        );
    }
    let (state_value, after) = translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block,
        after.or(prev),
        loc.clone(),
    )?;
    let values = state_values(
        ctx,
        state_value,
        block,
        after.or(prev),
        &loc,
        "pipeline lifecycle",
    )?;

    let (op, result, what) = match lifecycle {
        Lifecycle::ProducerAcquire => {
            let operation =
                CutePipelineProducerAcquireOp::new(ctx, pipeline, values.slot, values.phase, state);
            (operation.get_operation(), None, "pipeline_producer_acquire")
        }
        Lifecycle::ProducerExpectTx => {
            let operation = CutePipelineProducerExpectTxOp::new(ctx, pipeline, values.slot, state);
            (
                operation.get_operation(),
                Some(operation.completion_barrier(ctx)),
                "pipeline_producer_expect_tx",
            )
        }
        Lifecycle::ConsumerWait => {
            let operation =
                CutePipelineConsumerWaitOp::new(ctx, pipeline, values.slot, values.phase, state);
            (operation.get_operation(), None, "pipeline_consumer_wait")
        }
        Lifecycle::ConsumerRelease => {
            let operation = CutePipelineConsumerReleaseOp::new(ctx, pipeline, values.slot, state);
            (operation.get_operation(), None, "pipeline_consumer_release")
        }
        Lifecycle::ProducerTail => {
            let operation =
                CutePipelineProducerTailOp::new(ctx, pipeline, values.slot, values.phase, state);
            (operation.get_operation(), None, "pipeline_producer_tail")
        }
    };
    op.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, op, block, Some(values.anchor));
    if let Some(result) = result {
        let destination = destination.expect("expect-tx is the only value lifecycle operation");
        finish_value(
            ctx,
            op,
            result,
            destination,
            target,
            block,
            value_map,
            block_map,
            loc,
            what,
        )
    } else {
        finish(ctx, op, target, block_map, loc, what)
    }
}

macro_rules! lifecycle_emitter {
    ($name:ident, $kind:expr) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $name(
            ctx: &mut Context,
            body: &mir::Body,
            args: &[mir::Operand],
            target: &Option<usize>,
            block: Ptr<BasicBlock>,
            prev: Option<Ptr<Operation>>,
            value_map: &mut ValueMap,
            block_map: &[Ptr<BasicBlock>],
            loc: Location,
        ) -> TranslationResult<Ptr<Operation>> {
            emit_lifecycle(
                ctx, body, args, None, target, block, prev, value_map, block_map, loc, $kind,
            )
        }
    };
}

lifecycle_emitter!(emit_producer_acquire, Lifecycle::ProducerAcquire);
lifecycle_emitter!(emit_consumer_wait, Lifecycle::ConsumerWait);
lifecycle_emitter!(emit_consumer_release, Lifecycle::ConsumerRelease);
lifecycle_emitter!(emit_producer_tail, Lifecycle::ProducerTail);

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_producer_expect_tx(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_lifecycle(
        ctx,
        body,
        args,
        Some(destination),
        target,
        block,
        prev,
        value_map,
        block_map,
        loc,
        Lifecycle::ProducerExpectTx,
    )
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Barrier-ring configuration and producer/consumer cursor contracts.

use std::collections::{HashMap, HashSet};

use pliron::{
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{
    Result,
    cfg::{operation_dominates, operation_reaches},
    ghosts::pipeline_semantic_ids,
    invalid,
    provenance::{
        OrdinaryFacts, poison_missing_forwarding_sources, propagate_ordinary_plumbing,
        seed_unknown_integer_carriers,
    },
};
use crate::{
    attributes::{CutePipelineRoleAttr, CutePipelineStateAttr},
    ops::gemm_tma::CuteTmaCopy2dOp,
    ops::pipeline::{
        CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
        CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
        CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
        CuteTmaLoadPipelineMakeOp,
    },
    types::CuteTmaViewType,
};

/// Static ring facts and the constructor identifying one load pipeline.
#[derive(Clone, Copy)]
struct LoadPipelineFacts {
    /// Direct constructor that defines the compiler-only pipeline handle.
    make: Ptr<Operation>,
    /// Number of full and empty barrier slots in each ring.
    stages: u64,
    /// Total bytes expected by one full barrier for its stage.
    transaction_bytes: u32,
}

/// Require a direct pipeline constructor and recover its ring configuration.
fn resolve_load_pipeline(ctx: &Context, value: Value) -> Result<LoadPipelineFacts> {
    let Some(make) = value.defining_op() else {
        return Err(invalid(
            "TMA load-pipeline handle reached a block argument; a direct tma_load_pipeline_make result is required",
        ));
    };
    let producer_id = Operation::get_opid(make, ctx);
    if producer_id != CuteTmaLoadPipelineMakeOp::get_opid_static() {
        return Err(invalid(format!(
            "TMA load-pipeline handle is produced by `{producer_id}` instead of `cute.tma_load_pipeline_make`"
        )));
    }
    let make_op = CuteTmaLoadPipelineMakeOp::wrap(make);
    if make_op.pipeline(ctx) != value {
        return Err(invalid(
            "TMA load-pipeline operand is not the make operation's pipeline result",
        ));
    }
    let pipeline = make_op
        .pipeline_type(ctx)
        .ok_or_else(|| invalid("TMA load-pipeline make has no pipeline type"))?;
    Ok(LoadPipelineFacts {
        make,
        stages: pipeline.stages,
        transaction_bytes: pipeline.transaction_bytes,
    })
}

/// Read the producer/consumer role and ring size from a recognized cursor operation.
fn pipeline_state_for_op(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Option<CutePipelineStateAttr> {
    if *opid == CutePipelineStateNewOp::get_opid_static() {
        CutePipelineStateNewOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineStateSlotOp::get_opid_static() {
        CutePipelineStateSlotOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineStateAdvanceOp::get_opid_static() {
        CutePipelineStateAdvanceOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerAcquireOp::get_opid_static() {
        CutePipelineProducerAcquireOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        CutePipelineProducerExpectTxOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineConsumerWaitOp::get_opid_static() {
        CutePipelineConsumerWaitOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        CutePipelineConsumerReleaseOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerTailOp::get_opid_static() {
        CutePipelineProducerTailOp::wrap(operation).state(ctx)
    } else {
        None
    }
}

/// Find a lifecycle operation's pipeline handle; cursor-only operations return `None`.
fn pipeline_operand(ctx: &Context, operation: Ptr<Operation>, opid: &OpId) -> Option<Value> {
    if *opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
        Some(CuteTmaLoadPipelineInitOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerAcquireOp::get_opid_static() {
        Some(CutePipelineProducerAcquireOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        Some(CutePipelineProducerExpectTxOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineConsumerWaitOp::get_opid_static() {
        Some(CutePipelineConsumerWaitOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        Some(CutePipelineConsumerReleaseOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerTailOp::get_opid_static() {
        Some(CutePipelineProducerTailOp::wrap(operation).pipeline(ctx))
    } else {
        None
    }
}

/// Sum the static byte sizes of every direct TMA copy completed by one
/// `expect_tx` result. The result may not escape through any other operand.
fn expected_tma_transaction_bytes(ctx: &Context, operation: Ptr<Operation>) -> Result<u64> {
    let barrier = CutePipelineProducerExpectTxOp::wrap(operation).completion_barrier(ctx);
    let uses = barrier.uses(ctx);
    if uses.is_empty() {
        return Err(invalid(
            "cute.pipeline_producer_expect_tx needs at least one direct cute.tma_copy_2d user",
        ));
    }
    let mut copies = HashSet::new();
    let mut bytes = 0u64;
    for r#use in uses {
        let user = r#use.user_op();
        if Operation::get_opid(user, ctx) != CuteTmaCopy2dOp::get_opid_static()
            || r#use.find_index(ctx) != 4
            || CuteTmaCopy2dOp::wrap(user).completion_barrier(ctx) != barrier
        {
            return Err(invalid(
                "cute.pipeline_producer_expect_tx result may only be the completion-barrier operand of cute.tma_copy_2d",
            ));
        }
        if !copies.insert(user) {
            continue;
        }
        let copy = CuteTmaCopy2dOp::wrap(user);
        let source_ty = copy.source(ctx).get_type(ctx);
        let source_ty = source_ty.deref(ctx);
        let view = source_ty
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| invalid("TMA copy attached to expect_tx has no TMA view type"))?;
        let tile_bytes = view.tile_bytes(ctx).ok_or_else(|| {
            invalid("TMA copy attached to expect_tx has no static tile byte count")
        })?;
        bytes = bytes
            .checked_add(tile_bytes)
            .ok_or_else(|| invalid("TMA transaction byte total overflows u64"))?;
    }
    Ok(bytes)
}

/// Origin of one ordinary cursor component, including untrusted incoming values.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineFact {
    /// Ring slot derived from this cursor-creation operation.
    Slot(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Barrier parity derived from this cursor-creation operation.
    Phase(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Scalar without a proven cursor origin; it cannot satisfy lifecycle checks.
    Unknown(#[doc = "Original SSA value lacking semantic provenance."] Value),
}

/// Require an unambiguous slot or phase fact rooted at cursor creation.
fn pipeline_state_root(
    facts: &OrdinaryFacts<PipelineFact>,
    value: Value,
    slot: bool,
    what: &str,
) -> Result<Ptr<Operation>> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no pipeline-state provenance after ordinary SSA forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges values from different pipeline-state stories"
        )));
    }
    match found.iter().next().copied() {
        Some(PipelineFact::Slot(root)) if slot => Ok(root),
        Some(PipelineFact::Phase(root)) if !slot => Ok(root),
        _ => Err(invalid(format!(
            "{what} is not derived from the required cute.pipeline_state_new result"
        ))),
    }
}

/// Track slot/phase origins through advances and forwarding, preserving unknown inputs for consumer checks.
fn build_pipeline_facts(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> OrdinaryFacts<PipelineFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        if Operation::get_opid(*operation, ctx) == CutePipelineStateNewOp::get_opid_static() {
            let state = CutePipelineStateNewOp::wrap(*operation);
            facts.insert(state.slot(ctx), PipelineFact::Slot(*operation));
            facts.insert(state.phase(ctx), PipelineFact::Phase(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CutePipelineStateAdvanceOp::get_opid_static()
            {
                continue;
            }
            let advance = CutePipelineStateAdvanceOp::wrap(*operation);
            let Some(slot_facts) = facts.at(advance.slot(ctx)).cloned() else {
                continue;
            };
            let Some(phase_facts) = facts.at(advance.phase(ctx)).cloned() else {
                continue;
            };
            for slot in &slot_facts {
                let PipelineFact::Slot(root) = slot else {
                    continue;
                };
                if phase_facts.contains(&PipelineFact::Phase(*root))
                    && CutePipelineStateNewOp::wrap(*root).state(ctx) == advance.state(ctx)
                {
                    changed |= facts.insert(advance.next_slot(ctx), PipelineFact::Slot(*root));
                    changed |= facts.insert(advance.next_phase(ctx), PipelineFact::Phase(*root));
                }
            }
        }
        if !changed {
            break;
        }
    }

    seed_unknown_integer_carriers(ctx, operations, &mut facts, 32, PipelineFact::Unknown);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, PipelineFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

/// Static lifecycle sites collected for one producer or consumer cursor root.
#[derive(Default)]
struct PipelineStateEvents {
    /// Sites that widen the cursor slot for addressing.
    slots: Vec<Ptr<Operation>>,
    /// Sites that advance this cursor's slot and phase.
    advances: Vec<Ptr<Operation>>,
    /// Producer sites waiting for an empty stage.
    acquires: Vec<Ptr<Operation>>,
    /// Producer sites announcing the stage's transaction bytes.
    expects: Vec<Ptr<Operation>>,
    /// Consumer sites waiting for a full stage.
    waits: Vec<Ptr<Operation>>,
    /// Consumer sites releasing the stage after reading.
    releases: Vec<Ptr<Operation>>,
    /// Producer exit sites draining the empty-barrier ring.
    tails: Vec<Ptr<Operation>>,
}

/// Require dominance from cursor creation through acquire/expect/advance and a reachable exit tail.
pub(super) fn require_producer_pipeline_order(
    ctx: &Context,
    root: Ptr<Operation>,
    acquire: Ptr<Operation>,
    expect: Ptr<Operation>,
    advance: Ptr<Operation>,
    tail: Ptr<Operation>,
) -> Result {
    if operation_dominates(ctx, root, acquire)
        && operation_dominates(ctx, acquire, expect)
        && operation_dominates(ctx, expect, advance)
        && operation_reaches(ctx, advance, tail)
    {
        Ok(())
    } else {
        Err(invalid(
            "producer pipeline CFG must order state_new -> acquire -> expect_tx -> advance, with tail reachable only after the producer loop",
        ))
    }
}

/// Validate handle ownership, complete cursor lifecycles, CFG order, and expected TMA byte totals.
pub(super) fn verify_pipeline_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = pipeline_semantic_ids();
    let facts = build_pipeline_facts(ctx, all_ops);
    let makes: Vec<_> = all_ops
        .iter()
        .copied()
        .filter(|operation| {
            Operation::get_opid(*operation, ctx) == CuteTmaLoadPipelineMakeOp::get_opid_static()
        })
        .collect();
    let mut init_by_make: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut events: HashMap<Ptr<Operation>, PipelineStateEvents> = HashMap::new();
    let mut binding: HashMap<Ptr<Operation>, Ptr<Operation>> = HashMap::new();

    for make in &makes {
        if make.deref(ctx).get_parent_block().is_none() {
            return Err(invalid("TMA load-pipeline make is not attached to a block"));
        }
        let handle = CuteTmaLoadPipelineMakeOp::wrap(*make).pipeline(ctx);
        let uses = handle.uses(ctx);
        if uses.is_empty() {
            return Err(invalid(
                "cute.tma_load_pipeline_make has no lifecycle consumer",
            ));
        }
        for r#use in uses {
            let user = r#use.user_op();
            let user_id = Operation::get_opid(user, ctx);
            if !ids.contains(&user_id)
                || r#use.find_index(ctx) != 0
                || pipeline_operand(ctx, user, &user_id) != Some(handle)
            {
                return Err(invalid(format!(
                    "TMA load-pipeline handle has an invalid use by `{user_id}`"
                )));
            }
        }
    }

    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "pipeline operation `{opid}` is not attached to a block"
            )));
        }
        if opid == CutePipelineStateNewOp::get_opid_static() {
            events.entry(*operation).or_default();
            continue;
        }

        let state_root = if opid == CutePipelineStateSlotOp::get_opid_static() {
            let semantic = CutePipelineStateSlotOp::wrap(*operation);
            let root = pipeline_state_root(
                &facts,
                semantic.slot(ctx),
                true,
                "cute.pipeline_state_slot slot",
            )?;
            if semantic.state(ctx) != CutePipelineStateNewOp::wrap(root).state(ctx) {
                return Err(invalid(
                    "cute.pipeline_state_slot facts differ from its pipeline_state_new root",
                ));
            }
            events.entry(root).or_default().slots.push(*operation);
            Some(root)
        } else if opid == CutePipelineStateAdvanceOp::get_opid_static() {
            let semantic = CutePipelineStateAdvanceOp::wrap(*operation);
            let slot_root = pipeline_state_root(
                &facts,
                semantic.slot(ctx),
                true,
                "cute.pipeline_state_advance slot",
            )?;
            let phase_root = pipeline_state_root(
                &facts,
                semantic.phase(ctx),
                false,
                "cute.pipeline_state_advance phase",
            )?;
            if slot_root != phase_root
                || semantic.state(ctx) != CutePipelineStateNewOp::wrap(slot_root).state(ctx)
            {
                return Err(invalid(
                    "cute.pipeline_state_advance must consume slot and phase from one matching pipeline_state_new",
                ));
            }
            events
                .entry(slot_root)
                .or_default()
                .advances
                .push(*operation);
            Some(slot_root)
        } else if opid == CuteTmaLoadPipelineMakeOp::get_opid_static() {
            None
        } else if opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
            let pipeline = resolve_load_pipeline(
                ctx,
                CuteTmaLoadPipelineInitOp::wrap(*operation).pipeline(ctx),
            )?;
            if !makes.contains(&pipeline.make) {
                return Err(invalid(
                    "pipeline init refers to a make outside its semantic scope",
                ));
            }
            init_by_make
                .entry(pipeline.make)
                .or_default()
                .push(*operation);
            None
        } else {
            let state = pipeline_state_for_op(ctx, *operation, &opid)
                .ok_or_else(|| invalid(format!("pipeline operation `{opid}` has no state")))?;
            let operation_ref = operation.deref(ctx);
            let slot_root = pipeline_state_root(
                &facts,
                operation_ref.get_operand(1),
                true,
                &format!("`{opid}` slot"),
            )?;
            let needs_phase = opid == CutePipelineProducerAcquireOp::get_opid_static()
                || opid == CutePipelineConsumerWaitOp::get_opid_static()
                || opid == CutePipelineProducerTailOp::get_opid_static();
            if needs_phase {
                let phase_root = pipeline_state_root(
                    &facts,
                    operation_ref.get_operand(2),
                    false,
                    &format!("`{opid}` phase"),
                )?;
                if phase_root != slot_root {
                    return Err(invalid(format!(
                        "`{opid}` mixes slot and phase from different pipeline states"
                    )));
                }
            }
            if CutePipelineStateNewOp::wrap(slot_root).state(ctx) != Some(state) {
                return Err(invalid(format!(
                    "`{opid}` state facts differ from its pipeline_state_new root"
                )));
            }

            let handle = pipeline_operand(ctx, *operation, &opid)
                .ok_or_else(|| invalid(format!("`{opid}` has no load-pipeline handle")))?;
            let pipeline = resolve_load_pipeline(ctx, handle)?;
            if !makes.contains(&pipeline.make) {
                return Err(invalid(
                    "pipeline lifecycle operation refers to a make outside its semantic scope",
                ));
            }
            if state.stages != pipeline.stages {
                return Err(invalid(format!(
                    "pipeline state on `{opid}` has {} stages but its handle has {}",
                    state.stages, pipeline.stages
                )));
            }
            if let Some(previous) = binding.insert(slot_root, pipeline.make)
                && previous != pipeline.make
            {
                return Err(invalid(
                    "one ordinary pipeline state is used with multiple load-pipeline handles",
                ));
            }

            let root_events = events.entry(slot_root).or_default();
            if opid == CutePipelineProducerAcquireOp::get_opid_static() {
                root_events.acquires.push(*operation);
            } else if opid == CutePipelineProducerExpectTxOp::get_opid_static() {
                root_events.expects.push(*operation);
                let copied = expected_tma_transaction_bytes(ctx, *operation)?;
                if copied != u64::from(pipeline.transaction_bytes) {
                    return Err(invalid(format!(
                        "cute.pipeline_producer_expect_tx promises {} bytes, but its direct TMA copies move {copied} bytes",
                        pipeline.transaction_bytes
                    )));
                }
            } else if opid == CutePipelineConsumerWaitOp::get_opid_static() {
                root_events.waits.push(*operation);
            } else if opid == CutePipelineConsumerReleaseOp::get_opid_static() {
                root_events.releases.push(*operation);
            } else if opid == CutePipelineProducerTailOp::get_opid_static() {
                root_events.tails.push(*operation);
            }
            Some(slot_root)
        };

        if let Some(root) = state_root {
            let root_state = CutePipelineStateNewOp::wrap(root)
                .state(ctx)
                .expect("locally verified pipeline state root");
            if let Some(state) = pipeline_state_for_op(ctx, *operation, &opid)
                && state != root_state
            {
                return Err(invalid(format!(
                    "`{opid}` does not carry its pipeline-state root's role and stage count"
                )));
            }
        }
    }

    let mut roles_by_make: HashMap<Ptr<Operation>, (usize, usize)> = HashMap::new();
    for (root, root_events) in &events {
        let state = CutePipelineStateNewOp::wrap(*root)
            .state(ctx)
            .expect("locally verified pipeline state root");
        let Some(make) = binding.get(root).copied() else {
            return Err(invalid(
                "cute.pipeline_state_new is not connected to any load-pipeline lifecycle",
            ));
        };
        if root_events.slots.is_empty() || root_events.advances.is_empty() {
            return Err(invalid(
                "each pipeline state needs state_slot and state_advance operations",
            ));
        }
        let roles = roles_by_make.entry(make).or_default();
        match state.role {
            CutePipelineRoleAttr::Producer => {
                roles.0 += 1;
                if root_events.acquires.len() != 1
                    || root_events.expects.len() != 1
                    || root_events.advances.len() != 1
                    || root_events.slots.len() != 1
                    || root_events.tails.len() != 1
                    || !root_events.waits.is_empty()
                    || !root_events.releases.is_empty()
                {
                    return Err(invalid(
                        "producer pipeline state needs acquire, expect_tx, advance, and exactly one tail, with no consumer events",
                    ));
                }
                let acquire = root_events.acquires[0];
                let expect = root_events.expects[0];
                let advance = root_events.advances[0];
                let tail = root_events.tails[0];
                require_producer_pipeline_order(ctx, *root, acquire, expect, advance, tail)?;
                let barrier = CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(ctx);
                for r#use in barrier.uses(ctx) {
                    let copy = r#use.user_op();
                    if !operation_dominates(ctx, expect, copy)
                        || !operation_dominates(ctx, copy, advance)
                    {
                        return Err(invalid(
                            "every TMA copy must be CFG-ordered after expect_tx and before producer state advance",
                        ));
                    }
                }
            }
            CutePipelineRoleAttr::Consumer => {
                roles.1 += 1;
                if root_events.waits.len() != 1
                    || root_events.releases.len() != 1
                    || root_events.advances.len() != 1
                    || root_events.slots.len() != 1
                    || !root_events.acquires.is_empty()
                    || !root_events.expects.is_empty()
                    || !root_events.tails.is_empty()
                {
                    return Err(invalid(
                        "consumer pipeline state needs wait, release, and advance, with no producer events",
                    ));
                }
                let wait = root_events.waits[0];
                let release = root_events.releases[0];
                let advance = root_events.advances[0];
                if !operation_dominates(ctx, *root, wait)
                    || !operation_dominates(ctx, wait, release)
                    || !operation_dominates(ctx, release, advance)
                    || !operation_reaches(ctx, advance, wait)
                {
                    return Err(invalid(
                        "consumer pipeline CFG must order state_new -> wait -> release -> advance -> next wait",
                    ));
                }
            }
        }
    }
    for make in makes {
        if init_by_make
            .get(&make)
            .is_none_or(|initializers| initializers.len() != 1)
        {
            return Err(invalid(
                "each tma_load_pipeline_make needs exactly one initialization",
            ));
        }
        if roles_by_make.get(&make).copied() != Some((1, 1)) {
            return Err(invalid(
                "each tma_load_pipeline_make needs exactly one complete producer state and one complete consumer state",
            ));
        }
        let initializer = init_by_make[&make][0];
        for (root, bound_make) in &binding {
            if *bound_make == make && !operation_dominates(ctx, initializer, *root) {
                return Err(invalid(
                    "load-pipeline initialization must dominate both producer and consumer state creation",
                ));
            }
        }
    }
    Ok(())
}

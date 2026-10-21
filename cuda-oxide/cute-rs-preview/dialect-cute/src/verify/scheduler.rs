/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent output-tile grids and their ordinary cursor provenance.

use std::collections::{HashMap, HashSet};

use dialect_mir::ops::{MirNotOp, control_flow::MirCondBranchOp};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{
    Result,
    cfg::{block_reaches, operation_reaches},
    ghosts::scheduler_semantic_ids,
    invalid,
    provenance::{
        OrdinaryFacts, poison_missing_forwarding_sources, propagate_ordinary_plumbing,
        seed_unknown_integer_carriers,
    },
};
use crate::{
    attributes::CuteTileGridAttr,
    ops::scheduler::{
        CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp,
        CuteSchedulerNew1dOp, CuteWorkTileCoordinatesOp,
    },
    types::CuteWorkTileType,
};

/// Read the static output grid from a recognized scheduler operation.
fn scheduler_grid(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Result<CuteTileGridAttr> {
    let grid = if *opid == CuteSchedulerNew1dOp::get_opid_static() {
        CuteSchedulerNew1dOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerHasWorkOp::get_opid_static() {
        CuteSchedulerHasWorkOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerCurrentOp::get_opid_static() {
        CuteSchedulerCurrentOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerAdvanceOp::get_opid_static() {
        CuteSchedulerAdvanceOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteWorkTileCoordinatesOp::get_opid_static() {
        CuteWorkTileCoordinatesOp::wrap(operation)
            .work_tile(ctx)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<CuteWorkTileType>()
            .map(|tile| tile.grid)
    } else {
        None
    };
    grid.ok_or_else(|| invalid(format!("`{opid}` is missing its scheduler tile grid")))
}

/// Origin of an ordinary persistent-scheduler state component.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum SchedulerFact {
    /// Linear tile cursor derived from one scheduler constructor.
    Current(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Physical-grid stride derived from the same scheduler constructor.
    Stride(#[doc = "Operation identifying the semantic origin of this carrier."] Ptr<Operation>),
    /// Scalar without scheduler provenance, invalid in a scheduler-state merge.
    Unknown(#[doc = "Original SSA value lacking semantic provenance."] Value),
}

/// Require exactly one current-index or stride origin of the requested kind.
pub(super) fn scheduler_root(
    facts: &OrdinaryFacts<SchedulerFact>,
    value: Value,
    current: bool,
    what: &str,
) -> Result<Ptr<Operation>> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no scheduler provenance after ordinary SSA forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges values from different scheduler stories"
        )));
    }
    match found.iter().next().copied() {
        Some(SchedulerFact::Current(root)) if current => Ok(root),
        Some(SchedulerFact::Stride(root)) if !current => Ok(root),
        _ => Err(invalid(format!(
            "{what} is not derived from the required scheduler_new_1d result"
        ))),
    }
}

/// Propagate scheduler origins through advances and ordinary carriers before poisoning unknown edges.
pub(super) fn build_scheduler_facts(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> OrdinaryFacts<SchedulerFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        if Operation::get_opid(*operation, ctx) == CuteSchedulerNew1dOp::get_opid_static() {
            let scheduler = CuteSchedulerNew1dOp::wrap(*operation);
            facts.insert(scheduler.current(ctx), SchedulerFact::Current(*operation));
            facts.insert(scheduler.stride(ctx), SchedulerFact::Stride(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CuteSchedulerAdvanceOp::get_opid_static() {
                continue;
            }
            let advance = CuteSchedulerAdvanceOp::wrap(*operation);
            let Some(current_facts) = facts.at(advance.current(ctx)).cloned() else {
                continue;
            };
            let Some(stride_facts) = facts.at(advance.stride(ctx)).cloned() else {
                continue;
            };
            for current in &current_facts {
                let SchedulerFact::Current(root) = current else {
                    continue;
                };
                if stride_facts.contains(&SchedulerFact::Stride(*root)) {
                    changed |= facts.insert(advance.next(ctx), SchedulerFact::Current(*root));
                }
            }
        }
        if !changed {
            break;
        }
    }

    // A factless scalar is an ordinary, unrelated value.  Seeding it only
    // after the semantic fixed point lets a valid loop recurrence establish
    // itself first, while still poisoning a merge with an arbitrary edge.
    seed_unknown_integer_carriers(ctx, operations, &mut facts, 64, SchedulerFact::Unknown);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, SchedulerFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

/// Find a branch whose has-work condition guards the selected scheduler cursor.
fn scheduler_guard(
    ctx: &Context,
    has_work: Value,
) -> Option<(Ptr<Operation>, Ptr<BasicBlock>, Ptr<BasicBlock>)> {
    let mut value = has_work;
    let mut negated = false;
    loop {
        let uses = value.uses(ctx);
        if uses.len() != 1 || uses[0].find_index(ctx) != 0 {
            return None;
        }
        let user = uses[0].user_op();
        let opid = Operation::get_opid(user, ctx);
        if opid == MirNotOp::get_opid_static() {
            let operation = user.deref(ctx);
            if operation.get_num_operands() != 1 || operation.get_num_results() != 1 {
                return None;
            }
            value = operation.get_result(0);
            negated = !negated;
            continue;
        }
        if opid != MirCondBranchOp::get_opid_static() {
            return None;
        }
        let operation = user.deref(ctx);
        if operation.get_num_successors() != 2 || operation.get_operand(0) != value {
            return None;
        }
        let work_index = usize::from(negated);
        return Some((
            user,
            operation.get_successor(work_index),
            operation.get_successor(1 - work_index),
        ));
    }
}

/// Check one scheduler's state origins, loop advance, work guard, and coordinate consumers.
pub(super) fn verify_scheduler_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = scheduler_semantic_ids();
    let facts = build_scheduler_facts(ctx, all_ops);
    let mut roots = HashSet::new();
    let mut has_work_roots = HashSet::new();
    let mut current_roots = HashSet::new();
    let mut advance_roots = HashSet::new();
    let mut coordinate_roots = HashSet::new();
    let mut has_work_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut current_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut advance_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut coordinates_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut currents = Vec::new();
    let mut coordinates = Vec::new();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "scheduler operation `{opid}` is not attached to a block"
            )));
        }
        let grid = scheduler_grid(ctx, *operation, &opid)?;
        if opid == CuteSchedulerNew1dOp::get_opid_static() {
            roots.insert(*operation);
        } else if opid == CuteSchedulerHasWorkOp::get_opid_static() {
            let semantic = CuteSchedulerHasWorkOp::wrap(*operation);
            let root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_has_work current",
            )?;
            if CuteSchedulerNew1dOp::wrap(root).tile_grid(ctx) != Some(grid) {
                return Err(invalid(
                    "cute.scheduler_has_work grid differs from its scheduler_new_1d root",
                ));
            }
            if !semantic.has_work(ctx).is_used(ctx) {
                return Err(invalid(
                    "cute.scheduler_has_work result is unused; it must guard the scheduler body",
                ));
            }
            has_work_roots.insert(root);
            has_work_by_root.entry(root).or_default().push(*operation);
        } else if opid == CuteSchedulerCurrentOp::get_opid_static() {
            let semantic = CuteSchedulerCurrentOp::wrap(*operation);
            let root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_current current",
            )?;
            if CuteSchedulerNew1dOp::wrap(root).tile_grid(ctx) != Some(grid) {
                return Err(invalid(
                    "cute.scheduler_current grid differs from its scheduler_new_1d root",
                ));
            }
            current_roots.insert(root);
            current_by_root.entry(root).or_default().push(*operation);
            currents.push(*operation);
        } else if opid == CuteWorkTileCoordinatesOp::get_opid_static() {
            coordinates.push(*operation);
        } else if opid == CuteSchedulerAdvanceOp::get_opid_static() {
            let semantic = CuteSchedulerAdvanceOp::wrap(*operation);
            let current_root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_advance current",
            )?;
            let stride_root = scheduler_root(
                &facts,
                semantic.stride(ctx),
                false,
                "cute.scheduler_advance stride",
            )?;
            if current_root != stride_root
                || CuteSchedulerNew1dOp::wrap(current_root).tile_grid(ctx) != Some(grid)
            {
                return Err(invalid(
                    "cute.scheduler_advance must use current, stride, and grid from one scheduler_new_1d",
                ));
            }
            advance_roots.insert(current_root);
            advance_by_root
                .entry(current_root)
                .or_default()
                .push(*operation);
        }
    }

    for current in &currents {
        let tile = CuteSchedulerCurrentOp::wrap(*current).work_tile(ctx);
        let uses = tile.uses(ctx);
        if uses.is_empty()
            || uses.iter().any(|r#use| {
                Operation::get_opid(r#use.user_op(), ctx)
                    != CuteWorkTileCoordinatesOp::get_opid_static()
                    || r#use.find_index(ctx) != 0
            })
        {
            return Err(invalid(
                "cute.scheduler_current work tile must be used only by at least one cute.work_tile_coordinates",
            ));
        }
    }
    for coordinates in coordinates {
        let tile = CuteWorkTileCoordinatesOp::wrap(coordinates).work_tile(ctx);
        let Some(producer) = tile.defining_op() else {
            return Err(invalid(
                "cute.work_tile_coordinates reached a block argument; a direct scheduler_current result is required",
            ));
        };
        if Operation::get_opid(producer, ctx) != CuteSchedulerCurrentOp::get_opid_static()
            || !currents.contains(&producer)
        {
            return Err(invalid(
                "cute.work_tile_coordinates must use a scheduler_current attached to this module",
            ));
        }
        let root = scheduler_root(
            &facts,
            CuteSchedulerCurrentOp::wrap(producer).current(ctx),
            true,
            "cute.work_tile_coordinates scheduler current",
        )?;
        coordinate_roots.insert(root);
        coordinates_by_root
            .entry(root)
            .or_default()
            .push(coordinates);
    }

    for root in roots {
        if !has_work_roots.contains(&root)
            || !current_roots.contains(&root)
            || !coordinate_roots.contains(&root)
            || !advance_roots.contains(&root)
        {
            return Err(invalid(
                "each cute.scheduler_new_1d needs a complete has_work -> current -> coordinates -> advance story",
            ));
        }
        let has_work = &has_work_by_root[&root];
        let current = &current_by_root[&root];
        let coordinates = &coordinates_by_root[&root];
        let advances = &advance_by_root[&root];
        if has_work.len() != 1
            || current.len() != 1
            || coordinates.len() != 1
            || advances.len() != 1
        {
            return Err(invalid(
                "each scheduler root needs exactly one static has_work/current/coordinates/advance protocol",
            ));
        }
        let Some((guard, work_successor, exit_successor)) =
            scheduler_guard(ctx, CuteSchedulerHasWorkOp::wrap(has_work[0]).has_work(ctx))
        else {
            return Err(invalid(
                "cute.scheduler_has_work must exclusively guard one conditional branch (optionally through mir.not)",
            ));
        };
        let current_block = current[0]
            .deref(ctx)
            .get_parent_block()
            .expect("attached scheduler current");
        if !operation_reaches(ctx, root, has_work[0])
            || !block_reaches(ctx, work_successor, current_block)
            || block_reaches(ctx, exit_successor, current_block)
            || !operation_reaches(ctx, current[0], coordinates[0])
            || !operation_reaches(ctx, coordinates[0], advances[0])
            || !operation_reaches(ctx, advances[0], guard)
        {
            return Err(invalid(
                "scheduler CFG must be new -> guarded work current -> coordinates -> advance -> next guard, with the exit edge outside the work body",
            ));
        }
    }
    Ok(())
}

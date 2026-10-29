/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function scopes, reachability, dominance, and source-order protocol checks.

use std::collections::{HashMap, HashSet};

use dialect_mir::ops::{MirGotoOp, function::MirFuncOp};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

use super::{Result, invalid};

/// The semantic verifier treats every MIR function as an independent story
/// namespace.  Tests and tools may also place operations directly below a
/// module, in which case the verifier root is their namespace.
fn semantic_scope(
    ctx: &Context,
    root: Ptr<Operation>,
    operation: Ptr<Operation>,
) -> Ptr<Operation> {
    let mut current = operation;
    while let Some(parent) = current.deref(ctx).get_parent_op(ctx) {
        if Operation::get_opid(parent, ctx) == MirFuncOp::get_opid_static() {
            return parent;
        }
        if parent == root {
            return root;
        }
        current = parent;
    }
    root
}

/// Partition operations by their enclosing MIR function, keeping standalone IR in the root scope.
pub(super) fn operations_by_scope(
    ctx: &Context,
    root: Ptr<Operation>,
    all_ops: &[Ptr<Operation>],
) -> Vec<Vec<Ptr<Operation>>> {
    let mut scopes: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    for operation in all_ops {
        let scope = semantic_scope(ctx, root, *operation);
        scopes.entry(scope).or_default().push(*operation);
    }
    scopes.into_values().collect()
}

/// Test whether any successor path in the same region reaches the destination block.
pub(super) fn block_reaches(
    ctx: &Context,
    start: Ptr<BasicBlock>,
    destination: Ptr<BasicBlock>,
) -> bool {
    let Some(region) = start.deref(ctx).get_parent_region() else {
        return false;
    };
    if destination.deref(ctx).get_parent_region() != Some(region) {
        return false;
    }
    let mut worklist = vec![start];
    let mut visited = HashSet::new();
    while let Some(block) = worklist.pop() {
        if !visited.insert(block) {
            continue;
        }
        if block == destination {
            return true;
        }
        let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        worklist.extend(terminator.deref(ctx).successors());
    }
    false
}

/// Locate an operation's parent block and source-order index, or return `None` if detached.
fn operation_position(
    ctx: &Context,
    operation: Ptr<Operation>,
) -> Option<(Ptr<BasicBlock>, usize)> {
    let block = operation.deref(ctx).get_parent_block()?;
    let index = block
        .deref(ctx)
        .iter(ctx)
        .position(|candidate| candidate == operation)?;
    Some((block, index))
}

/// Existential CFG reachability with source order inside one block.  This is
/// used only to connect role-specific semantic milestones; each role's local
/// sequence is checked separately by `require_linear_protocol_order` where the
/// importer promises a straight-line call chain.
pub(super) fn operation_reaches(
    ctx: &Context,
    source: Ptr<Operation>,
    destination: Ptr<Operation>,
) -> bool {
    let Some((source_block, source_index)) = operation_position(ctx, source) else {
        return false;
    };
    let Some((destination_block, destination_index)) = operation_position(ctx, destination) else {
        return false;
    };
    if source_block == destination_block {
        return source_index < destination_index;
    }
    let Some(terminator) = source_block.deref(ctx).get_terminator(ctx) else {
        return false;
    };
    terminator
        .deref(ctx)
        .successors()
        .any(|successor| block_reaches(ctx, successor, destination_block))
}

/// Require source order in one block or dominance from the region entry across blocks.
pub(super) fn operation_dominates(
    ctx: &Context,
    source: Ptr<Operation>,
    destination: Ptr<Operation>,
) -> bool {
    let Some((source_block, source_index)) = operation_position(ctx, source) else {
        return false;
    };
    let Some((destination_block, destination_index)) = operation_position(ctx, destination) else {
        return false;
    };
    if source_block == destination_block {
        return source_index < destination_index;
    }
    let Some(region) = source_block.deref(ctx).get_parent_region() else {
        return false;
    };
    if destination_block.deref(ctx).get_parent_region() != Some(region) {
        return false;
    }
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(entry) = blocks.first().copied() else {
        return false;
    };
    let all: HashSet<_> = blocks.iter().copied().collect();
    let mut dominators: HashMap<Ptr<BasicBlock>, HashSet<Ptr<BasicBlock>>> = blocks
        .iter()
        .copied()
        .map(|block| {
            if block == entry {
                (block, HashSet::from([entry]))
            } else {
                (block, all.clone())
            }
        })
        .collect();

    loop {
        let mut changed = false;
        for block in blocks.iter().copied().filter(|block| *block != entry) {
            let predecessors: Vec<_> = block
                .uses(ctx)
                .into_iter()
                .filter_map(|edge| edge.user_op().deref(ctx).get_parent_block())
                .collect();
            let mut next = if let Some(first) = predecessors.first() {
                dominators.get(first).cloned().unwrap_or_default()
            } else {
                HashSet::new()
            };
            for predecessor in predecessors.iter().skip(1) {
                let predecessor_dominators =
                    dominators.get(predecessor).cloned().unwrap_or_default();
                next.retain(|candidate| predecessor_dominators.contains(candidate));
            }
            next.insert(block);
            if dominators.get(&block) != Some(&next) {
                dominators.insert(block, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dominators
        .get(&destination_block)
        .is_some_and(|set| set.contains(&source_block))
}

/// Follow one straight-line source sequence across the block splits emitted
/// for Rust call terminators. Every crossed edge must be an unconditional
/// goto into a block with no side entry.
pub(super) fn require_linear_protocol_order(
    ctx: &Context,
    label: &str,
    ordered: &[Ptr<Operation>],
) -> Result<Vec<Ptr<Operation>>> {
    let Some(first) = ordered.first() else {
        return Err(invalid(format!("{label} has no operations to order")));
    };
    let first_block = first
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| invalid(format!("{label} contains an operation outside a block")))?;
    let region = first_block
        .deref(ctx)
        .get_parent_region()
        .ok_or_else(|| invalid(format!("{label} starts in a block outside a region")))?;
    if ordered.iter().any(|operation| {
        operation
            .deref(ctx)
            .get_parent_block()
            .and_then(|block| block.deref(ctx).get_parent_region())
            != Some(region)
    }) {
        return Err(invalid(format!(
            "{label} operations must stay in one region"
        )));
    }

    let mut current = first_block;
    let mut visited = HashSet::new();
    let mut next_expected = 0usize;
    let mut range = Vec::new();
    loop {
        if !visited.insert(current) {
            return Err(invalid(format!(
                "{label} loops before reaching its final operation"
            )));
        }
        let operations: Vec<_> = current.deref(ctx).iter(ctx).collect();
        let start = if current == first_block {
            operations
                .iter()
                .position(|operation| operation == first)
                .ok_or_else(|| {
                    invalid(format!(
                        "{label} contains an operation missing from its parent block"
                    ))
                })?
        } else {
            0
        };
        for operation in operations.into_iter().skip(start) {
            range.push(operation);
            if ordered.get(next_expected) == Some(&operation) {
                next_expected += 1;
                if next_expected == ordered.len() {
                    return Ok(range);
                }
            } else if ordered.contains(&operation) {
                return Err(invalid(format!(
                    "{label} is not in the required source order"
                )));
            }
        }

        let terminator = current
            .deref(ctx)
            .get_terminator(ctx)
            .ok_or_else(|| invalid(format!("{label} ends a block before its final operation")))?;
        if Operation::get_opid(terminator, ctx) != MirGotoOp::get_opid_static()
            || terminator.deref(ctx).get_num_successors() != 1
        {
            return Err(invalid(format!(
                "{label} crosses an extra exit instead of one unconditional successor"
            )));
        }
        let successor = terminator.deref(ctx).get_successor(0);
        if successor.deref(ctx).get_parent_region() != Some(region) {
            return Err(invalid(format!("{label} leaves its source region")));
        }
        if successor.preds(ctx).len() != 1 {
            return Err(invalid(format!(
                "{label} crosses a block with a side entry"
            )));
        }
        current = successor;
    }
}

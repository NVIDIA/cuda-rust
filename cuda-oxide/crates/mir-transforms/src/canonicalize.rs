/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small loop-shape rewrites needed before unrolling.
//!
//! A source loop can have several paths back to its header, usually because it
//! contains more than one `continue`. The unroller is much simpler when there is
//! one latch, so we route every back-edge through a synthetic block first:
//!
//! ```text
//!   continue_a(values_a) ---> header(args)
//!   continue_b(values_b) ---> header(args)
//!
//! becomes
//!
//!   continue_a(values_a) -+-> unified_latch(args) -> header(args)
//!   continue_b(values_b) -+
//! ```
//!
//! Each old edge keeps the values it passed to the header. The new latch receives
//! those values as block arguments and forwards them unchanged. The caller must
//! recompute dominance, loop structure, and induction facts after this rewrite.

use dialect_mir::ops::MirGotoOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::context::Context;
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::value::Value;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analyses::loop_info::{LoopId, LoopInfo};

/// Result of trying to put one loop into the single-latch form the unroller
/// consumes.
pub(crate) enum CanonicalizeOutcome {
    /// The loop already had one unconditional back-edge.
    Unchanged,
    /// The IR was normalized. Cached analyses must be discarded.
    Changed,
    /// The CFG did not provide enough well-formed edge information to rewrite
    /// safely. The caller warns and skips the requested unroll.
    Unsupported(String),
}

/// Route direct outside uses of header-carried values through block arguments.
///
/// Rust mem2reg can legally leave a header argument in use after an early
/// `break`: the header dominates the shared post-loop block. Full unrolling
/// cannot replace that use with one final value because each break path needs
/// the value from its own copy. This small LCSSA-style rewrite propagates the
/// current value from every loop exit through outside forwarding and join
/// blocks to each original user.
///
/// Only header arguments are handled: they dominate every loop block, so each
/// exit edge has an unambiguous value. Other loop definitions remain a
/// conservative skip in the unroller.
pub(crate) fn close_header_liveouts(
    ctx: &mut Context,
    info: &LoopInfo,
    id: LoopId,
) -> CanonicalizeOutcome {
    let lp = &info.loops()[id];
    let header_args: Vec<Value> = lp.header.deref(ctx).arguments().collect();

    // Snapshot only the original direct outside uses. Successor operands added
    // below are deliberately not candidates for replacement.
    let mut liveouts = Vec::new();
    let mut user_blocks = FxHashSet::default();
    for header_arg in header_args {
        let mut outside_uses = Vec::new();
        for r#use in header_arg.uses(ctx) {
            let Some(block) = r#use.user_op().deref(ctx).get_parent_block() else {
                return CanonicalizeOutcome::Unsupported(
                    "a loop-carried value has a use outside any basic block".into(),
                );
            };
            if !lp.blocks.contains(&block) {
                user_blocks.insert(block);
                outside_uses.push(r#use);
            }
        }
        if !outside_uses.is_empty() {
            liveouts.push((header_arg, outside_uses));
        }
    }
    if liveouts.is_empty() {
        return CanonicalizeOutcome::Unchanged;
    }

    // Walk backward from every user to the loop boundary. If an outside path
    // comes from before the loop, the header value is not available there and a
    // local block-argument rewrite is insufficient.
    let mut propagation_blocks = FxHashSet::default();
    let mut worklist: Vec<_> = user_blocks.into_iter().collect();
    while let Some(block) = worklist.pop() {
        if !propagation_blocks.insert(block) {
            continue;
        }
        let incoming_edges = block.uses(ctx);
        if incoming_edges.is_empty() {
            return CanonicalizeOutcome::Unsupported(
                "a loop-carried live-out is reachable from outside the loop".into(),
            );
        }
        for edge_use in incoming_edges {
            let term = edge_use.user_op();
            let Some(source) = term.deref(ctx).get_parent_block() else {
                return CanonicalizeOutcome::Unsupported(
                    "a live-out predecessor has no basic block".into(),
                );
            };
            let opobj = Operation::get_op_dyn(term, ctx);
            if op_cast::<dyn BranchOpInterface>(opobj.as_ref()).is_none() {
                return CanonicalizeOutcome::Unsupported(
                    "a live-out edge does not expose branch operands".into(),
                );
            }
            if !lp.blocks.contains(&source) {
                worklist.push(source);
            }
        }
    }

    for (header_arg, original_uses) in liveouts {
        // Give every block in the propagation slice a value for this header
        // argument before wiring edges, so cycles in outside control flow are
        // harmless.
        let mut block_values = FxHashMap::default();
        let mut arg_indices = FxHashMap::default();
        for &block in &propagation_blocks {
            let index = BasicBlock::push_argument(block, ctx, header_arg.get_type(ctx));
            block_values.insert(block, block.deref(ctx).get_argument(index));
            arg_indices.insert(block, index);
        }

        for &block in &propagation_blocks {
            let target_index = arg_indices[&block];
            for edge_use in block.uses(ctx) {
                let term = edge_use.user_op();
                let source = term
                    .deref(ctx)
                    .get_parent_block()
                    .expect("validated predecessor block");
                let incoming = if lp.blocks.contains(&source) {
                    header_arg
                } else {
                    block_values[&source]
                };
                let opobj = Operation::get_op_dyn(term, ctx);
                let branch = op_cast::<dyn BranchOpInterface>(opobj.as_ref())
                    .expect("validated branch interface");
                let appended =
                    branch.add_successor_operand(ctx, edge_use.find_index(ctx), incoming);
                debug_assert_eq!(appended, target_index);
            }
        }

        for original_use in original_uses {
            let block = original_use
                .user_op()
                .deref(ctx)
                .get_parent_block()
                .expect("validated live-out user block");
            header_arg.replace_use_with(ctx, original_use, &block_values[&block]);
        }
    }

    CanonicalizeOutcome::Changed
}

/// Merge all back-edges of loop `id` through one block with the header's
/// argument signature. This is the `insertUniqueBackedgeBlock` part of LLVM's
/// LoopSimplify, kept deliberately small for the unroller's needs.
pub(crate) fn merge_backedges(
    ctx: &mut Context,
    info: &LoopInfo,
    id: LoopId,
) -> CanonicalizeOutcome {
    let lp = &info.loops()[id];

    let header = lp.header;
    let nargs = header.deref(ctx).get_num_arguments();
    let arg_types = header
        .deref(ctx)
        .arguments()
        .map(|arg| arg.get_type(ctx))
        .collect();

    // Validate every edge before mutating anything. A block can name the header
    // more than once, so record successor slots rather than only source blocks.
    let mut seen_blocks = FxHashSet::default();
    let mut backedges = Vec::new();
    for &source in &lp.latches {
        if !seen_blocks.insert(source) {
            continue;
        }
        let Some(term) = source.deref(ctx).get_terminator(ctx) else {
            return CanonicalizeOutcome::Unsupported(
                "a loop back-edge block has no terminator".into(),
            );
        };
        let opobj = Operation::get_op_dyn(term, ctx);
        let Some(branch) = op_cast::<dyn BranchOpInterface>(opobj.as_ref()) else {
            return CanonicalizeOutcome::Unsupported(
                "a loop back-edge terminator does not expose branch operands".into(),
            );
        };
        for (succ_idx, succ) in term.deref(ctx).successors().enumerate() {
            if succ == header {
                if branch.successor_operands(ctx, succ_idx).len() != nargs {
                    return CanonicalizeOutcome::Unsupported(
                        "a loop back-edge carries the wrong number of header values".into(),
                    );
                }
                backedges.push((term, succ_idx));
            }
        }
    }
    if backedges.is_empty() {
        return CanonicalizeOutcome::Unsupported(
            "LoopInfo reported a loop but no back-edge to its header was found".into(),
        );
    }
    if backedges.len() == 1 {
        let (term, succ_idx) = backedges[0];
        let successors: Vec<_> = term.deref(ctx).successors().collect();
        if succ_idx == 0
            && successors == [header]
            && Operation::get_op::<MirGotoOp>(term, ctx).is_some()
        {
            return CanonicalizeOutcome::Unchanged;
        }
    }

    let unified = BasicBlock::new(ctx, None, arg_types);
    unified.insert_before(ctx, header);

    // Build the forwarding edge before retargeting old edges. With recorded
    // successor slots either order is correct; doing this first also makes the
    // intended block shape explicit throughout the mutation.
    let args = unified.deref(ctx).arguments().collect();
    let goto = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        args,
        vec![header],
        0,
    );
    goto.insert_at_back(unified, ctx);

    for (term, succ_idx) in backedges {
        Operation::replace_successor(term, ctx, succ_idx, unified);
    }
    CanonicalizeOutcome::Changed
}

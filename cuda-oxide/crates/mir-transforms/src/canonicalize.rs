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
//!
//! A range `for` loop needs one more rewrite, [`thread_iterator_exit_test`]: its
//! exit test is a match on the `Option` from `Iterator::next`.

use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{
    MirCastOp, MirCondBranchOp, MirConstantOp, MirConstructEnumOp, MirEnumPayloadOp, MirEqOp,
    MirGetDiscriminantOp, MirGotoOp, MirNeOp, MirStorageDeadOp, MirStorageLiveOp,
};
use dialect_mir::types::MirEnumType;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::r#type::{Typed, TypedHandle};
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

/// The reason reported for a loop whose header does not exit and that
/// [`thread_iterator_exit_test`] does not recognize.
fn unrecognized_iterator_loop() -> CanonicalizeOutcome {
    CanonicalizeOutcome::Unsupported(
        "its exit test is not in the loop header, and it is not a range `for` loop over `a..b`"
            .into(),
    )
}

/// Bound on the blocks followed through one discriminant match, so a cycle of
/// match-like blocks cannot hang the pass.
const MAX_MATCH_BLOCKS: usize = 16;

/// One header successor of a range `for` loop and where the rewrite sends it.
struct IteratorArm {
    /// The header successor that builds the `Option` (`Some` or `None`).
    block: Ptr<BasicBlock>,
    /// The values the arm passes to the match block's arguments.
    join_operands: Vec<Value>,
    /// The `mir.construct_enum` that builds the matched value.
    construct: Ptr<Operation>,
    /// The variant it builds.
    variant: usize,
    /// The block the match reaches for that variant.
    target: Ptr<BasicBlock>,
    /// The operands of the edge into `target`, before the arm's values replace
    /// the match block's arguments.
    target_operands: Vec<Value>,
}

/// Rewrite the exit test of a range `for` loop into a header test.
///
/// With `Range::next` inlined, and rustc's JumpThreading off (cargo-oxide turns
/// it off because it may duplicate a block that holds a barrier), mem2reg leaves
/// `for i in a..b { body }` in this shape:
///
/// ```text
///   header(i):   c = not(i < b);        cond_br c [none, some]
///   some:        i1 = i + 1; o = Some(i);  goto join(o, i1)
///   none:        o = None;                 goto join(o, i)
///   join(o, n):  d = discriminant(o);   cond_br d == 0 [exit, check]
///   check:       cond_br d == 1 [body, unreachable]
///   body:        x = payload(o); ...;   goto header(n)
/// ```
///
/// The header does not exit, and `n` merges `i + 1` with `i`, so `i` is not an
/// induction variable. Each arm builds a known variant, so the match always
/// takes the same branch after it. The rewrite sends each arm straight to that
/// branch's target, with the arm's values:
///
/// ```text
///   header(i):   c = not(i < b);        cond_br c [none, some]
///   some:        i1 = i + 1;            goto body     (payload(o) is now i)
///   none:                               goto exit
///   body:        ...;                   goto header(i1)
/// ```
///
/// No block is copied; the caller's CFG cleanup removes the match blocks.
/// `(a..b).rev()` leaves the same shape, with the counter stepping down.
///
/// Returns `Unchanged` when the header exits, and `Unsupported`, with the IR
/// untouched, when the header does not exit and the loop misses this shape or
/// a value from the match is read where both the body and the exit reach.
pub(crate) fn thread_iterator_exit_test(
    ctx: &mut Context,
    info: &LoopInfo,
    id: LoopId,
) -> CanonicalizeOutcome {
    let lp = &info.loops()[id];
    let Some(header_term) = lp.header.deref(ctx).get_terminator(ctx) else {
        return CanonicalizeOutcome::Unchanged;
    };
    if Operation::get_op::<MirCondBranchOp>(header_term, ctx).is_none() {
        return CanonicalizeOutcome::Unchanged;
    }
    let arm_blocks: Vec<_> = header_term.deref(ctx).successors().collect();
    if arm_blocks.len() != 2
        || arm_blocks[0] == arm_blocks[1]
        || !arm_blocks.iter().all(|arm| lp.blocks.contains(arm))
    {
        return CanonicalizeOutcome::Unchanged;
    }

    // Each arm is entered only from the header and ends in `goto join(..)`.
    let mut join = None;
    for &arm in &arm_blocks {
        if arm.uses(ctx).len() != 1 {
            return unrecognized_iterator_loop();
        }
        let Some(term) = arm.deref(ctx).get_terminator(ctx) else {
            return unrecognized_iterator_loop();
        };
        if Operation::get_op::<MirGotoOp>(term, ctx).is_none() {
            return unrecognized_iterator_loop();
        }
        let target = term.deref(ctx).get_successor(0);
        if *join.get_or_insert(target) != target {
            return unrecognized_iterator_loop();
        }
    }
    let Some(join) = join else {
        return unrecognized_iterator_loop();
    };
    if join == lp.header || arm_blocks.contains(&join) || join.uses(ctx).len() != 2 {
        return unrecognized_iterator_loop();
    }

    // The match block reads the discriminant of exactly one of its arguments.
    let join_args: Vec<Value> = join.deref(ctx).arguments().collect();
    let mut tested = None;
    for op in join.deref(ctx).iter(ctx) {
        if Operation::get_op::<MirGetDiscriminantOp>(op, ctx).is_some() {
            let operand = op.deref(ctx).get_operand(0);
            let Some(index) = join_args.iter().position(|&arg| arg == operand) else {
                return unrecognized_iterator_loop();
            };
            if tested
                .replace(index)
                .is_some_and(|previous| previous != index)
            {
                return unrecognized_iterator_loop();
            }
        }
    }
    let Some(tested) = tested else {
        return unrecognized_iterator_loop();
    };

    // Follow the match for the variant each arm builds.
    let mut match_blocks = FxHashSet::default();
    let mut forwarded = FxHashMap::default();
    let mut arms = Vec::with_capacity(2);
    for &arm in &arm_blocks {
        let term = arm.deref(ctx).get_terminator(ctx).expect("checked above");
        let join_operands: Vec<Value> = term.deref(ctx).operands().collect();
        let Some(&matched) = join_operands.get(tested) else {
            return unrecognized_iterator_loop();
        };
        let Some(construct) = matched
            .defining_op()
            .filter(|&op| Operation::get_op::<MirConstructEnumOp>(op, ctx).is_some())
        else {
            return unrecognized_iterator_loop();
        };
        let Some(variant) = MirConstructEnumOp::new(construct)
            .get_attr_construct_enum_variant_index(ctx)
            .map(|attr| attr.0 as usize)
        else {
            return unrecognized_iterator_loop();
        };
        let discriminant = {
            let ty = matched.get_type(ctx);
            let ty = ty.deref(ctx);
            let Some(discriminant) = ty
                .downcast_ref::<MirEnumType>()
                .and_then(|enum_ty| enum_ty.variant_discriminants.get(variant).copied())
            else {
                return unrecognized_iterator_loop();
            };
            discriminant
        };
        let Some((target, branch, succ_idx)) = follow_discriminant_match(
            ctx,
            join,
            join_args[tested],
            discriminant,
            &mut match_blocks,
            &mut forwarded,
        ) else {
            return unrecognized_iterator_loop();
        };
        let Some(target_operands) = successor_operands(ctx, branch, succ_idx) else {
            return unrecognized_iterator_loop();
        };
        arms.push(IteratorArm {
            block: arm,
            join_operands,
            construct,
            variant,
            target,
            target_operands,
        });
    }
    if arms[0].target == arms[1].target || arms.iter().any(|arm| match_blocks.contains(&arm.target))
    {
        return unrecognized_iterator_loop();
    }

    // Blocks each target reaches without passing through the match again. A
    // value the match produces is replaced by one arm's value only where one
    // target alone reaches the use.
    let reach: Vec<FxHashSet<Ptr<BasicBlock>>> = arms
        .iter()
        .map(|arm| reachable_outside(ctx, arm.target, &match_blocks))
        .collect();
    let arm_of_block = |block: Ptr<BasicBlock>| -> Option<usize> {
        match (reach[0].contains(&block), reach[1].contains(&block)) {
            (true, false) => Some(0),
            (false, true) => Some(1),
            _ => None,
        }
    };

    // What a match block's argument holds on `arm`'s path. `None` for a value
    // computed inside the match.
    let arm_value = |arm: &IteratorArm, value: Value| -> Option<Value> {
        let value = forwarded.get(&value).copied().unwrap_or(value);
        if let Some(index) = join_args.iter().position(|&arg| arg == value) {
            Some(arm.join_operands[index])
        } else if value_block(ctx, value).is_some_and(|block| match_blocks.contains(&block)) {
            None
        } else {
            Some(value)
        }
    };

    // Plan every replacement before mutating anything.
    let mut replacements = Vec::new();
    for &match_block in &match_blocks {
        let args: Vec<Value> = match_block.deref(ctx).arguments().collect();
        for arg in args {
            for r#use in arg.uses(ctx) {
                let Some(block) = r#use.user_op().deref(ctx).get_parent_block() else {
                    return unrecognized_iterator_loop();
                };
                if match_blocks.contains(&block) {
                    continue;
                }
                let Some(arm) = arm_of_block(block) else {
                    return CanonicalizeOutcome::Unsupported(
                        "a value from its iterator's `next` is read where both the loop body and the loop exit reach"
                            .into(),
                    );
                };
                let Some(value) = arm_value(&arms[arm], arg) else {
                    return unrecognized_iterator_loop();
                };
                replacements.push((arg, r#use, value));
            }
        }
    }
    for &block in &match_blocks {
        for op in block.deref(ctx).iter(ctx) {
            for result in op.deref(ctx).results() {
                for r#use in result.uses(ctx) {
                    let user_block = r#use.user_op().deref(ctx).get_parent_block();
                    if !user_block.is_some_and(|b| match_blocks.contains(&b)) {
                        return unrecognized_iterator_loop();
                    }
                }
            }
        }
    }
    let mut new_edges = Vec::with_capacity(2);
    for arm in &arms {
        let operands: Option<Vec<Value>> = arm
            .target_operands
            .iter()
            .map(|&operand| arm_value(arm, operand))
            .collect();
        let Some(operands) = operands else {
            return unrecognized_iterator_loop();
        };
        new_edges.push(operands);
    }

    for (arm, operands) in arms.iter().zip(new_edges) {
        let old = arm
            .block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("checked above");
        Operation::erase(old, ctx);
        let goto = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![arm.target],
            0,
        );
        goto.insert_at_back(arm.block, ctx);
    }
    for (arg, r#use, value) in replacements {
        arg.replace_use_with(ctx, r#use, &value);
    }

    // `payload(o)` on an arm's own variant is the field the arm stored.
    for (index, arm) in arms.iter().enumerate() {
        let built = arm.construct.deref(ctx).get_result(0);
        for r#use in built.uses(ctx) {
            let user = r#use.user_op();
            let Some(payload) = Operation::get_op::<MirEnumPayloadOp>(user, ctx) else {
                continue;
            };
            let in_arm_region = user
                .deref(ctx)
                .get_parent_block()
                .is_some_and(|block| arm_of_block(block) == Some(index));
            let variant = payload
                .get_attr_payload_variant_index(ctx)
                .map(|attr| attr.0 as usize);
            let field = payload
                .get_attr_payload_field_index(ctx)
                .map(|attr| attr.0 as usize);
            if !in_arm_region || variant != Some(arm.variant) {
                continue;
            }
            let Some(stored) =
                field.and_then(|field| arm.construct.deref(ctx).operands().nth(field))
            else {
                continue;
            };
            let payload_value = user.deref(ctx).get_result(0);
            payload_value.replace_all_uses_with(ctx, &stored);
        }
    }
    CanonicalizeOutcome::Changed
}

/// Follow a discriminant match from `join` for a value whose discriminant is
/// `discriminant`. Returns the first block the match does not decide, with the
/// branch op and successor index that enter it. Every block the match passes
/// through is added to `match_blocks`, and the arguments of each block after
/// `join` to `forwarded`, mapped to the value that reaches them from `join`.
/// `None` when a block holds anything but the match itself.
fn follow_discriminant_match(
    ctx: &Context,
    join: Ptr<BasicBlock>,
    tested: Value,
    discriminant: u64,
    match_blocks: &mut FxHashSet<Ptr<BasicBlock>>,
    forwarded: &mut FxHashMap<Value, Value>,
) -> Option<(Ptr<BasicBlock>, Ptr<Operation>, usize)> {
    let mut known = FxHashMap::default();
    let mut block = join;
    let mut step = evaluate_match_block(ctx, block, tested, discriminant, &mut known)?;
    for _ in 0..MAX_MATCH_BLOCKS {
        match_blocks.insert(block);
        let (branch, succ_idx) = step;
        let next = branch.deref(ctx).get_successor(succ_idx);
        // A later match block has no other way in.
        if next != join && next.uses(ctx).len() == 1 {
            let mut next_known = known.clone();
            if let Some(next_step) =
                evaluate_match_block(ctx, next, tested, discriminant, &mut next_known)
            {
                let incoming = successor_operands(ctx, branch, succ_idx)?;
                for (arg, value) in next.deref(ctx).arguments().zip(incoming) {
                    forwarded.insert(arg, forwarded.get(&value).copied().unwrap_or(value));
                }
                known = next_known;
                block = next;
                step = next_step;
                continue;
            }
        }
        return Some((next, branch, succ_idx));
    }
    None
}

/// Evaluate one block of a discriminant match. Returns its conditional branch
/// and the successor index taken, or `None` when the block holds an operation
/// other than storage markers, integer constants, the discriminant read,
/// integer casts and equality tests on those values.
fn evaluate_match_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    tested: Value,
    discriminant: u64,
    known: &mut FxHashMap<Value, u128>,
) -> Option<(Ptr<Operation>, usize)> {
    for op in block.deref(ctx).iter(ctx) {
        if Operation::get_op::<MirStorageLiveOp>(op, ctx).is_some()
            || Operation::get_op::<MirStorageDeadOp>(op, ctx).is_some()
        {
            continue;
        }
        if Operation::get_op::<MirCondBranchOp>(op, ctx).is_some() {
            let taken = *known.get(&op.deref(ctx).get_operand(0))?;
            return Some((op, if taken == 1 { 0 } else { 1 }));
        }
        let result = (op.deref(ctx).get_num_results() == 1).then(|| op.deref(ctx).get_result(0))?;
        let width = integer_width(ctx, result)?;
        let value = if let Some(constant) = Operation::get_op::<MirConstantOp>(op, ctx) {
            constant.get_attr_value(ctx)?.value().to_u128()
        } else if Operation::get_op::<MirGetDiscriminantOp>(op, ctx).is_some() {
            if op.deref(ctx).get_operand(0) != tested {
                return None;
            }
            u128::from(discriminant)
        } else if let Some(cast) = Operation::get_op::<MirCastOp>(op, ctx) {
            if !cast
                .get_attr_cast_kind(ctx)
                .is_some_and(|kind| *kind == MirCastKindAttr::IntToInt)
            {
                return None;
            }
            let source = op.deref(ctx).get_operand(0);
            let bits = *known.get(&source)?;
            let source_width = integer_width(ctx, source)?;
            if width > source_width && integer_is_signed(ctx, source)? {
                sign_extend(bits, source_width)
            } else {
                bits
            }
        } else if Operation::get_op::<MirEqOp>(op, ctx).is_some()
            || Operation::get_op::<MirNeOp>(op, ctx).is_some()
        {
            let lhs = *known.get(&op.deref(ctx).get_operand(0))?;
            let rhs = *known.get(&op.deref(ctx).get_operand(1))?;
            let equal = lhs == rhs;
            u128::from(equal == Operation::get_op::<MirEqOp>(op, ctx).is_some())
        } else {
            return None;
        };
        known.insert(result, truncate(value, width));
    }
    None
}

/// The operands `branch` passes to its successor `index`.
fn successor_operands(ctx: &Context, branch: Ptr<Operation>, index: usize) -> Option<Vec<Value>> {
    let opobj = Operation::get_op_dyn(branch, ctx);
    let branch = op_cast::<dyn BranchOpInterface>(opobj.as_ref())?;
    Some(branch.successor_operands(ctx, index))
}

/// Blocks reachable from `start` (included) without entering `stop`.
fn reachable_outside(
    ctx: &Context,
    start: Ptr<BasicBlock>,
    stop: &FxHashSet<Ptr<BasicBlock>>,
) -> FxHashSet<Ptr<BasicBlock>> {
    let mut seen = FxHashSet::default();
    let mut worklist = vec![start];
    while let Some(block) = worklist.pop() {
        if stop.contains(&block) || !seen.insert(block) {
            continue;
        }
        if let Some(term) = block.deref(ctx).get_terminator(ctx) {
            worklist.extend(term.deref(ctx).successors());
        }
    }
    seen
}

/// The block that defines `value`: its block for an argument, its operation's
/// block for a result.
fn value_block(ctx: &Context, value: Value) -> Option<Ptr<BasicBlock>> {
    match value.defining_op() {
        Some(op) => op.deref(ctx).get_parent_block(),
        None => value.defining_block(),
    }
}

fn integer_type(ctx: &Context, value: Value) -> Option<TypedHandle<IntegerType>> {
    TypedHandle::<IntegerType>::from_handle(value.get_type(ctx), ctx).ok()
}

fn integer_width(ctx: &Context, value: Value) -> Option<u32> {
    let width = integer_type(ctx, value)?.deref(ctx).width();
    (1..=128).contains(&width).then_some(width)
}

fn integer_is_signed(ctx: &Context, value: Value) -> Option<bool> {
    Some(integer_type(ctx, value)?.deref(ctx).signedness() == Signedness::Signed)
}

/// The low `width` bits of `bits`.
fn truncate(bits: u128, width: u32) -> u128 {
    if width == 128 {
        bits
    } else {
        bits & ((1u128 << width) - 1)
    }
}

/// `bits`, a `width`-bit two's-complement value, sign-extended to 128 bits.
fn sign_extend(bits: u128, width: u32) -> u128 {
    if width == 128 || bits & (1u128 << (width - 1)) == 0 {
        bits
    } else {
        bits | !((1u128 << width) - 1)
    }
}

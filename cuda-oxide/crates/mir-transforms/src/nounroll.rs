/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[nounroll]`: keep an annotated loop rolled through LLVM, libNVVM, and
//! ptxas.
//!
//! The frontend records the request as a `mir.nounroll_hint` operation inside
//! the annotated loop's body. The hint has no side effects interface, so DCE
//! keeps it, and the unroll pass ignores it: when an enclosing `#[unroll]` loop
//! is unrolled, each body copy gets its own copy of the inner loop and of its
//! hint.
//!
//! [`resolve_nounroll_hints`] runs just before lowering, after every MIR
//! transform that can clone or reshape loops. For each hint it finds the
//! innermost loop that contains it, routes that loop's back-edges through one
//! latch block ending in `mir.goto header`, removes the hint, and returns that
//! goto. The lowering attaches `llvm.loop.unroll.disable` to the branch the goto
//! becomes.
//!
//! The loop needs its own latch because LLVM reads loop metadata from the
//! latch's terminator. With several latches, LLVM ignores the metadata unless
//! every latch carries the same node. A branch that ends an inner loop and also
//! jumps to the outer header would apply the request to both loops.

use dialect_mir::ops::control_flow::MirNoUnrollHintOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::ControlFlowGraph;
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::result::Result;
use rustc_hash::FxHashSet;

use crate::analyses::loop_info::LoopInfo;
use crate::canonicalize::{CanonicalizeOutcome, merge_backedges};
use crate::unroll::{collect_functions, hint_location};

/// Map every `mir.nounroll_hint` in `module` to its loop and return the latch
/// branch of each marked loop.
///
/// Each returned operation is a `mir.goto` whose only successor is its loop's
/// header, and it is the only back-edge of that loop. Every hint is erased,
/// including hints that could not be applied. Hints in unreachable blocks are
/// dropped silently; the others that cannot be applied produce a warning on
/// stderr, in the same form as the unroll pass's warnings.
///
/// Functions without hints are left untouched. In a function with hints, this
/// may insert one forwarding latch block per marked loop.
pub fn resolve_nounroll_hints(
    module: Ptr<Operation>,
    ctx: &mut Context,
) -> Result<Vec<Ptr<Operation>>> {
    let mut latches = Vec::new();
    for func_op in collect_functions(module, ctx) {
        let region = func_op.deref(ctx).get_region(0);

        // A hint in dead code has no loop to keep rolled. WGMMA counted-loop
        // fusion leaves a fused K loop's blocks unreachable this way, after
        // moving the request onto the fused operation. Merging back-edges
        // below only adds blocks, so reachability is computed once
        let reachable = reachable_blocks(ctx, region);
        for (hint, block) in collect_hints(ctx, region) {
            if !reachable.contains(&block) {
                Operation::erase(hint, ctx);
            }
        }

        // One loop per round. Merging back-edges changes the CFG, so loop
        // facts are recomputed before the next hint is placed
        loop {
            let hints = collect_hints(ctx, region);
            let Some(&(hint, block)) = hints.first() else {
                break;
            };

            let info = loop_info(module, ctx, region)?;
            let Some(loop_id) = info.innermost_loop(block) else {
                warn_not_applied(
                    ctx,
                    hint,
                    "the annotation is not inside a recognizable loop",
                );
                Operation::erase(hint, ctx);
                continue;
            };

            match merge_backedges(ctx, &info, loop_id) {
                CanonicalizeOutcome::Changed => continue,
                CanonicalizeOutcome::Unsupported(reason) => {
                    warn_not_applied(
                        ctx,
                        hint,
                        &format!("could not normalize its back-edges: {reason}"),
                    );
                }
                CanonicalizeOutcome::Unchanged => {
                    // `Unchanged` means the loop has exactly one back-edge, a
                    // `mir.goto` to the header
                    let latch = info.loops()[loop_id].latches[0];
                    let Some(branch) = latch.deref(ctx).get_terminator(ctx) else {
                        return pliron::input_err_noloc!(
                            "a normalized loop latch has no terminator"
                        );
                    };
                    latches.push(branch);
                }
            }

            // a loop can hold several hints, for example after block merging
            for (other, other_block) in hints {
                if info.innermost_loop(other_block) == Some(loop_id) {
                    Operation::erase(other, ctx);
                }
            }
        }
    }

    Ok(latches)
}

/// The blocks of `region` reachable from its entry block.
fn reachable_blocks(ctx: &Context, region: Ptr<Region>) -> FxHashSet<Ptr<BasicBlock>> {
    let mut reachable = FxHashSet::default();
    let mut worklist: Vec<_> = region.deref(ctx).iter(ctx).take(1).collect();
    while let Some(block) = worklist.pop() {
        if reachable.insert(block) {
            worklist.extend(region.successors(ctx, &block));
        }
    }
    reachable
}

/// Compute fresh loop facts for `region`.
fn loop_info(module: Ptr<Operation>, ctx: &mut Context, region: Ptr<Region>) -> Result<LoopInfo> {
    let mut analyses = AnalysisManager::default();
    let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
    let dom = dom_info.get_dom_tree(ctx, region);
    Ok(LoopInfo::compute(ctx, region, dom))
}

/// The `mir.nounroll_hint` ops in `region`, each with the block it sits in.
fn collect_hints(ctx: &Context, region: Ptr<Region>) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>)> {
    let mut out = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        for op in block.deref(ctx).iter(ctx) {
            if Operation::get_op::<MirNoUnrollHintOp>(op, ctx).is_some() {
                out.push((op, block));
            }
        }
    }
    out
}

/// Report a `#[nounroll]` that could not be applied, with the annotation's
/// position on a `-->` line when the hint carries one.
fn warn_not_applied(ctx: &Context, hint: Ptr<Operation>, reason: &str) {
    eprintln!(
        "{}",
        not_applied_warning(reason, hint_location(ctx, hint).as_deref())
    );
}

/// The full text of one "not applied" warning.
fn not_applied_warning(reason: &str, location: Option<&str>) -> String {
    let mut warning =
        format!("warning: #[nounroll] requested but the loop may still be unrolled: {reason}");
    if let Some(location) = location {
        warning.push_str("\n  --> ");
        warning.push_str(location);
    }
    warning
}

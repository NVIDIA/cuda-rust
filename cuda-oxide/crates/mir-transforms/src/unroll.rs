/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Loop unrolling, switched on by a `#[unroll]` annotation.
//!
//! Unrolling means making copies of a loop body so the loop runs fewer times
//! (or not at all), trading bigger code for less per-iteration overhead and more
//! chances to optimise. For example:
//!
//! ```text
//!   i = 0; while i < 4 { f(i); i += 1 }   becomes:   f(0); f(1); f(2); f(3);
//! ```
//!
//! `#[unroll]` requests full unrolling when the iteration count is known at
//! compile time. `#[unroll(N)]` requests `N` body copies per trip and leaves a
//! small remainder loop for leftover iterations. The frontend records the
//! request as a `mir.unroll_hint` operation inside that loop.
//!
//! The current analysis recognizes explicit counted `while` loops, including an
//! exit test that adds a constant to the counter (`while i + 2 <= n`).
//! Range-based `for` loops are not yet recognized.
//!
//! Several `continue` paths are supported: the pass joins their back-edges
//! before unrolling. Full `#[unroll]` also preserves early `break` paths and
//! multiple exit targets. Partial `#[unroll(N)]` warns and leaves the loop not
//! unrolled when it has an extra exit. Partial unrolling currently requires a
//! positive step, a `<` or `<=` test, and a loop-invariant bound.
//!
//! To bound compile time and memory, one request may create at most 1,024 body
//! copies, 8,192 cloned blocks, and 65,536 cloned operations. Larger requests
//! warn and are not unrolled.
//!
//! If an annotated loop contains another loop, only the annotated loop is
//! unrolled. The inner loop is copied intact into each body copy and remains a
//! loop. Give the inner loop its own annotation if it should be unrolled too.
//!
//! This pass is the reference example for writing an optimisation pass in oxide.
//! It builds on two reusable analyses, [`LoopInfo`](crate::analyses::loop_info)
//! (finds the loops) and [`induction`] (finds the counters and how many times
//! each loop runs), and on pliron's IR cloning (`pliron::irbuild::cloning`) to
//! duplicate the body.

use dialect_mir::ops::arithmetic::{MirAddOp, MirBitAndOp, MirRemOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::{MirCondBranchOp, MirGotoOp, MirUnrollHintOp};
use dialect_mir::ops::function::MirFuncOp;
use dialect_mir::ops::{MirStorageDeadOp, MirStorageLiveOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::graph::ControlFlowGraph;
use pliron::graph::dominance::DomInfo;
use pliron::irbuild::{
    cloning::{IrMapping, clone_blocks_into},
    listener::DummyListener,
    rewriter::IRRewriter,
};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::opts::constants::sccp::sccp;
use pliron::opts::dce::{SideEffects, dce};
use pliron::opts::simplify_cfg::simplify_cfg;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed, TypedHandle};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_hash::FxHashSet;
use std::num::NonZero;

use crate::analyses::induction::{self, ArgKind, CmpPred};
use crate::analyses::loop_info::LoopInfo;
use crate::canonicalize::{CanonicalizeOutcome, close_header_liveouts, merge_backedges};

/// Hard safety limit on how many body copies one annotation may request.
/// Explicit annotations still need a bound: accepting an arbitrary `u32`
/// factor or `u64` trip count would let valid source exhaust the compiler.
const MAX_UNROLL_COPIES: u64 = 1_024;

/// Hard safety limit on the total number of basic blocks cloned by one unroll.
/// This separately bounds large and nested bodies even when their copy count is
/// below [`MAX_UNROLL_COPIES`].
const MAX_CLONED_BLOCKS: u64 = 8_192;

/// Hard safety limit on the total number of operations cloned by one unroll.
/// A block budget alone is insufficient because one block may contain an
/// arbitrarily large straight-line body.
const MAX_CLONED_OPS: u64 = 65_536;

fn verbose() -> bool {
    std::env::var("CUDA_OXIDE_VERBOSE").is_ok()
}

/// How the author spelled the request, for diagnostics: `#[unroll]` (full) or
/// `#[unroll(N)]` (partial by N).
fn unroll_kind(factor: u32) -> String {
    if factor == 0 {
        "#[unroll]".to_string()
    } else {
        format!("#[unroll({factor})]")
    }
}

/// Check code-growth budgets whose inputs have already been counted safely.
fn check_growth_budget(
    copies: u64,
    blocks_per_copy: u64,
    ops_per_copy: u64,
) -> std::result::Result<(), String> {
    if copies > MAX_UNROLL_COPIES {
        return Err(format!(
            "unrolling would create {copies} body copies; the safety limit is {MAX_UNROLL_COPIES}"
        ));
    }
    let total_blocks = copies
        .checked_mul(blocks_per_copy)
        .ok_or_else(|| "the requested unroll is too large to budget safely".to_string())?;
    if total_blocks > MAX_CLONED_BLOCKS {
        return Err(format!(
            "unrolling would clone {total_blocks} body blocks ({copies} copies x {blocks_per_copy} blocks); the safety limit is {MAX_CLONED_BLOCKS} cloned blocks"
        ));
    }
    let total_ops = copies
        .checked_mul(ops_per_copy)
        .ok_or_else(|| "the requested unroll is too large to budget safely".to_string())?;
    if total_ops > MAX_CLONED_OPS {
        return Err(format!(
            "unrolling would clone {total_ops} operations ({copies} copies x {ops_per_copy} operations); the safety limit is {MAX_CLONED_OPS} cloned operations"
        ));
    }
    Ok(())
}

/// Count one body copy and check every code-growth budget before cloning.
fn check_clone_budget(
    ctx: &Context,
    copies: u64,
    body_blocks: &[Ptr<BasicBlock>],
) -> std::result::Result<(), String> {
    let blocks_per_copy = u64::try_from(body_blocks.len())
        .map_err(|_| "the loop body is too large to budget safely".to_string())?;
    let mut ops_per_copy = 0u64;
    for &block in body_blocks {
        let block_ops = u64::try_from(block.deref(ctx).iter(ctx).count())
            .map_err(|_| "the loop body has too many operations to budget safely".to_string())?;
        ops_per_copy = ops_per_copy
            .checked_add(block_ops)
            .ok_or_else(|| "the loop body has too many operations to budget safely".to_string())?;
    }
    check_growth_budget(copies, blocks_per_copy, ops_per_copy)
}

/// Unroll each loop that carries an in-body `mir.unroll_hint`, which the
/// frontend plants for `#[unroll]` and `#[unroll(N)]`. Other functions are left
/// untouched.
///
/// Within each annotated function, this finds loops with [`LoopInfo`], maps
/// every hint to its loop, joins multiple back-edges through one latch, analyzes
/// the loop counter, and performs full or partial unrolling. It recomputes loop
/// and induction facts after normalization. Annotated nested loops are processed
/// from the inside out. Afterward, SCCP, CFG simplification, and dead-code
/// elimination clean only that function: constant index expressions fold, dead
/// branches disappear, and unreachable original loop blocks are removed.
///
/// Unsupported loop shapes produce a warning and are not unrolled.
pub fn unroll_annotated_loops(
    module: Ptr<Operation>,
    ctx: &mut Context,
    // The caller threads in the manager mem2reg used. We deliberately do not use
    // it: the CFG-normalization step below changes block structure, which would
    // stale that manager's cached dominator trees. We build a fresh manager
    // afterwards instead. (Kept in the signature to match pliron's pass shape.)
    _analyses: &mut AnalysisManager,
) -> Result<()> {
    // Nothing annotated => leave every function byte-for-byte untouched.
    let has_hints = collect_functions(module, ctx)
        .iter()
        .any(|&f| !collect_hints(ctx, f.deref(ctx).get_region(0)).is_empty());
    if !has_hints {
        return Ok(());
    }

    let mut changed = false;
    for func_op in collect_functions(module, ctx) {
        let region = func_op.deref(ctx).get_region(0);
        if collect_hints(ctx, region).is_empty() {
            continue;
        }
        let mut function_changed = false;

        // Normalize only the annotated function. The marker call often gets its
        // own MIR block; `simplify_cfg` merges that block back into the loop body
        // without changing unrelated functions in the module.
        simplify_cfg(func_op, ctx)?;

        // Unroll one annotated loop per round, innermost-first, recomputing the
        // loop analyses each round. Each unroll rewrites the CFG -- and for a loop
        // that *contains* inner loops it clones those inner loops into every copy
        // -- so a single shared `LoopInfo` snapshot would go stale. A fresh
        // dominator tree + `LoopInfo` per round keeps every query correct; we run
        // `simplify_cfg` first so the recompute sees only live blocks (a full
        // unroll leaves the original loop unreachable, and we must not let those
        // dead blocks confuse dominance). Innermost-first means an annotated inner
        // loop is unrolled, and its hint consumed, before its enclosing loop is
        // cloned, so inner hints are never duplicated into the copies. (`dce` is
        // deliberately NOT run here: it would delete the side-effect-free
        // `mir.unroll_hint` markers before we get to collect them.)
        let mut skip_simplify_once = false;
        loop {
            // Drop blocks the previous round made unreachable before recomputing.
            // Do not immediately simplify a newly inserted unified latch: a CFG
            // cleanup could fold that forwarding block away before the fresh
            // loop analysis gets to use it.
            if skip_simplify_once {
                skip_simplify_once = false;
            } else {
                simplify_cfg(func_op, ctx)?;
            }

            let hints = collect_hints(ctx, region);
            if hints.is_empty() {
                break;
            }

            let info = {
                let mut analyses = AnalysisManager::default();
                let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
                let dom = dom_info.get_dom_tree(ctx, region);
                LoopInfo::compute(ctx, region, dom)
            };

            // Pick the innermost annotated loop still to do (smallest body wins;
            // in a reducible CFG a container's body is a strict superset of its
            // child's, so "smallest body" is exactly "innermost").
            let mut best: Option<(usize, usize, u32)> = None; // (loop_id, body_size, factor)
            for (_op, block, factor) in &hints {
                if let Some(loop_id) = info.innermost_loop(*block) {
                    let size = info.loops()[loop_id].blocks.len();
                    if best.is_none_or(|(_, bs, _)| size < bs) {
                        best = Some((loop_id, size, *factor));
                    }
                }
            }
            let Some((loop_id, _size, factor)) = best else {
                // No remaining hint sits in a recognizable loop. The author asked
                // for unrolling, so say so loudly, then drop the markers and stop.
                for (op, _block, factor) in &hints {
                    eprintln!(
                        "warning: {} requested but the loop was not unrolled: the annotation is not inside a recognizable loop",
                        unroll_kind(*factor)
                    );
                    op.unlink(ctx);
                }
                break;
            };
            let kind = unroll_kind(factor);

            let Some(ph) = info.preheader(ctx, region, loop_id) else {
                // The author asked for unrolling; never silently do nothing.
                for (op, block, _f) in &hints {
                    if info.innermost_loop(*block) == Some(loop_id) {
                        op.unlink(ctx);
                    }
                }
                eprintln!(
                    "warning: {kind} requested but the loop was not unrolled: it has no single preheader (it is entered from more than one place)"
                );
                continue;
            };

            // A grouped main loop plus a remainder loop needs explicit merging
            // for every early-exit value. Keep that as a follow-up: full unroll
            // can preserve those paths directly, partial unroll warns and skips.
            if factor != 0 {
                let lp = &info.loops()[loop_id];
                let exiting = info.exiting_blocks(ctx, region, loop_id);
                let exits = info.exit_blocks(ctx, region, loop_id);
                if exiting.len() != 1 || exiting[0] != lp.header || exits.len() != 1 {
                    for (op, block, _f) in &hints {
                        if info.innermost_loop(*block) == Some(loop_id) {
                            op.unlink(ctx);
                        }
                    }
                    eprintln!(
                        "warning: {kind} requested but the loop was not unrolled: partial unrolling does not yet support an early `break` or multiple exits"
                    );
                    continue;
                }
            }

            // Full unroll needs path-specific values at early exits. Route any
            // directly used header arguments through outside block arguments
            // before cloning, then recompute analyses over the normalized IR.
            if factor == 0 {
                match close_header_liveouts(ctx, &info, loop_id) {
                    CanonicalizeOutcome::Unchanged => {}
                    CanonicalizeOutcome::Changed => {
                        function_changed = true;
                        changed = true;
                        skip_simplify_once = true;
                        continue;
                    }
                    CanonicalizeOutcome::Unsupported(reason) => {
                        for (op, block, _f) in &hints {
                            if info.innermost_loop(*block) == Some(loop_id) {
                                op.unlink(ctx);
                            }
                        }
                        eprintln!(
                            "warning: {kind} requested but the loop was not unrolled: could not route its live-out values: {reason}"
                        );
                        continue;
                    }
                }
            }

            // Normalize every continue/back-edge through one unconditional
            // latch. Leave the hint in place when the CFG changes so the next
            // round can recompute dominance, loops, and induction facts first.
            match merge_backedges(ctx, &info, loop_id) {
                CanonicalizeOutcome::Unchanged => {}
                CanonicalizeOutcome::Changed => {
                    function_changed = true;
                    changed = true;
                    skip_simplify_once = true;
                    continue;
                }
                CanonicalizeOutcome::Unsupported(reason) => {
                    for (op, block, _f) in &hints {
                        if info.innermost_loop(*block) == Some(loop_id) {
                            op.unlink(ctx);
                        }
                    }
                    eprintln!(
                        "warning: {kind} requested but the loop was not unrolled: could not normalize its back-edges: {reason}"
                    );
                    continue;
                }
            }

            // Consume this loop's hint(s) before any cloning so the markers are
            // never copied into the unrolled bodies.
            for (op, block, _f) in &hints {
                if info.innermost_loop(*block) == Some(loop_id) {
                    op.unlink(ctx);
                }
            }

            let rec = induction::analyze(ctx, &info, loop_id, ph);
            if verbose() {
                eprintln!(
                    "loop-unroll: loop#{loop_id} factor={factor} trip={:?} primary_iv={:?}",
                    rec.trip_count, rec.primary_iv,
                );
            }
            let outcome = if factor == 0 {
                // Full unroll: only works when the trip count is a constant.
                full_unroll(ctx, &info, region, loop_id, ph, &rec)?
            } else {
                // Partial unroll by `factor`, with a remainder loop for the tail.
                partial_unroll(ctx, &info, region, loop_id, ph, &rec, factor)?
            };
            match outcome {
                UnrollOutcome::Unrolled => {
                    function_changed = true;
                    changed = true;
                }
                // Requested but unsupported shape: report exactly why, loudly, so
                // it is never a silent no-op.
                UnrollOutcome::Skipped(reason) => {
                    eprintln!("warning: {kind} requested but the loop was not unrolled: {reason}");
                }
            }
        }
        if function_changed {
            // Fold and clean only this function. `sccp` folds constant index
            // arithmetic and branch conditions, `simplify_cfg` removes dead
            // paths and original loop blocks, and `dce` removes unused ops.
            // The outer filter ensures this cleanup never touches a function
            // with no unroll annotation.
            sccp(func_op, ctx)?;
            simplify_cfg(func_op, ctx)?;
            dce(func_op, ctx)?;
        }
    }

    if changed {
        // The input was verified before this pass, so a failure here is a bug in
        // the unroller. Verify after cleanup, when unreachable original loop
        // blocks are gone and the dominator-based verifier sees the final CFG.
        pliron::operation::verify_operation(module, ctx)?;
    }
    Ok(())
}

/// What happened when we tried to unroll one loop.
enum UnrollOutcome {
    /// The loop was unrolled; the IR changed.
    Unrolled,
    /// The loop was not unrolled, with a plain-English reason. The caller turns
    /// this into a loud warning (the author asked for unrolling, so we never
    /// silently do nothing). Earlier normalization may still have rewritten its CFG.
    Skipped(String),
}

/// The facts the unroller needs about a loop, gathered once and shared by full
/// and partial unroll, plus the checks that the loop is in the shape we support.
///
/// v1 supports a loop with **arbitrary internal control flow** (the body may be
/// many blocks: `if`/`else`, `match`, `&&`/`||`) and may contain nested
/// loops. Multiple source latches have already been merged into one canonical
/// latch. Full unrolling accepts early exits and multiple exit targets. Partial
/// unrolling additionally requires the header test to be the only exit. Both
/// modes require a recognized counter, a dedicated preheader, and a header with
/// only pure guard calculations before its conditional branch. `analyze_shape`
/// returns `Err(reason)` for anything else.
///
/// We deliberately do **not** clone the header. The header holds the loop's
/// carried values as block arguments (the counter and any accumulators); the
/// body reads those by dominance. Cloning only the body and substituting those
/// arguments per copy keeps the counter a literal in each copy (the property that
/// lets `i & 3` fold). A header that *computes* a value the body reads would
/// break that, so we reject it (`Err`) and leave header-cloning to a follow-up.
struct LoopShape {
    header: Ptr<BasicBlock>,
    latch: Ptr<BasicBlock>,
    /// The outside successor of the header's ordinary loop test.
    normal_exit: Ptr<BasicBlock>,
    /// Whether some body block can leave without going through the latch.
    has_early_exits: bool,
    /// The header's single in-loop successor: the first block of the body.
    body_entry: Ptr<BasicBlock>,
    /// Every body block (the loop minus the header) forward-reachable from
    /// `body_entry`, in a deterministic visit order. `clone_blocks_into` is
    /// order-independent, so the order is not load-bearing; we keep a stable
    /// order only for readable IR dumps. Its length (vs the body-block count) is
    /// what detects unreachable / irreducible body blocks (see
    /// [`reachable_body_blocks`]).
    body_blocks_ordered: Vec<Ptr<BasicBlock>>,
    /// The header's block arguments (the loop-carried values, counter included).
    header_args: Vec<Value>,
    nargs: usize,
    /// preheader -> header operands (the loop's initial carried values).
    init_ops: Vec<Value>,
    /// latch -> header operands (the updated carried values each iteration).
    recur_ops: Vec<Value>,
    /// header -> body_entry operands (args the header passes into the body).
    entry_ops: Vec<Value>,
    /// header -> normal_exit operands (the normal-completion live-out values).
    normal_exit_ops: Vec<Value>,
    iv_idx: usize,
    iv_init: i128,
    iv_step: i128,
    iv_type: TypeHandle,
    /// The boolean type of the header's exit test (for any new comparison).
    i1_type: TypeHandle,
    /// The preheader's terminator (a plain branch to the header).
    preheader_term: Ptr<Operation>,
    /// Every block of the loop (header included). Used to tell loop-internal uses
    /// of the carried values from out-of-loop (live-out) uses.
    loop_blocks: FxHashSet<Ptr<BasicBlock>>,
}

/// True if `v` is the result of an operation located in one of `set`'s blocks
/// (a block argument, having no defining op, is not "defined in" the set this
/// way). Used to tell loop-variant values from loop-invariant ones.
fn defined_in_loop(ctx: &Context, v: Value, set: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    v.defining_op()
        .and_then(|d| d.deref(ctx).get_parent_block())
        .map(|b| set.contains(&b))
        .unwrap_or(false)
}

/// The blocks of `set` forward-reachable from `entry`, following only edges that
/// stay inside `set`, in a deterministic visit order.
///
/// `clone_blocks_into` is order-independent (it records every clone block, block
/// argument, and op result before wiring any operand or successor), so the order
/// here is not required for cloning; a plain reachability walk is enough. We keep
/// the result ordered only so IR dumps are stable. Its main job is the soundness
/// check at the call site: if it is shorter than `set`, some body block reaches
/// the latch but is not reachable from the single entry, i.e. irreducible /
/// multi-entry control flow the v1 shape does not support.
fn reachable_body_blocks(
    ctx: &Context,
    region: Ptr<Region>,
    entry: Ptr<BasicBlock>,
    set: &FxHashSet<Ptr<BasicBlock>>,
) -> Vec<Ptr<BasicBlock>> {
    let mut visited: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
    let mut order: Vec<Ptr<BasicBlock>> = Vec::new();
    let mut stack: Vec<Ptr<BasicBlock>> = vec![entry];
    while let Some(b) = stack.pop() {
        if !visited.insert(b) {
            continue;
        }
        order.push(b);
        for s in region.successors(ctx, &b) {
            if set.contains(&s) && !visited.contains(&s) {
                stack.push(s);
            }
        }
    }
    order
}

/// Gather the loop facts and confirm the v1 shape. See [`LoopShape`].
fn analyze_shape(
    ctx: &Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
    allow_early_exits: bool,
) -> std::result::Result<LoopShape, String> {
    let l = &info.loops()[id];
    // A loop that *contains* an inner loop can be unrolled: the inner loop's
    // blocks are part of this loop's body (`l.blocks`), so cloning the body
    // clones the inner loop wholesale and it stays a loop in each copy -- we do
    // not recurse into it or unroll it. Unrolling the inner loop too is a
    // separate `#[unroll]` on the inner loop, which the driver processes
    // innermost-first; and the driver recomputes `LoopInfo` after each unroll so
    // the cloned inner loops are re-registered cleanly. So `l.children` is no
    // longer a bail; inner loops are opaque body blocks to the clone.
    if l.latches.len() != 1 {
        return Err(format!(
            "the loop has {} back-edges after canonicalization; expected one unified latch",
            l.latches.len()
        ));
    }
    let header = l.header;
    let latch = l.latches[0];

    let iv_idx = match rec.primary_iv {
        Some(iv_idx) => iv_idx,
        None if rec
            .args
            .iter()
            .any(|arg| matches!(arg, ArgKind::BasicIv { .. })) =>
        {
            return Err("the loop has a counter, but its exit test is not of the form `counter <op> bound` (or `counter + const <op> bound`)".into());
        }
        None => return Err("no recognized induction variable (loop counter)".into()),
    };
    let (iv_init, iv_step) = match &rec.args[iv_idx] {
        ArgKind::BasicIv { init, step } => (*init, *step),
        _ => return Err("the loop counter is not a simple induction variable".into()),
    };

    // The header has one ordinary body successor and one ordinary exit. Other
    // body blocks may have their own outside successors on the full-unroll path.
    let header_term = header
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the header has no terminator")?;
    if Operation::get_op::<MirCondBranchOp>(header_term, ctx).is_none() {
        return Err("the loop header must end in a mir.cond_br exit test".into());
    }
    // Full unrolling removes the header, and partial unrolling bypasses it for
    // grouped iterations. Either transform would change observable header work.
    // Storage lifetime markers are harmless: lowering erases them and emits no
    // runtime instruction.
    for op in header.deref(ctx).iter(ctx) {
        let harmless_marker = Operation::get_op::<MirStorageLiveOp>(op, ctx).is_some()
            || Operation::get_op::<MirStorageDeadOp>(op, ctx).is_some();
        if op == header_term || harmless_marker {
            continue;
        }
        let opobj = Operation::get_op_dyn(op, ctx);
        let may_have_side_effects = op_cast::<dyn SideEffects>(opobj.as_ref())
            .is_none_or(|effects| effects.has_side_effects(ctx));
        if may_have_side_effects {
            return Err(
                "the loop header contains an operation with possible side effects; move that work into the body before unrolling"
                    .into(),
            );
        }
    }
    let successors: Vec<Ptr<BasicBlock>> = header_term.deref(ctx).successors().collect();
    let in_loop: Vec<_> = successors
        .iter()
        .copied()
        .filter(|s| l.blocks.contains(s))
        .collect();
    let outside: Vec<_> = successors
        .iter()
        .copied()
        .filter(|s| !l.blocks.contains(s))
        .collect();
    if in_loop.len() != 1 || outside.len() != 1 {
        return Err("the loop header must have one body edge and one normal exit edge".into());
    }
    let body_entry = in_loop[0];
    let normal_exit = outside[0];

    let exiting = info.exiting_blocks(ctx, region, id);
    let exits = info.exit_blocks(ctx, region, id);
    let has_early_exits = exiting.iter().any(|&block| block != header);
    if !allow_early_exits && (has_early_exits || exits.len() != 1) {
        return Err(
            "partial unrolling does not yet support an early `break` or multiple exits".into(),
        );
    }

    // The preheader must end in a plain branch to the header (dedicated preheader).
    let preheader_term = preheader
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the preheader has no terminator")?;
    let p_succs: Vec<Ptr<BasicBlock>> = preheader_term.deref(ctx).successors().collect();
    if Operation::get_op::<MirGotoOp>(preheader_term, ctx).is_none() || p_succs != [header] {
        return Err("the loop preheader must end in a mir.goto to the header".into());
    }

    // The latch must end in a plain back-edge to the header.
    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the latch has no terminator")?;
    let l_succs: Vec<Ptr<BasicBlock>> = latch_term.deref(ctx).successors().collect();
    if Operation::get_op::<MirGotoOp>(latch_term, ctx).is_none() || l_succs != [header] {
        return Err("the loop latch must end in a mir.goto back to the header".into());
    }

    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs)
        .map(|i| header.deref(ctx).get_argument(i))
        .collect();
    let iv_type = header_args[iv_idx].get_type(ctx);
    let i1_type = header_term.deref(ctx).get_operand(0).get_type(ctx);

    let init_ops = induction::edge_operands(ctx, preheader, header)
        .filter(|v| v.len() == nargs)
        .ok_or("preheader carried-value arity mismatch")?;
    let recur_ops = induction::edge_operands(ctx, latch, header)
        .filter(|v| v.len() == nargs)
        .ok_or("latch carried-value arity mismatch")?;
    let entry_ops = induction::edge_operands(ctx, header, body_entry).unwrap_or_default();
    let normal_exit_ops = induction::edge_operands(ctx, header, normal_exit).unwrap_or_default();

    // Clean-header check (see [`LoopShape`]): the body must not read any value the
    // header *computes*. Reading the header's block arguments is fine (we
    // substitute those per copy); reading a header op result is not.
    let body_blocks: FxHashSet<Ptr<BasicBlock>> =
        l.blocks.iter().copied().filter(|&b| b != header).collect();
    let defined_in_header = |ctx: &Context, v: Value| -> bool {
        v.defining_op()
            .map(|d| d.deref(ctx).get_parent_block() == Some(header))
            .unwrap_or(false)
    };
    for &b in &body_blocks {
        for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            let nops = op.deref(ctx).get_num_operands();
            for o in 0..nops {
                if defined_in_header(ctx, op.deref(ctx).get_operand(o)) {
                    return Err("the header computes a value the body reads; this shape needs header cloning (a follow-up)".into());
                }
            }
        }
    }
    for &v in &entry_ops {
        if defined_in_header(ctx, v) {
            return Err("the header passes a computed value into the body; this shape needs header cloning (a follow-up)".into());
        }
    }
    // Same for live-outs: a header *block argument* the exit reads is handled (we
    // substitute the final value), but a header op *result* on the exit edge
    // would dangle once full unroll deletes the header. Reject it loudly.
    for &v in &normal_exit_ops {
        if defined_in_header(ctx, v) {
            return Err("the header computes a live-out value the exit reads; this shape needs header cloning (a follow-up)".into());
        }
    }

    let body_blocks_ordered = reachable_body_blocks(ctx, region, body_entry, &body_blocks);
    if body_blocks_ordered.len() != body_blocks.len() {
        return Err("the loop body has blocks unreachable from its entry (irreducible control flow); not supported".into());
    }

    Ok(LoopShape {
        header,
        latch,
        normal_exit,
        has_early_exits,
        body_entry,
        body_blocks_ordered,
        header_args,
        nargs,
        init_ops,
        recur_ops,
        entry_ops,
        normal_exit_ops,
        iv_idx,
        iv_init,
        iv_step,
        iv_type,
        i1_type,
        preheader_term,
        loop_blocks: l.blocks.clone(),
    })
}

/// One cloned copy of the loop body.
struct CopyResult {
    /// The clone of `body_entry`: where control enters this copy.
    entry: Ptr<BasicBlock>,
    /// The clone of the latch's terminator (its back-edge `goto header`), which
    /// the caller repoints to the next copy (or the exit).
    latch_term: Ptr<Operation>,
    /// The carried values this copy produces, to feed the next copy (the latch's
    /// back-edge operands, mapped through this copy's substitution).
    next_running: Vec<Value>,
    /// The operands to pass when branching into `entry` (the header -> body_entry
    /// operands, mapped through this copy's substitution).
    entry_args: Vec<Value>,
    /// This copy's cloned blocks, in the same order as
    /// `LoopShape::body_blocks_ordered`.
    blocks: Vec<Ptr<BasicBlock>>,
}

/// Clone one copy of the loop body, substituting `subst[a]` for header argument
/// `a` (so `subst` carries this copy's counter value and accumulators). The
/// clone's internal branches and block-argument passes are remapped
/// automatically by [`clone_blocks_into`]; the caller wires the boundary edges.
fn clone_one_copy(
    ctx: &mut Context,
    region: Ptr<Region>,
    s: &LoopShape,
    subst: &[Value],
) -> CopyResult {
    let mut mapper = IrMapping::new();
    for (a, &hv) in s.header_args.iter().enumerate() {
        mapper.map_value(hv, subst[a]);
    }
    // Operands for the edge that enters this copy: the header -> body_entry
    // operands with this copy's carried values substituted in. Computed before
    // cloning (they reference only header args / outer values, never body values).
    let entry_args: Vec<Value> = s
        .entry_ops
        .iter()
        .map(|&v| mapper.lookup_value_or_default(v))
        .collect();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    clone_blocks_into(
        &s.body_blocks_ordered,
        region,
        ctx,
        &mut rewriter,
        &mut mapper,
    );
    let entry = mapper.lookup_block_or_default(s.body_entry);
    let latch = mapper.lookup_block_or_default(s.latch);
    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .expect("a cloned latch has a terminator");
    let next_running: Vec<Value> = s
        .recur_ops
        .iter()
        .map(|&v| mapper.lookup_value_or_default(v))
        .collect();
    let blocks: Vec<Ptr<BasicBlock>> = s
        .body_blocks_ordered
        .iter()
        .map(|&b| mapper.lookup_block_or_default(b))
        .collect();
    CopyResult {
        entry,
        latch_term,
        next_running,
        entry_args,
        blocks,
    }
}

/// Repoint a single-successor branch (`goto`) at `new_succ`, replacing all its
/// edge operands with `operands`.
fn rewire_goto(
    ctx: &mut Context,
    term: Ptr<Operation>,
    new_succ: Ptr<BasicBlock>,
    operands: &[Value],
) {
    Operation::replace_successor(term, ctx, 0, new_succ);
    let n = term.deref(ctx).get_num_operands();
    for _ in 0..n {
        Operation::remove_operand(term, ctx, 0);
    }
    for &v in operands {
        Operation::push_operand(term, ctx, v);
    }
}

/// Whether `value` is read directly by an operation outside this loop.
///
/// Passing it as an exit-edge operand is safe: that use belongs to the branch
/// inside the loop. The corresponding exit block argument is the value outside
/// code should read.
fn has_direct_outside_use(
    ctx: &Context,
    value: Value,
    loop_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> bool {
    value.uses(ctx).iter().any(|r#use| {
        r#use
            .user_op()
            .deref(ctx)
            .get_parent_block()
            .is_none_or(|block| !loop_blocks.contains(&block))
    })
}

/// Check the live-out rule before cloning anything. With an early exit there is
/// no single "final" loop value, so every loop definition must leave through an
/// exit edge and an exit block argument. Without an early exit, header arguments
/// may still be replaced by their normal final values as the original path did.
fn liveouts_are_safe(ctx: &Context, shape: &LoopShape) -> bool {
    for &block in &shape.loop_blocks {
        for value in block.deref(ctx).arguments() {
            let replaceable_header_arg =
                !shape.has_early_exits && shape.header_args.contains(&value);
            if !replaceable_header_arg && has_direct_outside_use(ctx, value, &shape.loop_blocks) {
                return false;
            }
        }
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            for value in op.deref(ctx).results() {
                if has_direct_outside_use(ctx, value, &shape.loop_blocks) {
                    return false;
                }
            }
        }
    }
    true
}

/// Mathematical bounds representable by the IV type. Signless integers follow
/// the dialect's existing unsigned comparison convention. Unsigned 128-bit
/// values above `i128::MAX` are conservatively outside this analysis.
fn integer_value_bounds(ctx: &Context, ty: TypeHandle) -> Option<(i128, i128)> {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).ok()?;
    let width = typed.deref(ctx).width();
    if width == 0 || width > 128 {
        return None;
    }
    match typed.deref(ctx).signedness() {
        Signedness::Signed if width == 128 => Some((i128::MIN, i128::MAX)),
        Signedness::Signed => {
            let half = 1i128.checked_shl(width - 1)?;
            Some((-half, half - 1))
        }
        Signedness::Unsigned | Signedness::Signless if width >= 127 => Some((0, i128::MAX)),
        Signedness::Unsigned | Signedness::Signless => Some((0, (1i128.checked_shl(width)? - 1))),
    }
}

/// Full unrolling uses a mathematical recurrence. Prove that the fixed-width IV
/// reaches the same final value without wrapping; otherwise the original loop
/// may continue after the trip count this analysis computed.
fn full_iv_stays_in_range(ctx: &Context, shape: &LoopShape, trip: i128) -> bool {
    let Some((min, max)) = integer_value_bounds(ctx, shape.iv_type) else {
        return false;
    };
    let Some(delta) = trip.checked_mul(shape.iv_step) else {
        return false;
    };
    let Some(final_iv) = shape.iv_init.checked_add(delta) else {
        return false;
    };
    (min..=max).contains(&shape.iv_init) && (min..=max).contains(&final_iv)
}

/// With an exit test `IV + offset <pred> bound`, the header computes
/// `init + k*step + offset` for every `k` from 0 to the trip count. That sum
/// changes monotonically in `k`, so it stays in the IV type's range exactly when
/// its first and last values do; otherwise the fixed-width test can wrap and
/// disagree with the trip count.
fn full_exit_test_stays_in_range(
    ctx: &Context,
    shape: &LoopShape,
    trip: i128,
    offset: i128,
) -> bool {
    if offset == 0 {
        return true;
    }
    let Some((min, max)) = integer_value_bounds(ctx, shape.iv_type) else {
        return false;
    };
    let Some(final_iv) = trip
        .checked_mul(shape.iv_step)
        .and_then(|delta| shape.iv_init.checked_add(delta))
    else {
        return false;
    };
    [shape.iv_init, final_iv].iter().all(|&iv| {
        iv.checked_add(offset)
            .is_some_and(|tested| (min..=max).contains(&tested))
    })
}

/// A grouped positive-IV span must be small enough to cross the type boundary
/// at most once. The runtime guard can then detect that crossing reliably.
fn partial_span_is_representable(ctx: &Context, ty: TypeHandle, span: i128) -> bool {
    let Some((_min, max)) = integer_value_bounds(ctx, ty) else {
        return false;
    };
    (0..=max).contains(&span)
}

/// Fully unroll a loop whose iteration count is known at compile time, so no
/// loop is left at all. Works for any body shape the [`LoopShape`] checks allow,
/// including bodies with internal `if`/`else`/`match`, nested loops, early
/// exits, and multiple exit targets.
///
/// For a trip count `T`, it lays down `T` copies of the body, chained one into
/// the next: copy 0 entered from the preheader, copy `k`'s latch flowing into
/// copy `k+1`, and the last copy's latch flowing to the header's normal exit. In copy `k`
/// the counter is the literal `init + k*step`, and the other carried values are
/// threaded from each copy to the next. The original loop blocks become
/// unreachable and `simplify_cfg` deletes them.
///
/// Early exits are cloned with the body. Their outside targets stay unchanged,
/// while their edge operands are remapped to the current copy. An exit from copy
/// `k` therefore skips every later copy, just as `break` skips later iterations.
/// Values made in the body must leave through exit-edge operands and block
/// arguments. A small normalization handles direct outside uses of header-carried
/// values before cloning.
fn full_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
) -> Result<UnrollOutcome> {
    let s = match analyze_shape(ctx, info, region, id, preheader, rec, true) {
        Ok(s) => s,
        Err(why) => return Ok(UnrollOutcome::Skipped(why)),
    };
    let trip_count = match rec.trip_count {
        Some(t) => t,
        None => {
            return Ok(UnrollOutcome::Skipped(
                "full #[unroll] needs a compile-time-constant trip count; this loop's count is only known at runtime (use #[unroll(N)] for partial unrolling)".into(),
            ));
        }
    };
    if let Err(reason) = check_clone_budget(ctx, trip_count, &s.body_blocks_ordered) {
        return Ok(UnrollOutcome::Skipped(reason));
    }
    if !liveouts_are_safe(ctx, &s) {
        return Ok(UnrollOutcome::Skipped(
            "full unrolling requires values made in the loop body to leave through exit block arguments".into(),
        ));
    }
    let trip = i128::from(trip_count);
    if !full_iv_stays_in_range(ctx, &s, trip) {
        return Ok(UnrollOutcome::Skipped(
            "the loop counter may wrap before the computed full-unroll trip count is reached"
                .into(),
        ));
    }
    if !full_exit_test_stays_in_range(ctx, &s, trip, rec.iv_offset) {
        return Ok(UnrollOutcome::Skipped(
            "the exit test's `counter + const` may wrap in the counter's type, so the computed trip count may be wrong".into(),
        ));
    }

    // Precompute every literal before changing the CFG. Besides keeping all
    // arithmetic checked, this guarantees that an unsupported recurrence cannot
    // leave behind a partly unrolled loop.
    let copies = match usize::try_from(trip_count) {
        Ok(copies) => copies,
        Err(_) => {
            return Ok(UnrollOutcome::Skipped(
                "the full-unroll copy count does not fit this target's address space".into(),
            ));
        }
    };
    let Some(literal_count) = copies.checked_add(1) else {
        return Ok(UnrollOutcome::Skipped(
            "the full-unroll counter table is too large to allocate safely".into(),
        ));
    };
    let mut iv_literals = Vec::with_capacity(literal_count);
    for k in 0..=trip_count {
        let Some(delta) = i128::from(k).checked_mul(s.iv_step) else {
            return Ok(UnrollOutcome::Skipped(
                "the full-unroll counter arithmetic overflows the analysis range".into(),
            ));
        };
        let Some(value) = s.iv_init.checked_add(delta) else {
            return Ok(UnrollOutcome::Skipped(
                "the full-unroll counter arithmetic overflows the analysis range".into(),
            ));
        };
        iv_literals.push(value);
    }

    // Carried values flowing into the next copy; start at the loop's initial
    // values. `prev_tail` is the branch we must point at the next copy (the
    // preheader first, then each copy's latch).
    let mut running: Vec<Value> = s.init_ops.clone();
    let mut prev_tail = s.preheader_term;

    for &iv_literal in iv_literals.iter().take(copies) {
        // The counter in copy k is the literal init + k*step. Materialize it just
        // before the branch into this copy, so it dominates the copy.
        let iv_val = make_const(ctx, s.iv_type, iv_literal, prev_tail);
        let mut subst = running.clone();
        subst[s.iv_idx] = iv_val;
        let c = clone_one_copy(ctx, region, &s, &subst);
        rewire_goto(ctx, prev_tail, c.entry, &c.entry_args);
        prev_tail = c.latch_term;
        running = c.next_running;
    }

    // The loop has finished: the counter's final value is the literal
    // init + T*step. Use it for any live-out reads of the counter.
    let final_iv = make_const(ctx, s.iv_type, iv_literals[copies], prev_tail);
    running[s.iv_idx] = final_iv;

    // Branch the last copy to the exit, feeding the exit's own block arguments (if
    // any) the final carried values.
    let exit_args: Vec<Value> = s
        .normal_exit_ops
        .iter()
        .map(|&v| match s.header_args.iter().position(|&h| h == v) {
            Some(a) => running[a],
            None => v,
        })
        .collect();
    rewire_goto(ctx, prev_tail, s.normal_exit, &exit_args);

    // The exit (and code after it) may also read the carried values *directly* by
    // dominance (this IR is not loop-closed SSA, so a header block argument can be
    // used outside the loop). The original header is now dead, so those reads must
    // be repointed to the final unrolled values. Uses inside the loop are left
    // alone; their blocks are unreachable and get deleted by `simplify_cfg`.
    if !s.has_early_exits {
        for (a, &replacement) in running.iter().enumerate().take(s.nargs) {
            s.header_args[a].replace_some_uses_with(
                ctx,
                |ctx, u| match u.user_op().deref(ctx).get_parent_block() {
                    Some(b) => !s.loop_blocks.contains(&b),
                    None => true,
                },
                &replacement,
            );
        }
    }

    Ok(UnrollOutcome::Unrolled)
}

/// Build an integer constant op (`mir.constant`) of type `ty` holding `value`,
/// place it just before the op `before`, and hand back the value it produces.
fn make_const(ctx: &mut Context, ty: TypeHandle, value: i128, before: Ptr<Operation>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("IV is an integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i128(value, NonZero::new(width).expect("non-zero width"));
    let attr = IntegerAttr::new(typed, apint);
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    op.insert_before(ctx, before);
    op.deref(ctx).get_result(0)
}

/// Partially unroll a loop: do `factor` iterations' worth of work per trip, and
/// keep a small "remainder" loop for the iterations left over when the total
/// doesn't divide evenly by `factor`. Works for any body shape the [`LoopShape`]
/// checks allow (multi-block bodies included).
///
/// Unlike full unroll, this works even when the iteration count is only known at
/// runtime. The original loop is reused as the **remainder loop**, running the
/// last `trip % factor` iterations one at a time. In front of it we build a new
/// **main loop** whose body is `factor` copies of the loop body chained
/// together, advancing the counter by `factor*step` each trip. The main loop
/// keeps going only while a whole group of `factor` more iterations still fits;
/// once fewer than that remain, control falls into the remainder loop.
///
/// Multiple source latches are supported after normalization. An early `break`
/// or another exit is not yet supported here: the main and remainder loops would
/// need their path-specific exit values merged. Such a request warns and skips.
/// Only counting-up loops (test `<` or `<=`, positive step) are handled.
///
/// What it produces (the loop was entered as `preheader -> header`):
/// ```text
///   preheader -> main_h(init...)
///   main_h(acc, i):                       (i = counter, acc = carried values)
///       if (i + (factor-1)*step) + off <pred> bound  -> copy0   (a full group fits)
///       else                                         -> header  (run the remainder)
///   copy0 .. copy(factor-1): the body, factor times, chained; the last copy's
///       latch loops back to main_h with (acc', i + factor*step)
///   header/.../latch: the original loop, now just the leftover tail
/// ```
fn partial_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
    factor: u32,
) -> Result<UnrollOutcome> {
    if factor < 2 {
        return Ok(UnrollOutcome::Skipped(format!(
            "unroll factor {factor} is too small (need 2 or more)"
        )));
    }
    let n = i128::from(factor);
    let s = match analyze_shape(ctx, info, region, id, preheader, rec, false) {
        Ok(s) => s,
        Err(why) => return Ok(UnrollOutcome::Skipped(why)),
    };
    if let Err(reason) = check_clone_budget(ctx, u64::from(factor), &s.body_blocks_ordered) {
        return Ok(UnrollOutcome::Skipped(reason));
    }
    let factor_usize = match usize::try_from(factor) {
        Ok(factor) => factor,
        Err(_) => {
            return Ok(UnrollOutcome::Skipped(
                "the partial-unroll factor does not fit this target's address space".into(),
            ));
        }
    };
    // Partial unroll needs an up-counting loop with a known bound value.
    if s.iv_step <= 0 || !matches!(rec.continue_pred, Some(CmpPred::Lt) | Some(CmpPred::Le)) {
        return Ok(UnrollOutcome::Skipped(
            "partial #[unroll(N)] supports only up-counting loops (test < or <=, positive step) for now".into(),
        ));
    }
    let Some(last_span) = (n - 1).checked_mul(s.iv_step) else {
        return Ok(UnrollOutcome::Skipped(
            "the partial-unroll counter step is too large to analyze safely".into(),
        ));
    };
    let Some(group_step) = n.checked_mul(s.iv_step) else {
        return Ok(UnrollOutcome::Skipped(
            "the partial-unroll counter step is too large to analyze safely".into(),
        ));
    };
    if !partial_span_is_representable(ctx, s.iv_type, last_span) {
        return Ok(UnrollOutcome::Skipped(
            "the partial-unroll group spans too much of the counter type to guard wraparound safely"
                .into(),
        ));
    }
    let pred = rec.continue_pred.unwrap();
    let bound = match rec.bound_value {
        Some(b) => b,
        None => {
            return Ok(UnrollOutcome::Skipped(
                "partial #[unroll(N)] needs the loop bound as a value".into(),
            ));
        }
    };
    // The guard we build below, `i + (N-1)*step <pred> bound`, is only correct if
    // `bound` is the same on every iteration. A constant bound is always fine (we
    // re-materialize it in the main header below, so where its op sits does not
    // matter). A non-constant bound that is a loop-carried header argument or is
    // computed inside the loop can change within a group of N iterations, which
    // would make the guard admit too many iterations (a miscompile), and would
    // not dominate the new main header either. Bail loudly on those.
    let bound_is_const = induction::const_i128(ctx, bound).is_some();
    if !bound_is_const
        && (s.header_args.contains(&bound) || defined_in_loop(ctx, bound, &s.loop_blocks))
    {
        return Ok(UnrollOutcome::Skipped(
            "partial #[unroll(N)] needs a loop-invariant bound (the loop's limit must be the same on every iteration); this loop's limit changes inside the loop".into(),
        ));
    }

    // Validate each per-copy offset before creating the main header or cloning
    // any body block, so arithmetic failure is always a clean skip.
    let mut copy_offsets = Vec::with_capacity(factor_usize);
    for j in 0..factor {
        let Some(offset) = i128::from(j).checked_mul(s.iv_step) else {
            return Ok(UnrollOutcome::Skipped(
                "the partial-unroll counter offset overflows the analysis range".into(),
            ));
        };
        copy_offsets.push(offset);
    }

    let arg_types: Vec<TypeHandle> = s.header_args.iter().map(|a| a.get_type(ctx)).collect();
    // The new main-loop header, taking the same carried values as the original.
    let main_h = BasicBlock::new(ctx, None, arg_types);
    main_h.insert_before(ctx, s.header);
    let mh_args: Vec<Value> = (0..s.nargs)
        .map(|i| main_h.deref(ctx).get_argument(i))
        .collect();
    let mh_iv = mh_args[s.iv_idx];

    // Lay down `factor` copies of the body, threading non-IV carried values from
    // one copy to the next. Give each copy an explicit `mh_iv + j*step` counter.
    // This is semantically the same recurrence, and it keeps the affine relation
    // visible even when a synthetic unified latch forwards its counter through a
    // block argument.
    let mut running: Vec<Value> = mh_args.clone();
    let mut copies: Vec<CopyResult> = Vec::with_capacity(factor_usize);
    for (j, offset) in copy_offsets.iter().copied().enumerate() {
        let copy_iv = if j == 0 {
            mh_iv
        } else {
            let offset_value = append_const(ctx, s.iv_type, offset, main_h);
            append_add(ctx, s.iv_type, mh_iv, offset_value, main_h)
        };
        let mut subst = running.clone();
        subst[s.iv_idx] = copy_iv;
        let c = clone_one_copy(ctx, region, &s, &subst);
        running = c.next_running.clone();
        copies.push(c);
    }
    let next_offset = append_const(ctx, s.iv_type, group_step, main_h);
    running[s.iv_idx] = append_add(ctx, s.iv_type, mh_iv, next_offset, main_h);

    // Chain copy j's latch into copy j+1's entry; the last copy's latch loops
    // back to main_h carrying `running` (counter now mh_iv + factor*step).
    for j in 0..(factor_usize - 1) {
        let next_entry = copies[j + 1].entry;
        let next_args = copies[j + 1].entry_args.clone();
        rewire_goto(ctx, copies[j].latch_term, next_entry, &next_args);
    }
    let last_latch = copies[factor_usize - 1].latch_term;
    rewire_goto(ctx, last_latch, main_h, &running);

    // The main-loop counter is mh_iv = init + (factor*step)*t, so it is always
    // init plus a multiple of factor*step. That lets us replace counter-derived
    // index ops -- `(counter +/- const) & mask` and `(counter +/- const) % 2^k`
    // -- in the copies with literals (`fold_constant_index_in_copies`), the main
    // payoff of unrolling. Scan every cloned copy block.
    let copy_blocks: Vec<Ptr<BasicBlock>> = copies
        .iter()
        .flat_map(|c| c.blocks.iter().copied())
        .collect();
    fold_constant_index_in_copies(ctx, &copy_blocks, mh_iv, s.iv_init, group_step);

    // main_h guard: stay in the main loop only while a whole group of `factor`
    // iterations still fits. The last copy in a group uses counter
    // mh_iv + (factor-1)*step, so the group fits when that still passes the test
    // and did not wrap below mh_iv. True -> copy 0 (enter the main body); False
    // -> the original header (the remainder loop), passing the current carried
    // values. The remainder then preserves the source loop's exact wrap behavior.
    // A constant bound may be defined inside the (original) loop, which does not
    // dominate `main_h`; re-materialize it here. A non-constant bound was already
    // checked to be defined outside the loop, so it dominates `main_h` as-is.
    let guard_bound = match induction::const_i128(ctx, bound) {
        Some(c) => append_const(ctx, bound.get_type(ctx), c, main_h),
        None => bound,
    };
    let last_off = append_const(ctx, s.iv_type, last_span, main_h);
    let last_iv = append_add(ctx, s.iv_type, mh_iv, last_off, main_h);
    // With an exit test `IV + off <pred> bound`, compare what the source test
    // computes for the last copy. That sum is only meaningful if it did not
    // wrap for any copy in the group. It moves monotonically with the counter,
    // so a positive `off` can only wrap at the last copy and a negative one
    // only at the first; one more comparison rules that out.
    let (last_tested, offset_no_wrap) = match rec.iv_offset {
        0 => (last_iv, None),
        off if off > 0 => {
            let off_value = append_const(ctx, s.iv_type, off, main_h);
            let tested = append_add(ctx, s.iv_type, last_iv, off_value, main_h);
            let no_wrap = append_cmp(ctx, CmpPred::Gt, tested, last_iv, s.i1_type, main_h);
            (tested, Some(no_wrap))
        }
        off => {
            let off_value = append_const(ctx, s.iv_type, off, main_h);
            let tested = append_add(ctx, s.iv_type, last_iv, off_value, main_h);
            let first_tested = append_add(ctx, s.iv_type, mh_iv, off_value, main_h);
            let no_wrap = append_cmp(ctx, CmpPred::Lt, first_tested, mh_iv, s.i1_type, main_h);
            (tested, Some(no_wrap))
        }
    };
    let within_bound = append_cmp(ctx, pred, last_tested, guard_bound, s.i1_type, main_h);
    let no_wrap = append_cmp(ctx, CmpPred::Ge, last_iv, mh_iv, s.i1_type, main_h);
    let mut cont = append_bitand(ctx, s.i1_type, within_bound, no_wrap, main_h);
    if let Some(offset_no_wrap) = offset_no_wrap {
        cont = append_bitand(ctx, s.i1_type, cont, offset_no_wrap, main_h);
    }
    let entry0 = copies[0].entry;
    let entry0_args = copies[0].entry_args.clone();
    let (flat, segs) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cont], entry0_args, mh_args.clone()]);
    let cbr = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![entry0, s.header],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(cbr, ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(ctx, segs);
    cbr.insert_at_back(main_h, ctx);

    // Finally, make the preheader branch into the new main loop instead of the
    // original header, reusing the same initial values it already passed. The
    // original loop stays in place and becomes the remainder.
    rewire_goto(ctx, s.preheader_term, main_h, &s.init_ops);
    Ok(UnrollOutcome::Unrolled)
}

/// If `v` is the counter `iv` plus or minus some constants (e.g. `iv`, `iv + 1`,
/// `iv + 4 - 2`), return that net constant offset; `None` otherwise. So `iv + 1`
/// gives `Some(1)` and `iv` gives `Some(0)`. The caller uses this to spot values
/// that track the counter with a known fixed offset.
fn affine_offset(ctx: &Context, v: Value, iv: Value) -> Option<i128> {
    if v == iv {
        return Some(0);
    }
    let def = v.defining_op()?;
    if Operation::get_op::<MirAddOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if let (Some(o), Some(c)) = (affine_offset(ctx, a, iv), induction::const_i128(ctx, b)) {
            return o.checked_add(c);
        }
        if let (Some(o), Some(c)) = (affine_offset(ctx, b, iv), induction::const_i128(ctx, a)) {
            return o.checked_add(c);
        }
    } else if Operation::get_op::<MirSubOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if let (Some(o), Some(c)) = (affine_offset(ctx, a, iv), induction::const_i128(ctx, b)) {
            return o.checked_sub(c);
        }
    }
    None
}

/// Peephole: in each partial-unroll copy, replace a counter-derived index with
/// the constant it always equals.
///
/// After unrolling by N, the main counter `iv` only takes values N apart:
/// `init, init+N, init+2N, ...`. Copy `j` of the body uses `iv + j`. Two index
/// shapes are then the same constant on every iteration, so we replace each with
/// a literal (note `x & (2^k - 1)` and `x % 2^k` are the same operation):
///
/// ```text
///   iv & MASK   (MASK = 2^k - 1)      keep the low k bits
///   iv % M      (M a power of two)    same thing: x % 2^k == x & (2^k - 1)
/// ```
///
/// Both read only the low k bits, and a multiple of N has *fixed* low k bits
/// exactly when the window `2^k` divides the unroll step `N*step`. Example,
/// N = 4, `(iv + j) & 3` (the gemm pipeline-stage index):
///
/// ```text
///   iv = 0, 4, 8, 12, ...   all end in ...00, so:
///     (iv + 0) & 3 = 0       (iv + 2) & 3 = 2
///     (iv + 1) & 3 = 1       (iv + 3) & 3 = 3
///   => each copy's stage index is a compile-time constant.
/// ```
///
/// Fires only when ALL of these hold (otherwise the value genuinely changes each
/// iteration, so there is nothing to fold and we skip):
///
/// - the op is `(iv +/- consts) & MASK` or `(iv +/- consts) % M`;
/// - the window (`MASK + 1`, or `M`) is a power of two -- so it reads only low
///   bits and is therefore immune to the type's wraparound;
/// - that window divides the unroll step `N*step` -- so the low bits never move;
/// - for `%`, the operand type is unsigned (signed `%` follows the dividend's
///   sign, which breaks the equality with the masked low bits);
/// - `M > 0` -- never fold `% 0` (rem-by-zero is a Rust panic).
///
/// Deliberately NOT handled: non-power-of-two `% M` (e.g. `% 3`). The congruence
/// still holds on paper, but `%` by a non-power-of-two is not wraparound-safe:
/// near the type's max the counter wraps by `2^width`, which is not a multiple
/// of `M`, shifting the result. Documented gap, left for later.
///
/// Full unroll never needs this: there `iv` is a literal per copy, so ordinary
/// constant folding already turns `i & 3` / `i % 3` into a number. The leftover
/// dead `& MASK` / `% M` ops are removed by the unroll pass's later `dce`.
///
/// Parameters: `iv` is the counter; `init` its start; `step_jump` the unroll step
/// (`N * original_step`), so `iv = init + step_jump * t` for iteration `t`.
/// `blocks` are all the cloned copy blocks (one copy may span several blocks).
fn fold_constant_index_in_copies(
    ctx: &mut Context,
    blocks: &[Ptr<BasicBlock>],
    iv: Value,
    init: i128,
    step_jump: i128,
) {
    if step_jump <= 0 {
        return;
    }
    for &block in blocks {
        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some((offset, window)) = counter_index_window(ctx, op, iv) else {
                continue;
            };
            // Foldable only when the window is a power of two that divides the
            // unroll step: then the counter's low bits never move, so the index
            // is the same every iteration (and power-of-two makes it wrap-safe).
            // The power-of-two check also rejects a non-low-bit `& C` (e.g.
            // `x & 5` is not `x % 6`).
            if window <= 0 || (window & (window - 1)) != 0 || step_jump % window != 0 {
                continue;
            }
            let Some(base) = init.checked_add(offset) else {
                continue;
            };
            let folded = base.rem_euclid(window);
            let result = op.deref(ctx).get_result(0);
            let ty = result.get_type(ctx);
            let lit = make_const(ctx, ty, folded, op);
            result.replace_all_uses_with(ctx, &lit);
        }
    }
}

/// If `op` is a counter-derived index we might fold -- `(iv +/- consts) & MASK`
/// or `(iv +/- consts) % M` -- return `(offset, window)`, where `offset` is the
/// counter's constant offset and `window` is `MASK + 1` (for `&`) or `M` (for
/// `%`). The caller still checks that `window` is a power of two dividing the
/// unroll step. Returns `None` for any other op.
fn counter_index_window(ctx: &Context, op: Ptr<Operation>, iv: Value) -> Option<(i128, i128)> {
    if Operation::get_op::<MirBitAndOp>(op, ctx).is_some() {
        // `&` is commutative: either operand may be the counter.
        let a = op.deref(ctx).get_operand(0);
        let b = op.deref(ctx).get_operand(1);
        let (offset, mask) = if let (Some(o), Some(m)) =
            (affine_offset(ctx, a, iv), induction::const_i128(ctx, b))
        {
            (o, m)
        } else if let (Some(o), Some(m)) =
            (affine_offset(ctx, b, iv), induction::const_i128(ctx, a))
        {
            (o, m)
        } else {
            return None;
        };
        if mask < 0 {
            return None;
        }
        // window = MASK + 1.
        Some((offset, mask.checked_add(1)?))
    } else if Operation::get_op::<MirRemOp>(op, ctx).is_some() {
        // `%` is NOT commutative: only the dividend (operand 0) may be the
        // counter, and only for unsigned types.
        let dividend = op.deref(ctx).get_operand(0);
        let divisor = op.deref(ctx).get_operand(1);
        if !is_unsigned_int(ctx, dividend) {
            return None;
        }
        let (Some(offset), Some(m)) = (
            affine_offset(ctx, dividend, iv),
            induction::const_i128(ctx, divisor),
        ) else {
            return None;
        };
        if m <= 0 {
            return None;
        }
        // window = M.
        Some((offset, m))
    } else {
        None
    }
}

/// True if `v` has an unsigned (or signless) integer type. Keeps the `%` fold off
/// signed remainders, whose result follows the dividend's sign rather than the
/// masked low bits.
fn is_unsigned_int(ctx: &Context, v: Value) -> bool {
    let ty = v.get_type(ctx);
    match TypedHandle::<IntegerType>::from_handle(ty, ctx) {
        Ok(t) => t.deref(ctx).signedness() != Signedness::Signed,
        Err(_) => false,
    }
}

/// Build an integer constant `value` of type `ty`, add it as the last op of
/// `block`, and return the value it produces. (Same as [`make_const`] but
/// appends to a block instead of inserting before a given op.)
fn append_const(ctx: &mut Context, ty: TypeHandle, value: i128, block: Ptr<BasicBlock>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i128(value, NonZero::new(width).expect("non-zero width"));
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(typed, apint));
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build `a + b` (an integer `mir.add` of type `ty`), add it as the last op of
/// `block`, and return its result value.
fn append_add(
    ctx: &mut Context,
    ty: TypeHandle,
    a: Value,
    b: Value,
    block: Ptr<BasicBlock>,
) -> Value {
    let op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![ty],
        vec![a, b],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build boolean `a & b`, add it as the last op of `block`, and return its
/// result. Both inputs and the result have `i1_type`.
fn append_bitand(
    ctx: &mut Context,
    i1_type: TypeHandle,
    a: Value,
    b: Value,
    block: Ptr<BasicBlock>,
) -> Value {
    let op = Operation::new(
        ctx,
        MirBitAndOp::get_concrete_op_info(),
        vec![i1_type],
        vec![a, b],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build the comparison `a <pred> b` (a boolean, of type `i1_type`), add it as
/// the last op of `block`, and return its result value.
fn append_cmp(
    ctx: &mut Context,
    pred: CmpPred,
    a: Value,
    b: Value,
    i1_type: TypeHandle,
    block: Ptr<BasicBlock>,
) -> Value {
    let info = match pred {
        CmpPred::Lt => MirLtOp::get_concrete_op_info(),
        CmpPred::Le => MirLeOp::get_concrete_op_info(),
        CmpPred::Gt => MirGtOp::get_concrete_op_info(),
        CmpPred::Ge => MirGeOp::get_concrete_op_info(),
    };
    let op = Operation::new(ctx, info, vec![i1_type], vec![a, b], vec![], 0);
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// All function ops in the module (each `mir.func`).
fn collect_functions(module: Ptr<Operation>, ctx: &Context) -> Vec<Ptr<Operation>> {
    let mut out = Vec::new();
    let module_region = module.deref(ctx).get_region(0);
    let blocks: Vec<Ptr<BasicBlock>> = module_region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirFuncOp>(op, ctx).is_some() {
                out.push(op);
            }
        }
    }
    out
}

/// Find the `mir.unroll_hint` ops in `region`, each with the block it sits in
/// (used to locate the enclosing loop) and its requested factor (0 = full).
fn collect_hints(
    ctx: &Context,
    region: Ptr<Region>,
) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>, u32)> {
    let mut out = Vec::new();
    let blocks: Vec<Ptr<BasicBlock>> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if let Some(hint) = Operation::get_op::<MirUnrollHintOp>(op, ctx) {
                out.push((op, block, hint.factor(ctx)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::check_growth_budget;

    #[test]
    fn growth_budget_counts_operations_not_only_blocks() {
        let err = check_growth_budget(1_024, 1, 65).unwrap_err();
        assert!(err.contains("65536 cloned operations"));
        assert!(
            check_growth_budget(1_024, 1, 64).is_ok(),
            "the exact operation limit remains allowed"
        );
    }
}

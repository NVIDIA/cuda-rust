/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tests for the unroll pass itself. Rather than poke at the
//! internal shape analysis, these build a counted loop, plant an `#[unroll]`
//! marker (`mir.unroll_hint`) in its body, run `unroll_annotated_loops`, and
//! check the result through the public `LoopInfo` analysis.
//!
//! The key observable: fully unrolling the only loop in a function leaves a
//! function with no loops at all.

mod common;

use common::{
    OffsetBound, counted_loop, counted_loop_from_step, early_exit_counted_loop,
    early_exit_with_direct_liveout, mir_ctx, multi_latch_counted_loop, multiple_exit_counted_loop,
    nested_counted_loop, offset_counted_loop, u32t,
};
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirCallOp, MirCondBranchOp, MirConstantOp, MirGeOp, MirGtOp, MirLeOp,
    MirLtOp, MirNotOp, MirReturnOp, MirSubOp, MirUnrollHintOp,
};
use mir_transforms::analyses::induction::{CmpPred, analyze};
use mir_transforms::unroll::unroll_annotated_loops;
use pliron::attribute::Attribute;
use pliron::builtin::attributes::{IntegerAttr, StringAttr};
use pliron::builtin::ops::ConstantOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::graph::{ControlFlowGraph, dominance::DomInfo};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::r#type::{Typed, TypedHandle};
use pliron::value::Value;
use std::collections::HashMap;

use mir_transforms::analyses::loop_info::LoopInfo;

/// How many natural loops are left in `lp`'s region.
fn loop_count(
    ctx: &pliron::context::Context,
    region: pliron::context::Ptr<pliron::region::Region>,
) -> usize {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt).loops().len()
}

fn operations(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect()
}

fn hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirUnrollHintOp>(op, ctx).is_some())
        .count()
}

fn cond_branch_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirCondBranchOp>(op, ctx).is_some())
        .count()
}

fn return_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirReturnOp>(op, ctx).is_some())
        .count()
}

fn constant_i128(ctx: &Context, value: pliron::value::Value) -> Option<i128> {
    let def = value.defining_op()?;
    if let Some(c) = Operation::get_op::<MirConstantOp>(def, ctx) {
        return c.get_attr_value(ctx).map(|a| a.value().to_i128());
    }
    let attr = Operation::get_op::<ConstantOp>(def, ctx)?.get_value(ctx);
    (&*attr as &dyn Attribute)
        .downcast_ref::<IntegerAttr>()
        .map(|a| a.value().to_i128())
}

fn sole_return_constant(ctx: &Context, region: Ptr<Region>) -> Option<i128> {
    let returns: Vec<_> = operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirReturnOp>(op, ctx).is_some())
        .collect();
    if returns.len() != 1 || returns[0].deref(ctx).get_num_operands() != 1 {
        return None;
    }
    constant_i128(ctx, returns[0].deref(ctx).get_operand(0))
}

fn loop_info(ctx: &Context, region: Ptr<Region>) -> LoopInfo {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt)
}

/// A full-unroll hint on a constant-trip loop deletes the loop entirely:
/// `while i < 4 { .. }` becomes four straight-line copies with no back-edge.
#[test]
fn full_unroll_removes_the_loop() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4); // while i < 4 -> trip count 4

    assert_eq!(loop_count(&ctx, lp.region), 1, "starts with one loop");

    // Plant a full-unroll marker (factor 0 = full) in the loop body.
    let hint = MirUnrollHintOp::new(&mut ctx, 0);
    hint.get_operation().insert_at_front(lp.latch, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("unroll pass succeeds");

    assert_eq!(
        loop_count(&ctx, lp.region),
        0,
        "fully unrolling the only loop should leave no loop"
    );
}

/// With no `#[unroll]` marker the pass is a no-op: the loop is left intact.
#[test]
fn no_hint_leaves_the_loop_intact() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("unroll pass succeeds");

    assert_eq!(
        loop_count(&ctx, lp.region),
        1,
        "no hint => the loop is untouched"
    );
}

/// Fully unrolling an OUTER loop that *contains* an inner loop clones the inner
/// loop wholesale once per outer iteration: the outer loop disappears, the inner
/// loop stays a loop, and there is one inner-loop copy per outer iteration. This
/// is the capability the `!children` bail used to forbid.
#[test]
fn nested_outer_full_unroll_clones_the_inner_loop() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2); // outer trip 3, inner trip 2

    assert_eq!(
        loop_count(&ctx, lp.region),
        2,
        "starts with outer + inner loop"
    );

    // Full-unroll marker on the OUTER loop (its body block dominating the inner).
    let hint = MirUnrollHintOp::new(&mut ctx, 0);
    hint.get_operation().insert_at_front(lp.outer_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("nested outer unroll");

    // The transform must leave valid IR (it clones inner-loop blocks + back-edges
    // and recomputes loop structure; a bug here would corrupt the CFG).
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid IR after nested unroll");

    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert!(
        !info.loops().iter().any(|l| l.header == lp.outer_header),
        "the outer loop is gone (fully unrolled)"
    );
    assert_eq!(
        info.loops().len(),
        3,
        "the inner loop is cloned once per outer iteration (3 inner loops, 0 outer)"
    );
}

/// Fully unrolling only the inner loop removes that loop and leaves its outer
/// container intact.
#[test]
fn nested_inner_full_unroll_keeps_outer_loop() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.inner_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("nested inner unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after unrolling the inner loop");
    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert_eq!(info.loops().len(), 1, "only the outer loop should remain");
    assert_eq!(info.loops()[0].header, lp.outer_header);
}

/// Partially unrolling an outer loop that contains an inner loop is valid: the
/// outer loop becomes a main loop (stepping by N) plus a remainder, and each
/// copy carries its own clone of the inner loop. We assert the IR verifies and
/// that loops survive the transform (the gemm K-loop shape, minus the fold which
/// `unroll_smoke` checks numerically).
#[test]
fn nested_outer_partial_unroll_is_valid() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 8, 2);

    assert_eq!(loop_count(&ctx, lp.region), 2);

    let hint = MirUnrollHintOp::new(&mut ctx, 4); // partial unroll by 4
    hint.get_operation().insert_at_front(lp.outer_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("nested outer partial unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after nested partial unroll");
    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert_eq!(
        info.loops().len(),
        7,
        "main outer + remainder outer + original inner + four cloned inner loops"
    );
    assert_eq!(info.top_level().len(), 2, "main loop + remainder loop");
    let mut child_counts: Vec<usize> = info
        .top_level()
        .iter()
        .map(|&id| info.loops()[id].children.len())
        .collect();
    child_counts.sort_unstable();
    assert_eq!(child_counts, [1, 4], "one and four nested inner loops");
}

/// When both loops are annotated, the driver must consume the inner hint first.
/// Otherwise cloning the outer body would duplicate the inner hint and either
/// unroll it repeatedly or leave marker operations behind.
#[test]
fn nested_inner_and_outer_full_unroll_innermost_first() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.outer_body, &ctx);
    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.inner_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("nested inner + outer unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after unrolling both nested loops");
    assert_eq!(
        loop_count(&ctx, lp.region),
        0,
        "both fully unrolled loops should be gone"
    );
}

#[test]
fn full_unroll_handles_multi_latch_continue() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 1);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.loops()[id].latches.len(), 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll of multi-latch loop");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-latch full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(sole_return_constant(&ctx, lp.region), Some(5));
}

#[test]
fn partial_unroll_handles_multi_latch_continue() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 5, 1, 1);

    MirUnrollHintOp::new(&mut ctx, 2)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial unroll of multi-latch loop");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-latch partial unroll");
    assert_eq!(
        loop_count(&ctx, lp.region),
        2,
        "partial unroll should build a main loop and keep a remainder"
    );
}

#[test]
fn inconsistent_multi_latch_steps_are_skipped() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("inconsistent recurrence is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_preserves_an_early_break() {
    let mut ctx = mir_ctx();
    let lp = early_exit_counted_loop(&mut ctx, 4, 2);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.exiting_blocks(&ctx, lp.region, id).len(), 2);
    assert_eq!(before.exit_blocks(&ctx, lp.region, id), vec![lp.exit]);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.body, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll with early break");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after early-break full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(
        sole_return_constant(&ctx, lp.region),
        Some(3),
        "0 + 1 + 2, then the early break bypasses the fourth copy"
    );
}

#[test]
fn full_unroll_preserves_multiple_exit_targets() {
    let mut ctx = mir_ctx();
    let lp = multiple_exit_counted_loop(&mut ctx, 4);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.exiting_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(before.exit_blocks(&ctx, lp.region, id).len(), 3);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.check_a, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll with multiple exits");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-exit full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(
        cond_branch_count(&ctx, lp.region),
        8,
        "two runtime break checks survive in each of four copies"
    );
    assert_eq!(
        lp.region.predecessors(&ctx, &lp.exit_a).len(),
        4,
        "each copy keeps its first early-exit edge"
    );
    assert_eq!(
        lp.region.predecessors(&ctx, &lp.exit_b).len(),
        4,
        "each copy keeps its second early-exit edge"
    );
    assert_eq!(return_count(&ctx, lp.region), 3);
}

#[test]
fn partial_unroll_with_extra_exits_is_skipped() {
    let mut ctx = mir_ctx();
    let lp = multiple_exit_counted_loop(&mut ctx, 4);

    MirUnrollHintOp::new(&mut ctx, 2)
        .get_operation()
        .insert_at_front(lp.check_a, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial multi-exit is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    let after = loop_info(&ctx, lp.region);
    assert_eq!(after.loops().len(), 1, "no main loop was introduced");
    let id = after.innermost_loop(lp.header).unwrap();
    assert_eq!(after.loops()[id].latches, vec![lp.latch]);
    assert_eq!(after.exiting_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(after.exit_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(cond_branch_count(&ctx, lp.region), 3);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_routes_a_direct_header_liveout_through_exit_arguments() {
    let mut ctx = mir_ctx();
    let lp = early_exit_with_direct_liveout(&mut ctx, 4);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.body, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll canonicalizes direct header live-outs");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after live-out canonicalization and unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_skips_a_counter_that_wraps_before_exit() {
    let mut ctx = mir_ctx();
    let lp = counted_loop_from_step(&mut ctx, i64::from(u32::MAX) - 1, i64::from(u32::MAX), 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("wrapping full-unroll request is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1);
    assert_eq!(hint_count(&ctx, lp.region), 0);
}

#[test]
fn partial_unroll_guards_the_last_copy_against_wraparound() {
    let mut ctx = mir_ctx();
    let lp = counted_loop_from_step(&mut ctx, i64::from(u32::MAX) - 4, i64::from(u32::MAX), 1);

    MirUnrollHintOp::new(&mut ctx, 4)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial unroll near the unsigned boundary");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR with a no-wrap group guard");
    assert_eq!(loop_count(&ctx, lp.region), 2);
    assert!(
        operations(&ctx, lp.region)
            .iter()
            .any(|&op| Operation::get_op::<MirGeOp>(op, &ctx).is_some()),
        "main-loop guard must compare last_iv >= current_iv"
    );
    assert!(
        operations(&ctx, lp.region)
            .iter()
            .any(|&op| Operation::get_op::<MirBitAndOp>(op, &ctx).is_some()),
        "main-loop guard must require both in-bounds and no-wrap"
    );
}

#[test]
fn side_effecting_loop_header_is_skipped() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    let header_term = lp
        .header
        .deref(&ctx)
        .get_terminator(&ctx)
        .expect("loop header terminator");
    let side_effect = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let side_effect = MirCallOp::new(side_effect);
    side_effect.set_attr_callee(&ctx, StringAttr::new("header_effect".into()));
    let signature = FunctionType::get(&ctx, vec![], vec![]);
    side_effect.set_external_callee_signature(&mut ctx, signature.into());
    side_effect.get_operation().insert_before(&ctx, header_term);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("a side-effecting header is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn huge_partial_unroll_factor_is_skipped_before_cloning() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    MirUnrollHintOp::new(&mut ctx, u32::MAX)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("an oversized partial-unroll request is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

fn op_count<T: Op>(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<T>(op, ctx).is_some())
        .count()
}

/// The trip count the induction analysis computes for `lp`.
fn analyzed_trip_count(ctx: &Context, lp: &common::CountedLoop) -> Option<u64> {
    let info = loop_info(ctx, lp.region);
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(ctx, lp.region, id).unwrap();
    analyze(ctx, &info, id, ph).trip_count
}

/// Plant an unroll hint (`factor` 0 = full) in the latch and run the pass.
fn unroll_offset_loop(ctx: &mut Context, lp: &common::CountedLoop, factor: u32) {
    MirUnrollHintOp::new(ctx, factor)
        .get_operation()
        .insert_at_front(lp.latch, ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, ctx, &mut analyses).expect("unroll pass succeeds");
    pliron::operation::verify_operation(lp.module, ctx).expect("valid IR after the pass");
}

/// `while i + 1 <= 4 { acc += i; i += 1 }` fully unrolls: no loop is left and
/// the function returns the constant `0 + 1 + 2 + 3`.
#[test]
fn full_unroll_of_a_counter_offset_exit_test_folds_the_sum() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, 1, CmpPred::Le, OffsetBound::Const(4));

    unroll_offset_loop(&mut ctx, &lp, 0);

    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(sole_return_constant(&ctx, lp.region), Some(6));
}

/// `while i + 2 <= u32::MAX` from `u32::MAX - 5`: the analysis counts four
/// trips, but at the fifth test `i + 2` wraps to 0, which is still `<= MAX`, so
/// the source loop keeps going. Full unroll must skip it.
#[test]
fn full_unroll_skips_a_counter_offset_that_wraps_at_the_last_test() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let max = i128::from(u32::MAX);
    let lp = offset_counted_loop(
        &mut ctx,
        u32,
        max - 5,
        1,
        2,
        CmpPred::Le,
        OffsetBound::Const(max),
    );
    assert_eq!(
        analyzed_trip_count(&ctx, &lp),
        Some(4),
        "the analysis recognizes the test; only the wrap check may refuse it"
    );

    unroll_offset_loop(&mut ctx, &lp, 0);

    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

/// `while i - 1 < 4` with an unsigned counter from 0: `0 - 1` wraps to
/// `u32::MAX`, so the source loop runs zero times, not five. Full unroll must
/// skip it.
#[test]
fn full_unroll_skips_a_counter_offset_that_wraps_at_the_first_test() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, -1, CmpPred::Lt, OffsetBound::Const(4));
    assert_eq!(
        analyzed_trip_count(&ctx, &lp),
        Some(5),
        "the analysis recognizes the test; only the wrap check may refuse it"
    );

    unroll_offset_loop(&mut ctx, &lp, 0);

    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

/// `#[unroll(4)]` on `while i + 1 <= n` with a runtime `n` builds a main loop
/// and keeps the source loop as the remainder.
#[test]
fn partial_unroll_of_a_counter_offset_exit_test_keeps_a_remainder() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, 1, CmpPred::Le, OffsetBound::Param);

    unroll_offset_loop(&mut ctx, &lp, 4);

    assert_eq!(loop_count(&ctx, lp.region), 2, "main loop + remainder");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

/// From `u32::MAX - 5`, `while i + 4 <= n` never runs for `n = 10`, but in a
/// group of four the last counter is `u32::MAX - 2` and `(u32::MAX - 2) + 4`
/// wraps to 1, which passes `<= 10`. The main-loop guard must also require
/// `last + 4 > last`, so it has three tests joined by two `&`.
#[test]
fn partial_unroll_guards_a_positive_counter_offset_against_wraparound() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let max = i128::from(u32::MAX);
    let lp = offset_counted_loop(
        &mut ctx,
        u32,
        max - 5,
        1,
        4,
        CmpPred::Le,
        OffsetBound::Param,
    );

    unroll_offset_loop(&mut ctx, &lp, 4);

    assert_eq!(loop_count(&ctx, lp.region), 2, "main loop + remainder");
    assert_eq!(
        op_count::<MirGtOp>(&ctx, lp.region),
        1,
        "the guard checks that last + offset did not wrap"
    );
    assert_eq!(
        op_count::<MirBitAndOp>(&ctx, lp.region),
        2,
        "in bounds, counter no-wrap, and offset no-wrap"
    );
}

/// With `while i - 1 < n` from 0, the first test computes `0 - 1`, which wraps.
/// The main-loop guard must also require `first - 1 < first`, so the group
/// is left to the remainder loop whenever that subtraction wraps.
#[test]
fn partial_unroll_guards_a_negative_counter_offset_against_wraparound() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, -1, CmpPred::Lt, OffsetBound::Param);

    unroll_offset_loop(&mut ctx, &lp, 4);

    assert_eq!(loop_count(&ctx, lp.region), 2, "main loop + remainder");
    assert_eq!(
        op_count::<MirLtOp>(&ctx, lp.region),
        3,
        "the source test, the main-loop bound test, and first - 1 < first"
    );
    assert_eq!(
        op_count::<MirBitAndOp>(&ctx, lp.region),
        2,
        "in bounds, counter no-wrap, and offset no-wrap"
    );
}

/// `while i + 1 < acc + 8` has a counter but no counted exit test. Full unroll
/// skips it and leaves the loop as it was.
#[test]
fn counter_with_an_unrecognized_exit_test_is_skipped() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, 1, CmpPred::Lt, OffsetBound::AccPlus(8));

    unroll_offset_loop(&mut ctx, &lp, 0);

    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

/// Evaluate the generated guard with the MIR integer type's wrapping semantics.
/// Inputs substitute the new header's counter and the original runtime bound.
fn evaluate_integer(ctx: &Context, value: Value, inputs: &HashMap<Value, i128>) -> i128 {
    let raw = if let Some(&input) = inputs.get(&value) {
        input
    } else if let Some(constant) = constant_i128(ctx, value) {
        constant
    } else {
        let op = value.defining_op().expect("guard value has a definition");
        let lhs = evaluate_integer(ctx, op.deref(ctx).get_operand(0), inputs);
        if Operation::get_op::<MirNotOp>(op, ctx).is_some() {
            !lhs
        } else {
            let rhs = evaluate_integer(ctx, op.deref(ctx).get_operand(1), inputs);
            if Operation::get_op::<MirAddOp>(op, ctx).is_some() {
                lhs + rhs
            } else if Operation::get_op::<MirSubOp>(op, ctx).is_some() {
                lhs - rhs
            } else if Operation::get_op::<MirBitAndOp>(op, ctx).is_some() {
                lhs & rhs
            } else if Operation::get_op::<MirLtOp>(op, ctx).is_some() {
                i128::from(lhs < rhs)
            } else if Operation::get_op::<MirLeOp>(op, ctx).is_some() {
                i128::from(lhs <= rhs)
            } else if Operation::get_op::<MirGtOp>(op, ctx).is_some() {
                i128::from(lhs > rhs)
            } else if Operation::get_op::<MirGeOp>(op, ctx).is_some() {
                i128::from(lhs >= rhs)
            } else {
                panic!("unexpected operation in the integer guard")
            }
        }
    };
    let ty = TypedHandle::<IntegerType>::from_handle(value.get_type(ctx), ctx).unwrap();
    let width = ty.deref(ctx).width();
    assert!(
        (1..=32).contains(&width),
        "this evaluator covers narrow integers"
    );
    let modulus = 1i128 << width;
    let bits = raw.rem_euclid(modulus);
    if ty.deref(ctx).signedness() == Signedness::Signed && bits >= modulus / 2 {
        bits - modulus
    } else {
        bits
    }
}

#[test]
fn partial_offset_guard_admits_exactly_the_non_wrapping_in_bounds_groups() {
    for signedness in [Signedness::Signed, Signedness::Unsigned] {
        let (min, max, bounds, offsets) = if signedness == Signedness::Signed {
            (
                -128,
                127,
                vec![-128, -127, -1, 0, 1, 126, 127],
                vec![-127, -2, -1, 0, 1, 2, 127],
            )
        } else {
            (
                0,
                255,
                vec![0, 1, 2, 127, 254, 255],
                vec![-255, -2, -1, 0, 1, 2, 255],
            )
        };
        for offset in offsets {
            for (step, factor) in [(1, 4), (3, 2), (16, 4)] {
                for pred in [CmpPred::Lt, CmpPred::Le] {
                    let mut ctx = mir_ctx();
                    let ty = IntegerType::get(&ctx, 8, signedness);
                    let lp = offset_counted_loop(
                        &mut ctx,
                        ty,
                        min,
                        step,
                        offset,
                        pred,
                        OffsetBound::Param,
                    );
                    unroll_offset_loop(&mut ctx, &lp, factor);
                    assert_eq!(loop_count(&ctx, lp.region), 2);
                    let entry = lp.preheader.deref(&ctx).get_terminator(&ctx).unwrap();
                    let main = entry.deref(&ctx).get_successor(0);
                    let branch = main.deref(&ctx).get_terminator(&ctx).unwrap();
                    assert!(Operation::get_op::<MirCondBranchOp>(branch, &ctx).is_some());
                    let guard = branch.deref(&ctx).get_operand(0);
                    let counter = main.deref(&ctx).get_argument(1);
                    let bound_arg = lp.preheader.deref(&ctx).get_argument(0);

                    for first in min..=max {
                        for &bound in &bounds {
                            let expected = (0..factor).all(|j| {
                                let iv = first + i128::from(j) * step;
                                let tested = iv + offset;
                                (min..=max).contains(&iv)
                                    && (min..=max).contains(&tested)
                                    && match pred {
                                        CmpPred::Lt => tested < bound,
                                        CmpPred::Le => tested <= bound,
                                        _ => unreachable!(),
                                    }
                            });
                            let inputs = HashMap::from([(counter, first), (bound_arg, bound)]);
                            let accepted = evaluate_integer(&ctx, guard, &inputs) != 0;
                            assert_eq!(
                                accepted, expected,
                                "{signedness:?}: first={first}, step={step}, offset={offset}, bound={bound}, factor={factor}, pred={pred:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn full_offset_unroll_preserves_signed_descending_and_high_bit_unsigned_results() {
    let cases = [
        (Signedness::Signed, 4, -1, 1, CmpPred::Gt, 0, 10),
        (Signedness::Signed, -3, 1, -2, CmpPred::Le, 2, 4),
        (Signedness::Unsigned, 0, 1, 128, CmpPred::Lt, 132, 6),
    ];
    for (signedness, start, step, offset, pred, bound, expected) in cases {
        let mut ctx = mir_ctx();
        let ty = IntegerType::get(&ctx, 8, signedness);
        let lp = offset_counted_loop(
            &mut ctx,
            ty,
            start,
            step,
            offset,
            pred,
            OffsetBound::Const(bound),
        );
        unroll_offset_loop(&mut ctx, &lp, 0);
        assert_eq!(loop_count(&ctx, lp.region), 0);
        assert_eq!(sole_return_constant(&ctx, lp.region), Some(expected));
    }
}

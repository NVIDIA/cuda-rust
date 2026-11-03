/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for `#[nounroll]` latch placement. Each test plants a
//! `mir.nounroll_hint` where the frontend would, runs the unroll pass when the
//! case involves one, then runs `resolve_nounroll_hints` and checks which loop
//! latches came back.
//!
//! The property that matters for LLVM: every marked loop ends up with exactly
//! one latch, that latch is the returned branch, and no unmarked loop's latch
//! is returned.

mod common;

use common::{block, goto, mir_ctx, multi_latch_counted_loop, nested_counted_loop};
use dialect_mir::ops::{MirGotoOp, MirNoUnrollHintOp, MirUnrollHintOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use rustc_hash::FxHashSet;

use mir_transforms::analyses::loop_info::LoopInfo;
use mir_transforms::nounroll::resolve_nounroll_hints;
use mir_transforms::unroll::unroll_annotated_loops;

fn loop_info(ctx: &Context, region: Ptr<Region>) -> LoopInfo {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt)
}

fn plant_nounroll(ctx: &mut Context, block: Ptr<BasicBlock>) {
    MirNoUnrollHintOp::new(ctx)
        .get_operation()
        .insert_at_front(block, ctx);
}

fn nounroll_hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .filter(|&op| Operation::get_op::<MirNoUnrollHintOp>(op, ctx).is_some())
        .count()
}

/// Run the resolver and check the shared contract: hints are gone, the module
/// verifies, and every returned branch is a `mir.goto` to a loop header that is
/// that loop's only back-edge. Returns the headers of the marked loops.
fn resolve(
    ctx: &mut Context,
    module: Ptr<Operation>,
    region: Ptr<Region>,
) -> FxHashSet<Ptr<BasicBlock>> {
    let latches = resolve_nounroll_hints(module, ctx).expect("resolve nounroll hints");
    pliron::operation::verify_operation(module, ctx).expect("valid IR after latch placement");
    assert_eq!(nounroll_hint_count(ctx, region), 0, "every hint is erased");

    let info = loop_info(ctx, region);
    let mut headers = FxHashSet::default();
    for &branch in &latches {
        assert!(
            Operation::get_op::<MirGotoOp>(branch, ctx).is_some(),
            "a marked latch ends in mir.goto"
        );
        let latch = branch
            .deref(ctx)
            .get_parent_block()
            .expect("branch in a block");
        let successors: Vec<_> = branch.deref(ctx).successors().collect();
        let [header] = successors[..] else {
            panic!("a marked latch has one successor, got {}", successors.len());
        };
        let lp = info
            .loops()
            .iter()
            .find(|lp| lp.header == header)
            .expect("a marked latch jumps to a loop header");
        assert_eq!(
            lp.latches,
            [latch],
            "the marked branch is the loop's only latch"
        );
        assert!(headers.insert(header), "one marked latch per loop");
    }
    headers
}

/// A hint marks the loop whose body holds it, not the loop around it, and the
/// enclosing loop's latch is not returned.
#[test]
fn nounroll_marks_only_the_annotated_loop_of_a_nest() {
    for annotate_inner in [true, false] {
        let mut ctx = mir_ctx();
        let lp = nested_counted_loop(&mut ctx, 3, 2);
        let (body, expected) = if annotate_inner {
            (lp.inner_body, lp.inner_header)
        } else {
            (lp.outer_body, lp.outer_header)
        };
        plant_nounroll(&mut ctx, body);

        let headers = resolve(&mut ctx, lp.module, lp.region);

        assert_eq!(headers.into_iter().collect::<Vec<_>>(), [expected]);
        assert_eq!(
            loop_info(&ctx, lp.region).loops().len(),
            2,
            "both loops remain"
        );
    }
}

/// Two `continue` paths are two back-edges. LLVM ignores loop metadata unless
/// every latch carries the same node, so the resolver routes both through one
/// new latch block and returns its branch.
#[test]
fn nounroll_merges_several_back_edges_into_one_latch() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 1);
    assert_eq!(loop_info(&ctx, lp.region).loops()[0].latches.len(), 2);
    plant_nounroll(&mut ctx, lp.normal_latch);

    let headers = resolve(&mut ctx, lp.module, lp.region);

    assert_eq!(headers.into_iter().collect::<Vec<_>>(), [lp.header]);
}

/// A full `#[unroll]` of an outer loop copies the inner `#[nounroll]` loop,
/// hint included, into each body copy. Every copy must stay rolled.
#[test]
fn nounroll_inside_a_fully_unrolled_loop_marks_every_copy() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);
    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.outer_body, &ctx);
    plant_nounroll(&mut ctx, lp.inner_body);

    unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default())
        .expect("unroll the outer loop");
    assert_eq!(nounroll_hint_count(&ctx, lp.region), 3, "one hint per copy");

    let headers = resolve(&mut ctx, lp.module, lp.region);

    let info = loop_info(&ctx, lp.region);
    assert_eq!(info.loops().len(), 3, "three inner copies, no outer loop");
    assert!(
        info.loops()
            .iter()
            .all(|inner| headers.contains(&inner.header))
    );
}

/// A partial `#[unroll(4)]` of an outer loop leaves a main loop and a remainder
/// loop. The main loop holds four inner copies and the remainder holds one. All
/// five stay rolled, and the main and remainder loops themselves are unmarked.
#[test]
fn nounroll_inside_a_partially_unrolled_loop_marks_main_and_remainder_copies() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 8, 2);
    MirUnrollHintOp::new(&mut ctx, 4)
        .get_operation()
        .insert_at_front(lp.outer_body, &ctx);
    plant_nounroll(&mut ctx, lp.inner_body);

    unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default())
        .expect("partially unroll the outer loop");

    let headers = resolve(&mut ctx, lp.module, lp.region);

    let info = loop_info(&ctx, lp.region);
    let outer: FxHashSet<_> = info
        .top_level()
        .iter()
        .map(|&id| info.loops()[id].header)
        .collect();
    assert_eq!(outer.len(), 2, "main loop + remainder loop");
    let inner: FxHashSet<_> = info
        .loops()
        .iter()
        .map(|lp| lp.header)
        .filter(|header| !outer.contains(header))
        .collect();
    assert_eq!(inner.len(), 5);
    assert_eq!(headers, inner, "exactly the inner copies are marked");
}

/// A hint outside every loop has nothing to mark. It is dropped with a warning
/// instead of failing the compile.
#[test]
fn nounroll_outside_a_loop_marks_nothing() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);
    plant_nounroll(&mut ctx, lp.preheader);

    let headers = resolve(&mut ctx, lp.module, lp.region);

    assert!(headers.is_empty());
}

/// Dead code that branches into a loop puts that loop's blocks in its reach
/// backwards, but a hint there has no loop to keep rolled. WGMMA counted-loop
/// fusion leaves a fused K loop's blocks unreachable like this.
#[test]
fn nounroll_in_unreachable_code_marks_nothing() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);
    let dead = block(&mut ctx, lp.region, vec![]);
    goto(&mut ctx, dead, lp.outer_latch, vec![]);
    plant_nounroll(&mut ctx, dead);

    let headers = resolve(&mut ctx, lp.module, lp.region);

    assert!(headers.is_empty());
}

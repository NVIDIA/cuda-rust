/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests automatic unrolling and silent rejection of ineligible loops.

mod common;

use common::{block, cond_br, func, goto, i1, iconst, mir_ctx, ret, u32t};
use dialect_mir::ops::{
    MirAddOp, MirAllocaOp, MirArrayElementAddrOp, MirBitXorOp, MirLtOp, MirNoUnrollHintOp,
    MirNotOp, MirStoreOp, MirUnrollHintOp,
};
use dialect_mir::types::{MirArrayType, MirPtrType};
use mir_transforms::analyses::loop_info::LoopInfo;
use mir_transforms::unroll::{unroll_annotated_and_automatic_loops, unroll_annotated_loops};
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::r#type::TypeHandle;

macro_rules! append {
    ($ctx:expr, $block:expr, $op:ty, $results:expr, $operands:expr $(,)?) => {{
        let op = Operation::new(
            $ctx,
            <$op>::get_concrete_op_info(),
            $results,
            $operands,
            vec![],
            0,
        );
        op.insert_at_back($block, $ctx);
        op.deref($ctx).get_result(0)
    }};
}

/// How the store index is computed from the counter `i` and the function's
/// runtime argument `param`.
#[derive(Clone, Copy)]
enum Index {
    /// `arr[i]`: folds to a constant in every unrolled copy.
    Counter,
    /// `arr[i ^ param]`: computed in the loop, but never a constant.
    CounterXorParam,
}

#[derive(Clone, Copy, PartialEq)]
enum Marker {
    None,
    NoUnroll,
}

struct Shape {
    /// The trip count, or `None` for the runtime argument `param`.
    trip: Option<i64>,
    /// Number of `u32` elements in the local array.
    len: u64,
    index: Index,
    marker: Marker,
}

fn build(ctx: &mut Context, shape: &Shape) -> (Ptr<Operation>, Ptr<Region>) {
    let u32 = u32t(ctx);
    let i1 = i1(ctx);
    let (module, region) = func(ctx, vec![u32.into()], vec![]);
    let preheader = block(ctx, region, vec![u32.into()]);
    let header = block(ctx, region, vec![u32.into()]);
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    let param = preheader.deref(ctx).get_argument(0);
    let array: TypeHandle = MirArrayType::get(ctx, u32.into(), shape.len).into();
    let array_ptr: TypeHandle = MirPtrType::get_generic(ctx, array, true).into();
    let elem_ptr: TypeHandle = MirPtrType::get_generic(ctx, u32.into(), true).into();
    let arr = append!(ctx, preheader, MirAllocaOp, vec![array_ptr], vec![],);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![i0]);

    let i = header.deref(ctx).get_argument(0);
    let n = match shape.trip {
        Some(trip) => iconst(ctx, header, u32, trip),
        None => param,
    };
    let more = append!(ctx, header, MirLtOp, vec![i1.into()], vec![i, n],);
    let done = append!(ctx, header, MirNotOp, vec![i1.into()], vec![more],);
    cond_br(ctx, header, done, exit, latch);

    if shape.marker == Marker::NoUnroll {
        MirNoUnrollHintOp::new(ctx)
            .get_operation()
            .insert_at_back(latch, ctx);
    }
    let idx = match shape.index {
        Index::Counter => i,
        Index::CounterXorParam => {
            append!(ctx, latch, MirBitXorOp, vec![u32.into()], vec![i, param],)
        }
    };
    let p = append!(
        ctx,
        latch,
        MirArrayElementAddrOp,
        vec![elem_ptr],
        vec![arr, idx],
    );
    let store = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![p, i],
        vec![],
        0,
    );
    store.insert_at_back(latch, ctx);
    let one = iconst(ctx, latch, u32, 1);
    let next = append!(ctx, latch, MirAddOp, vec![u32.into()], vec![i, one],);
    goto(ctx, latch, header, vec![next]);
    ret(ctx, exit);
    (module, region)
}

fn loop_count(ctx: &Context, region: Ptr<Region>) -> usize {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt).loops().len()
}

fn hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .filter(|&op| Operation::get_op::<MirUnrollHintOp>(op, ctx).is_some())
        .count()
}

fn run(shape: Shape) -> (usize, usize) {
    let mut ctx = mir_ctx();
    let (module, region) = build(&mut ctx, &shape);
    assert_eq!(loop_count(&ctx, region), 1, "the test loop is built");
    let warnings =
        unroll_annotated_and_automatic_loops(module, &mut ctx, &mut AnalysisManager::default())
            .expect("the pass runs");
    assert_eq!(
        hint_count(&ctx, region),
        0,
        "every planted hint is consumed"
    );
    (loop_count(&ctx, region), warnings.len())
}

#[test]
fn a_counter_indexed_small_array_loop_is_unrolled() {
    let left = run(Shape {
        trip: Some(8),
        len: 8,
        index: Index::Counter,
        marker: Marker::None,
    });
    assert_eq!(left, (0, 0), "the loop is fully unrolled, without warnings");
}

#[test]
fn a_runtime_trip_count_stays_rolled_without_a_warning() {
    let left = run(Shape {
        trip: None,
        len: 8,
        index: Index::Counter,
        marker: Marker::None,
    });
    assert_eq!(left, (1, 0));
}

#[test]
fn an_index_that_never_folds_stays_rolled_without_a_warning() {
    let left = run(Shape {
        trip: Some(8),
        len: 8,
        index: Index::CounterXorParam,
        marker: Marker::None,
    });
    assert_eq!(left, (1, 0));
}

#[test]
fn an_array_over_the_byte_limit_stays_rolled() {
    let len = mir_transforms::auto_unroll::AUTO_UNROLL_MAX_ARRAY_BYTES / 4 + 1;
    let left = run(Shape {
        trip: Some(8),
        len,
        index: Index::Counter,
        marker: Marker::None,
    });
    assert_eq!(left, (1, 0));
}

#[test]
fn a_trip_count_over_the_copy_limit_stays_rolled() {
    let trip = mir_transforms::auto_unroll::AUTO_UNROLL_MAX_COPIES as i64 + 1;
    let left = run(Shape {
        trip: Some(trip),
        len: 128,
        index: Index::Counter,
        marker: Marker::None,
    });
    assert_eq!(left, (1, 0));
}

#[test]
fn nounroll_opts_a_loop_out() {
    let left = run(Shape {
        trip: Some(8),
        len: 8,
        index: Index::Counter,
        marker: Marker::NoUnroll,
    });
    assert_eq!(left.0, 1, "#[nounroll] keeps the loop rolled");
}

#[test]
fn the_annotation_only_entry_point_does_not_unroll_automatically() {
    let mut ctx = mir_ctx();
    let shape = Shape {
        trip: Some(8),
        len: 8,
        index: Index::Counter,
        marker: Marker::None,
    };
    let (module, region) = build(&mut ctx, &shape);
    let warnings = unroll_annotated_loops(module, &mut ctx, &mut AnalysisManager::default())
        .expect("the pass runs");
    assert_eq!((loop_count(&ctx, region), warnings.len()), (1, 0));
}

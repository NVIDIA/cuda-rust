/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the induction-variable analysis: given a counted loop, does it
//! classify each carried value (counter vs reduction), read off the counter's
//! `init`/`step`, recover the continue-predicate, and compute the trip count?

mod common;

use common::{
    CountedLoop, OffsetBound, counted_loop, counted_loop_from, i32t, i128t, mir_ctx,
    multi_latch_counted_loop, offset_counted_loop, offset_counted_loop_iv_on_right, u32t,
};
use mir_transforms::analyses::induction::{ArgKind, CmpPred, analyze};
use mir_transforms::analyses::loop_info::LoopInfo;
use pliron::graph::dominance::DomInfo;

/// Run the analysis on `while i < n { acc += i; i += 1 }` and return its facts.
fn recurrences_for(n: i64) -> mir_transforms::analyses::induction::LoopRecurrences {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, n);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    analyze(&ctx, &info, id, ph)
}

/// Run the analysis on an already built loop.
fn recurrences_of(
    ctx: &pliron::context::Context,
    lp: &CountedLoop,
) -> mir_transforms::analyses::induction::LoopRecurrences {
    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(ctx, lp.region);
        LoopInfo::compute(ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(ctx, lp.region, id).unwrap();
    analyze(ctx, &info, id, ph)
}

#[test]
fn analyzes_counted_loop_recurrence() {
    // while i < 8 { acc += i; i += 1 }  =>  header args are (acc, i).
    let rec = recurrences_for(8);

    // `i` (header arg 1) is the counter: starts at 0, steps by 1.
    assert_eq!(rec.primary_iv, Some(1));
    match rec.args[1] {
        ArgKind::BasicIv { init, step } => {
            assert_eq!(init, 0);
            assert_eq!(step, 1);
        }
        ref other => panic!("i should be a BasicIv, got {other:?}"),
    }

    // The loop continues while `i < 8`, so 8 iterations, bound 8.
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(8));
    assert_eq!(rec.trip_count, Some(8));

    // `acc` (header arg 0) is carried but updated by `acc + i`, so it is a
    // reduction, not the counter.
    assert!(
        matches!(rec.args[0], ArgKind::Reduction),
        "acc should be a reduction, got {:?}",
        rec.args[0]
    );
}

/// The trip count tracks the bound: `while i < n` runs `n` times (init 0, step 1).
#[test]
fn trip_count_tracks_the_bound() {
    for n in [1, 4, 16, 100] {
        let rec = recurrences_for(n);
        assert_eq!(rec.bound, Some(n as i128), "bound for n={n}");
        assert_eq!(rec.trip_count, Some(n as u64), "trip count for n={n}");
    }
}

/// Unsigned constants must be zero-extended. The high bit of both values is set,
/// but this is still a four-trip loop.
#[test]
fn high_bit_unsigned_constants_keep_their_positive_values() {
    let mut ctx = mir_ctx();
    let start = 2_147_483_646i64;
    let bound = 2_147_483_650i64;
    let lp = counted_loop_from(&mut ctx, start, bound);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert!(matches!(
        rec.args[1],
        ArgKind::BasicIv {
            init: 2_147_483_646,
            step: 1
        }
    ));
    assert_eq!(rec.bound, Some(2_147_483_650));
    assert_eq!(rec.trip_count, Some(4));
}

#[test]
fn analyzes_matching_recurrence_on_every_latch() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 1);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    assert_eq!(info.loops()[id].latches.len(), 2);
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert_eq!(rec.primary_iv, Some(1));
    match rec.args[1] {
        ArgKind::BasicIv { init, step } => {
            assert_eq!(init, 0);
            assert_eq!(step, 1);
        }
        ref other => panic!("i should be a BasicIv on every latch, got {other:?}"),
    }
    assert!(
        matches!(rec.args[0], ArgKind::Reduction),
        "acc differs between continue and normal paths, so it is a reduction"
    );
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(4));
    assert_eq!(rec.trip_count, Some(4));
}

#[test]
fn rejects_inconsistent_iv_steps_across_latches() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 2);

    let mut dom = DomInfo::default();
    let info = {
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    let id = info.innermost_loop(lp.header).unwrap();
    assert_eq!(info.loops()[id].latches.len(), 2);
    let ph = info.preheader(&ctx, lp.region, id).unwrap();
    let rec = analyze(&ctx, &info, id, ph);

    assert_eq!(rec.primary_iv, None, "there is no single affine counter");
    assert!(
        !matches!(rec.args[1], ArgKind::BasicIv { .. }),
        "different latch steps must not be guessed from an arbitrary latch"
    );
    assert_eq!(rec.trip_count, None);
}

/// `while i + 1 <= 4` tests the counter plus one. The analysis records the
/// offset and normalizes the limit to `i <= 3`, so the loop runs four times.
#[test]
fn counter_plus_constant_exit_test_is_recognized() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, 1, CmpPred::Le, OffsetBound::Const(4));
    let rec = recurrences_of(&ctx, &lp);

    assert_eq!(rec.primary_iv, Some(1));
    assert_eq!(rec.iv_offset, 1);
    assert_eq!(rec.continue_pred, Some(CmpPred::Le));
    assert_eq!(rec.bound, Some(3));
    assert_eq!(rec.trip_count, Some(4));
}

/// `while i + 2 < 8` with `i += 2` runs for `i = 0, 2, 4`: three trips.
#[test]
fn counter_offset_and_step_combine_in_the_trip_count() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 2, 2, CmpPred::Lt, OffsetBound::Const(8));
    let rec = recurrences_of(&ctx, &lp);

    assert_eq!(rec.primary_iv, Some(1));
    assert_eq!(rec.iv_offset, 2);
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(6));
    assert_eq!(rec.trip_count, Some(3));
}

/// `while 8 >= i + 2` is the same test as `while i + 2 <= 8`: the predicate is
/// swapped and the counter found on the right.
#[test]
fn counter_offset_on_the_right_side_is_swapped() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp =
        offset_counted_loop_iv_on_right(&mut ctx, u32, 0, 1, 2, CmpPred::Ge, OffsetBound::Const(8));
    let rec = recurrences_of(&ctx, &lp);

    assert_eq!(rec.primary_iv, Some(1));
    assert_eq!(rec.iv_offset, 2);
    assert_eq!(rec.continue_pred, Some(CmpPred::Le));
    assert_eq!(rec.bound, Some(6));
    assert_eq!(rec.trip_count, Some(7));
}

/// `while i - 1 < 4` has offset -1, so the normalized limit is `i < 5` and a
/// signed counter from 0 runs five times.
#[test]
fn counter_minus_constant_exit_test_raises_the_limit() {
    let mut ctx = mir_ctx();
    let i32 = i32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, i32, 0, 1, -1, CmpPred::Lt, OffsetBound::Const(4));
    let rec = recurrences_of(&ctx, &lp);

    assert_eq!(rec.primary_iv, Some(1));
    assert_eq!(rec.iv_offset, -1);
    assert_eq!(rec.continue_pred, Some(CmpPred::Lt));
    assert_eq!(rec.bound, Some(5));
    assert_eq!(rec.trip_count, Some(5));
}

/// `while i - 1 <= i128::MAX` would need the limit `i128::MAX + 1`, which the
/// analysis cannot represent. It reports no exit test rather than a wrong one.
#[test]
fn counter_offset_whose_limit_overflows_is_not_recognized() {
    let mut ctx = mir_ctx();
    let i128 = i128t(&mut ctx);
    let lp = offset_counted_loop(
        &mut ctx,
        i128,
        0,
        1,
        -1,
        CmpPred::Le,
        OffsetBound::Const(i128::MAX),
    );
    let rec = recurrences_of(&ctx, &lp);

    assert!(matches!(rec.args[1], ArgKind::BasicIv { init: 0, step: 1 }));
    assert_eq!(rec.primary_iv, None);
    assert_eq!(rec.bound, None);
    assert_eq!(rec.continue_pred, None);
    assert_eq!(rec.trip_count, None);
}

/// Unsigned constants in the offset and the limit must be zero-extended.
/// `while i + 0x8000_0000 < 0x8000_0004` is a four-trip loop.
#[test]
fn high_bit_unsigned_offset_keeps_its_positive_value() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(
        &mut ctx,
        u32,
        0,
        1,
        2_147_483_648,
        CmpPred::Lt,
        OffsetBound::Const(2_147_483_652),
    );
    let rec = recurrences_of(&ctx, &lp);

    assert_eq!(rec.primary_iv, Some(1));
    assert_eq!(rec.iv_offset, 2_147_483_648);
    assert_eq!(rec.bound, Some(4));
    assert_eq!(rec.trip_count, Some(4));
}

/// `while i + 1 < acc + 8` compares two values that both change in the loop.
/// Neither side is a limit, so there is no counted exit test, though `i` is
/// still a counter.
#[test]
fn offsets_on_both_sides_are_not_a_counted_exit_test() {
    let mut ctx = mir_ctx();
    let u32 = u32t(&mut ctx);
    let lp = offset_counted_loop(&mut ctx, u32, 0, 1, 1, CmpPred::Lt, OffsetBound::AccPlus(8));
    let rec = recurrences_of(&ctx, &lp);

    assert!(matches!(rec.args[1], ArgKind::BasicIv { init: 0, step: 1 }));
    assert_eq!(rec.primary_iv, None);
    assert_eq!(rec.bound_value, None);
    assert_eq!(rec.continue_pred, None);
    assert_eq!(rec.trip_count, None);
}

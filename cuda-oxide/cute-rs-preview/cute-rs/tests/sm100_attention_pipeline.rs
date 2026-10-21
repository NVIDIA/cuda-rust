/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks two-CTA barrier-ring addressing, persistent cursors and policy rejection."]
use core::mem::MaybeUninit;
use cuda_device::barrier::Barrier;
use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, TwoCtaRing,
};

/// Full and empty arrays must be disjoint, and ticket indices must wrap within each array.
#[test]
fn barrier_views_select_disjoint_arrays_and_wrap_slots() {
    // Address calculation is CPU-safe; this test never initializes or uses GPU barriers.
    let mut storage = [const { MaybeUninit::<Barrier>::uninit() }; 8];
    let base = storage.as_mut_ptr().cast::<Barrier>();
    let ring = unsafe { TwoCtaRing::<3, AsyncCompletion>::from_raw(base, 1) };
    for (ticket, full_offset, empty_offset) in
        [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 1, 4), (7, 2, 5)]
    {
        unsafe {
            assert_eq!(ring.full_at(ticket), base.add(full_offset));
            assert_eq!(ring.empty_at(ticket), base.add(empty_offset));
        }
    }
}

/// Compile checks must reject signals that do not match the ring edge’s memory-ordering policy.
#[test]
fn completion_policies_reject_incompatible_signals() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/compile_pass/two_cta_ring_valid_signal.rs");
    tests.pass("tests/compile_pass/two_cta_ring_non_tc_receivers.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_non_tc_wrong_signal.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_wrong_signal.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_zero_stages.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_policy_sealed.rs");
}

/// Cursors must retain slot, parity and barrier addresses across persistent work boundaries.
#[test]
fn cursors_preserve_phase_and_addresses_across_persistent_work() {
    let mut storage = [const { MaybeUninit::<Barrier>::uninit() }; 32];
    let base = storage.as_mut_ptr().cast::<Barrier>();
    let ring = unsafe { TwoCtaRing::<15, AsyncCompletion>::from_raw(base, 1) };
    let mut producer = RingCursor::<15, Producer>::new();
    let mut consumer = RingCursor::<15, Consumer>::new();
    let mut ticket = 0;
    for work_stages in [17, 1, 256, 4097] {
        for _ in 0..work_stages {
            assert_eq!(producer.index(), ticket % 15);
            assert_eq!(consumer.index(), ticket % 15);
            assert_eq!(producer.phase(), ((ticket / 15) & 1) ^ 1);
            assert_eq!(consumer.phase(), (ticket / 15) & 1);
            unsafe {
                assert_eq!(ring.full_at_cursor(producer), ring.full_at(ticket));
                assert_eq!(ring.full_at_cursor(consumer), ring.full_at(ticket));
                assert_eq!(ring.empty_at_cursor(producer), ring.empty_at(ticket));
                assert_eq!(ring.empty_at_cursor(consumer), ring.empty_at(ticket));
            }
            producer.advance();
            consumer.advance();
            ticket += 1;
        }
    }
    assert_eq!((producer.index(), producer.phase()), (6, 0));
    assert_eq!((consumer.index(), consumer.phase()), (6, 1));
}

/// Producer and consumer parities must flip after every one-stage handoff and remain opposite.
#[test]
fn one_stage_cursors_flip_opposite_phases_on_every_advance() {
    let mut producer = RingCursor::<1, Producer>::default();
    let mut consumer = RingCursor::<1, Consumer>::default();
    for (producer_phase, consumer_phase) in [(1, 0), (0, 1), (1, 0), (0, 1)] {
        assert_eq!((producer.index(), producer.phase()), (0, producer_phase));
        assert_eq!((consumer.index(), consumer.phase()), (0, consumer_phase));
        producer.advance();
        consumer.advance();
    }
}

/// Compile checks must reject cursor/ring stage mismatches, role swaps and empty rings.
#[test]
fn cursors_reject_wrong_stages_roles_and_zero_stages() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/compile_pass/two_cta_ring_valid_cursor.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_cursor_wrong_stages.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_cursor_wrong_role.rs");
    tests.compile_fail("tests/compile_fail/two_cta_ring_cursor_zero_stages.rs");
}

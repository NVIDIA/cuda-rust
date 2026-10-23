/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Stable first-index selection and sticky invalid-input reduction contracts.

use cute_rs::reduction::ArgMax;

#[test]
fn first_index_ties_are_independent_of_partition_order() {
    let a = ArgMax::empty().observe(3.0, 70).observe(3.0, 2);
    let b = ArgMax::empty().observe(-8.0, 0).observe(3.0, 33);
    for selected in [a.merge(b), b.merge(a)] {
        assert_eq!(selected.index(), 2);
        assert_eq!(selected.value(), 3.0);
        assert!(selected.is_valid());
    }
}

#[test]
fn signed_zero_ties_choose_the_first_index() {
    let selected = ArgMax::empty().observe(-0.0, 8).observe(0.0, 3);
    assert_eq!(selected.index(), 3);
    assert_eq!(selected.value().to_bits(), 0);
}

#[test]
fn nonfinite_input_is_sticky_across_observations_and_merges() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = ArgMax::empty().observe(bad, 0).observe(5.0, 1);
        let valid = ArgMax::empty().observe(10.0, 2);
        assert!(!invalid.is_valid());
        assert!(!invalid.merge(valid).is_valid());
        assert!(!valid.merge(invalid).is_valid());
    }
}

#[test]
fn empty_is_an_identity_for_finite_partitions() {
    let value = ArgMax::empty().observe(-5.0, 9);
    assert_eq!(ArgMax::empty().merge(value).index(), 9);
    assert_eq!(value.merge(ArgMax::empty()).index(), 9);
    let _: unsafe fn(ArgMax, u32) -> ArgMax = ArgMax::reduce_warp;
}

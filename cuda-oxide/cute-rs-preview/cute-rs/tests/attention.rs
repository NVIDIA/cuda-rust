/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attention and cache extents are checked without allocating device memory.
#![cfg(feature = "host")]

use cute_rs::attention::{AttentionError, AttentionShape, AttentionWindow, CacheShape};

#[test]
fn gqa_dimensions_preserve_independent_query_and_kv_widths() {
    let shape = AttentionShape::<16, 8, 128>::new().unwrap();
    assert_eq!(
        (shape.query_heads(), shape.kv_heads(), shape.head_dim()),
        (16, 8, 128)
    );
    assert_eq!(
        (shape.query_width(), shape.kv_width(), shape.groups()),
        (2048, 1024, 2)
    );
    assert_eq!(shape.query_len(3), Ok(6144));
    assert_eq!(shape.kv_len(3), Ok(3072));
    let small = AttentionShape::<6, 2, 5>::new().unwrap();
    assert_eq!(
        (small.query_width(), small.kv_width(), small.groups()),
        (30, 10, 3)
    );
}

#[test]
fn attention_rejects_zero_dimensions_and_incomplete_query_groups() {
    assert_eq!(
        AttentionShape::<0, 1, 8>::new(),
        Err(AttentionError::ZeroDimension)
    );
    assert_eq!(
        AttentionShape::<1, 0, 8>::new(),
        Err(AttentionError::ZeroDimension)
    );
    assert_eq!(
        AttentionShape::<1, 1, 0>::new(),
        Err(AttentionError::ZeroDimension)
    );
    assert_eq!(
        AttentionShape::<3, 2, 8>::new(),
        Err(AttentionError::InvalidHeadGrouping)
    );
    assert_eq!(
        AttentionShape::<1, 2, 8>::new(),
        Err(AttentionError::InvalidHeadGrouping)
    );
}

#[test]
fn attention_checks_width_multiplication_and_bf16_addressability() {
    assert_eq!(
        AttentionShape::<{ usize::MAX }, 1, 2>::new(),
        Err(AttentionError::ExtentOverflow)
    );
    assert_eq!(
        AttentionShape::<1, 1, { isize::MAX as usize / 2 + 1 }>::new(),
        Err(AttentionError::ExtentOverflow)
    );
    let widest = AttentionShape::<1, 1, { isize::MAX as usize / 2 }>::new().unwrap();
    assert_eq!(widest.query_len(1), Ok(isize::MAX as usize / 2));
    assert_eq!(widest.query_len(2), Err(AttentionError::ExtentOverflow));
}

#[test]
fn token_lengths_accept_empty_inputs_and_check_both_tensor_extents() {
    let shape = AttentionShape::<2, 1, 1>::new().unwrap();
    assert_eq!(shape.query_len(0), Ok(0));
    assert_eq!(shape.kv_len(0), Ok(0));
    assert_eq!(shape.query_len(65), Ok(130));
    assert_eq!(shape.kv_len(65), Ok(65));
    assert_eq!(
        shape.query_len(usize::MAX),
        Err(AttentionError::ExtentOverflow)
    );
    assert_eq!(
        shape.kv_len(isize::MAX as usize / 2),
        Ok(isize::MAX as usize / 2)
    );
    assert_eq!(
        shape.kv_len(isize::MAX as usize / 2 + 1),
        Err(AttentionError::ExtentOverflow)
    );
}

#[test]
fn cache_ranges_count_tokens_and_accept_non_tile_aligned_appends() {
    let shape = CacheShape::<3, 5>::new(65).unwrap();
    assert_eq!(
        (shape.capacity(), shape.width(), shape.allocation_len()),
        (65, 15, 975)
    );
    assert_eq!(shape.append_range(0, 17), Ok(0..17));
    assert_eq!(shape.append_range(17, 48), Ok(17..65));
    assert_eq!(shape.append_range(65, 0), Ok(65..65));
    // A range is metadata; requesting it does not mutate a cache cursor.
    assert_eq!(shape.append_range(0, 17), Ok(0..17));
}

#[test]
fn zero_capacity_keeps_head_dimensions_valid_and_permits_empty_append() {
    let empty = CacheShape::<8, 128>::new(0).unwrap();
    assert_eq!(
        (empty.capacity(), empty.width(), empty.allocation_len()),
        (0, 1024, 0)
    );
    assert_eq!(empty.append_range(0, 0), Ok(0..0));
    assert_eq!(
        empty.append_range(0, 1),
        Err(AttentionError::CapacityExceeded)
    );
    assert_eq!(
        CacheShape::<0, 128>::new(0),
        Err(AttentionError::ZeroDimension)
    );
    assert_eq!(
        CacheShape::<8, 0>::new(0),
        Err(AttentionError::ZeroDimension)
    );
    assert_eq!(
        CacheShape::<1, { isize::MAX as usize / 2 + 1 }>::new(0),
        Err(AttentionError::ExtentOverflow)
    );
}

#[test]
fn append_rejects_invalid_cursors_capacity_exhaustion_and_overflow() {
    let shape = CacheShape::<2, 3>::new(65).unwrap();
    assert_eq!(
        shape.append_range(66, 0),
        Err(AttentionError::InvalidCursor)
    );
    assert_eq!(
        shape.append_range(64, 2),
        Err(AttentionError::CapacityExceeded)
    );
    assert_eq!(
        shape.append_range(65, 1),
        Err(AttentionError::CapacityExceeded)
    );
    assert_eq!(
        shape.append_range(1, usize::MAX),
        Err(AttentionError::ExtentOverflow)
    );
}

#[test]
fn cache_allocation_checks_full_bf16_extent_before_allocation() {
    let max_elements = isize::MAX as usize / 2;
    let largest = CacheShape::<1, 1>::new(max_elements).unwrap();
    assert_eq!(largest.allocation_len(), max_elements);
    assert_eq!(
        largest.append_range(max_elements - 1, 1),
        Ok(max_elements - 1..max_elements)
    );
    assert_eq!(
        CacheShape::<1, 1>::new(max_elements + 1),
        Err(AttentionError::ExtentOverflow)
    );
    assert_eq!(
        CacheShape::<1, 2>::new(usize::MAX),
        Err(AttentionError::ExtentOverflow)
    );
    assert_eq!(
        CacheShape::<{ usize::MAX }, 2>::new(0),
        Err(AttentionError::ExtentOverflow)
    );
}

#[test]
fn causal_window_tracks_absolute_query_positions_in_a_cached_prefix() {
    let window = AttentionWindow::causal(3, 19, 16).unwrap();
    assert_eq!(window.query_tokens(), 3);
    assert_eq!(window.key_tokens(), 19);
    assert_eq!(window.causal_start(), Some(16));
    assert_eq!(window.visible_keys(0), Ok(17));
    assert_eq!(window.visible_keys(1), Ok(18));
    assert_eq!(window.visible_keys(2), Ok(19));

    // Extra key rows remain masked; a window need not end at the cache cursor.
    let with_future_keys = AttentionWindow::causal(3, 25, 16).unwrap();
    assert_eq!(with_future_keys.visible_keys(2), Ok(19));
    let decode = AttentionWindow::causal(1, 65, 64).unwrap();
    assert_eq!(decode.visible_keys(0), Ok(65));
}

#[test]
fn noncausal_window_exposes_all_keys_even_with_more_queries_than_keys() {
    let window = AttentionWindow::noncausal(5, 3).unwrap();
    assert_eq!(window.query_tokens(), 5);
    assert_eq!(window.key_tokens(), 3);
    assert_eq!(window.causal_start(), None);
    assert_eq!(window.visible_keys(0), Ok(3));
    assert_eq!(window.visible_keys(4), Ok(3));
}

#[test]
fn empty_query_windows_are_valid_but_have_no_accessible_rows() {
    for window in [
        AttentionWindow::causal(0, 0, 0).unwrap(),
        AttentionWindow::causal(0, 65, 65).unwrap(),
        AttentionWindow::noncausal(0, 0).unwrap(),
        AttentionWindow::noncausal(0, 65).unwrap(),
    ] {
        assert_eq!(window.query_tokens(), 0);
        assert_eq!(window.visible_keys(0), Err(AttentionError::InvalidQueryRow));
    }
    assert_eq!(
        AttentionWindow::causal(0, 65, 66),
        Err(AttentionError::InvalidWindow)
    );
}

#[test]
fn window_rejects_queries_without_keys_and_out_of_bounds_causal_ranges() {
    assert_eq!(
        AttentionWindow::noncausal(1, 0),
        Err(AttentionError::MissingKeys)
    );
    assert_eq!(
        AttentionWindow::causal(1, 0, 0),
        Err(AttentionError::MissingKeys)
    );
    assert_eq!(
        AttentionWindow::causal(3, 19, 17),
        Err(AttentionError::InvalidWindow)
    );
    assert_eq!(
        AttentionWindow::causal(1, usize::MAX, usize::MAX),
        Err(AttentionError::ExtentOverflow)
    );
    for window in [
        AttentionWindow::causal(3, 19, 16).unwrap(),
        AttentionWindow::noncausal(3, 19).unwrap(),
    ] {
        assert_eq!(window.visible_keys(3), Err(AttentionError::InvalidQueryRow));
        assert_eq!(
            window.visible_keys(usize::MAX),
            Err(AttentionError::InvalidQueryRow)
        );
    }
}

#[test]
fn window_checks_token_arithmetic_without_claiming_allocation_extents() {
    let boundary = AttentionWindow::causal(1, usize::MAX, usize::MAX - 1).unwrap();
    assert_eq!(boundary.visible_keys(0), Ok(usize::MAX));
    let empty = AttentionWindow::causal(0, usize::MAX, usize::MAX).unwrap();
    assert_eq!(empty.causal_start(), Some(usize::MAX));
    let noncausal = AttentionWindow::noncausal(1, usize::MAX).unwrap();
    assert_eq!(noncausal.visible_keys(0), Ok(usize::MAX));
    // Actual tensor extents remain a separate check at the operator boundary.
    assert_eq!(
        AttentionShape::<1, 1, 1>::new()
            .unwrap()
            .kv_len(noncausal.key_tokens()),
        Err(AttentionError::ExtentOverflow)
    );
}

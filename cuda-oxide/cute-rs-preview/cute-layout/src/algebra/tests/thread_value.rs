/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Contract tests for the thread/value algebra family.

use super::*;

/// Checks the documented six-thread, four-values-per-thread example and
/// then verifies that its 24 assignments cover the `4 x 6` tile exactly.
#[test]
fn make_layout_tv_matches_dsl_docstring() {
    let thr = l("(2,3):(3,1)");
    let val = l("(2,2):(2,1)");
    let (tiler, tv) = make_layout_tv(&thr, &val).unwrap();
    assert_eq!(tiler, vec![4, 6]);
    assert_eq!(tv.to_string(), "((3,2),(2,2)):((8,2),(4,1))");
    // No two `(thread, value)` pairs may own the same tile cell, and no
    // cell may be left unowned.
    let mut seen: Vec<i64> = (0..tv.size())
        .map(|i| tv.call(&IntTuple::Leaf(i)))
        .collect();
    seen.sort_unstable();
    let expect: Vec<i64> = (0..24).collect();
    assert_eq!(seen, expect);
}

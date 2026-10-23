/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Contract tests for the tiling algebra family.

use super::*;

/// Zipped division changes the coordinate structure, not the address. The
/// inside-tile and tile-number coordinates are recombined mode by mode and
/// compared with the original layout.
#[test]
fn zipped_divide_is_definitional() {
    let a = l("(8,6):(6,1)");
    let r = zipped_divide(&a, &[4, 3]).unwrap();
    for t0 in 0..4 {
        for t1 in 0..3 {
            for r0 in 0..2 {
                for r1 in 0..2 {
                    let coord = IntTuple::Tuple(vec![
                        IntTuple::Tuple(vec![IntTuple::Leaf(t0), IntTuple::Leaf(t1)]),
                        IntTuple::Tuple(vec![IntTuple::Leaf(r0), IntTuple::Leaf(r1)]),
                    ]);
                    let orig = IntTuple::Tuple(vec![
                        IntTuple::Leaf(t0 + 4 * r0),
                        IntTuple::Leaf(t1 + 3 * r1),
                    ]);
                    assert_eq!(
                        r.call(&coord),
                        a.call(&orig),
                        "at t=({t0},{t1}) r=({r0},{r1})"
                    );
                }
            }
        }
    }
}

/// Logical product must repeat the inner block at the tiler’s expected displacements.
#[test]
fn logical_product_repeats_the_block() {
    // Six copies of a four-cell block must reach all 24 offsets exactly
    // once.
    let r = logical_product(&l("4:1"), &l("6:1")).unwrap();
    let mut seen: Vec<i64> = (0..r.size()).map(|i| r.call(&IntTuple::Leaf(i))).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..24).collect::<Vec<_>>());
}

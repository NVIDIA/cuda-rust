/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Contract tests for the tiling algebra family.

use super::*;
use crate::sm100_smem_crd2idx;

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

/// `tile_to_shape` and `sm100_smem_layout` must match the canonical SM100 K-major and MN-major
/// swizzled shared-memory layouts and agree element-for-element with `sm100_smem_crd2idx`.
#[test]
fn sm100_smem_layout_and_crd2idx_match_tile_to_shape() {
    use alloc::string::ToString;
    let k_major = sm100_smem_layout(128, 128, false).unwrap();
    assert_eq!(
        k_major.outer().to_string(),
        "((8,16),(64,2)):((64,512),(1,8192))"
    );
    let mn_major = sm100_smem_layout(128, 128, true).unwrap();
    assert_eq!(
        mn_major.outer().to_string(),
        "((64,2),(8,16)):((1,8192),(64,512))"
    );

    for r in 0..64u32 {
        for c in 0..64u32 {
            let coord = IntTuple::Tuple(vec![IntTuple::Leaf(r.into()), IntTuple::Leaf(c.into())]);
            let expected = sm100_smem_layout(64, 64, false)
                .unwrap()
                .checked_call(&coord)
                .unwrap() as usize;
            assert_eq!(sm100_smem_crd2idx::<64, 64, false>(r, c), expected);
        }
    }
    for r in 0..128u32 {
        for c in 0..32u32 {
            let coord = IntTuple::Tuple(vec![IntTuple::Leaf(r.into()), IntTuple::Leaf(c.into())]);
            let expected = sm100_smem_layout(128, 32, true)
                .unwrap()
                .checked_call(&coord)
                .unwrap() as usize;
            assert_eq!(sm100_smem_crd2idx::<128, 32, true>(r, c), expected);
        }
    }
}

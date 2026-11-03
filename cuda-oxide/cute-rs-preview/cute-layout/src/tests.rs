/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Foundation layout, representation and rejection contracts.

use super::*;
use alloc::vec;

/// Parse a test layout literal and fail immediately if the fixture is malformed.
fn t(v: &[i64]) -> IntTuple {
    IntTuple::Tuple(v.iter().map(|x| IntTuple::Leaf(*x)).collect())
}

/// Tuple coordinates in a row-major layout must use row stride 32 and adjacent columns.
#[test]
fn row_major_8x32() {
    // Eight rows of 32 adjacent values: moving one row adds 32 and moving
    // one column adds 1.
    let l = Layout::new(t(&[8, 32]), t(&[32, 1]));
    assert_eq!(l.size(), 256);
    assert_eq!(l.call(&t(&[0, 0])), 0);
    assert_eq!(l.call(&t(&[1, 0])), 32);
    assert_eq!(l.call(&t(&[3, 7])), 3 * 32 + 7);
}

/// Flat integer coordinates must unpack with the leftmost shape mode changing fastest.
#[test]
fn integer_coord_is_colexicographic() {
    // The leftmost mode changes first, so flat coordinate 5 becomes
    // `(row 5, column 0)` and maps to `5 * 32`.
    let l = Layout::new(t(&[8, 32]), t(&[32, 1]));
    assert_eq!(l.call(&IntTuple::Leaf(5)), 160);
    // After all eight rows, flat coordinate 9 becomes `(1,1)`.
    assert_eq!(l.call(&IntTuple::Leaf(9)), 33);
}

/// Layout formatting and parsing must preserve nested shape/stride structure.
#[test]
fn display_and_parse_roundtrip() {
    use alloc::string::ToString;
    for text in ["(8,32):(32,1)", "4:1", "((2,4),32):((128,32),1)"] {
        let l: Layout = text.parse().unwrap();
        assert_eq!(l.to_string(), text);
    }
    assert!("(8,32):(32)".parse::<Layout>().is_err()); // not congruent
    assert!("(8,32)".parse::<Layout>().is_err()); // missing stride
}

/// Identity detection must inspect the complete flat-coordinate offset sequence.
#[test]
fn identity_map_detection() {
    assert!(Layout::contiguous(4).is_identity_map());
    // Parentheses around a single mode do not change its mapping.
    let l: Layout = "(4):(1)".parse().unwrap();
    assert!(l.is_identity_map());
    // This shape/stride order makes its colexicographic flat coordinates
    // walk through adjacent offsets.
    let rm: Layout = "(32,8):(1,32)".parse().unwrap();
    assert!(rm.is_identity_map());
    // A stride of two skips every other offset.
    let s2: Layout = "4:2".parse().unwrap();
    assert!(!s2.is_identity_map());

    let overflowing_size = Layout::new(t(&[i64::MAX, 2]), t(&[1, 1]));
    assert!(!overflowing_size.is_identity_map());
    let overflowing_address = Layout::new(IntTuple::Leaf(3), IntTuple::Leaf(i64::MAX));
    assert_eq!(overflowing_address.checked_call(&IntTuple::Leaf(2)), None);
}

/// `get` must return each top-level mode as `modes` lists it, and `rank` must count them.
#[test]
fn get_and_rank_follow_the_top_level_modes() {
    use alloc::string::ToString;
    for text in ["(8,32):(32,1)", "4:1", "((2,4),32):((128,32),1)"] {
        let l: Layout = text.parse().unwrap();
        let modes = l.modes();
        assert_eq!(l.rank(), modes.len(), "{text}");
        for (mode, expected) in modes.into_iter().enumerate() {
            assert_eq!(l.get(mode), expected, "{text} mode {mode}");
        }
    }
    let nested: Layout = "((2,4),32):((128,32),1)".parse().unwrap();
    assert_eq!(nested.get(0).to_string(), "(2,4):(128,32)");
    assert_eq!(nested.get(1).to_string(), "32:1");
    assert_eq!(Layout::contiguous(4).get(0), Layout::contiguous(4));
    assert_eq!(Layout::contiguous(4).rank(), 1);
}

/// `get` past the rank must panic rather than return a wrong mode.
#[test]
#[should_panic(expected = "out of range")]
fn get_rejects_a_mode_past_the_rank() {
    let l: Layout = "(8,32):(32,1)".parse().unwrap();
    let _ = l.get(2);
}

/// Flattening a tuple must list its integers left to right and drop the grouping.
#[test]
fn flatten_and_is_congruent_follow_the_tuple_structure() {
    let nested: IntTuple = "((2,3),4)".parse().unwrap();
    assert_eq!(nested.flatten(), vec![2, 3, 4]);
    assert_eq!(IntTuple::Leaf(7).flatten(), vec![7]);
    assert!(nested.is_congruent(&"((8,5),6)".parse().unwrap()));
    assert!(!nested.is_congruent(&"(8,(5,6))".parse().unwrap()));
}

/// Nested coordinate groups must retain their matching shape and stride structure.
#[test]
fn nested_shape() {
    // The first top-level mode contains two nested coordinate parts; the
    // second top-level mode is a single part.
    let shape = IntTuple::Tuple(vec![t(&[2, 4]), IntTuple::Leaf(32)]);
    let stride = IntTuple::Tuple(vec![t(&[128, 32]), IntTuple::Leaf(1)]);
    let l = Layout::new(shape, stride);
    assert_eq!(l.size(), 256);
    // Every nested coordinate is multiplied by its matching nested stride.
    let c = IntTuple::Tuple(vec![t(&[1, 2]), IntTuple::Leaf(5)]);
    assert_eq!(l.call(&c), 128 + 64 + 5);
}

/// `idx2crd` must invert the colexicographic flat-coordinate walk for flat and nested shapes.
#[test]
fn idx2crd_roundtrips_with_crd2idx() {
    for text in [
        "(8,32):(32,1)",
        "(64,2):(1,64)",
        "((2,4),32):((128,32),1)",
        "((8,16),(64,2)):((64,512),(1,8192))",
    ] {
        let l: Layout = text.parse().unwrap();
        for flat in 0..l.size() {
            let coord = idx2crd(flat, &l.shape);
            assert!(coord.is_congruent(&l.shape));
            assert_eq!(
                l.call(&coord),
                l.call(&IntTuple::Leaf(flat)),
                "{text} @ {flat}"
            );
        }
    }
}

/// Allocation-free 2D/3D and grouped-2D coordinate helpers must match `Layout` evaluation.
#[test]
fn const_coordinate_helpers_match_layout_algebra() {
    let col_major = Layout::col_major(t(&[64, 2]));
    for tid in 0..128u32 {
        let (r, c) = idx2crd_col_major_2d(tid, 64);
        assert_eq!(
            idx2crd(i64::from(tid), &col_major.shape),
            t(&[i64::from(r), i64::from(c)])
        );
    }

    let row_major = Layout::row_major(t(&[4, 8]));
    for r in 0..4usize {
        for c in 0..8usize {
            let idx = crd2idx_row_major_2d(r, c, 8);
            assert_eq!(idx as i64, row_major.call(&t(&[r as i64, c as i64])));
            assert_eq!(idx2crd_row_major_2d(idx as u32, 8), (r as u32, c as u32));
        }
    }

    // Grouped-M (G=8) 2D tile grid must visit every (m_tile, n_tile) in 0..mtiles x 0..ntiles
    // exactly once, both when mtiles < 8 and when mtiles >= 8.
    for mtiles in [1u32, 2, 4, 8, 16, 20, 64] {
        let ntiles = 24u32;
        let mut seen = alloc::vec![false; (mtiles * ntiles) as usize];
        for tile in 0..(mtiles * ntiles) {
            let (m_tile, n_tile) = idx2crd_grouped_2d::<8>(tile, mtiles, ntiles);
            assert!(m_tile < mtiles && n_tile < ntiles);
            let linear = crd2idx_row_major_2d(m_tile as usize, n_tile as usize, ntiles as usize);
            assert!(
                !seen[linear],
                "duplicate tile ({m_tile}, {n_tile}) for mtiles={mtiles}"
            );
            seen[linear] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }
}

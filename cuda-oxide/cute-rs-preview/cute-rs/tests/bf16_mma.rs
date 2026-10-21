/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent checks of BF16 operand coverage, register order and layout evaluation.
use cute_rs::Bf16;

#[test]
fn static_atom_layouts_match_cute_algebra_for_every_lane_and_value() {
    use cute_rs::bf16_mma::{ALayout, BLayout, CLayout};
    use cute_rs::static_layout::StaticLayout;
    fn check<L: StaticLayout>(description: &str) {
        let oracle: cute_rs::layout::Layout = description.parse().unwrap();
        for flat in 0..L::SIZE {
            assert_eq!(
                L::offset(flat),
                oracle
                    .checked_call(&cute_rs::layout::IntTuple::Leaf(flat as i64))
                    .unwrap() as usize
            );
        }
    }
    check::<ALayout>("((4,8),((2,2),2)):((2,16),((1,128),8))");
    check::<BLayout>("((4,8),(2,2)):((2,16),(1,8))");
    check::<CLayout>("((4,8),(2,2)):((2,8),(1,64))");
}

#[test]
fn packed_output_partition_visits_only_live_decode_rows() {
    use cute_rs::bf16_mma::PackedDecode;
    #[repr(align(4))]
    struct Aligned([Bf16; 256]);
    let data = Aligned([Bf16::from_bits(0); 256]);
    for rows in [1, 2] {
        let source = PackedDecode::new(&data.0, &data.0, rows, 16, 8, 16, 16).unwrap();
        let mut seen = vec![0; rows * 8];
        for lane in 0..32 {
            Bf16Mma16x8x16::get_slice(lane)
                .partition_c()
                .store_decode_with(MmaAccumulator::zero(), &source, |r, c, _| {
                    seen[r * 8 + c] += 1;
                });
        }
        assert_eq!(seen, vec![1; rows * 8]);
    }
}
use cute_rs::bf16_mma::{Bf16Mma16x8x16, MmaAccumulator};
use cute_rs::markers::{Composed, RowMajor, Swizzle};

#[test]
fn packed_decode_proof_rejects_incomplete_or_misaligned_operands() {
    use cute_rs::bf16_mma::PackedDecode;
    #[repr(align(4))]
    struct Aligned([Bf16; 1024]);
    let data = Aligned([Bf16::from_bits(0); 1024]);
    assert!(PackedDecode::new(&data.0, &data.0, 1, 16, 8, 16, 16).is_some());
    assert!(PackedDecode::new(&data.0, &data.0, 2, 32, 16, 34, 34).is_some());
    for (rows, width, outputs, is, ws) in [
        (0, 16, 8, 16, 16),
        (3, 16, 8, 16, 16),
        (1, 15, 8, 16, 16),
        (1, 16, 7, 16, 16),
        (1, 16, 8, 15, 16),
        (1, 16, 8, 16, 17),
    ] {
        assert!(PackedDecode::new(&data.0, &data.0, rows, width, outputs, is, ws).is_none());
    }
    assert!(PackedDecode::new(&data.0[1..], &data.0, 1, 16, 8, 16, 16).is_none());
    assert!(PackedDecode::new(&data.0[..15], &data.0, 1, 16, 8, 16, 16).is_none());
    assert!(PackedDecode::new(&data.0, &data.0[..127], 1, 16, 8, 16, 16).is_none());
}

#[test]
fn validated_decode_callback_runs_once_only_for_proven_storage() {
    use cute_rs::bf16_mma::PackedDecode;
    #[repr(align(4))]
    struct Aligned([Bf16; 128]);
    let data = Aligned([Bf16::from_bits(0); 128]);
    let mut calls = 0;
    assert!(PackedDecode::with_validated(
        &data.0,
        &data.0,
        1,
        16,
        8,
        16,
        16,
        |proof| {
            calls += 1;
            let mut count = 0;
            Bf16Mma16x8x16::get_slice(0)
                .partition_c()
                .store_decode_with(MmaAccumulator::zero(), proof, |_, _, _| count += 1);
            assert_eq!(count, 2);
        }
    ));
    assert!(!PackedDecode::with_validated(
        &data.0[..15],
        &data.0,
        1,
        16,
        8,
        16,
        16,
        |_| calls += 1
    ));
    assert_eq!(calls, 1);
}

#[test]
fn operand_partition_matches_ptx_coordinates_and_covers_every_element() {
    let mut a_seen = [0; 256];
    let mut b_seen = [0; 128];
    let mut c_seen = [0; 128];
    for lane in 0..32 {
        let mma = Bf16Mma16x8x16::get_slice(lane);
        let mut a_coords = Vec::new();
        let mut b_coords = Vec::new();
        mma.partition_a().load_with(|r, k| {
            a_coords.push((r, k));
            a_seen[r * 16 + k] += 1;
            Bf16::from_bits(0)
        });
        mma.partition_b().load_with(|n, k| {
            b_coords.push((n, k));
            b_seen[n * 16 + k] += 1;
            Bf16::from_bits(0)
        });
        let row = lane / 4;
        let col = lane % 4 * 2;
        assert_eq!(
            a_coords,
            [
                (row, col),
                (row, col + 1),
                (row + 8, col),
                (row + 8, col + 1),
                (row, col + 8),
                (row, col + 9),
                (row + 8, col + 8),
                (row + 8, col + 9)
            ]
        );
        assert_eq!(
            b_coords,
            [(row, col), (row, col + 1), (row, col + 8), (row, col + 9)]
        );
        let mut c_coords = Vec::new();
        mma.partition_c()
            .store_with(MmaAccumulator::zero(), |r, c, v| {
                c_coords.push((r, c));
                c_seen[r * 8 + c] += 1;
                assert_eq!(v.to_bits(), 0);
            });
        assert_eq!(
            c_coords,
            [
                (row, col),
                (row, col + 1),
                (row + 8, col),
                (row + 8, col + 1)
            ]
        );
    }
    assert_eq!(a_seen, [1; 256]);
    assert_eq!(b_seen, [1; 128]);
    assert_eq!(c_seen, [1; 128]);
}

#[test]
fn shared_layout_offset_matches_independent_cute_layout_oracle() {
    type L = Composed<Swizzle<3, 3, 3>, 0, RowMajor<64, 128>>;
    let layout: cute_rs::layout::Layout = "(64,128):(128,1)".parse().unwrap();
    let swizzle = cute_rs::layout::Swizzle::new(3, 3, 3);
    for r in 0..64 {
        for c in 0..128 {
            let plain = layout
                .checked_call(&cute_rs::layout::IntTuple::Tuple(vec![
                    cute_rs::layout::IntTuple::Leaf(r),
                    cute_rs::layout::IntTuple::Leaf(c),
                ]))
                .unwrap();
            assert_eq!(
                cute_rs::static_layout::offset_2d::<L>(r as usize, c as usize),
                swizzle.apply(plain) as usize
            );
        }
    }
}

#[test]
fn static_composition_applies_nonzero_offset_before_negative_swizzle() {
    use cute_rs::static_layout::StaticLayout;
    type L = Composed<Swizzle<2, 1, -3>, 7, RowMajor<3, 5>>;
    let inner: cute_rs::layout::Layout = "(3,5):(5,1)".parse().unwrap();
    let swizzle = cute_rs::layout::Swizzle::new(2, 1, -3);
    for flat in 0..L::SIZE {
        let plain = inner
            .checked_call(&cute_rs::layout::IntTuple::Leaf(flat as i64))
            .unwrap();
        let expected = swizzle.apply(plain + 7) as usize;
        assert_eq!(L::offset(flat), expected);
        assert_eq!(
            cute_rs::static_layout::offset_2d::<L>(flat % 3, flat / 3),
            expected
        );
    }
}

#[test]
fn mma_rejects_swapped_operand_roles() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/bf16_mma_partition_roles.rs");
    tests.compile_fail("tests/compile_fail/packed_decode_borrow_escape.rs");
    tests.compile_fail("tests/compile_fail/packed_decode_callback_escape.rs");
    tests.compile_fail("tests/compile_fail/packed_decode_callback_mutation.rs");
}

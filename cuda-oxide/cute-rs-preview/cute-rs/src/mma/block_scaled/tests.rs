/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks SM120 accumulator, B-fragment and scale-provider mappings."]
use super::tiled::{a_atom_128, b_pair_atom_128};
use super::{Mxf4AccumulatorTile2x8, Mxf4BTileK64, Mxfp4TiledMma};
use crate::block_scaled::{Mkl, Nkl, SharedScaleAtom, Sm120ScaleAtom};
use crate::mma::AccC;

/// Accumulator and lazy B views must retain the intended register and pointer carrier sizes.
#[test]
fn accumulator_and_lazy_b_keep_their_physical_carriers_explicit() {
    assert_eq!(
        core::mem::size_of::<Mxf4AccumulatorTile2x8>(),
        16 * core::mem::size_of::<AccC>()
    );
    assert_eq!(
        core::mem::size_of::<Mxf4BTileK64<'static, ()>>(),
        // shared base + capacity + warp-N + K-half
        4 * core::mem::size_of::<usize>()
    );
}

/// Warp row bands and B-pair indices must match CuTe’s 128-by-128 fragment assignment.
#[test]
fn cta_128_fragment_permutation_matches_cute() {
    assert_eq!(
        [
            [16 * a_atom_128(0, 0), 16 * a_atom_128(0, 1)],
            [16 * a_atom_128(1, 0), 16 * a_atom_128(1, 1)],
            [16 * a_atom_128(2, 0), 16 * a_atom_128(2, 1)],
            [16 * a_atom_128(3, 0), 16 * a_atom_128(3, 1)],
        ],
        [[0, 64], [16, 80], [32, 96], [48, 112]]
    );
    assert_eq!(
        [
            [
                16 * b_pair_atom_128(0, 0),
                16 * b_pair_atom_128(0, 1),
                16 * b_pair_atom_128(0, 2),
                16 * b_pair_atom_128(0, 3),
            ],
            [
                16 * b_pair_atom_128(1, 0),
                16 * b_pair_atom_128(1, 1),
                16 * b_pair_atom_128(1, 2),
                16 * b_pair_atom_128(1, 3),
            ],
        ],
        [[0, 32, 64, 96], [16, 48, 80, 112]]
    );
}

/// Each lane’s scale load must select exactly the canonical two A and eight B words.
#[test]
fn canonical_scale_atom_loads_exact_two_a_and_eight_b_packs() {
    let mut a_words =
        core::array::from_fn::<_, { Sm120ScaleAtom::WORDS }, _>(|word| 0xa500_0000 | word as u32);
    let mut b_words =
        core::array::from_fn::<_, { Sm120ScaleAtom::WORDS }, _>(|word| 0xb500_0000 | word as u32);
    // Both arrays are aligned, contain one complete scale block, and stay
    // alive while the views are used.
    let a = unsafe {
        SharedScaleAtom::<Mkl>::from_raw_parts(a_words.as_mut_ptr(), Sm120ScaleAtom::WORDS)
    };
    // The separate B array follows the same rules.
    let b = unsafe {
        SharedScaleAtom::<Nkl>::from_raw_parts(b_words.as_mut_ptr(), Sm120ScaleAtom::WORDS)
    };

    // Lane 0 provides A rows 0/64 and B rows
    // 0,8,32,40,64,72,96,104 for warp (0,0).
    let lane0 = Mxfp4TiledMma::<()>::get_slice(0);
    // The arrays are valid and lane/warp positions are in range.
    let first = unsafe { lane0.load_scale_atom_128(&a, &b, 0, 0) };
    assert_eq!(
        [first.a0.bits(), first.a1.bits()],
        [0xa500_0000, 0xa500_0002]
    );
    assert_eq!(
        [
            first.b0.bits(),
            first.b1.bits(),
            first.b2.bits(),
            first.b3.bits(),
            first.b4.bits(),
            first.b5.bits(),
            first.b6.bits(),
            first.b7.bits(),
        ],
        [
            0xb500_0000,
            0xb500_0020,
            0xb500_0001,
            0xb500_0021,
            0xb500_0002,
            0xb500_0022,
            0xb500_0003,
            0xb500_0023,
        ]
    );

    // Lane 31 checks the last rows and words for warp position (3,1).
    let lane31 = Mxfp4TiledMma::<()>::get_slice(31);
    // The arrays are valid and lane/warp positions are in range.
    let last = unsafe { lane31.load_scale_atom_128(&a, &b, 3, 1) };
    assert_eq!([last.a0.bits(), last.a1.bits()], [0xa500_007d, 0xa500_007f]);
    assert_eq!(
        [
            last.b0.bits(),
            last.b1.bits(),
            last.b2.bits(),
            last.b3.bits(),
            last.b4.bits(),
            last.b5.bits(),
            last.b6.bits(),
            last.b7.bits(),
        ],
        [
            0xb500_005c,
            0xb500_007c,
            0xb500_005d,
            0xb500_007d,
            0xb500_005e,
            0xb500_007e,
            0xb500_005f,
            0xb500_007f,
        ]
    );
}

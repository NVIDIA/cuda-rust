/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks packed-value carriers and dense versus Blackwell scale layouts."]
use super::*;
use crate::numeric::{Float4E2M1FN, Float8E8M0FNU};

/// Canonical scale coordinates must match known Blackwell offsets, including block boundaries.
#[test]
fn canonical_scale_layout_matches_blackwell_golden_offsets() {
    let layout = Sm1xxBlockScaleKMajor::<16>::new(256, 128);

    assert_eq!(layout.crd2idx(0, 0, 0), 0);
    assert_eq!(layout.crd2idx(0, 31, 0), 496);
    assert_eq!(layout.crd2idx(0, 32, 0), 4);
    assert_eq!(layout.crd2idx(0, 127, 3), 511);
    assert_eq!(layout.crd2idx(0, 0, 4), 512);
    assert_eq!(layout.crd2idx(0, 128, 0), 1_024);
    assert_eq!(layout.crd2idx(1, 0, 0), 2_048);
    assert_eq!(layout.storage_len(2), 4_096);
}

/// M/K/L and N/K/L scale/value views must retain distinct type identities.
#[test]
fn matrix_and_vector_modes_remain_distinct_types() {
    let values = [0u8; 128];
    let scales = [0u8; CANONICAL_SCALE_ATOM_BYTES];

    let matrix = BlockScaledTensor::<Float4E2M1FN, Float8E8M0FNU, 16, KMajor<Mkl>>::from_slices(
        &values, &scales, 4, 64,
    );
    let vector = BlockScaledTensor::<Float4E2M1FN, Float8E8M0FNU, 16, KMajor<Nkl>>::from_slices(
        &values, &scales, 1, 64,
    );

    assert_eq!(matrix.k_tile_count(), 1);
    assert_eq!(vector.k_tile_count(), 1);
    assert_eq!(matrix.thread_row(0, 3).k_tile(0).value_base, 96);
    assert_eq!(vector.thread_row(0, 0).k_tile(0).value_base, 0);
}

/// Logical K extents must survive conversion to packed value and scale copy carriers.
#[test]
fn mxf4_tensor_keeps_logical_k_and_materializes_packed_carriers() {
    let values = [f16::from_bits(0); 128];
    let scales = [127u8; 16];
    let matrix = Mxf4BlockScaledTensor::<Mkl>::from_slices(&values, &scales, 4, 64);

    let copy = matrix.values_for_copy();
    assert_eq!(copy.carrier().rows, 4);
    assert_eq!(copy.carrier().cols, 16);
    assert_eq!(copy.carrier().leading_dim.elements, 16);
    assert_eq!(matrix.scales.layout.groups(), 2);
    assert_eq!(matrix.scales.layout.storage_len(), 8);
    assert_eq!(matrix.load_scale_pair(3, 0).bits(), 0x7f7f);
}

/// Packed scale words must retain low-byte-first K-group order when selecting either half.
#[test]
fn staged_scale_pack_preserves_little_endian_k_order() {
    let pack = ScalePack4::<Mkl>::from_bits(0x8382_8180);
    assert_eq!(pack.pair::<0>().bits(), 0x8180);
    assert_eq!(pack.pair::<1>().bits(), 0x8382);

    /// Four-byte-aligned scale storage for testing u32 scale carriers.
    #[repr(align(4))]
    struct AlignedScales(#[doc = "Scale bytes arranged in increasing K-group order."] [u8; 32]);
    let values = [f16::from_bits(0); 256];
    let scales = AlignedScales([127; 32]);
    let matrix = Mxf4BlockScaledTensor::<Mkl>::from_slices(&values, &scales.0, 4, 256);
    // Safe because the wrapper gives the bytes u32 alignment and each row
    // has 8 groups, which can be split into groups of 4.
    let copy = unsafe { matrix.scale_words_for_copy() };
    assert_eq!(copy.carrier().rows, 4);
    assert_eq!(copy.carrier().cols, 2);
    assert_eq!(copy.carrier().leading_dim.elements, 2);
}

/// Dense row scales and padded Blackwell scale blocks must report their different storage requirements.
#[test]
fn dense_and_canonical_scale_layouts_are_distinct() {
    let dense = DenseBlockScaleKMajor::<Mkl, 32>::new(128, 128);
    let canonical = Sm1xxBlockScaleKMajor::<32>::new(128, 128);

    assert_eq!(dense.crd2idx(32, 0), 128);
    assert_eq!(dense.storage_len(), 512);
    assert_eq!(canonical.crd2idx(0, 32, 0), 4);
    assert_eq!(canonical.storage_len(1), 512);
}

/// Per-row shared scale words must match the canonical 512-byte atom permutation.
#[test]
fn shared_scale_atom_words_match_blackwell_golden_layout() {
    let canonical = Sm1xxBlockScaleKMajor::<32>::new(128, 128);
    let mut seen = [false; Sm120ScaleAtom::WORDS];

    for row in 0..Sm120ScaleAtom::ROWS {
        let word = Sm120ScaleAtom::word_offset(row);
        assert_eq!(word * 4, canonical.crd2idx(0, row, 0));
        for group in 0..CANONICAL_SCALE_K_GROUPS {
            assert_eq!(word * 4 + group, canonical.crd2idx(0, row, group));
        }
        assert!(!seen[word]);
        seen[word] = true;
    }

    assert!(seen.into_iter().all(|present| present));
    assert_eq!(Sm120ScaleAtom::BYTES, 512);
    assert_eq!(
        [
            Sm120ScaleAtom::word_offset(0),
            Sm120ScaleAtom::word_offset(31),
            Sm120ScaleAtom::word_offset(32),
            Sm120ScaleAtom::word_offset(63),
            Sm120ScaleAtom::word_offset(64),
            Sm120ScaleAtom::word_offset(95),
            Sm120ScaleAtom::word_offset(96),
            Sm120ScaleAtom::word_offset(127),
        ],
        [0, 124, 1, 125, 2, 126, 3, 127]
    );
}

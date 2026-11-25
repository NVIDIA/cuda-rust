/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression cases for the composed layout validators.

use super::thread_value::tv_coord;
use super::*;
use crate::Swizzle;
use crate::algebra::make_layout_tv;
use crate::{ComposedLayout, IntTuple, Layout, OffsetUnit};
use alloc::string::ToString;
use alloc::{vec, vec::Vec};

/// Parse a test layout literal and fail immediately if the fixture is malformed.
fn l(text: &str) -> Layout {
    text.parse().unwrap()
}

/// Construct the checked cooperative-copy plan used by the live decode canary.
fn canary_plan() -> Result<CooperativeCopyPlan, CooperativePlanError> {
    let thread = l("(6,1):(1,0)");
    let value = l("(1,4):(0,1)");
    let tile = l("(6,4):(4,1)");
    let smem =
        ComposedLayout::new(Swizzle::new(3, 4, 3), 8, tile.clone(), OffsetUnit::Elements).unwrap();
    validate_cooperative_copy_plan(16, &thread, &value, &tile, &smem, 4)
}

/// The complete copy plan must match the canary’s thread ownership and shared capacity.
#[test]
fn high_level_plan_matches_the_live_decode_canary() {
    let plan = canary_plan().unwrap();
    assert_eq!(plan.tiler_shape, vec![6, 4]);
    assert_eq!(plan.thread_count, 6);
    assert_eq!(plan.values_per_thread, 4);
    assert_eq!(plan.tile_elements, 24);
    assert_eq!(plan.minimum_smem_capacity, 32);
}

/// The global source map must be canonical row-major even if a digit permutation covers the same cells.
#[test]
fn high_level_plan_rejects_a_digit_permuted_row_major_tile() {
    let thread = l("(1,1):(1,0)");
    // The logical columns are 0,1,2,3, but this nested digit mode visits
    // them as 0,2,1,3. Giving TV and TileL the same permutation makes the
    // two mistakes cancel and appear contiguous:
    //
    // TV order:       0 2 1 3
    // TileL maps to:  0 1 2 3
    //
    // A real row-major GmemMatrix still uses logical order 0,1,2,3, so
    // the high-level validator must reject the permuted TileL itself.
    let digit_permutation = l("(1,(2,2)):(0,(2,1))");
    let (_, tv) = make_layout_tv(&thread, &digit_permutation).unwrap();
    let apparent_offsets: Vec<_> = (0..4)
        .map(|value| {
            let cell = tv.checked_call(&tv_coord(0, value)).unwrap();
            digit_permutation
                .checked_call(&IntTuple::Leaf(cell))
                .unwrap()
        })
        .collect();
    assert_eq!(apparent_offsets, vec![0, 1, 2, 3]);

    let smem = ComposedLayout::from_layout(digit_permutation.clone(), OffsetUnit::Elements);
    let error = validate_cooperative_copy_plan(
        16,
        &thread,
        &digit_permutation,
        &digit_permutation,
        &smem,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("not canonical row-major"), "{error}");
}

/// Plan validation must reject excessive threads and huge strides hidden in size-one modes.
#[test]
fn high_level_plan_bounds_threads_and_hidden_raw_strides() {
    let thread = l("(1025,1):(1,0)");
    let value = l("(1,1):(0,0)");
    let tile = l("(1025,1):(1,0)");
    let smem = ComposedLayout::from_layout(tile.clone(), OffsetUnit::Elements);
    let error = validate_cooperative_copy_plan(4, &thread, &value, &tile, &smem, 4)
        .unwrap_err()
        .to_string();
    assert!(error.contains("CUDA block limit 1024"), "{error}");

    // A size-one mode has only coordinate zero, which hides its stride:
    //
    // 0 * i64::MAX = 0
    //
    // Raw-stride validation sees i64::MAX before algebra can multiply it.
    let hidden_stride = l("(6,1):(1,9223372036854775807)");
    let tile = l("(6,4):(4,1)");
    let smem = ComposedLayout::from_layout(tile.clone(), OffsetUnit::Elements);
    let error =
        validate_cooperative_copy_plan(16, &hidden_stride, &l("(1,4):(0,1)"), &tile, &smem, 4)
            .unwrap_err()
            .to_string();
    assert!(error.contains("layout stride"), "{error}");
}

/// Thread IDs may be permuted but must cover a compact interval without holes.
#[test]
fn compact_threads_allow_permutation_but_reject_gaps() {
    validate_compact_thread_layout(&l("(2,2):(2,1)")).unwrap();
    validate_compact_thread_layout(&l("(2,2):(1,2)")).unwrap();

    assert!(matches!(
        validate_compact_thread_layout(&l("4:2")),
        Err(ValidationError::ThreadOutOfRange { .. })
    ));
    assert!(matches!(
        validate_compact_thread_layout(&l("(2,2):(0,1)")),
        Err(ValidationError::DuplicateThread { .. })
    ));

    let too_large = Layout::contiguous(MAX_STATIC_VALIDATION_ELEMENTS + 1);
    assert!(matches!(
        validate_compact_thread_layout(&too_large),
        Err(ValidationError::CountTooLarge { .. })
    ));
}

/// Thread/value ownership must reject duplicate owners, missing cells and invalid extents.
#[test]
fn tv_layout_must_be_an_exact_partition() {
    let valid = l("(4,2):(2,1)");
    validate_tv_exact_coverage(&valid, 4, 2, 8).unwrap();

    let duplicate = l("(4,2):(1,1)");
    assert!(matches!(
        validate_tv_exact_coverage(&duplicate, 4, 2, 8),
        Err(ValidationError::DuplicateCell { .. })
    ));

    let out_of_range = l("(4,2):(3,1)");
    assert!(matches!(
        validate_tv_exact_coverage(&out_of_range, 4, 2, 8),
        Err(ValidationError::TvOutOfRange { .. })
    ));
}

/// The constructed thread/value map must satisfy coverage and compactness proofs.
#[test]
fn make_layout_tv_result_passes_all_static_checks() {
    let threads = l("(2,3):(3,1)");
    let values = l("(2,2):(2,1)");
    let (tiler, tv) = crate::algebra::make_layout_tv(&threads, &values).unwrap();
    validate_tv_layout(&threads, &values, &tiler, &tv).unwrap();
}

/// The tile’s value count must divide into complete instruction-width copy atoms.
#[test]
fn atom_counts_must_divide() {
    validate_atom_compatibility(256, 4, 8, 2).unwrap();
    assert_eq!(
        validate_atom_compatibility(250, 4, 8, 2),
        Err(ValidationError::AtomDoesNotDivide {
            axis: "thread",
            copy_count: 250,
            atom_count: 8,
        })
    );
    assert!(matches!(
        validate_atom_compatibility(256, 3, 8, 2),
        Err(ValidationError::AtomDoesNotDivide { axis: "value", .. })
    ));
}

/// Complete adjacent aligned per-thread atoms must pass static contiguity validation.
#[test]
fn contiguous_aligned_atoms_pass() {
    // Four threads each own four f32 values: one 16-byte atom.
    //
    // cell:  0  1  2  3 | 4  5  6  7 | ...
    // owner: T0 T0 T0 T0 |T1 T1 T1 T1 | ...
    let tv = l("(4,4):(4,1)");
    let elements =
        ComposedLayout::new(Swizzle::new(2, 2, 3), 0, l("16:1"), OffsetUnit::Elements).unwrap();
    let bytes = elements.to_byte_offsets(4).unwrap();
    validate_per_thread_atom_contiguity(&tv, 4, 4, 16, 4, &bytes, 16).unwrap();
}

/// A byte swizzle preserving four low bits must keep aligned 16-byte atoms intact.
#[test]
fn m_four_swizzle_moves_whole_sixteen_byte_atoms() {
    // The map spans enough bytes to reach the high bits that feed the
    // XOR. A smaller map would leave every source bit zero and make the
    // swizzle look like the identity by accident.
    let tv = l("(32,4):(4,1)");
    let memory =
        ComposedLayout::new(Swizzle::new(3, 4, 3), 0, l("128:4"), OffsetUnit::Bytes).unwrap();
    validate_per_thread_atom_contiguity(&tv, 32, 4, 16, 4, &memory, 16).unwrap();
}

/// A scattered per-thread value map must be rejected as a single vector transaction.
#[test]
fn scattered_values_are_not_one_atom() {
    // Coverage is complete, but each thread is raked through the tile:
    //
    // cell:  0  1  2  3 | 4  5  6  7 | ...
    // owner: T0 T1 T2 T3 |T0 T1 T2 T3 | ...
    let tv = l("(4,4):(1,4)");
    let memory = ComposedLayout::from_layout(l("16:4"), OffsetUnit::Bytes);
    assert!(matches!(
        validate_per_thread_atom_contiguity(&tv, 4, 4, 16, 4, &memory, 16),
        Err(ValidationError::AtomNotContiguous { .. })
    ));
}

/// Swizzles that split or scatter an instruction atom must be rejected.
#[test]
fn swizzle_must_preserve_the_whole_atom() {
    let tv = l("(2,4):(4,1)");
    let memory = ComposedLayout::new(
        // M=3 leaves the lowest three address bits unchanged, which
        // guarantees only 2^3 = 8 adjacent bytes. Here the XOR splits a
        // 16-byte transaction, and the exact byte proof catches it.
        Swizzle::new(1, 3, 3),
        64,
        l("8:4"),
        OffsetUnit::Bytes,
    )
    .unwrap();
    assert!(matches!(
        validate_per_thread_atom_contiguity(&tv, 2, 4, 16, 4, &memory, 16),
        Err(ValidationError::AtomStartMisaligned { .. })
            | Err(ValidationError::AtomNotContiguous { .. })
    ));
}

/// Overflow in an expected atom address must return a validation error instead of wrapping.
#[test]
fn expected_atom_address_overflow_is_a_validation_error() {
    // The swizzle maps MAX-1 to MAX, while the next input remains valid:
    //
    // input:   MAX-1  MAX
    // output:  MAX    MAX-1
    let tv = l("(1,7):(7,1)");
    let memory = ComposedLayout::new(
        Swizzle::new(1, 0, 1),
        i64::MAX - 1,
        l("7:1"),
        OffsetUnit::Bytes,
    )
    .unwrap();
    assert_eq!(
        validate_per_thread_atom_contiguity(&tv, 1, 7, 7, 1, &memory, 7),
        Err(ValidationError::ArithmeticOverflow(
            "expected atom byte address"
        ))
    );
}

/// Unswizzled u16 tiles must retain TMA’s minimum shared-base alignment.
#[test]
fn tma_identity_u16_keeps_the_hardware_minimum_alignment() {
    let layout = ComposedLayout::new(
        Swizzle::IDENTITY,
        0,
        l("(1,256):(256,1)"),
        OffsetUnit::Elements,
    )
    .unwrap();

    assert_eq!(tma_phase_alignment_bytes(&layout, 2), Ok(16));
}

/// B64 byte tiles must require a shared base aligned to the full 512-byte swizzle period.
#[test]
fn tma_b64_u8_needs_one_complete_512_byte_phase() {
    let layout = ComposedLayout::new(
        Swizzle::new(2, 4, 3),
        0,
        l("(128,64):(64,1)"),
        OffsetUnit::Elements,
    )
    .unwrap();

    assert_eq!(tma_phase_alignment_bytes(&layout, 1), Ok(512));
}

/// B128 FP16 tiles must require a shared base aligned to the full 1024-byte swizzle period.
#[test]
fn tma_b128_f16_needs_one_complete_1024_byte_phase() {
    let layout = ComposedLayout::new(
        Swizzle::new(3, 3, 3),
        0,
        l("(128,64):(64,1)"),
        OffsetUnit::Elements,
    )
    .unwrap();

    assert_eq!(tma_phase_alignment_bytes(&layout, 2), Ok(1024));
}

/// Phase-alignment queries must reject tile/swizzle combinations outside TMA’s supported modes.
#[test]
fn tma_phase_alignment_rejects_unencodable_inputs() {
    let column_major =
        ComposedLayout::new(Swizzle::IDENTITY, 0, l("(2,4):(1,2)"), OffsetUnit::Elements).unwrap();
    assert!(tma_phase_alignment_bytes(&column_major, 2).is_err());

    let b64_u8 = ComposedLayout::new(
        Swizzle::new(2, 4, 3),
        0,
        l("(128,64):(64,1)"),
        OffsetUnit::Elements,
    )
    .unwrap();
    assert!(
        tma_phase_alignment_bytes(&b64_u8, 2).is_err(),
        "the element width changes the byte-unit swizzle and row pitch"
    );
    assert!(tma_phase_alignment_bytes(&b64_u8, 0).is_err());

    let byte_offsets = ComposedLayout::new(
        Swizzle::IDENTITY,
        0,
        l("(1,256):(256,1)"),
        OffsetUnit::Bytes,
    )
    .unwrap();
    assert!(tma_phase_alignment_bytes(&byte_offsets, 2).is_err());
}

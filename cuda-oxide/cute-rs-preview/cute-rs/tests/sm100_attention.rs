/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks attention shared layouts, typed MMA shapes and complete tile-array contracts."]
use cute_rs::attention::sm100::{MmaSs, MmaTs, SharedLayout, SharedTileArray, SmemLayout};
use cute_rs::{OperandA, OperandB};

/// Keep explicit SM100 instructions behind CTM in both kernels and their CuTe helpers.
/// CuTe compiler markers for structured MMA/TMA remain a separate supported path.
#[test]
fn sm100_attention_uses_ctm_for_explicit_device_operations() {
    let cute_rs_preview = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let paths = [
        "examples/sm100_fmha_prefill_cute/src/kernel.rs",
        "examples/sm100_mla_decode_fp16_cute/src/kernel.rs",
        "cute-rs/src/arch.rs",
        "cute-rs/src/attention/sm100/pipeline.rs",
        "cute-rs/src/attention/sm100/tmem.rs",
        "cute-rs/src/attention/sm100/tma.rs",
        "cute-rs/src/attention/sm100/__compiler.rs",
    ];
    for path in paths {
        let source = std::fs::read_to_string(cute_rs_preview.join(path)).unwrap();
        assert!(
            !source.contains("cuda_device::"),
            "{path} bypasses the CTM instruction/storage interface"
        );
    }
}

/// Enumerate one shared layout and require each storage element to have exactly one owner.
fn check_bijection<L: SmemLayout>() {
    let mut offsets = Vec::new();
    for row in 0..L::ROWS {
        for col in 0..L::COLS {
            offsets.push(L::crd2idx(row, col));
        }
    }
    offsets.sort_unstable();
    assert_eq!(
        offsets,
        (0..(L::ROWS * L::COLS) as usize).collect::<Vec<_>>()
    );
}

/// Supported K-major and MN-major tiled swizzles must cover each element exactly once.
#[test]
fn tiled_swizzles_cover_storage_without_overlap() {
    check_bijection::<SharedLayout<128, 128>>();
    check_bijection::<SharedLayout<128, 128, true>>();
    check_bijection::<SharedLayout<64, 64>>();
    check_bijection::<SharedLayout<64, 32>>();
    check_bijection::<SharedLayout<128, 32, true>>();
}

/// TMA chunk coordinates and MN-major coordinates must select identical physical FP16 offsets.
#[test]
fn tma_contiguous_chunks_and_transposed_v_share_the_same_storage() {
    /// K-major 128-by-128 tile used to compare chunked TMA and transposed coordinates.
    type K = SharedLayout<128, 128>;
    assert_eq!(K::crd2idx(0, 0), 0);
    assert_eq!(K::crd2idx(0, 64), 8192);
    assert_eq!(K::crd2idx(1, 0), 72);
    for row in 0..32 {
        for col in 0..128 {
            assert_eq!(
                SharedLayout::<32, 128>::crd2idx(row, col),
                SharedLayout::<128, 32, true>::crd2idx(col, row)
            );
        }
    }
    assert_eq!(SharedLayout::<64, 32>::SWIZZLE_BITS, 2);
    assert_eq!(SharedLayout::<128, 128>::SWIZZLE_BITS, 3);
}

/// Supported shared/shared and tensor/shared attention shapes must construct without runtime data.
#[test]
fn independent_qk_and_pv_configurations_are_constructible() {
    let _ = MmaSs::<SharedLayout<128, 128>, SharedLayout<128, 128>, 128, 128, 128, 1>::new();
    let _ = MmaTs::<SharedLayout<128, 128, true>, 128, 128, 128, 1>::new();
    let _ = MmaSs::<SharedLayout<64, 64>, SharedLayout<64, 64>, 128, 128, 64, 2>::new();
    let _ = MmaSs::<SharedLayout<64, 32>, SharedLayout<128, 32, true>, 128, 256, 32, 2>::new();
}

// Host storage models the aligned allocation without accessing it as GPU memory.
/// Full-swizzle-period-aligned buffer for three complete shared-layout test tiles.
#[repr(align(1024))]
struct SharedStorage(
    #[doc = "FP16-sized storage words backing the complete tile array."] [u16; 3 * 16384],
);

/// Check each complete tile’s base address and swizzle mapping within an aligned array.
fn check_tile_array<L: SmemLayout, Role>() {
    let mut storage = SharedStorage([0; 3 * 16384]);
    let base = storage.0.as_mut_ptr();
    // SAFETY: storage covers three complete tiles and meets either alignment.
    let tiles = unsafe { SharedTileArray::<L, Role, 3>::from_raw(base.cast()) };
    let tile_elements = (L::ROWS * L::COLS) as usize;
    let period_elements = 1usize << (L::SWIZZLE_BITS + 6);
    assert_eq!(tile_elements % period_elements, 0);
    let mut seen = vec![false; 3 * tile_elements];
    for tile_index in 0..3 {
        // SAFETY: the index is in the array's static extent.
        let tile = unsafe { tiles.tile_unchecked(tile_index) };
        assert_eq!(
            tile.as_ptr().cast::<u16>(),
            base.wrapping_add(tile_index as usize * tile_elements)
        );
        for row in 0..L::ROWS {
            for col in 0..L::COLS {
                let offset = tile_index as usize * tile_elements + L::crd2idx(row, col);
                assert!(!seen[offset], "overlapping physical tile elements");
                seen[offset] = true;
                // The XOR swizzle is unchanged when translated by whole periods.
                let mask = (1usize << L::SWIZZLE_BITS) - 1;
                let unswizzled = offset ^ (((offset >> 6) & mask) << 3);
                let local = L::crd2idx(row, col);
                let local_unswizzled = local ^ (((local >> 6) & mask) << 3);
                assert_eq!(
                    unswizzled,
                    tile_index as usize * tile_elements + local_unswizzled
                );
            }
        }
    }
    assert!(seen.into_iter().all(|value| value));
}

/// Every array element must preserve the complete tile mapping and expected byte displacement.
#[test]
fn complete_tile_arrays_preserve_addresses_and_swizzles() {
    check_tile_array::<SharedLayout<64, 64>, OperandB>();
    check_tile_array::<SharedLayout<64, 32>, OperandA>();
    check_tile_array::<SharedLayout<128, 32, true>, OperandB>();
    check_tile_array::<SharedLayout<128, 128>, OperandA>();
    check_tile_array::<SharedLayout<128, 128, true>, OperandB>();
    check_tile_array::<SharedLayout<8, 32>, OperandA>();
    check_tile_array::<SharedLayout<8, 64>, OperandA>();
}

/// Compile checks must reject wrong A/B roles, zero tile counts and overflowing arrays.
#[test]
fn tile_arrays_reject_operand_role_changes_and_invalid_extents() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/sm100_attention_tile_array_role.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_tile_array_extent.rs");
}

// Const evaluation rejects inbounds pointer arithmetic after an allocation
// expires. A safe raw-address accessor must not require that allocation live.
/// Require shared tile-array construction and selection to remain usable in const evaluation.
const _: () = {
    let tile = {
        let mut storage = SharedStorage([0; 3 * 16384]);
        // SAFETY: two full tiles fit in this aligned, currently live storage.
        let tiles = unsafe {
            SharedTileArray::<SharedLayout<64, 64>, OperandA, 2>::from_raw(
                storage.0.as_mut_ptr().cast(),
            )
        };
        // SAFETY: index one is within the two-tile extent; no asynchronous use.
        unsafe { tiles.tile_unchecked(1) }
    };
    let _ = tile.as_ptr();
};

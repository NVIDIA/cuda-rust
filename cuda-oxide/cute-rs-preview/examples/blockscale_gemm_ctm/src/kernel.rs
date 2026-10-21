/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit three-stage SM120 MXFP4 GEMM. No CuTe layout or pipeline operations.
use crate::physical;
use ctm_rs::mbarrier::BarrierAddressExt;
use ctm_rs::memory::DynamicSharedArray;
use ctm_rs::{cuda_module, kernel, launch_bounds, launch_contract};
use ctm_rs::{
    mbarrier::{self, Barrier, GeneralBarrier, ManagedBarrier, Ready, TmaBarrier, Uninit},
    memory::{F16Tile128, Mxf4ScaleTile, Mxf4Tile128, PhysicalFormat, SharedAddress, SharedRegion},
    mma::{self, Accumulator16x8, AccumulatorTile16x8, ScalePairA, ScalePairB},
    thread,
    tma::{self, Coordinates2D, TensorMap},
};
/// Output row count for the fixed benchmark problem.
pub const M: usize = 2048;
/// Output column count for the fixed benchmark problem.
pub const N: usize = 2048;
/// Reduction extent in logical FP4 elements before packing.
pub const K: usize = 1024;
/// Number of 32-element K groups, each with one UE8M0 scale.
pub const KB: usize = K / 32;
/// Number of 128-element K tiles consumed for each output tile.
pub const K_STAGES: usize = K / 128;
/// Threads required in each CTA by the kernel launch contract.
pub const THREADS: usize = 288;
/// Number of 128x128 output tiles assigned persistently across the launched CTAs.
pub const TOTAL_TILES: usize = (M / 128) * (N / 128);
/// Total dynamic shared-memory allocation in bytes required per CTA.
pub const DYNAMIC_SMEM_BYTES: usize = 86_016;
/// Number of shared input slots overlapped by TMA and matrix instructions.
const STAGES: usize = 3;
/// Byte footprint of one 128-row packed FP4 value stage.
const DATA_BYTES: usize = 8192;
/// Byte footprint of one 128-row, four-group scale stage.
const SCALE_BYTES: usize = 512;
/// Byte offset of the staged B values after all A stages.
const B_OFFSET: usize = 24576;
/// Byte offset of the A-scale ring after both value rings.
const SFA_OFFSET: usize = 49152;
/// Byte offset of the B-scale ring after the A-scale ring.
const SFB_OFFSET: usize = 50688;
/// Byte offset of the full/empty barrier rings in dynamic shared memory.
const BARRIERS: usize = 52224;
/// Aligned byte offset of the two shared FP16 output halves.
const OUTPUT: usize = 53248;
/// Total value and scale bytes expected by one full-stage barrier.
const TX_BYTES: u32 = 17408;
/// Compile-time check that the two output halves end at the declared shared-memory size.
const _: () = assert!(OUTPUT + 2 * F16Tile128::FOOTPRINT_BYTES == DYNAMIC_SMEM_BYTES);
/// Compile-time check that every B-scale stage supports aligned 16-byte vector loads.
const _: () = assert!(SFB_OFFSET.is_multiple_of(16) && SCALE_BYTES.is_multiple_of(16));

/// One lane's scale words for four pairs of B fragments.
struct BScaleFragments {
    /// Four pairs of packed u32 scale words, with low/high halves selecting the K64 half.
    words: [[u32; 2]; 4],
}

impl BScaleFragments {
    /// Load the lane's eight words as two aligned 16-byte groups.
    ///
    /// # Safety
    /// `stage` must point to a readable, 16-byte-aligned 512-byte scale stage
    /// whose TMA writes have completed. `wn < 2` and `lane < 32` are required.
    #[inline(always)]
    unsafe fn load(stage: *const u32, wn: usize, lane: usize) -> Self {
        // q <= 23 and q+8 <= 31, so scale_word reduces to 4*q and 4*(q+8).
        // Each start is 16-byte aligned; even the last vector ends at byte 512.
        let q = 16 * wn + lane / 4;
        let even = unsafe { *stage.add(physical::scale_word(q)).cast::<u128>() };
        let odd = unsafe { *stage.add(physical::scale_word(q + 8)).cast::<u128>() };
        let mut words = [[0; 2]; 4];
        mma::static_for!(PAIR in [0, 1, 2, 3] {
            // Narrow after constant shifts, before the runtime K-half selection.
            words[PAIR] = [(even >> (32 * PAIR)) as u32, (odd >> (32 * PAIR)) as u32];
        });
        Self { words }
    }

    /// Select the two UE8M0 scale pairs for K64 half 0 or 1.
    #[inline(always)]
    fn pair<const PAIR: usize>(&self, half: usize) -> [ScalePairB; 2] {
        const { assert!(PAIR < 4) }
        [
            ScalePairB::from_bits((self.words[PAIR][0] >> (16 * half)) as u16),
            ScalePairB::from_bits((self.words[PAIR][1] >> (16 * half)) as u16),
        ]
    }
}

/// Borrow a typed shared tile; the caller provides live, aligned storage covering its full footprint.
#[inline(always)]
unsafe fn region<'s, F: PhysicalFormat>(pointer: *mut u8) -> SharedRegion<'s, F> {
    unsafe { SharedRegion::from_raw_parts_or_trap(pointer.cast(), F::FOOTPRINT_BYTES) }
}
/// Retry CTA parity probes on a live initialized barrier until the phase completes.
#[inline(always)]
unsafe fn wait<K>(bar: &ManagedBarrier<Ready, K>, phase: u32) {
    unsafe { while !mbarrier::try_wait_parity(bar, phase) {} }
}

// BAND/COL identify fragments; the example retains its explicit physical mapping.
/// Compute an FP16 element address for BAND<2, COL<8, warp<8, and lane<32.
#[inline(always)]
const fn output_fragment_row<const BAND: usize, const COL: usize>(
    warp: usize,
    lane: usize,
) -> usize {
    let row = 16 * (warp % 4) + 64 * BAND + (lane & 15);
    // Resolve the table before MIR import: a runtime array promotion would
    // otherwise require device global-allocation support.
    let column = 16 * (warp / 4) + const { physical::N_OFFSETS[COL] };
    physical::output_element(row, column)
}

/// Store a converted accumulator through stmatrix; callers provide the complete warp and live aligned output tile.
#[inline(always)]
unsafe fn store_fragment<const BAND: usize, const COL: usize>(
    output: *mut f16,
    warp: usize,
    lane: usize,
    fragment: Accumulator16x8,
) {
    unsafe {
        mma::store_f16(
            SharedAddress::from_raw(
                output
                    .add(output_fragment_row::<BAND, COL>(warp, lane))
                    .cast(),
            ),
            fragment,
        );
    }
}

/// SM120 block-scaled GEMM entry point built from typed instruction primitives.
#[cuda_module]
pub mod kernels {

    use super::*;
    /// Compute persistent 128x128 tiles from five live tensor maps using three load stages and eight compute warps.
    #[launch_bounds(288)]
    #[launch_contract(domain=1, block=(288,1,1), dynamic_shared=86016,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    pub fn blockscale_gemm(
        #[grid_constant] a_map: &TensorMap<u8, Mxf4Tile128>,
        #[grid_constant] b_map: &TensorMap<u8, Mxf4Tile128>,
        #[grid_constant] sa_map: &TensorMap<u16, Mxf4ScaleTile>,
        #[grid_constant] sb_map: &TensorMap<u16, Mxf4ScaleTile>,
        #[grid_constant] c_map: &TensorMap<f16, F16Tile128>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let base = DynamicSharedArray::<u8, 1024>::get_raw();
        unsafe {
            let full = base.add(BARRIERS).cast::<Barrier>();
            let empty = full.add(STAGES);
            if tid == 0 {
                let mut s = 0;
                while s < STAGES {
                    ManagedBarrier::<Uninit, TmaBarrier>::from_static(full.add(s))
                        .init_instruction(1);
                    ManagedBarrier::<Uninit, GeneralBarrier>::from_static(empty.add(s))
                        .init_instruction(8);
                    s += 1;
                }
            }
            mbarrier::fence_mbarrier_init_release_cluster();
            mbarrier::barrier_cta_sync(0, THREADS as u32);
            let mut tile = thread::blockIdx_x() as usize;
            let stride = thread::gridDim_x() as usize;
            let mut slot = 0usize;
            if warp < 8 {
                let wm = warp % 4;
                let wn = warp / 4;
                let mut phase = 0u32;
                while tile < TOTAL_TILES {
                    let tile_m = tile % (M / 128);
                    let tile_n = tile / (M / 128);
                    let mut accumulators = AccumulatorTile16x8::<2, 8>::zero();
                    let mut stage = 0;
                    while stage < K_STAGES {
                        let ready =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(slot));
                        wait(&ready, phase);
                        let a = base.add(slot * DATA_BYTES);
                        let b = base.add(B_OFFSET + slot * DATA_BYTES);
                        let sa = base.add(SFA_OFFSET + slot * SCALE_BYTES).cast::<u32>();
                        let sb = base.add(SFB_OFFSET + slot * SCALE_BYTES).cast::<u32>();
                        let a_provider = lane / 4 + 8 * (lane & 1);
                        let sa0 = *sa.add(physical::scale_word(16 * wm + a_provider));
                        let sa1 = *sa.add(physical::scale_word(16 * wm + a_provider + 64));
                        let scales_b = BScaleFragments::load(sb, wn, lane);
                        let mut half = 0;
                        while half < 2 {
                            let a0 = mma::load_mxf4_a(SharedAddress::from_raw(
                                a.add(physical::matrix_row(wm, half, lane)).cast(),
                            ));
                            let a1 = mma::load_mxf4_a(SharedAddress::from_raw(
                                a.add(physical::matrix_row(wm + 4, half, lane)).cast(),
                            ));
                            let scale_a0 = ScalePairA::from_bits((sa0 >> (16 * half)) as u16);
                            let scale_a1 = ScalePairA::from_bits((sa1 >> (16 * half)) as u16);
                            // Finish all four MMAs before loading the next B pair.
                            mma::static_for!(PAIR in [0, 1, 2, 3] {
                                let (b0, b1) = mma::load_mxf4_b_pair(SharedAddress::from_raw(
                                    b.add(physical::matrix_row(wn + 2 * PAIR, half, lane)).cast(),
                                ));
                                accumulators.mma_mxf4_pair::<PAIR>(
                                    [a0, a1],
                                    [b0, b1],
                                    [scale_a0, scale_a1],
                                    scales_b.pair::<PAIR>(half),
                                );
                            });
                            half += 1;
                        }
                        if lane == 0 {
                            mbarrier::arrive(
                                &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                                    empty.add(slot),
                                ),
                            );
                        }
                        slot += 1;
                        if slot == STAGES {
                            slot = 0;
                            phase ^= 1;
                        }
                        stage += 1;
                    }
                    // The producer warp never joins the 256-thread output barriers.
                    if tid == 0 {
                        tma::bulk_wait_group_read::<0>();
                    }
                    mbarrier::barrier_cta_sync(2, 256);
                    let output = base.add(OUTPUT).cast::<f16>();
                    mma::static_for!(BAND in [0, 1] {
                        mma::static_for!(COL in [0, 1, 2, 3, 4, 5, 6, 7] {
                            store_fragment::<BAND, COL>(
                                output, warp, lane, accumulators.fragment::<BAND, COL>(),
                            );
                        });
                    });
                    mbarrier::fence_proxy_async_shared_cta();
                    mbarrier::barrier_cta_sync(2, 256);
                    if tid == 0 {
                        let left = region::<F16Tile128>(base.add(OUTPUT));
                        let right = region::<F16Tile128>(base.add(OUTPUT + 16384));
                        tma::store_2d(
                            &left,
                            c_map,
                            Coordinates2D::new((tile_n * 128) as i32, (tile_m * 128) as i32),
                        );
                        tma::store_2d(
                            &right,
                            c_map,
                            Coordinates2D::new((tile_n * 128 + 64) as i32, (tile_m * 128) as i32),
                        );
                        tma::bulk_commit_group();
                    }
                    tile += stride;
                }
                if tid == 0 {
                    tma::bulk_wait_group_read::<0>();
                }
            } else if lane == 0 {
                let mut phase = 1u32;
                while tile < TOTAL_TILES {
                    let tile_m = tile % (M / 128);
                    let tile_n = tile / (M / 128);
                    let mut stage = 0;
                    while stage < K_STAGES {
                        wait(
                            &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(slot)),
                            phase,
                        );
                        let ready =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(slot));
                        mbarrier::arrive_expect_tx(&ready, TX_BYTES);
                        let a = region::<Mxf4Tile128>(base.add(slot * DATA_BYTES));
                        let b = region::<Mxf4Tile128>(base.add(B_OFFSET + slot * DATA_BYTES));
                        let sa = region::<Mxf4ScaleTile>(base.add(SFA_OFFSET + slot * SCALE_BYTES));
                        let sb = region::<Mxf4ScaleTile>(base.add(SFB_OFFSET + slot * SCALE_BYTES));
                        tma::load_2d(
                            &a,
                            a_map,
                            Coordinates2D::new((stage * 64) as i32, (tile_m * 128) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &b,
                            b_map,
                            Coordinates2D::new((stage * 64) as i32, (tile_n * 128) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &sa,
                            sa_map,
                            Coordinates2D::new(0, (tile_m * K_STAGES + stage) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &sb,
                            sb_map,
                            Coordinates2D::new(0, (tile_n * K_STAGES + stage) as i32),
                            ready.shared_address(),
                        );
                        slot += 1;
                        if slot == STAGES {
                            slot = 0;
                            phase ^= 1;
                        }
                        stage += 1;
                    }
                    tile += stride;
                }
                let mut tail = 0;
                while tail < STAGES {
                    wait(
                        &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(slot)),
                        phase,
                    );
                    slot += 1;
                    if slot == STAGES {
                        slot = 0;
                        phase ^= 1;
                    }
                    tail += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! CPU checks for vectorized scale loading and constant-index fragment placement.

    use super::*;

    /// Aligned scale-ring test storage with distinguishable words and K halves.
    #[repr(align(16))]
    struct ScaleStages(
        #[doc = "Three 512-byte scale stages represented as consecutive u32 words."]
        [u32; 128 * STAGES],
    );

    /// Fill the scale ring with tags distinguishing each stage, word, and 16-bit half.
    fn tagged_scales() -> ScaleStages {
        // Distinct low/high halves also distinguish adjacent words and stages.
        ScaleStages(core::array::from_fn(|index| {
            ((0x8000 | index as u32) << 16) | (0x1200 | index as u32)
        }))
    }

    /// Compare vectorized scale loads with scalar physical addressing across every legal provider and K half.
    #[test]
    fn vector_scale_loads_match_scalar_mapping_for_all_stages_warps_lanes_and_halves() {
        let storage = tagged_scales();
        for slot in 0..STAGES {
            let stage = unsafe { storage.0.as_ptr().add(slot * 128) };
            for warp in 0..8 {
                let wn = warp / 4;
                for lane in 0..32 {
                    let q = 16 * wn + lane / 4;
                    for row in [q, q + 8] {
                        let offset = 4 * physical::scale_word(row);
                        assert_eq!((stage as usize + offset) % 16, 0);
                        assert!(offset + 16 <= SCALE_BYTES);
                    }
                    let scales = unsafe { BScaleFragments::load(stage, wn, lane) };
                    for half in 0..2 {
                        mma::static_for!(PAIR in [0, 1, 2, 3] {
                            let actual = scales.pair::<PAIR>(half);
                            for (side, scale) in actual.into_iter().enumerate() {
                                let row = q + physical::N_OFFSETS[2 * PAIR + side];
                                let word = storage.0[slot * 128 + physical::scale_word(row)];
                                assert_eq!(scale.bits(), (word >> (16 * half)) as u16);
                            }
                        });
                    }
                }
            }
        }
    }

    /// Check the final vector provider reaches the last stage words without reordering either K half.
    #[test]
    fn last_scale_provider_preserves_vector_word_order_and_k_halves() {
        let storage = tagged_scales();
        // wn=1, lane=31: q=23, the two vectors start at words 92 and 124.
        let scales = unsafe { BScaleFragments::load(storage.0.as_ptr(), 1, 31) };
        assert_eq!(scales.pair::<0>(0).map(ScalePairB::bits), [0x125c, 0x127c]);
        assert_eq!(scales.pair::<0>(1).map(ScalePairB::bits), [0x805c, 0x807c]);
        assert_eq!(scales.pair::<3>(0).map(ScalePairB::bits), [0x125f, 0x127f]);
        assert_eq!(scales.pair::<3>(1).map(ScalePairB::bits), [0x805f, 0x807f]);
    }

    /// Check compile-time fragment addressing agrees with the independent physical map for every warp/lane.
    #[test]
    fn const_fragment_store_addresses_match_the_physical_mapping_for_every_lane() {
        for warp in 0..8 {
            for lane in 0..32 {
                mma::static_for!(BAND in [0, 1] {
                    mma::static_for!(COL in [0, 1, 2, 3, 4, 5, 6, 7] {
                        assert_eq!(
                            output_fragment_row::<BAND, COL>(warp, lane),
                            physical::output_row(warp, lane, BAND, COL),
                        );
                    });
                });
            }
        }
    }
}

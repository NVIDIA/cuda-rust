/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 CuTe GEMM, emitted through the CUTLASS MLIR translation backend.
//! The two CTAs collectively compute 256x352, split into 256x192 and 256x160
//! MMA instructions sharing the A operand collector. Five AB stages overlap
//! TMA with MMA; four epilogue warps drain TMEM through two shared C stages.

use cuda_device::{
    DynamicSharedArray, cluster_launch, cuda_module, kernel, launch_bounds, launch_contract,
};
use cuda_device::{barrier::*, cluster, thread, tma::prefetch_tma_descriptor, warp};
use cute_rs::{
    Composed, OperandA, OperandB, RowMajor, Sm100AccumulatorPipeline, Sm100ClusterTmaCopy,
    Sm100SharedTile, Sm100TiledMma, Sm100TmaMmaPipeline, Sm100TmaStorePipeline, Sm100Tmem,
    Sm100TmemEpilogue, Swizzle, TmaDesc,
};

// These element layouts also encode the host TMA descriptors. FP16 shifts
// the swizzle's low-bit position by one when lowering to byte addresses.
pub type ASmem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
pub type B0Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
pub type B1Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
pub type CSmem = Composed<Swizzle<2, 3, 3>, 0, RowMajor<128, 32>>;

pub const THREADS: u32 = 192;
const STAGES: usize = 5;
const A_BYTES: usize = 128 * 64 * 2;
const B_BYTES: usize = 176 * 64 * 2;
const B_OFFSET: usize = A_BYTES * STAGES;
const C_OFFSET: usize = B_OFFSET + B_BYTES * STAGES;
const C_STAGE_BYTES: usize = 128 * 32 * 2;
const BARRIER_OFFSET: usize = C_OFFSET + 2 * C_STAGE_BYTES;
pub const DYNAMIC_SMEM_BYTES: usize = BARRIER_OFFSET + 128;
const _: () = assert!(DYNAMIC_SMEM_BYTES == 211_072);

type MainloopPipeline = Sm100TmaMmaPipeline<5, 77_824>;
type AccumulatorPipeline = Sm100AccumulatorPipeline<4>;
type TiledMma = Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>;

// Wait for both CTAs before freeing tensor memory. The CuTe pipelines below
// track when input tiles and accumulators can be reused.
#[inline(always)]
unsafe fn wait_deallocation(bar: *const Barrier) {
    unsafe { while !mbarrier_try_wait_parity_cluster(bar, 0) {} }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Compute output tiles with a (2,1,1) cluster and four typed tensor maps.
    ///
    /// Each map is passed by value; all threads borrow its read-only bytes.
    /// The tensor allocations must remain live until the stream finishes.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 211072,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    #[allow(clippy::too_many_arguments)] // Keep each tensor map explicit in the signature.
    pub fn fp16_gemm_256x352(
        #[grid_constant] a: &TmaDesc<f16, ASmem>,
        #[grid_constant] b0: &TmaDesc<f16, B0Smem>,
        #[grid_constant] b1: &TmaDesc<f16, B1Smem>,
        #[grid_constant] c: &TmaDesc<f16, CSmem>,
        bias: *const f16,
        m: u64,
        n: u64,
        k: u64,
        has_bias: u64,
    ) {
        // The CUTLASS launch ABI uses 64-bit scalar arguments. The host checks
        // that each dimension fits before the kernel converts it to u32.
        let m = m as u32;
        let n = n as u32;
        let k = k as u32;
        let has_bias = has_bias != 0;
        let tid = thread::threadIdx_x();
        let warp_id = tid / 32;
        let rank = cluster::block_rank();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        // Shared storage: A[5], B[5], C[2], full[5], empty[5], acc_empty,
        // acc_full, dealloc, tmem_token. Tile bases are 1024-byte aligned.
        unsafe {
            let full = smem.add(BARRIER_OFFSET).cast::<Barrier>();
            let empty = full.add(STAGES);
            let acc_empty = empty.add(STAGES);
            let acc_full = acc_empty.add(1);
            let dealloc = acc_full.add(1);
            let tmem_token = dealloc.add(1).cast::<u32>();
            let pipeline = MainloopPipeline::from_raw(full, empty);
            let accumulator = AccumulatorPipeline::from_raw(acc_empty, acc_full);
            let mma = TiledMma::new();
            let stores = Sm100TmaStorePipeline::<2>::new();
            if warp_id == 4 {
                prefetch_tma_descriptor((a as *const TmaDesc<f16, ASmem>).cast());
                prefetch_tma_descriptor((b0 as *const TmaDesc<f16, B0Smem>).cast());
                prefetch_tma_descriptor((b1 as *const TmaDesc<f16, B1Smem>).cast());
                prefetch_tma_descriptor((c as *const TmaDesc<f16, CSmem>).cast());
            }
            if warp_id == 0 && warp::is_elected_sync(u32::MAX) {
                mbarrier_init(dealloc, 32);
                accumulator.init();
                pipeline.init();
            }
            fence_mbarrier_init_release_cluster();
            cluster::barrier_cluster_arrive_relaxed();

            // Each cluster visits output tiles in M-first order.
            let mtiles = m / 256;
            let total_tiles = mtiles * n.div_ceil(352);
            let mut tile = thread::blockIdx_x() / 2;
            let tile_stride = thread::gridDim_x() / 2;
            let ktiles = k.div_ceil(64);
            cluster::barrier_cluster_wait();

            if warp_id == 5 {
                let mut stage = 0usize;
                let mut phase = 1u32;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let mut kt = 0;
                    while kt < ktiles {
                        pipeline.producer_acquire(stage as u32, phase);
                        if rank == 0 && warp::is_elected_sync(u32::MAX) {
                            pipeline.producer_expect(stage as u32);
                        }
                        // Native CuTe cluster TMA performs its own warp election.
                        // Every producer lane must participate in each copy.
                        let barrier = pipeline.full_barrier(stage as u32);
                        let mut a_tile = Sm100SharedTile::<ASmem, OperandA>::from_raw(
                            smem.add(stage * A_BYTES).cast(),
                        );
                        let mut b0_tile = Sm100SharedTile::<B0Smem, OperandB>::from_raw(
                            smem.add(B_OFFSET + stage * B_BYTES).cast(),
                        );
                        let mut b1_tile = Sm100SharedTile::<B1Smem, OperandB>::from_raw(
                            smem.add(B_OFFSET + stage * B_BYTES + 96 * 64 * 2).cast(),
                        );
                        Sm100ClusterTmaCopy::copy(a, &mut a_tile, row, kt * 64, barrier, rank);
                        Sm100ClusterTmaCopy::copy(
                            b0,
                            &mut b0_tile,
                            col + rank * 96,
                            kt * 64,
                            barrier,
                            rank,
                        );
                        Sm100ClusterTmaCopy::copy(
                            b1,
                            &mut b1_tile,
                            col + 192 + rank * 80,
                            kt * 64,
                            barrier,
                            rank,
                        );
                        stage += 1;
                        if stage == STAGES {
                            stage = 0;
                            phase ^= 1;
                        }
                        kt += 1;
                    }
                    tile += tile_stride;
                }
                // The next stage plus STAGES-1 identifies the last slot and
                // its release phase, including wraparound and short K tails.
                let mut tail = 0;
                while tail < STAGES - 1 {
                    stage += 1;
                    if stage == STAGES {
                        stage = 0;
                        phase ^= 1;
                    }
                    tail += 1;
                }
                if warp::is_elected_sync(u32::MAX) {
                    pipeline.producer_acquire(stage as u32, phase);
                }
            } else if warp_id == 4 {
                barrier_cta_sync(2, 160);
                let tmem = *tmem_token;
                let mut stage = 0usize;
                let mut phase = 0u32;
                let mut acc_phase = 1u32;
                while tile < total_tiles {
                    if rank == 0 {
                        accumulator.producer_acquire(acc_phase);
                        let mut accumulate = false;
                        let mut kt = 0;
                        while kt < ktiles {
                            pipeline.consumer_wait(stage as u32, phase);
                            let a_tile = Sm100SharedTile::<ASmem, OperandA>::from_raw(
                                smem.add(stage * A_BYTES).cast(),
                            );
                            let b0_tile = Sm100SharedTile::<B0Smem, OperandB>::from_raw(
                                smem.add(B_OFFSET + stage * B_BYTES).cast(),
                            );
                            let b1_tile = Sm100SharedTile::<B1Smem, OperandB>::from_raw(
                                smem.add(B_OFFSET + stage * B_BYTES + 96 * 64 * 2).cast(),
                            );
                            if warp::is_elected_sync(u32::MAX) {
                                // Each of the four K16 steps reuses A across
                                // the two N partitions, then releases it.
                                mma.gemm(tmem, &a_tile, &b0_tile, &b1_tile, accumulate);
                                pipeline.consumer_release(stage as u32);
                            }
                            accumulate = true;
                            stage += 1;
                            if stage == STAGES {
                                stage = 0;
                                phase ^= 1;
                            }
                            kt += 1;
                        }
                        if warp::is_elected_sync(u32::MAX) {
                            accumulator.producer_commit();
                        }
                    }
                    acc_phase ^= 1;
                    tile += tile_stride;
                }
                if rank == 0 && warp::is_elected_sync(u32::MAX) {
                    accumulator.producer_acquire(acc_phase);
                }
            } else {
                if warp_id == 0 {
                    Sm100Tmem::<512>::allocate(tmem_token);
                }
                barrier_cta_sync(2, 160);
                let tmem = *tmem_token;
                let mut phase = 0u32;
                let mut epi_stage = 0usize;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let row_bias = if has_bias {
                        *bias.add((row + tid) as usize) as f32
                    } else {
                        0.0
                    };
                    accumulator.consumer_wait(phase);
                    let mut subtile = 0u32;
                    while subtile < 11 {
                        epi_stage ^= 1;
                        let dst = smem.add(C_OFFSET + epi_stage * C_STAGE_BYTES);
                        Sm100TmemEpilogue::<CSmem>::store(
                            Sm100TmemEpilogue::<CSmem>::slice_address(tmem, warp_id, subtile),
                            dst.cast(),
                            tid,
                            row_bias,
                        );
                        fence_proxy_async_shared_cta();
                        barrier_cta_sync(1, 128);
                        if tid == 0 {
                            Sm100TmemEpilogue::<CSmem>::store_async(
                                c,
                                dst.cast(),
                                row,
                                col + subtile * 32,
                            );
                            stores.producer_commit();
                            stores.producer_acquire();
                        }
                        barrier_cta_sync(1, 128);
                        subtile += 1;
                    }
                    accumulator.consumer_release();
                    phase ^= 1;
                    tile += tile_stride;
                }
                if warp_id == 0 {
                    // Thread 0 owns every TMA store group.
                    if tid == 0 {
                        stores.producer_tail();
                    }
                    barrier_cta_sync(3, 32);
                    let peer = cluster::map_shared_rank(dealloc, rank ^ 1);
                    mbarrier_arrive_cluster(peer as u64);
                    wait_deallocation(dealloc);
                    Sm100Tmem::<512>::deallocate(tmem);
                }
            }
        }
    }
}

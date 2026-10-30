/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 CuTe GEMM, emitted through the CUTLASS MLIR translation backend.
//!
//! The two CTAs collectively compute a `256 × 352` output tile, split into
//! `256 × 192` and `256 × 160` MMA instructions sharing the A operand collector.
//! Five AB stages overlap cluster TMA with MMA; four epilogue warps drain TMEM
//! through two shared C stages:
//!
//! ```text
//! Global A/B ──► Cluster TMA (Warp 5) ──► Shared A/B [5 stages]
//!                                               │
//!                                               ▼  `PipelineTmaUmma<5, 77_824>`
//!                                         `Sm100TiledMma` (Warp 4, CTA 0)
//!                                               │
//!                                               ▼  `PipelineUmmaAsync<4>`
//! Global C   ◄── TMA Store (Thread 0) ◄── `Sm100TmemEpilogue` (Warps 0..3)
//! ```

use ctm_rs::mbarrier::Barrier;
use ctm_rs::memory::DynamicSharedArray;
use ctm_rs::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract, thread};
use cute_rs::arch::{
    NamedBarrier, block_idx_in_cluster, cluster_arrive_relaxed, cluster_wait, elect_one_sync,
    fence_view_async_shared, mbarrier_init_fence,
};
use cute_rs::sm100::PipelineTmaStore;
use cute_rs::tensor::partition::StridedTensor;
use cute_rs::{
    ComposedLayout, OperandA, OperandB, PipelineTmaUmma, PipelineUmmaAsync, RowMajor,
    Sm100ClusterTmaCopy, Sm100DeallocBarrier, Sm100SharedTile, Sm100TiledMma, Sm100Tmem,
    Sm100TmemEpilogue, Swizzle, TensorMap,
};

// These element layouts also encode the host TMA descriptors. FP16 shifts
// the swizzle's low-bit position by one when lowering to byte addresses.
/// Per-CTA 128x64 FP16 A tile with the shared swizzle used by its tensor map.
pub type ASmem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
/// Per-CTA 96x64 FP16 B tile feeding the 192-column MMA partition.
pub type B0Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
/// Per-CTA 80x64 FP16 B tile feeding the 160-column MMA partition.
pub type B1Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
/// Per-CTA 128x32 FP16 output tile with the B64 shared swizzle.
pub type CSmem = ComposedLayout<Swizzle<2, 3, 3>, 0, RowMajor<128, 32>>;

/// Threads required in each CTA by the kernel launch contract.
pub const THREADS: u32 = 192;
/// Number of A/B shared-memory stages overlapped by TMA and MMA.
const STAGES: usize = 5;
/// Shared bytes occupied by one CTA's A stage.
const A_BYTES: usize = 128 * 64 * 2;
/// Shared bytes occupied by both B partitions in one CTA stage.
const B_BYTES: usize = 176 * 64 * 2;
/// Byte offset of the staged B ring after all staged A tiles.
const B_OFFSET: usize = A_BYTES * STAGES;
/// Byte offset of the two shared output stages after the A/B rings.
const C_OFFSET: usize = B_OFFSET + B_BYTES * STAGES;
/// Shared bytes occupied by one FP16 output stage.
const C_STAGE_BYTES: usize = 128 * 32 * 2;
/// Byte offset of load, accumulator, and deallocation barriers after output storage.
const BARRIER_OFFSET: usize = C_OFFSET + 2 * C_STAGE_BYTES;
/// Total dynamic shared-memory allocation in bytes required per CTA.
pub const DYNAMIC_SMEM_BYTES: usize = BARRIER_OFFSET + 128;
/// Compile-time consistency check for the schedule's dynamic shared-memory reservation.
const _: () = assert!(DYNAMIC_SMEM_BYTES == 211_072);

/// Five-stage two-CTA TMA/MMA pipeline expecting 77,824 transaction bytes per stage.
type MainloopPipeline = PipelineTmaUmma<5, 77_824>;
/// Accumulator handoff between the MMA producer and four epilogue warps per CTA.
type AccumulatorPipeline = PipelineUmmaAsync<4>;
/// Paired 256x192 and 256x160 MMA instructions sharing the A collector.
type TiledMma = Sm100TiledMma<ASmem, B0Smem, B1Smem, 192, 160>;

/// Two-CTA SM100 GEMM entry point and its explicit launch/storage contract.
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
        #[grid_constant] a: &TensorMap<f16, ASmem>,
        #[grid_constant] b0: &TensorMap<f16, B0Smem>,
        #[grid_constant] b1: &TensorMap<f16, B1Smem>,
        #[grid_constant] c: &TensorMap<f16, CSmem>,
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
        let rank = block_idx_in_cluster();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        // Shared storage: A[5], B[5], C[2], full[5], empty[5], acc_empty,
        // acc_full, dealloc, tmem_token. Tile bases are 1024-byte aligned.
        unsafe {
            let bias_tensor = StridedTensor::from_raw_parts(bias, m as usize, 1);
            let full = smem.add(BARRIER_OFFSET).cast::<Barrier>();
            let empty = full.add(STAGES);
            let acc_empty = empty.add(STAGES);
            let acc_full = acc_empty.add(1);
            let dealloc_ptr = acc_full.add(1);
            let tmem_token = dealloc_ptr.add(1).cast::<u32>();
            let dealloc = Sm100DeallocBarrier::from_raw(dealloc_ptr);
            let pipeline = MainloopPipeline::from_raw(full, empty);
            let accumulator = AccumulatorPipeline::from_raw(acc_empty, acc_full);
            let mma = TiledMma::new();
            let stores = PipelineTmaStore::<2>::new();
            if warp_id == 4 {
                a.prefetch_descriptor();
                b0.prefetch_descriptor();
                b1.prefetch_descriptor();
                c.prefetch_descriptor();
            }
            if warp_id == 0 && elect_one_sync() {
                dealloc.init();
                accumulator.init();
                pipeline.init();
            }
            mbarrier_init_fence();
            cluster_arrive_relaxed();

            // Each cluster visits output tiles in M-first order.
            let mtiles = m / 256;
            let total_tiles = mtiles * n.div_ceil(352);
            let mut tile = thread::blockIdx_x() / 2;
            let tile_stride = thread::gridDim_x() / 2;
            let ktiles = k.div_ceil(64);
            cluster_wait();

            if warp_id == 5 {
                let mut stage = 0usize;
                let mut phase = 1u32;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let mut kt = 0;
                    while kt < ktiles {
                        pipeline.producer_acquire(stage as u32, phase);
                        if rank == 0 && elect_one_sync() {
                            pipeline.producer_expect_tx(stage as u32);
                        }
                        // Native CuTe cluster TMA performs its own warp election.
                        // Every producer lane must participate in each copy.
                        let barrier = pipeline.producer_get_barrier(stage as u32);
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
                if elect_one_sync() {
                    pipeline.producer_acquire(stage as u32, phase);
                }
            } else if warp_id == 4 {
                NamedBarrier::<2, 160>::sync();
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
                            if elect_one_sync() {
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
                        if elect_one_sync() {
                            accumulator.producer_commit();
                        }
                    }
                    acc_phase ^= 1;
                    tile += tile_stride;
                }
                if rank == 0 && elect_one_sync() {
                    accumulator.producer_acquire(acc_phase);
                }
            } else {
                if warp_id == 0 {
                    Sm100Tmem::<512>::alloc(tmem_token);
                }
                NamedBarrier::<2, 160>::sync();
                let tmem = *tmem_token;
                let mut phase = 0u32;
                let mut epi_stage = 0usize;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let row_bias = if has_bias {
                        bias_tensor.get_unchecked((row + tid) as usize) as f32
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
                        fence_view_async_shared();
                        NamedBarrier::<1, 128>::sync();
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
                        NamedBarrier::<1, 128>::sync();
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
                    dealloc.sync_cluster(rank);
                    Sm100Tmem::<512>::dealloc(tmem);
                }
            }
        }
    }
}

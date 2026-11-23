/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rubin (`sm_107a`) and Blackwell (`sm_100a`/`sm_103a`) persistent two-CTA `BFloat16` dense GEMM
//! (`C = A * B^T` with `FP32` accumulation) with 8-stage cluster TMA, 2-stage `TMEM` ping-pong,
//! and Programmatic Dependent Launch (`PDL`) in `cute-rs`.
//!
//! # Kernel Pipeline
//!
//! ```text
//!   PDL Prologue (when ENABLE_PDL = true):
//!     prefetch_descriptor(A, B) -> griddepcontrol_wait() -> griddepcontrol_launch_dependents()
//!
//!   Warp 5 (TMA Producer):  [ Stream A (64x64 BF16) & B (128x64 BF16) -> 8-stage SMEM Ring ]
//!   Warp 4 (MMA Consumer):  [ Tile t -> TMEM_0 (cols 0..256) ] [ Tile t+1 -> TMEM_1 (256..512) ]
//!   Warps 0..3 (Epilogue):                                     [ Load TMEM_0 -> BF16 -> GMEM C ]
//! ```

#![allow(unsafe_op_in_unsafe_fn)]

use cute_rs::arch::{
    NamedBarrier, block_idx_in_cluster, cluster_arrive_relaxed, cluster_wait, elect_one_sync,
    griddepcontrol_launch_dependents, griddepcontrol_wait, make_warp_uniform, mbarrier_init_fence,
    setmaxregister_decrease, setmaxregister_increase, tcgen05_fence_before_thread_sync, thread_idx,
    warp_idx,
};
use cute_rs::nvgpu::cpasync::TmaCopy;
use cute_rs::nvgpu::tcgen05::{MmaSs, SharedTileArray, TmemFragment, tmem_row_address};
use cute_rs::pipeline::sm100::{
    AsyncCompletion, AsyncCopyReuse, Consumer, Producer, RingCursor, TensorMemory, TwoCtaRing,
};
use cute_rs::tensor::partition::StridedTensorMut;
use cute_rs::utils::tmem_allocator::{Sm100DeallocBarrier, Sm100Tmem};
use cute_rs::{
    BFloat16, ColMajor, ComposedLayout, GlobalStoreTile, OperandA, OperandB,
    PersistentTileSchedulerParams, RowMajor, Swizzle, TensorMap, size_in_bytes,
};
use cutlass_primitives as prims;
use prims::mbarrier::Barrier;
use prims::memory::DynamicSharedArray;
use prims::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract};

/// Per-CTA `64 x 64` `BFloat16` K-major `A` activation tile with 128-byte swizzle (`8,192` B/stage).
/// Across the two-CTA `(2, 1, 1)` cluster, both CTAs cover `M = 128` rows by `K = 64` columns.
pub type ALayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>;

/// Per-CTA `128 x 64` `BFloat16` K-major `B` weight tile with 128-byte swizzle (`16,384` B/stage).
/// Across the two-CTA `(2, 1, 1)` cluster, both CTAs cover `N = 256` rows by `K = 64` columns.
pub type BLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;

/// Group-two `128 x 256 x 64` `BFloat16` Tensor Core (`tcgen05.MmaF16BF16Op`) tiled MMA.
/// Verifies at compile time that `ALayout` (`64x64`) and `BLayout` (`128x64`) match `(128, 256, 64)`.
pub type Bf16TiledMma = MmaSs<ALayout, BLayout, 128, 256, 64, 2, BFloat16>;

/// Persistent two-CTA `128 x 256` tile scheduler with `G = 8` M-tile L2 swizzling.
/// Keeps `1,024 x K` of `A` hot in L2 cache while sweeping `B` columns across SMs.
pub type TileScheduler = PersistentTileSchedulerParams<128, 256, 8, 2>;

/// Column-major `(64, 2) : (1, 64)` thread layout partitioning the `128` epilogue threads (`Warps 0..3`).
/// Maps flat thread index `tid` (`0..128`) to `(lane_row 0..64, col_half 0..2)` of `128` columns each.
pub type EpilogueThreadLayout = ColMajor<64, 2>;

/// Eight-stage two-CTA TMA-to-MMA shared-memory ring (`full[0..8]` and `empty[0..8]` barriers).
/// Overlaps 2-CTA cluster TMA loads of `A` and `B` with Tensor Core `tcgen05.mma` consumption.
pub type MainloopRing = TwoCtaRing<8, AsyncCompletion, AsyncCopyReuse>;

/// Two-stage `TMEM` accumulator ring (`TMEM_0` at `+0`, `TMEM_1` at `+128` per half) for ping-pong.
/// Allows `Warp 4` to accumulate Tile `t + 1` into stage `1` while `Warps 0..3` drain stage `0`.
pub type AccumulatorRing = TwoCtaRing<2, TensorMemory>;

/// Threads per CTA (`192` threads = `4` epilogue warps + `1` MMA warp + `1` TMA producer warp).
/// Matches the warp-specialized persistent schedule of CuTeDSL `PersistentDenseGemmKernel`.
pub const THREADS: u32 = 192;

/// Number of shared-memory input stages in the mainloop ring (`8` stages).
/// Uses `8 * (8 KB + 16 KB) = 192 KB` of dynamic shared memory per CTA.
const STAGES: u32 = 8;

/// Bytes occupied by one CTA's `A` tile stage (`8,192` bytes), derived from [`ALayout`].
/// Transferred by [`TmaCopy::<ALayout, 2, BFloat16>::load`] into the `A` shared-memory ring.
const A_BYTES: usize = size_in_bytes::<BFloat16, ALayout>() as usize;

/// Bytes occupied by one CTA's `B` tile stage (`16,384` bytes), derived from [`BLayout`].
/// Transferred by [`TmaCopy::<BLayout, 2, BFloat16>::load`] into the `B` shared-memory ring.
const B_BYTES: usize = size_in_bytes::<BFloat16, BLayout>() as usize;

/// Total cluster-wide TMA transaction bytes expected per stage across both CTAs (`49,152` bytes).
/// Announced to leader CTA 0's `full` barrier via `producer_expect_tx_cursor`.
const CLUSTER_STAGE_TX_BYTES: u32 = (2 * (A_BYTES + B_BYTES)) as u32;

/// Shared-memory byte offset of the 8-stage `B` ring (`8 * 8,192 = 65,536`).
/// Aligned to `1,024` bytes as required by the 128-byte swizzle period.
const B_OFFSET: usize = (STAGES as usize) * A_BYTES;

/// Shared-memory byte offset of the pipeline `mbarrier` slots and `TMEM` allocation token (`196,608`).
/// Holds `16` mainloop barriers, `4` accumulator barriers, `1` dealloc barrier, and `tmem_token`.
const CONTROL_OFFSET: usize = B_OFFSET + (STAGES as usize) * B_BYTES;

/// Total dynamic shared memory reserved per CTA (`196,864` bytes, `1,024`-byte aligned).
/// Leaves headroom below SM100/SM107 per-CTA shared-memory limits while fitting 8 TMA stages.
pub const DYNAMIC_SMEM_BYTES: u32 = (CONTROL_OFFSET + 256) as u32;

const _: () = {
    assert!(DYNAMIC_SMEM_BYTES == 196_864);
    assert!(CLUSTER_STAGE_TX_BYTES == 49_152);
};

/// Shared persistent two-CTA `BFloat16` GEMM schedule parameterized by `ENABLE_PDL`.
///
/// # Safety
/// Must be launched with `block = (192, 1, 1)`, `cluster = (2, 1, 1)`, `dynamic_shared = 196864`,
/// and valid 32-byte-aligned global buffers matching `(m, n, k)`.
#[inline(always)]
unsafe fn run_bf16_gemm<const ENABLE_PDL: bool>(
    a_tma: &TensorMap<BFloat16, ALayout>,
    b_tma: &TensorMap<BFloat16, BLayout>,
    c_out: *mut BFloat16,
    m: u32,
    n: u32,
    k: u32,
) {
    let (tid, _, _) = thread_idx();
    let warp_id = warp_idx();
    let rank = make_warp_uniform(block_idx_in_cluster());
    let smem = DynamicSharedArray::<u8, 1024>::get_raw();
    let bars = smem.add(CONTROL_OFFSET).cast::<Barrier>();
    let dealloc = Sm100DeallocBarrier::from_raw(bars.add(20));
    let token = bars.add(21).cast::<u32>();

    let mainloop = MainloopRing::from_raw(bars, 0);
    let acc_ring = AccumulatorRing::from_raw(bars, 16);

    if tid == 0 {
        a_tma.prefetch_descriptor();
        b_tma.prefetch_descriptor();
        mainloop.init(1, 1);
        acc_ring.init(1, 2);
        dealloc.init();
    }
    mbarrier_init_fence();
    cluster_arrive_relaxed();
    cluster_wait();

    if ENABLE_PDL {
        griddepcontrol_wait();
        griddepcontrol_launch_dependents();
    }

    let sched = TileScheduler::new_2d(m, n);
    let first = sched.first_tile();
    let stride = sched.stride();
    let tile_count = sched.tile_count();
    let ktiles = k / 64;

    if warp_id == 5 {
        setmaxregister_decrease::<40>();
        let a_tiles = SharedTileArray::<ALayout, OperandA, STAGES, BFloat16>::from_raw(smem.cast());
        let b_tiles = SharedTileArray::<BLayout, OperandB, STAGES, BFloat16>::from_raw(
            smem.add(B_OFFSET).cast(),
        );
        let mut cursor = RingCursor::<STAGES, Producer>::new();
        let mut tile = first;
        while tile < tile_count {
            let (m_tile, n_tile) = sched.tile_idx(tile);
            let (row, col) = sched.cta_tile_coord(m_tile, n_tile, rank);
            let mut kt = 0u32;
            while kt < ktiles {
                mainloop.acquire_cursor(cursor);
                if rank == 0 && elect_one_sync() {
                    mainloop.producer_expect_tx_cursor(cursor, CLUSTER_STAGE_TX_BYTES);
                }
                let bar = mainloop.full_at_cursor(cursor);
                let mut at = a_tiles.tile_unchecked(cursor.index());
                let mut bt = b_tiles.tile_unchecked(cursor.index());
                TmaCopy::<ALayout, 2, BFloat16>::load(
                    a_tma,
                    &mut at,
                    row,
                    (kt * 64) as i32,
                    bar,
                    rank,
                );
                TmaCopy::<BLayout, 2, BFloat16>::load(
                    b_tma,
                    &mut bt,
                    col,
                    (kt * 64) as i32,
                    bar,
                    rank,
                );
                cursor.advance();
                kt += 1;
            }
            tile += stride;
        }
        let mut tail = 0u32;
        while tail < STAGES - 1 {
            cursor.advance();
            tail += 1;
        }
        mainloop.acquire_cursor(cursor);
    } else if warp_id == 4 {
        setmaxregister_decrease::<40>();
        NamedBarrier::<2, 160>::sync();
        let tmem = *token;
        let mma = Bf16TiledMma::new();
        let a_tiles = SharedTileArray::<ALayout, OperandA, STAGES, BFloat16>::from_raw(smem.cast());
        let b_tiles = SharedTileArray::<BLayout, OperandB, STAGES, BFloat16>::from_raw(
            smem.add(B_OFFSET).cast(),
        );
        if rank == 0 {
            let mut smem_cursor = RingCursor::<STAGES, Consumer>::new();
            let mut acc_cursor = RingCursor::<2, Producer>::new();
            let mut tile = first;
            while tile < tile_count {
                acc_ring.acquire_cursor(acc_cursor);
                let dst_tmem = tmem + acc_cursor.index() * 128;
                let mut kt = 0u32;
                while kt < ktiles {
                    mainloop.wait_cursor(smem_cursor);
                    let at = a_tiles.tile_unchecked(smem_cursor.index());
                    let bt = b_tiles.tile_unchecked(smem_cursor.index());
                    if elect_one_sync() {
                        mma.gemm(dst_tmem, &at, &bt, kt != 0);
                        mainloop.mma_empty_cursor(smem_cursor);
                    }
                    smem_cursor.advance();
                    kt += 1;
                }
                if elect_one_sync() {
                    acc_ring.mma_full_cursor(acc_cursor);
                }
                acc_cursor.advance();
                tile += stride;
            }
            acc_cursor.advance();
            acc_ring.acquire_cursor(acc_cursor);
        }
    } else {
        setmaxregister_increase::<232>();
        if warp_id == 0 {
            Sm100Tmem::<512>::alloc(token);
        }
        NamedBarrier::<2, 160>::sync();
        let tmem = *token;
        let mut c_view =
            StridedTensorMut::<BFloat16>::from_raw_parts(c_out, (m as usize) * (n as usize), 1);
        let mut acc_cursor = RingCursor::<2, Consumer>::new();
        let mut tile = first;
        while tile < tile_count {
            let (m_tile, n_tile) = sched.tile_idx(tile);
            let (row, col_half) =
                sched.epilogue_thread_coord::<EpilogueThreadLayout>(m_tile, n_tile, rank, tid);
            acc_ring.wait_cursor(acc_cursor);
            let tmem_row = tmem_row_address(tmem, warp_id, acc_cursor.index() * 128);
            let fragment = TmemFragment::<128>::load(tmem_row);
            tcgen05_fence_before_thread_sync();
            NamedBarrier::<1, 128>::sync();
            if tid == 0 {
                acc_ring.thread_empty_cursor(acc_cursor);
            }

            let base_idx = (row as usize) * (n as usize) + (col_half as usize) * 128;
            let w0 = fragment
                .slice::<0, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w1 = fragment
                .slice::<16, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w2 = fragment
                .slice::<32, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w3 = fragment
                .slice::<48, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w4 = fragment
                .slice::<64, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w5 = fragment
                .slice::<80, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w6 = fragment
                .slice::<96, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            let w7 = fragment
                .slice::<112, 16>()
                .to::<BFloat16>()
                .bitcast::<u32, 8>()
                .into_array();
            c_view.store_global_no_allocate_words_at(base_idx, &GlobalStoreTile(w0));
            c_view.store_global_no_allocate_words_at(base_idx + 16, &GlobalStoreTile(w1));
            c_view.store_global_no_allocate_words_at(base_idx + 32, &GlobalStoreTile(w2));
            c_view.store_global_no_allocate_words_at(base_idx + 48, &GlobalStoreTile(w3));
            c_view.store_global_no_allocate_words_at(base_idx + 64, &GlobalStoreTile(w4));
            c_view.store_global_no_allocate_words_at(base_idx + 80, &GlobalStoreTile(w5));
            c_view.store_global_no_allocate_words_at(base_idx + 96, &GlobalStoreTile(w6));
            c_view.store_global_no_allocate_words_at(base_idx + 112, &GlobalStoreTile(w7));

            acc_cursor.advance();
            tile += stride;
        }
        if warp_id == 0 {
            dealloc.sync_cluster(rank);
            Sm100Tmem::<512>::dealloc(tmem);
        }
    }
}

/// Device entry points for both standard and Programmatic Dependent Launch (`PDL`) variants.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Persistent two-CTA `BFloat16` dense GEMM without `griddepcontrol`.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196864,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn sm107_bf16_gemm(
        #[grid_constant] a_tma: &TensorMap<BFloat16, ALayout>,
        #[grid_constant] b_tma: &TensorMap<BFloat16, BLayout>,
        c_out: *mut BFloat16,
        m: u64,
        n: u64,
        k: u64,
    ) {
        run_bf16_gemm::<false>(a_tma, b_tma, c_out, m as u32, n as u32, k as u32);
    }

    /// Persistent two-CTA `BFloat16` dense GEMM with `griddepcontrol` (`PDL`).
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 196864,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn sm107_bf16_gemm_pdl(
        #[grid_constant] a_tma: &TensorMap<BFloat16, ALayout>,
        #[grid_constant] b_tma: &TensorMap<BFloat16, BLayout>,
        c_out: *mut BFloat16,
        m: u64,
        n: u64,
        k: u64,
    ) {
        run_bf16_gemm::<true>(a_tma, b_tma, c_out, m as u32, n as u32, k as u32);
    }
}

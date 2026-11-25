/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vera Rubin (`sm_107a`) persistent `MXFP8` (`Float8E4M3FN` + `Float8E8M0FNU` `SF_VEC = 32`, `K = 64`)
//! block-scaled dense GEMM (`C = (A * SFA) * (B * SFB)^T` -> `BFloat16`) in `cute-rs`, comparing
//! single-M (`128 x 128`, `CollectorDiscard`) against Rubin hardware B-collector reuse
//! (`256 x 128`, `CollectorFill` + `CollectorLastUse`) with Programmatic Dependent Launch (`PDL`).
//!
//! # Rubin `SM107BlockScaledMmaMXF8F6F4Op` + B-Collector Reuse Pipeline
//!
//! ```text
//!   Each K = 128 TMA Stage (50,688 B/stage in B-Reuse Schedule):
//!     A0 (128x128 E4M3) + SFA0 (512 B UE8M0) ──┐
//!     A1 (128x128 E4M3) + SFA1 (512 B UE8M0) ──┼──► Warp 5 (PipelineTmaAsync) -> 3-Stage SMEM Ring
//!     B  (128x128 E4M3) + SFB  (512 B UE8M0) ──┘
//!
//!   Warp 4 (MMA Consumer, K = 64 per tcgen05.mma.kind::mxf8f6f4.block_scale):
//!     tcgen05.cp (SFA0, SFA1, SFB -> TMEM)
//!     for sub_k in [0, 1]:
//!       MmaBKeep::gemm (TMEM[  0..128], A0, B, SFA0, SFB, sub_k)  // CollectorOp::Fill
//!       MmaBReuse::gemm(TMEM[128..256], A1, B, SFA1, SFB, sub_k)  // CollectorOp::LastUse
//!
//!   Warps 0..3 (Epilogue):
//!     TmemFragment<128>::load -> TensorSSA::to::<BFloat16>() -> 256-bit Streaming Global Store
//! ```

#![allow(unsafe_op_in_unsafe_fn)]

use cute_rs::arch::{
    NamedBarrier, griddepcontrol_launch_dependents, griddepcontrol_wait, lane_idx,
    mbarrier_init_fence, setmaxregister_decrease, setmaxregister_increase,
    tcgen05_fence_after_thread_sync, thread_idx, warp_idx,
};
use cute_rs::nvgpu::tcgen05::{
    CollectorDiscard, CollectorFill, CollectorLastUse, SM107BlockScaledMmaMXF8F6F4Op,
    Sm100ScaleTile, Sm100SharedTile, TmemFragment, tmem_row_address,
};
use cute_rs::pipeline::sm100::{GeneralBarrier, SingleCtaRing};
use cute_rs::tensor::partition::StridedTensorMut;
use cute_rs::utils::tmem_allocator::TmemAllocator;
use cute_rs::{
    BFloat16, ColMajor, ComposedLayout, Consumer, Float8E4M3FN, GlobalStoreTile, Mkl, Nkl,
    OperandA, OperandB, PersistentTileSchedulerParams, PipelineState, PipelineTmaAsync, Producer,
    RowMajor, SmemTile, Swizzle, TensorMap, copy_tma_2d, crd2idx_col_major_2d, size_in_bytes,
};
use cutlass_primitives as prims;
use prims::memory::DynamicSharedArray;
use prims::{cuda_module, kernel, launch_bounds, launch_contract};

/// Per-M-half `128 x 128` `Float8E4M3FN` K-major tile with 128-byte swizzle (`16,384` bytes).
/// Covers `128` rows and `K = 128` (`2` Rubin `K = 64` `mxf8f6f4` MMA sub-steps).
pub type ATileSmem = ComposedLayout<Swizzle<3, 4, 3>, 0, RowMajor<128, 128>>;

/// `128 x 128` `Float8E4M3FN` K-major `B` tile with 128-byte swizzle (`16,384` bytes).
/// Reused across two `128`-row `A` halves in the `256 x 128` B-collector schedule.
pub type BTileSmem = ComposedLayout<Swizzle<3, 4, 3>, 0, RowMajor<128, 128>>;

/// `4 x 64` `u16` (`512`-byte) TMA transport layout for one `(32, 4, 4)` `UE8M0` scale atom.
/// Covers `128` rows by `4` scale groups (`4 * 32 = 128` `FP8` elements along `K`).
pub type ScaleAtomSmem = RowMajor<4, 64>;

/// Single-M (`128 x 128`) SM107 `MXF8F6F4` MMA atom discarding both A and B collectors.
pub type MmaNoReuse =
    SM107BlockScaledMmaMXF8F6F4Op<ATileSmem, BTileSmem, 128, 128, 1, CollectorDiscard>;

/// First-half (`M0 = 0..128`) SM107 `MXF8F6F4` MMA atom filling the hardware B-collector.
pub type MmaBKeep = SM107BlockScaledMmaMXF8F6F4Op<ATileSmem, BTileSmem, 128, 128, 1, CollectorFill>;

/// Second-half (`M1 = 128..256`) SM107 `MXF8F6F4` MMA atom consuming the retained B-collector.
pub type MmaBReuse =
    SM107BlockScaledMmaMXF8F6F4Op<ATileSmem, BTileSmem, 128, 128, 1, CollectorLastUse>;

/// Persistent `128 x 128` 1-CTA tile scheduler with `G = 8` M-tile L2 swizzling.
pub type Scheduler128 = PersistentTileSchedulerParams<128, 128, 8, 1>;

/// Persistent `256 x 128` 1-CTA tile scheduler with `G = 4` M-tile L2 swizzling for B-reuse.
pub type Scheduler256 = PersistentTileSchedulerParams<256, 128, 4, 1>;

/// Column-major `(128, 1)` thread layout mapping epilogue thread `tid` (`0..128`) to its row.
pub type EpilogueRowLayout = ColMajor<128, 1>;

/// Threads per CTA (`192` = `4` epilogue warps + `1` MMA warp + `1` TMA producer warp).
pub const THREADS: u32 = 192;

/// Number of TMA stages in the `128 x 128` baseline schedule (`6` stages).
const STAGES_128: usize = 6;
/// Bytes per `A` tile (`16,384` bytes), derived from [`ATileSmem`].
const A_BYTES: usize = size_in_bytes::<Float8E4M3FN, ATileSmem>() as usize;
/// Bytes per `B` tile (`16,384` bytes), derived from [`BTileSmem`].
const B_BYTES: usize = size_in_bytes::<Float8E4M3FN, BTileSmem>() as usize;
/// Bytes per `512`-byte `(32, 4, 4)` `UE8M0` scale atom, derived from [`ScaleAtomSmem`].
const SF_BYTES: usize = size_in_bytes::<u16, ScaleAtomSmem>() as usize;

/// Per-stage TMA transaction byte count for the `128 x 128` baseline (`33,792` bytes).
const TX_BYTES_128: u32 = (A_BYTES + B_BYTES + 2 * SF_BYTES) as u32;
/// Typed 6-stage TMA load pipeline (`33,792` B/stage) verified by `dialect-cute`.
type Pipeline128 = PipelineTmaAsync<STAGES_128, 1, TX_BYTES_128>;
/// Shared-memory offset of the 6-stage `B` ring in the `128 x 128` baseline (`98,304`).
const B_OFF_128: usize = STAGES_128 * A_BYTES;
/// Shared-memory offset of the 6-stage `SFA` ring in the `128 x 128` baseline (`196,608`).
const SFA_OFF_128: usize = B_OFF_128 + STAGES_128 * B_BYTES;
/// Shared-memory offset of the 6-stage `SFB` ring in the `128 x 128` baseline (`199,680`).
const SFB_OFF_128: usize = SFA_OFF_128 + STAGES_128 * SF_BYTES;
/// Shared-memory offset of barriers and `tmem_token` in the `128 x 128` baseline (`202,752`).
const CTRL_OFF_128: usize = SFB_OFF_128 + STAGES_128 * SF_BYTES;
/// Total dynamic shared memory for the `128 x 128` baseline (`203,008` bytes).
pub const DYNAMIC_SMEM_128: u32 = (CTRL_OFF_128 + 256) as u32;

/// Number of TMA stages in the `256 x 128` B-collector reuse schedule (`4` stages).
const STAGES_256: usize = 4;
/// Per-stage TMA transaction byte count for `256 x 128` B-reuse (`50,688` bytes: `2*A + B + 2*SFA + SFB`).
const TX_BYTES_256: u32 = (2 * A_BYTES + B_BYTES + 3 * SF_BYTES) as u32;
/// Typed 4-stage TMA load pipeline (`50,688` B/stage) verified by `dialect-cute`.
type Pipeline256 = PipelineTmaAsync<STAGES_256, 1, TX_BYTES_256>;
/// Shared-memory offset of `A1` ring in `256 x 128` B-reuse (`65,536`).
const A1_OFF_256: usize = STAGES_256 * A_BYTES;
/// Shared-memory offset of shared `B` ring in `256 x 128` B-reuse (`131,072`).
const B_OFF_256: usize = A1_OFF_256 + STAGES_256 * A_BYTES;
/// Shared-memory offset of `SFA0` ring in `256 x 128` B-reuse (`196,608`).
const SFA0_OFF_256: usize = B_OFF_256 + STAGES_256 * B_BYTES;
/// Shared-memory offset of `SFA1` ring in `256 x 128` B-reuse (`198,656`).
const SFA1_OFF_256: usize = SFA0_OFF_256 + STAGES_256 * SF_BYTES;
/// Shared-memory offset of `SFB` ring in `256 x 128` B-reuse (`200,704`).
const SFB_OFF_256: usize = SFA1_OFF_256 + STAGES_256 * SF_BYTES;
/// Shared-memory offset of barriers and `tmem_token` in `256 x 128` B-reuse (`202,752`).
const CTRL_OFF_256: usize = SFB_OFF_256 + STAGES_256 * SF_BYTES;
/// Total dynamic shared memory for the `256 x 128` B-reuse kernel (`203,008` bytes).
pub const DYNAMIC_SMEM_256: u32 = (CTRL_OFF_256 + 256) as u32;

const _: () = {
    assert!(TX_BYTES_128 == 33_792 && DYNAMIC_SMEM_128 == 203_008);
    assert!(TX_BYTES_256 == 50_688 && DYNAMIC_SMEM_256 == 203_008);
};

/// Store one `128 x 128` `FP32` `TMEM` accumulator slice at `col_offset` to `c_view` as `BFloat16`.
///
/// # Safety
/// All `128` epilogue lanes (`Warps 0..3`) must participate after waiting on the accumulator barrier.
#[inline(always)]
unsafe fn store_tmem_tile_128x128(
    tmem_base: u32,
    warp_id: u32,
    col_offset: u32,
    row_global: usize,
    col_global: usize,
    n_cols: usize,
    c_view: &mut StridedTensorMut<'_, BFloat16>,
) {
    let tmem_row = tmem_row_address(tmem_base, warp_id, col_offset);
    let fragment = TmemFragment::<128>::load(tmem_row);
    let base_idx = row_global * n_cols + col_global;
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
}

/// Rubin (`sm_107a`) `MXFP8` block-scaled GEMM entry points (`128x128` baseline and `256x128` B-reuse + PDL).
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Baseline `128 x 128 x 128` Rubin `MXFP8` (`E4M3 + UE8M0 SF=32`) persistent GEMM (`CollectorDiscard`)
    /// with 6-stage `PipelineTmaAsync` and 2-stage `TMEM` ping-pong.
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 203008,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 7),
    )]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sm107_mxfp8_gemm(
        #[grid_constant] a_tma: &TensorMap<Float8E4M3FN, ATileSmem>,
        #[grid_constant] b_tma: &TensorMap<Float8E4M3FN, BTileSmem>,
        #[grid_constant] sfa_tma: &TensorMap<u16, ScaleAtomSmem>,
        #[grid_constant] sfb_tma: &TensorMap<u16, ScaleAtomSmem>,
        c_out: *mut BFloat16,
        m: u64,
        n: u64,
        k: u64,
    ) {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let (tid, _, _) = thread_idx();
        let lane = lane_idx();
        let warp_id = warp_idx();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let pipeline = Pipeline128::from_shared_bytes(smem, CTRL_OFF_128);
        let smem_free_ring =
            SingleCtaRing::<6, GeneralBarrier>::from_raw(smem, CTRL_OFF_128 / 8 + 12);
        let acc_ring = SingleCtaRing::<2, GeneralBarrier>::from_raw(smem, CTRL_OFF_128 / 8 + 24);
        let tmem_slot = TmemAllocator::<512, 1>::from_raw_bytes(smem, CTRL_OFF_128 + 232);

        if tid == 0 {
            a_tma.prefetch_descriptor();
            b_tma.prefetch_descriptor();
            sfa_tma.prefetch_descriptor();
            sfb_tma.prefetch_descriptor();
            smem_free_ring.init(1, 1);
            acc_ring.init(1, 1);
        }
        if warp_id == 0 {
            TmemAllocator::<512, 1>::allocate(tmem_slot);
            TmemAllocator::<512, 1>::relinquish_alloc_permit();
        }
        mbarrier_init_fence();
        pipeline.init(0);

        let sched = Scheduler128::new_2d(m, n);
        let first = sched.first_tile();
        let stride = sched.stride();
        let tile_count = sched.tile_count();
        let k_stages = k / 128;

        if warp_id == 5 {
            setmaxregister_decrease::<40>();
            if lane == 0 {
                let mut prod = PipelineState::<Producer, STAGES_128>::new();
                let mut step = 0u32;
                let mut tile = first;
                while tile < tile_count {
                    let (m_tile, n_tile) = sched.tile_idx(tile);
                    let mut kt = 0u32;
                    while kt < k_stages {
                        if step >= STAGES_128 as u32 {
                            smem_free_ring.wait(step - STAGES_128 as u32);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let slot = prod.index();
                        let mut a_tile =
                            SmemTile::<Float8E4M3FN, ATileSmem>::from_stage_bytes(smem, 0, slot);
                        let mut b_tile = SmemTile::<Float8E4M3FN, BTileSmem>::from_stage_bytes(
                            smem, B_OFF_128, slot,
                        );
                        let mut sfa_tile = SmemTile::<u16, ScaleAtomSmem>::from_stage_bytes(
                            smem,
                            SFA_OFF_128,
                            slot,
                        );
                        let mut sfb_tile = SmemTile::<u16, ScaleAtomSmem>::from_stage_bytes(
                            smem,
                            SFB_OFF_128,
                            slot,
                        );
                        let sfa_atom = (m_tile * k_stages + kt) as usize;
                        let sfb_atom = (n_tile * k_stages + kt) as usize;
                        copy_tma_2d(a_tma, (m_tile as usize, kt as usize), &mut a_tile, bar);
                        copy_tma_2d(b_tma, (n_tile as usize, kt as usize), &mut b_tile, bar);
                        copy_tma_2d(sfa_tma, (sfa_atom, 0), &mut sfa_tile, bar);
                        copy_tma_2d(sfb_tma, (sfb_atom, 0), &mut sfb_tile, bar);
                        prod.advance();
                        step += 1;
                        kt += 1;
                    }
                    tile += stride;
                }
                pipeline.producer_tail(prod);
            }
        } else if warp_id == 4 {
            setmaxregister_decrease::<40>();
            let tmem = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot);
            let mma = MmaNoReuse::new();
            let mut cons = PipelineState::<Consumer, STAGES_128>::new();
            let mut step = 0u32;
            let mut acc_ticket = 0u32;
            let mut tile = first;
            while tile < tile_count {
                acc_ring.acquire(acc_ticket);
                let dst_tmem = tmem + (acc_ticket & 1) * 128;
                let mut kt = 0u32;
                while kt < k_stages {
                    pipeline.consumer_wait(&cons);
                    tcgen05_fence_after_thread_sync();
                    let slot = cons.index();
                    let sfa_col = tmem + 256 + (slot as u32) * 4;
                    let sfb_col = tmem + 280 + (slot as u32) * 4;
                    let a_tile =
                        Sm100SharedTile::<ATileSmem, OperandA>::from_stage_bytes(smem, 0, slot);
                    let b_tile = Sm100SharedTile::<BTileSmem, OperandB>::from_stage_bytes(
                        smem, B_OFF_128, slot,
                    );
                    let sfa_tile =
                        Sm100ScaleTile::<Mkl>::from_stage_bytes(smem, SFA_OFF_128, SF_BYTES, slot);
                    let sfb_tile =
                        Sm100ScaleTile::<Nkl>::from_stage_bytes(smem, SFB_OFF_128, SF_BYTES, slot);
                    if lane == 0 {
                        MmaNoReuse::copy(sfa_col, &sfa_tile, 0);
                        MmaNoReuse::copy(sfb_col, &sfb_tile, 0);
                        mma.gemm(dst_tmem, &a_tile, &b_tile, sfa_col, sfb_col, 0, kt != 0);
                        mma.gemm(dst_tmem, &a_tile, &b_tile, sfa_col, sfb_col, 1, true);
                        smem_free_ring.producer_commit(step);
                    }
                    pipeline.consumer_release(&cons);
                    cons.advance();
                    step += 1;
                    kt += 1;
                }
                acc_ring.commit_mma(acc_ticket);
                acc_ticket += 1;
                tile += stride;
            }
        } else {
            setmaxregister_increase::<232>();
            let tmem = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot);
            let mut c_view =
                StridedTensorMut::<BFloat16>::from_raw_parts(c_out, (m as usize) * (n as usize), 1);
            let mut acc_ticket = 0u32;
            let mut tile = first;
            while tile < tile_count {
                let (m_tile, n_tile) = sched.tile_idx(tile);
                let (row, col_tile) =
                    sched.epilogue_thread_coord::<EpilogueRowLayout>(m_tile, n_tile, 0, tid);
                acc_ring.wait(acc_ticket);
                tcgen05_fence_after_thread_sync();
                store_tmem_tile_128x128(
                    tmem,
                    warp_id,
                    (acc_ticket & 1) * 128,
                    row as usize,
                    (col_tile as usize) * 128,
                    n as usize,
                    &mut c_view,
                );
                NamedBarrier::<2, 128>::sync();
                if tid == 0 {
                    acc_ring.release_threads(acc_ticket);
                }
                acc_ticket += 1;
                tile += stride;
            }
            NamedBarrier::<2, 128>::sync();
            if warp_id == 0 {
                TmemAllocator::<512, 1>::free(tmem_slot);
            }
        }
    }

    /// `256 x 128 x 128` Rubin `MXFP8` (`E4M3 + UE8M0 SF=32`) persistent GEMM with hardware
    /// B-collector reuse (`CollectorFill` + `CollectorLastUse`) and Programmatic Dependent Launch (`PDL`).
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 203008,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 7),
    )]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sm107_mxfp8_gemm_breuse_pdl(
        #[grid_constant] a_tma: &TensorMap<Float8E4M3FN, ATileSmem>,
        #[grid_constant] b_tma: &TensorMap<Float8E4M3FN, BTileSmem>,
        #[grid_constant] sfa_tma: &TensorMap<u16, ScaleAtomSmem>,
        #[grid_constant] sfb_tma: &TensorMap<u16, ScaleAtomSmem>,
        c_out: *mut BFloat16,
        m: u64,
        n: u64,
        k: u64,
    ) {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let (tid, _, _) = thread_idx();
        let lane = lane_idx();
        let warp_id = warp_idx();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let pipeline = Pipeline256::from_shared_bytes(smem, CTRL_OFF_256);
        let smem_free_ring =
            SingleCtaRing::<4, GeneralBarrier>::from_raw(smem, CTRL_OFF_256 / 8 + 8);
        let acc_ring = SingleCtaRing::<1, GeneralBarrier>::from_raw(smem, CTRL_OFF_256 / 8 + 16);
        let tmem_slot = TmemAllocator::<512, 1>::from_raw_bytes(smem, CTRL_OFF_256 + 192);

        if tid == 0 {
            a_tma.prefetch_descriptor();
            b_tma.prefetch_descriptor();
            sfa_tma.prefetch_descriptor();
            sfb_tma.prefetch_descriptor();
            smem_free_ring.init(1, 1);
            acc_ring.init(1, 1);
        }
        if warp_id == 0 {
            TmemAllocator::<512, 1>::allocate(tmem_slot);
            TmemAllocator::<512, 1>::relinquish_alloc_permit();
        }
        mbarrier_init_fence();
        pipeline.init(0);

        griddepcontrol_wait();
        griddepcontrol_launch_dependents();

        let sched = Scheduler256::new_2d(m, n);
        let first = sched.first_tile();
        let stride = sched.stride();
        let tile_count = sched.tile_count();
        let k_stages = k / 128;

        if warp_id == 5 {
            setmaxregister_decrease::<40>();
            if lane == 0 {
                let mut prod = PipelineState::<Producer, STAGES_256>::new();
                let mut step = 0u32;
                let mut tile = first;
                while tile < tile_count {
                    let (m_tile256, n_tile) = sched.tile_idx(tile);
                    let m0 = (m_tile256 * 2) as usize;
                    let m1 = m0 + 1;
                    let mut kt = 0u32;
                    while kt < k_stages {
                        if step >= STAGES_256 as u32 {
                            smem_free_ring.wait(step - STAGES_256 as u32);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let slot = prod.index();
                        let mut a0_tile =
                            SmemTile::<Float8E4M3FN, ATileSmem>::from_stage_bytes(smem, 0, slot);
                        let mut a1_tile = SmemTile::<Float8E4M3FN, ATileSmem>::from_stage_bytes(
                            smem, A1_OFF_256, slot,
                        );
                        let mut b_tile = SmemTile::<Float8E4M3FN, BTileSmem>::from_stage_bytes(
                            smem, B_OFF_256, slot,
                        );
                        let mut sfa0_tile = SmemTile::<u16, ScaleAtomSmem>::from_stage_bytes(
                            smem,
                            SFA0_OFF_256,
                            slot,
                        );
                        let mut sfa1_tile = SmemTile::<u16, ScaleAtomSmem>::from_stage_bytes(
                            smem,
                            SFA1_OFF_256,
                            slot,
                        );
                        let mut sfb_tile = SmemTile::<u16, ScaleAtomSmem>::from_stage_bytes(
                            smem,
                            SFB_OFF_256,
                            slot,
                        );
                        let ks = k_stages as usize;
                        let ktu = kt as usize;
                        copy_tma_2d(a_tma, (m0, ktu), &mut a0_tile, bar);
                        copy_tma_2d(a_tma, (m1, ktu), &mut a1_tile, bar);
                        copy_tma_2d(b_tma, (n_tile as usize, ktu), &mut b_tile, bar);
                        copy_tma_2d(sfa_tma, (m0 * ks + ktu, 0), &mut sfa0_tile, bar);
                        copy_tma_2d(sfa_tma, (m1 * ks + ktu, 0), &mut sfa1_tile, bar);
                        copy_tma_2d(
                            sfb_tma,
                            ((n_tile as usize) * ks + ktu, 0),
                            &mut sfb_tile,
                            bar,
                        );
                        prod.advance();
                        step += 1;
                        kt += 1;
                    }
                    tile += stride;
                }
                pipeline.producer_tail(prod);
            }
        } else if warp_id == 4 {
            setmaxregister_decrease::<40>();
            let tmem = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot);
            let mma_keep = MmaBKeep::new();
            let mma_reuse = MmaBReuse::new();
            let mut cons = PipelineState::<Consumer, STAGES_256>::new();
            let mut step = 0u32;
            let mut acc_ticket = 0u32;
            let mut tile = first;
            while tile < tile_count {
                acc_ring.acquire(acc_ticket);
                let mut kt = 0u32;
                while kt < k_stages {
                    pipeline.consumer_wait(&cons);
                    tcgen05_fence_after_thread_sync();
                    let slot = cons.index();
                    let sfa0_col = tmem + 256 + (slot as u32) * 4;
                    let sfa1_col = tmem + 272 + (slot as u32) * 4;
                    let sfb_col = tmem + 288 + (slot as u32) * 4;
                    let a0_tile =
                        Sm100SharedTile::<ATileSmem, OperandA>::from_stage_bytes(smem, 0, slot);
                    let a1_tile = Sm100SharedTile::<ATileSmem, OperandA>::from_stage_bytes(
                        smem, A1_OFF_256, slot,
                    );
                    let b_tile = Sm100SharedTile::<BTileSmem, OperandB>::from_stage_bytes(
                        smem, B_OFF_256, slot,
                    );
                    let sfa0_tile =
                        Sm100ScaleTile::<Mkl>::from_stage_bytes(smem, SFA0_OFF_256, SF_BYTES, slot);
                    let sfa1_tile =
                        Sm100ScaleTile::<Mkl>::from_stage_bytes(smem, SFA1_OFF_256, SF_BYTES, slot);
                    let sfb_tile =
                        Sm100ScaleTile::<Nkl>::from_stage_bytes(smem, SFB_OFF_256, SF_BYTES, slot);
                    if lane == 0 {
                        MmaBKeep::copy(sfa0_col, &sfa0_tile, 0);
                        MmaBKeep::copy(sfa1_col, &sfa1_tile, 0);
                        MmaBKeep::copy(sfb_col, &sfb_tile, 0);
                        mma_keep.gemm(tmem, &a0_tile, &b_tile, sfa0_col, sfb_col, 0, kt != 0);
                        mma_reuse.gemm(
                            tmem + 128,
                            &a1_tile,
                            &b_tile,
                            sfa1_col,
                            sfb_col,
                            0,
                            kt != 0,
                        );
                        mma_keep.gemm(tmem, &a0_tile, &b_tile, sfa0_col, sfb_col, 1, true);
                        mma_reuse.gemm(tmem + 128, &a1_tile, &b_tile, sfa1_col, sfb_col, 1, true);
                        smem_free_ring.producer_commit(step);
                    }
                    pipeline.consumer_release(&cons);
                    cons.advance();
                    step += 1;
                    kt += 1;
                }
                acc_ring.commit_mma(acc_ticket);
                acc_ticket += 1;
                tile += stride;
            }
        } else {
            setmaxregister_increase::<232>();
            let tmem = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot);
            let mut c_view =
                StridedTensorMut::<BFloat16>::from_raw_parts(c_out, (m as usize) * (n as usize), 1);
            let mut acc_ticket = 0u32;
            let mut tile = first;
            while tile < tile_count {
                let (m_tile256, n_tile) = sched.tile_idx(tile);
                let row0 = crd2idx_col_major_2d(tid, m_tile256, 256) as usize;
                let row1 = row0 + 128;
                let col = crd2idx_col_major_2d(0, n_tile, 128) as usize;
                acc_ring.wait(acc_ticket);
                tcgen05_fence_after_thread_sync();
                store_tmem_tile_128x128(tmem, warp_id, 0, row0, col, n as usize, &mut c_view);
                store_tmem_tile_128x128(tmem, warp_id, 128, row1, col, n as usize, &mut c_view);
                NamedBarrier::<2, 128>::sync();
                if tid == 0 {
                    acc_ring.release_threads(acc_ticket);
                }
                acc_ticket += 1;
                tile += stride;
            }
            NamedBarrier::<2, 128>::sync();
            if warp_id == 0 {
                TmemAllocator::<512, 1>::free(tmem_slot);
            }
        }
    }
}

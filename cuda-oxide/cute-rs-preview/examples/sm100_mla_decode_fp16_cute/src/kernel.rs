/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paged latent attention, retaining the public CUTLASS v4.7 two-CTA schedule.
//! QK0, QK-next/PV-current, PV-last overlap with softmax and output correction.
//! All ring tickets continue across persistent work; resetting phases is invalid.
//! CuTe supplies typed MMA/TMA operations and pipeline policies, while CTM
//! supplies their explicit instructions, shared storage, and launch macros.

#![allow(unsafe_op_in_unsafe_fn)]

use ctm_rs::f32x2::{
    add_f32x2, fma_f32x2, mul_f32x2, pack_f32x2, unpack_f32x2_hi, unpack_f32x2_lo,
};
use ctm_rs::mbarrier::{Barrier, GeneralBarrier, ManagedBarrier, Ready};
use ctm_rs::memory::{DynamicSharedArray, SharedAddress, U32x4};
use ctm_rs::tcgen05::tmem::{self, TmemGuard, TmemReady, TmemUninit};
use ctm_rs::{
    async_copy, cluster, cluster_launch, convert, cuda_module, float, groups::CtaGroup2, kernel,
    launch_bounds, launch_contract, mbarrier, tcgen05, thread, warp,
};
use cute_rs::arch::assume_warp_uniform;
use cute_rs::attention::sm100::pipeline::{
    AsyncCompletion, AsyncCopyReuse, Consumer, LocalMemory, LocalShared, Producer, RingCursor,
    SharedReuse, SharedToAsync, TensorMemory, TwoCtaRing,
};
use cute_rs::attention::sm100::{
    MmaSs, SharedLayout, SharedTile, SharedTileArray, TmaCopy, TmemFragment,
};
use cute_rs::{OperandA, OperandB, TmaDesc};

/// Per-CTA 64-by-64 FP16 query tile, stored K-major for QK MMA.
pub type QLayout = SharedLayout<64, 64>;
/// Per-CTA 64-by-64 FP16 cache-key tile, stored K-major for QK MMA.
pub type KLayout = SharedLayout<64, 64>;
/// Per-CTA 64-by-32 FP16 probability chunk consumed by PV MMA.
pub type PLayout = SharedLayout<64, 32>;
/// Per-CTA 128-by-32 MN-major FP16 value chunk consumed by PV MMA.
pub type VLayout = SharedLayout<128, 32, true>;
/// Group-two M128/N128/K64 QK instruction using shared query and key tiles.
type QkMma = MmaSs<QLayout, KLayout, 128, 128, 64, 2>;
/// Group-two M128/N256/K32 PV instruction using shared probability and value tiles.
type PvMma = MmaSs<PLayout, VLayout, 128, 256, 32, 2>;

/// Threads in each persistent CTA, including dedicated load, MMA and reduction warps.
pub const THREADS: u32 = 384;
/// Required dynamic shared-memory bytes per CTA, with a 1024-byte-aligned base.
pub const DYNAMIC_SMEM_BYTES: u32 = 232_448;
/// Shared byte offset of eight latent-query chunks.
const Q: usize = 0;
/// Shared byte offset of the rotary-query chunk following the latent query.
const QR: usize = 65_536;
/// Shared byte offset of fifteen reusable key/value stages, each 8192 bytes.
const KV: usize = 73_728;
/// Shared byte offset of two probability buffers, each containing four K32 chunks.
const P: usize = 196_608;
/// Shared byte offset of four page-index slots, spaced 256 bytes apart.
const PT: usize = 229_376;
/// Shared byte offset of two 128-element FP32 softmax-maximum exchange buffers.
const MAX_EXCHANGE: usize = 230_400;
/// Shared byte offset of 128 FP32 partial sums used during final normalization.
const SUM_EXCHANGE: usize = 231_424;
/// Shared byte offset of the 54 full/empty mbarriers backing all pipeline rings.
const BARS: usize = 231_936;
/// Shared byte offset of the u32 slot receiving this CTA’s TMEM allocation address.
const TOKEN: usize = BARS + 54 * 8;
/// Convert the natural-exponential softmax scale to the base-2 instruction scale.
const LOG2_E: f32 = core::f32::consts::LOG2_E;

// Each pipeline carries its stage count and full/empty memory protocol in its type.
/// Single query buffer: TMA fills it, then MMA completion releases it for reuse.
type QueryRing = TwoCtaRing<1, AsyncCompletion, AsyncCopyReuse>;
/// Fifteen shared stages reused by TMA producers after MMA finishes reading each tile.
type KvRing = TwoCtaRing<15, AsyncCompletion, AsyncCopyReuse>;
/// Two TMEM score buffers passed from QK MMA to softmax threads.
type ScoreRing = TwoCtaRing<2, TensorMemory>;
/// Two shared probability buffers published by softmax and released after PV MMA.
type ProbabilityRing = TwoCtaRing<2, SharedToAsync, SharedReuse>;
/// Two local metadata buffers passed from softmax to output-correction threads.
type CorrectionRing = TwoCtaRing<2, LocalMemory>;
/// Single TMEM output accumulator handed between PV MMA and correction threads.
type OutputRing = TwoCtaRing<1, TensorMemory>;
/// Four local shared page-index slots filled by asynchronous copies and read by the load warp.
type PageRing = TwoCtaRing<4, LocalShared>;

/// Select the larger score or log-sum-exp value with Rust FP32 max semantics.
#[inline(always)]
fn maxf(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// A ternary reduction exposes independent maximum operations.
#[inline(always)]
fn score_max_ternary(values: &[f32; 64], previous: f32) -> f32 {
    /// Reduce three scores while exposing a balanced ternary maximum tree.
    #[inline(always)]
    fn max3(a: f32, b: f32, c: f32) -> f32 {
        a.max(b.max(c))
    }

    let mut first = [0.0f32; 21];
    let mut i = 0;
    while i < 21 {
        thread::__unroll_config::<0>();
        first[i] = max3(values[i * 3], values[i * 3 + 1], values[i * 3 + 2]);
        i += 1;
    }
    let mut second = [0.0f32; 7];
    i = 0;
    while i < 7 {
        thread::__unroll_config::<0>();
        second[i] = max3(first[i * 3], first[i * 3 + 1], first[i * 3 + 2]);
        i += 1;
    }
    max3(
        max3(second[0], second[1], second[2]),
        max3(second[3], second[4], second[5]),
        max3(second[6], values[63], previous),
    )
}

/// Flatten row, split and latent column to a usize FP32 workspace element offset.
#[inline(always)]
pub(crate) fn partial_offset(row: u32, split: u32, splits: u32, column: u32) -> usize {
    (row as usize * splits as usize + split as usize) * 512 + column as usize
}

/// Pack a warp’s 32-row group and a column displacement into a live TMEM base address.
#[inline(always)]
fn row_address(tmem: u32, warp: u32, columns: u32) -> u32 {
    tmem + ((warp * 32) << 16) + columns
}

/// Consume nine key chunks and publish one QK score tile, advancing persistent tickets.
///
/// # Safety
/// The leader CTA’s issuing warp must participate with initialized rings, live shared/TMEM storage and a completed query load.
#[inline(always)]
unsafe fn issue_qk(
    smem: *mut u8,
    tmem: u32,
    kv: KvRing,
    score: ScoreRing,
    kv_state: &mut RingCursor<15, Consumer>,
    score_ticket: &mut u32,
) {
    score.acquire(*score_ticket);
    let mma = QkMma::new();
    let queries = SharedTileArray::<QLayout, OperandA, 9>::from_raw(smem.add(Q).cast());
    let keys = SharedTileArray::<KLayout, OperandB, 15>::from_raw(smem.add(KV).cast());
    let mut chunk = 0;
    while chunk < 9 {
        kv.wait_cursor(*kv_state);
        let a = queries.tile_unchecked(chunk as u32);
        let b = keys.tile_unchecked(kv_state.index());
        if warp::is_elected_sync(u32::MAX) {
            mma.gemm(tmem + (*score_ticket % 2) * 64, &a, &b, chunk != 0);
            kv.mma_empty_cursor(*kv_state);
        }
        kv_state.advance();
        chunk += 1;
    }
    if warp::is_elected_sync(u32::MAX) {
        score.mma_full(*score_ticket);
    }
    *score_ticket += 1;
}

/// Consume eight value chunks and one probability buffer, then publish the updated output accumulator.
///
/// # Safety
/// The leader CTA’s issuing warp must participate with live storage and matching ring tickets. Set first only before the first accumulation.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // Independent pipeline tickets make ownership explicit.
unsafe fn issue_pv(
    smem: *mut u8,
    tmem: u32,
    kv: KvRing,
    probability: ProbabilityRing,
    output: OutputRing,
    kv_state: &mut RingCursor<15, Consumer>,
    p_ticket: &mut u32,
    o_ticket: &mut u32,
    first: bool,
) {
    output.acquire(*o_ticket);
    probability.wait(*p_ticket);
    let mma = PvMma::new();
    let probabilities = SharedTileArray::<PLayout, OperandA, 8>::from_raw(smem.add(P).cast());
    let values = SharedTileArray::<VLayout, OperandB, 15>::from_raw(smem.add(KV).cast());
    let mut kpart = 0;
    while kpart < 4 {
        let mut npart = 0;
        while npart < 2 {
            kv.wait_cursor(*kv_state);
            let a = probabilities.tile_unchecked((*p_ticket % 2) * 4 + kpart as u32);
            let b = values.tile_unchecked(kv_state.index());
            if warp::is_elected_sync(u32::MAX) {
                mma.gemm(
                    tmem + 128 + npart as u32 * 128,
                    &a,
                    &b,
                    !first || kpart != 0,
                );
                kv.mma_empty_cursor(*kv_state);
            }
            kv_state.advance();
            npart += 1;
        }
        kpart += 1;
    }
    if warp::is_elected_sync(u32::MAX) {
        probability.mma_empty(*p_ticket);
        output.mma_full(*o_ticket);
    }
    *p_ticket += 1;
    *o_ticket += 1;
}

/// Load one page’s nine key chunks and, on the first tile, the corresponding latent and rotary queries.
///
/// # Safety
/// Both CTAs’ load warps must supply their rank, matching live descriptors, shared storage and initialized rings.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // Typed maps and separate pipeline lifetimes are explicit.
unsafe fn load_qk(
    smem: *mut u8,
    q: &TmaDesc<f16, QLayout>,
    qr: &TmaDesc<f16, QLayout>,
    k: &TmaDesc<f16, KLayout>,
    kr: &TmaDesc<f16, KLayout>,
    query: QueryRing,
    kv: KvRing,
    pages: PageRing,
    q_ticket: u32,
    kv_state: &mut RingCursor<15, Producer>,
    page_ticket: u32,
    batch: u32,
    rank: u32,
    first: bool,
) {
    pages.wait(page_ticket);
    let page = *smem
        .add(PT + (page_ticket % 4) as usize * 256)
        .cast::<i32>();
    if first {
        query.acquire(q_ticket);
        if rank == 0 && warp::is_elected_sync(u32::MAX) {
            query.expect(q_ticket, 147456);
        }
    }
    if first {
        let mut chunk = 0usize;
        while chunk < 9 {
            let mut destination = SharedTile::<QLayout, OperandA>::from_raw(
                smem.add(if chunk < 8 { Q + chunk * 8192 } else { QR })
                    .cast(),
            );
            let descriptor = if chunk < 8 { q } else { qr };
            TmaCopy::<QLayout, 2>::load(
                descriptor,
                &mut destination,
                (batch * 128 + rank * 64) as i32,
                if chunk < 8 { chunk as i32 * 64 } else { 0 },
                query.full_at(q_ticket),
                rank,
            );
            chunk += 1;
        }
    }
    let mut chunk = 0usize;
    while chunk < 9 {
        kv.acquire_cursor(*kv_state);
        if rank == 0 && warp::is_elected_sync(u32::MAX) {
            kv.expect_cursor(*kv_state, 16384);
        }
        let mut destination = SharedTile::<KLayout, OperandB>::from_raw(
            smem.add(KV + kv_state.index() as usize * 8192).cast(),
        );
        TmaCopy::<KLayout, 2>::load(
            if chunk < 8 { k } else { kr },
            &mut destination,
            page * 128 + rank as i32 * 64,
            if chunk < 8 { chunk as i32 * 64 } else { 0 },
            kv.full_at_cursor(*kv_state),
            rank,
        );
        kv_state.advance();
        chunk += 1;
    }
}

/// Release the consumed page-index slot and load its eight value chunks into the KV ring.
///
/// # Safety
/// Both load warps must have waited for this page ticket and use matching rank, descriptor, storage and KV cursors.
#[inline(always)]
unsafe fn load_v(
    smem: *mut u8,
    v: &TmaDesc<f16, VLayout>,
    kv: KvRing,
    pages: PageRing,
    kv_state: &mut RingCursor<15, Producer>,
    page_ticket: u32,
    rank: u32,
) {
    let page = *smem
        .add(PT + (page_ticket % 4) as usize * 256)
        .cast::<i32>();
    // Every lane now owns its page index in a register; release the slot
    // before issuing V copies so the page producer can advance.
    mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
        pages.empty_at(page_ticket),
    ));
    let mut kpart = 0;
    while kpart < 4 {
        let mut npart = 0;
        while npart < 2 {
            kv.acquire_cursor(*kv_state);
            if rank == 0 && warp::is_elected_sync(u32::MAX) {
                kv.expect_cursor(*kv_state, 16384);
            }
            let mut destination = SharedTile::<VLayout, OperandB>::from_raw(
                smem.add(KV + kv_state.index() as usize * 8192).cast(),
            );
            TmaCopy::<VLayout, 2>::load(
                v,
                &mut destination,
                page * 128 + kpart * 32,
                npart * 256 + rank as i32 * 128,
                kv.full_at_cursor(*kv_state),
                rank,
            );
            kv_state.advance();
            npart += 1;
        }
        kpart += 1;
    }
}

/// Softmax's tickets span persistent work; only the row statistics reset per work item.
struct SoftmaxState {
    /// Next score buffer consumed; continues across persistent work items.
    score_ticket: u32,
    /// Next probability buffer produced; also selects its maximum-exchange buffer.
    probability_ticket: u32,
    /// Next metadata buffer published to output-correction threads.
    correction_ticket: u32,
    /// Running unscaled QK maximum for this row and split.
    maximum: f32,
    /// Running softmax denominator relative to the current row maximum.
    sum: f32,
}

/// Specialize the last-tile metadata handoff without changing the reduction order.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // The three typed pipelines have separate lifetimes.
unsafe fn softmax_tile<const LAST: bool>(
    smem: *mut u8,
    tmem: u32,
    tid: u32,
    wid: u32,
    lane: u32,
    tile: u32,
    length: u32,
    scale: f32,
    score: ScoreRing,
    probability: ProbabilityRing,
    correction: CorrectionRing,
    state: &mut SoftmaxState,
) {
    probability.acquire(state.probability_ticket);
    correction.acquire(state.correction_ticket);
    score.wait(state.score_ticket);
    let address = row_address(tmem, wid, (state.score_ticket % 2) * 64);
    let mut values = TmemFragment::<64>::load_deferred(address);
    let mut i = 0;
    // A nonlast local tile is at most ceil(length / 128) - 2, so it is full.
    // Only the final local tile can also be the partial final global tile.
    if LAST && tile * 128 + 128 > length {
        while i < 64 {
            thread::__unroll_config::<0>();
            if tile * 128 + (wid / 2) * 64 + i as u32 >= length {
                values.values[i] = f32::NEG_INFINITY;
            }
            i += 1;
        }
    }
    let mut new_maximum = score_max_ternary(&values.values, state.maximum);
    // Two exchange banks follow the P ring. Reacquiring P(i+2)
    // observes PV(i) completion, after every P(i) writer
    // has consumed max(i), including across persistent work.
    let exchange = smem
        .add(MAX_EXCHANGE + (state.probability_ticket % 2) as usize * 512)
        .cast::<f32>();
    *exchange.add(tid as usize) = new_maximum;
    mbarrier::barrier_cta_sync(2, 128);
    new_maximum = maxf(new_maximum, *exchange.add(((tid + 64) % 128) as usize));
    let alpha = float::ex2_approx_ftz_f32((state.maximum - new_maximum) * scale);
    let mut metadata = TmemFragment::<4> {
        values: [
            state.sum,
            new_maximum,
            alpha,
            if new_maximum == state.maximum {
                1.0
            } else {
                0.0
            },
        ],
    };
    if !LAST {
        metadata.store(row_address(
            tmem,
            wid,
            384 + (state.correction_ticket % 2) * 4,
        ));
        tcgen05::fence_before_thread_sync();
        mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
            correction.full_at(state.correction_ticket),
        ));
    }
    let row = lane + (wid % 2) * 32;
    let destination = smem
        .add(P + (state.probability_ticket % 2) as usize * 16384)
        .cast::<f16>();
    i = 0;
    while i < 64 {
        thread::__unroll_config::<0>();
        // Match the source's 128-bit universal P copy.
        // B64 swizzling preserves each aligned group of eight FP16 values.
        let mut packed = [0u32; 4];
        let mut pair = 0;
        while pair < 4 {
            thread::__unroll_config::<0>();
            let scores = fma_f32x2(
                pack_f32x2(values.values[i + pair * 2], values.values[i + pair * 2 + 1]),
                pack_f32x2(scale, scale),
                pack_f32x2(-new_maximum * scale, -new_maximum * scale),
            );
            let x = float::ex2_approx_ftz_f32(unpack_f32x2_lo(scores));
            let y = float::ex2_approx_ftz_f32(unpack_f32x2_hi(scores));
            values.values[i + pair * 2] = x;
            values.values[i + pair * 2 + 1] = y;
            packed[pair] = convert::cvt_f16x2_f32(x, y);
            pair += 1;
        }
        let key = (wid / 2) * 64 + i as u32;
        *destination
            .add((key / 32) as usize * 2048 + PLayout::offset(row, key % 32))
            .cast::<U32x4>() = U32x4(packed);
        i += 8;
    }
    probability.thread_full(state.probability_ticket);
    // Match the source: publish P before reducing the FP32 probabilities.
    let mut pair_sum = pack_f32x2(0.0, 0.0);
    i = 0;
    while i < 64 {
        thread::__unroll_config::<0>();
        pair_sum = add_f32x2(pair_sum, pack_f32x2(values.values[i], values.values[i + 1]));
        i += 2;
    }
    state.sum = unpack_f32x2_lo(pair_sum) + unpack_f32x2_hi(pair_sum) + state.sum * alpha;
    state.maximum = new_maximum;
    if LAST {
        metadata.values[0] = state.sum;
        metadata.store(row_address(
            tmem,
            wid,
            384 + (state.correction_ticket % 2) * 4,
        ));
        tcgen05::fence_before_thread_sync();
        mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
            correction.full_at(state.correction_ticket),
        ));
    }
    tmem::load_wait();
    score.thread_empty(state.score_ticket);
    state.score_ticket += 1;
    state.probability_ticket += 1;
    state.correction_ticket += 1;
}

/// Persistent two-CTA decode and the optional split-reduction kernel with host launch wrappers.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Decode fixed-length paged MLA; kernel math and schedule remain Rust code.
    #[kernel]
    #[launch_bounds(384, 1)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(domain=1,block=(384,1,1),dynamic_shared=232448,
                      dynamic_shared_alignment=1024,min_compute_capability=(10,0))]
    #[allow(clippy::too_many_arguments)]
    pub fn sm100_mla_decode_fp16_cute_rs(
        #[grid_constant] q: &TmaDesc<f16, QLayout>,
        #[grid_constant] qr: &TmaDesc<f16, QLayout>,
        #[grid_constant] k: &TmaDesc<f16, KLayout>,
        #[grid_constant] kr: &TmaDesc<f16, KLayout>,
        #[grid_constant] v: &TmaDesc<f16, VLayout>,
        page_table: *const i32,
        out: *mut f16,
        lse: *mut f32,
        partial_out: *mut f32,
        partial_lse: *mut f32,
        batch: u64,
        kv_length: u64,
        splits: u64,
        softmax_scale_bits: u64,
        output_scale_bits: u64,
    ) {
        unsafe {
            let batch = batch as u32;
            let length = kv_length as u32;
            let splits = splits as u32;
            let scale = f32::from_bits(softmax_scale_bits as u32) * LOG2_E;
            let output_scale = f32::from_bits(output_scale_bits as u32);
            let tid = thread::threadIdx_x();
            let wid = assume_warp_uniform(tid / 32);
            let lane = tid % 32;
            let rank = assume_warp_uniform(cluster::block_rank());
            let smem = DynamicSharedArray::<u8, 1024>::get_raw();
            let bars = smem.add(BARS).cast::<Barrier>();
            let query = QueryRing::from_raw(bars, 0);
            let kv = KvRing::from_raw(bars, 2);
            let score = ScoreRing::from_raw(bars, 32);
            let probability = ProbabilityRing::from_raw(bars, 36);
            let correction = CorrectionRing::from_raw(bars, 40);
            let output = OutputRing::from_raw(bars, 44);
            let pages = PageRing::from_raw(bars, 46);
            if tid == 0 {
                q.prefetch();
                qr.prefetch();
                k.prefetch();
                kr.prefetch();
                v.prefetch();
                query.init(1, 1);
                kv.init(1, 1);
                score.init(1, 256);
                probability.init(256, 1);
                correction.init(128, 128);
                output.init(1, 256);
                pages.init(32, 32);
            }
            mbarrier::fence_mbarrier_init_release_cluster();
            cluster::barrier_cluster_arrive_relaxed();
            cluster::barrier_cluster_wait();
            let token = smem.add(TOKEN).cast::<u32>();
            if wid == 8 {
                TmemGuard::<TmemUninit, 512, CtaGroup2>::from_raw_slot(token).alloc_collective();
                tcgen05::relinquish_alloc_permit_cg2();
            }
            if wid < 9 {
                mbarrier::barrier_cta_sync(1, 288);
            }
            let mut work = assume_warp_uniform(thread::blockIdx_x() / 2);
            let stride = assume_warp_uniform(thread::gridDim_x() / 2);
            let total_tiles = assume_warp_uniform(length.div_ceil(128));
            let per_split = assume_warp_uniform(total_tiles.div_ceil(splits));
            let mut q_ticket = 0;
            let mut s_ticket = 0;
            let mut p_ticket = 0;
            let mut c_ticket = 0;
            let mut o_ticket = 0;
            let mut page_ticket = 0;
            // Keep each register budget live through its complete role, as in
            // the reference. Reconverging before role dispatch forces spills.
            if wid == 10 {
                thread::setmaxnreg_dec::<96>();
                while work < batch * splits {
                    let b = work % batch;
                    let split = work / batch;
                    let first_tile = split * per_split;
                    let end_tile = (first_tile + per_split).min(total_tiles);
                    let count = end_tile.saturating_sub(first_tile);
                    if count > 0 {
                        let mut tile = first_tile;
                        while tile < end_tile {
                            pages.acquire(page_ticket);
                            let dst = smem
                                .add(PT + (page_ticket % 4) as usize * 256)
                                .cast::<u32>();
                            if lane == 0 {
                                async_copy::cp_async_ca_4(
                                    SharedAddress::from_raw(dst),
                                    &*page_table.add((b * total_tiles + tile) as usize).cast(),
                                );
                            }
                            async_copy::cp_async_mbarrier_arrive_noinc(&ManagedBarrier::<
                                Ready,
                                GeneralBarrier,
                            >::adopt_ready(
                                pages.full_at(page_ticket),
                            ));
                            page_ticket += 1;
                            tile += 1;
                        }
                    }
                    work += stride;
                }
            } else if wid == 9 {
                thread::setmaxnreg_dec::<96>();
                let mut kv_producer = RingCursor::<15, Producer>::new();
                while work < batch * splits {
                    let b = work % batch;
                    let split = work / batch;
                    let first_tile = split * per_split;
                    let end_tile = (first_tile + per_split).min(total_tiles);
                    let count = end_tile.saturating_sub(first_tile);
                    if count > 0 {
                        load_qk(
                            smem,
                            q,
                            qr,
                            k,
                            kr,
                            query,
                            kv,
                            pages,
                            q_ticket,
                            &mut kv_producer,
                            page_ticket,
                            b,
                            rank,
                            true,
                        );
                        let mut tile = 1;
                        while tile < count {
                            load_qk(
                                smem,
                                q,
                                qr,
                                k,
                                kr,
                                query,
                                kv,
                                pages,
                                q_ticket,
                                &mut kv_producer,
                                page_ticket + 1,
                                b,
                                rank,
                                false,
                            );
                            load_v(smem, v, kv, pages, &mut kv_producer, page_ticket, rank);
                            page_ticket += 1;
                            tile += 1;
                        }
                        load_v(smem, v, kv, pages, &mut kv_producer, page_ticket, rank);
                        page_ticket += 1;
                        q_ticket += 1;
                    }
                    work += stride;
                }
            } else if wid == 8 && rank == 0 {
                thread::setmaxnreg_dec::<96>();
                let mut kv_consumer = RingCursor::<15, Consumer>::new();
                while work < batch * splits {
                    let _b = work % batch;
                    let split = work / batch;
                    let first_tile = split * per_split;
                    let end_tile = (first_tile + per_split).min(total_tiles);
                    let count = end_tile.saturating_sub(first_tile);
                    if count > 0 {
                        let tmem = *token;
                        query.wait(q_ticket);
                        issue_qk(smem, tmem, kv, score, &mut kv_consumer, &mut s_ticket);
                        let mut tile = 1;
                        while tile < count {
                            issue_qk(smem, tmem, kv, score, &mut kv_consumer, &mut s_ticket);
                            issue_pv(
                                smem,
                                tmem,
                                kv,
                                probability,
                                output,
                                &mut kv_consumer,
                                &mut p_ticket,
                                &mut o_ticket,
                                tile == 1,
                            );
                            tile += 1;
                        }
                        if warp::is_elected_sync(u32::MAX) {
                            query.mma_empty(q_ticket);
                        }
                        q_ticket += 1;
                        issue_pv(
                            smem,
                            tmem,
                            kv,
                            probability,
                            output,
                            &mut kv_consumer,
                            &mut p_ticket,
                            &mut o_ticket,
                            count == 1,
                        );
                    }
                    work += stride;
                }
            } else if wid < 4 {
                thread::setmaxnreg_inc::<192>();
                let mut state = SoftmaxState {
                    score_ticket: s_ticket,
                    probability_ticket: p_ticket,
                    correction_ticket: c_ticket,
                    maximum: f32::NEG_INFINITY,
                    sum: 0.0,
                };
                while work < batch * splits {
                    let _b = work % batch;
                    let split = work / batch;
                    let first_tile = split * per_split;
                    let end_tile = (first_tile + per_split).min(total_tiles);
                    let count = end_tile.saturating_sub(first_tile);
                    if count > 0 {
                        let tmem = *token;
                        state.maximum = f32::NEG_INFINITY;
                        state.sum = 0.0;
                        let mut tile = first_tile;
                        while tile + 1 < end_tile {
                            softmax_tile::<false>(
                                smem,
                                tmem,
                                tid,
                                wid,
                                lane,
                                tile,
                                length,
                                scale,
                                score,
                                probability,
                                correction,
                                &mut state,
                            );
                            tile += 1;
                        }
                        softmax_tile::<true>(
                            smem,
                            tmem,
                            tid,
                            wid,
                            lane,
                            tile,
                            length,
                            scale,
                            score,
                            probability,
                            correction,
                            &mut state,
                        );
                    }
                    work += stride;
                }
            } else if wid < 8 {
                thread::setmaxnreg_inc::<208>();
                while work < batch * splits {
                    let b = work % batch;
                    let split = work / batch;
                    let first_tile = split * per_split;
                    let end_tile = (first_tile + per_split).min(total_tiles);
                    let count = end_tile.saturating_sub(first_tile);
                    if count > 0 {
                        let tmem = *token;
                        let cw = wid - 4;
                        let ct = tid - 128;
                        let row = lane + (cw % 2) * 32;
                        let head = rank * 64 + row;
                        let mut tile = 0;
                        while tile < count {
                            correction.wait(c_ticket);
                            let metadata = TmemFragment::<4>::load(row_address(
                                tmem,
                                cw,
                                384 + (c_ticket % 2) * 4,
                            ));
                            tcgen05::fence_before_thread_sync();
                            mbarrier::arrive(
                                &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                                    correction.empty_at(c_ticket),
                                ),
                            );
                            c_ticket += 1;
                            if tile > 0 {
                                output.wait(o_ticket);
                                if !warp::all(metadata.values[3] == 1.0) {
                                    let mut part = 0;
                                    #[unroll]
                                    while part < 8 {
                                        let address = row_address(tmem, cw, 128 + part * 32);
                                        let mut values = TmemFragment::<32>::load_deferred(address);
                                        let mut i = 0;
                                        #[unroll]
                                        while i < 32 {
                                            let pair = mul_f32x2(
                                                pack_f32x2(values.values[i], values.values[i + 1]),
                                                pack_f32x2(metadata.values[2], metadata.values[2]),
                                            );
                                            values.values[i] = unpack_f32x2_lo(pair);
                                            values.values[i + 1] = unpack_f32x2_hi(pair);
                                            i += 2;
                                        }
                                        values.store_deferred(address);
                                        part += 1;
                                    }
                                }
                                tmem::store_wait();
                                output.thread_empty(o_ticket);
                                o_ticket += 1;
                            }
                            if tile + 1 == count {
                                let exchange = smem.add(SUM_EXCHANGE).cast::<f32>();
                                *exchange.add(ct as usize) = metadata.values[0];
                                mbarrier::barrier_cta_sync(3, 128);
                                let denominator =
                                    metadata.values[0] + *exchange.add(((ct + 64) % 128) as usize);
                                mbarrier::barrier_cta_sync(3, 128);
                                output.wait(o_ticket);
                                let normalization = float::rcp_approx_ftz_f32(denominator);
                                let mut part = 0;
                                #[unroll]
                                while part < 8 {
                                    let mut values = TmemFragment::<32>::load_deferred(
                                        row_address(tmem, cw, 128 + part * 32),
                                    );
                                    let mut i = 0;
                                    #[unroll]
                                    while i < 32 {
                                        // The source rounds scale and normalization separately.
                                        let scaled = mul_f32x2(
                                            pack_f32x2(values.values[i], values.values[i + 1]),
                                            pack_f32x2(output_scale, output_scale),
                                        );
                                        let normalized = mul_f32x2(
                                            scaled,
                                            pack_f32x2(normalization, normalization),
                                        );
                                        values.values[i] = unpack_f32x2_lo(normalized);
                                        values.values[i + 1] = unpack_f32x2_hi(normalized);
                                        i += 2;
                                    }
                                    if splits == 1 {
                                        let mut i = 0;
                                        #[unroll]
                                        while i < 32 {
                                            let column = (part / 4) * 256
                                                + (cw / 2) * 128
                                                + (part % 4) * 32
                                                + i as u32;
                                            let mut packed = [0u32; 8];
                                            let mut pair = 0;
                                            #[unroll]
                                            while pair < 8 {
                                                packed[pair] = convert::cvt_f16x2_f32(
                                                    values.values[i + pair * 2],
                                                    values.values[i + pair * 2 + 1],
                                                );
                                                pair += 1;
                                            }
                                            cute_rs::store_global_no_allocate(
                                                out.add(((b * 128 + head) * 512 + column) as usize)
                                                    .cast::<u32>(),
                                                &cute_rs::GlobalStoreTile(packed),
                                            );
                                            i += 16;
                                        }
                                    } else {
                                        let mut i = 0;
                                        #[unroll]
                                        while i < 32 {
                                            let column = (part / 4) * 256
                                                + (cw / 2) * 128
                                                + (part % 4) * 32
                                                + i as u32;
                                            let mut packed = [0.0f32; 8];
                                            let mut element = 0;
                                            #[unroll]
                                            while element < 8 {
                                                packed[element] = values.values[i + element];
                                                element += 1;
                                            }
                                            cute_rs::store_global_no_allocate(
                                                partial_out.add(partial_offset(
                                                    b * 128 + head,
                                                    split,
                                                    splits,
                                                    column,
                                                )),
                                                &cute_rs::GlobalStoreTile(packed),
                                            );
                                            i += 8;
                                        }
                                    }
                                    part += 1;
                                }
                                if cw < 2 {
                                    let value = float::lg2_approx_f32(denominator)
                                        + scale * metadata.values[1];
                                    if splits == 1 {
                                        *lse.add((b * 128 + head) as usize) = value;
                                    } else {
                                        *partial_lse
                                            .add(((b * 128 + head) * splits + split) as usize) =
                                            value;
                                    }
                                }
                                tmem::load_wait();
                                output.thread_empty(o_ticket);
                                o_ticket += 1;
                            }
                            tile += 1;
                        }
                    }
                    work += stride;
                }
            } else {
                thread::setmaxnreg_dec::<96>();
            }
            // No shared or TMEM storage can be reclaimed before every pipeline
            // participant in both CTAs completes its final work item.
            cluster::barrier_cluster_arrive();
            cluster::barrier_cluster_wait();
            if wid == 8 {
                TmemGuard::<TmemReady, 512, CtaGroup2>::adopt_ready(token).dealloc_collective();
            }
        }
    }

    /// Merge normalized split outputs using base-2 log-sum-exp weights.
    #[kernel]
    #[launch_bounds(128, 1)]
    #[launch_contract(domain=1,block=(128,1,1),dynamic_shared=1024,
                      dynamic_shared_alignment=16,min_compute_capability=(10,0))]
    pub fn sm100_mla_decode_fp16_reduce_cute_rs(
        partial_out: *const f32,
        partial_lse: *const f32,
        out: *mut f16,
        lse: *mut f32,
        active_splits: u64,
        splits: u64,
    ) {
        unsafe {
            let row = thread::blockIdx_x();
            let tid = thread::threadIdx_x();
            let splits = splits as u32;
            // The fixed-length host contract supplies the nonempty split count.
            let active = active_splits as u32;
            let weights = DynamicSharedArray::<f32, 16>::get_raw().cast::<f32>();
            if tid < 32 {
                // The source retains all eight LSE candidates per lane in registers.
                let mut local_lse = [f32::NEG_INFINITY; 8];
                let mut maximum = f32::NEG_INFINITY;
                let mut i = 0usize;
                #[unroll]
                while i < 8 {
                    let split = tid + i as u32 * 32;
                    if split < active {
                        local_lse[i] = *partial_lse.add((row * splits + split) as usize);
                    }
                    maximum = maxf(maximum, local_lse[i]);
                    i += 1;
                }
                let mut mask = 16;
                while mask > 0 {
                    maximum = maxf(
                        maximum,
                        f32::from_bits(warp::shuffle_xor(maximum.to_bits(), mask)),
                    );
                    mask /= 2;
                }
                let mut sum = 0.0;
                i = 0;
                #[unroll]
                while i < 8 {
                    sum += float::ex2_approx_ftz_f32(local_lse[i] - maximum);
                    i += 1;
                }
                mask = 16;
                while mask > 0 {
                    sum += f32::from_bits(warp::shuffle_xor(sum.to_bits(), mask));
                    mask /= 2;
                }
                let global_lse = maximum + float::lg2_approx_f32(sum);
                if tid == 0 {
                    *lse.add(row as usize) = global_lse;
                }
                i = 0;
                #[unroll]
                while i < 8 {
                    let split = tid + i as u32 * 32;
                    if split < active {
                        *weights.add(split as usize) =
                            float::ex2_approx_ftz_f32(local_lse[i] - global_lse);
                    }
                    i += 1;
                }
            }
            mbarrier::barrier_cta_sync(4, 128);
            // Share each split weight across four independent output accumulators.
            // Each accumulator retains the reference's increasing-split sum order.
            let mut values = [0.0f32; 4];
            let mut split = 0;
            while split < active {
                let weight = *weights.add(split as usize);
                let mut element = 0;
                #[unroll]
                while element < 4 {
                    let c = tid + element as u32 * 128;
                    values[element] +=
                        *partial_out.add(partial_offset(row, split, splits, c)) * weight;
                    element += 1;
                }
                split += 1;
            }
            let mut element = 0;
            #[unroll]
            while element < 4 {
                let c = tid + element as u32 * 128;
                *out.add((row * 512 + c) as usize) = values[element] as f16;
                element += 1;
            }
        }
    }
}

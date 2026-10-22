/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-CTA FMHA schedule corresponding to CUTLASS v4.7.0 fmha.py.
//! Two independent softmax groups exchange the aliased S/P tensor-memory
//! regions. A separate correction group rescales the running output before
//! the MMA warp accumulates the next FP16 probability tile.
//! CuTe supplies typed MMA/TMA operations; CTM supplies explicit instructions,
//! shared storage, and the kernel launch macros.

use core::marker::PhantomData;
use ctm_rs::f32x2::{
    PackedF32x2, add_f32x2, fma_f32x2, mul_f32x2, pack_f32x2, unpack_f32x2_hi, unpack_f32x2_lo,
};
use ctm_rs::mbarrier::{Barrier, GeneralBarrier, ManagedBarrier, Ready, TmaBarrier, Uninit};
use ctm_rs::memory::{DynamicSharedArray, U32x4};
use ctm_rs::tcgen05::tmem::{self, TmemGuard, TmemReady, TmemUninit};
use ctm_rs::{
    barrier, convert, cuda_module, fence, float, kernel, launch_bounds, launch_contract, mbarrier,
    tcgen05, thread, warp,
};
use cute_rs::attention::sm100::{MmaSs, MmaTs, SharedLayout, SharedTile, TmaCopy, TmemFragment};
use cute_rs::{OperandA, OperandB, TensorMap};

/// K-major shared layout for a 128 × 128 FP16 query tile.
pub type QLayout = SharedLayout<128, 128>;
/// K-major shared layout for a 128 × 128 FP16 key tile.
pub type KLayout = SharedLayout<128, 128>;
/// MN-major shared layout for a 128 × 128 FP16 value tile.
pub type VLayout = SharedLayout<128, 128, true>;
/// K-major shared layout used to store a 128 × 128 FP16 output tile.
pub type OLayout = SharedLayout<128, 128>;
/// Single-CTA 128 × 128 × 128 QK MMA with both operands in shared memory.
type Qk = MmaSs<QLayout, KLayout, 128, 128, 128, 1>;
/// Single-CTA 128 × 128 × 128 PV MMA with probabilities in tensor memory.
type Pv = MmaTs<VLayout, 128, 128, 128, 1>;

/// Threads per CTA, partitioned among softmax, correction, MMA, and TMA roles.
pub const THREADS: u32 = 512;
/// Bytes in one 128 × 128 FP16 shared-memory tile.
const TILE_BYTES: usize = 128 * 128 * 2;
/// Byte offset of the two output tiles, after the barrier/token prefix.
const O_OFFSET: usize = 1024;
/// Byte offset of the two query tiles in dynamic shared memory.
const Q_OFFSET: usize = O_OFFSET + 2 * TILE_BYTES;
/// Byte offset of the three shared slots that alternate key and value tiles.
const KV_OFFSET: usize = Q_OFFSET + 2 * TILE_BYTES;
/// Total dynamic shared-memory bytes required by the launch contract.
pub const DYNAMIC_SMEM_BYTES: usize = KV_OFFSET + 3 * TILE_BYTES;
/// Byte offset of the TMEM allocation token, after 34 eight-byte barriers.
const TMEM_TOKEN_OFFSET: usize = 34 * 8;
/// Attention scale log2(e) / sqrt(128), allowing softmax to use base-two exponentials.
const SCALE_LOG2: f32 = 0.127_517_43; // log2(e) / sqrt(128)

/// One independent full/empty stage ring. A producer count starts at phase1
/// (initially empty), a consumer at phase0. Counts persist across work tiles.
/// The explicit arrival counts below are the source's participant contracts.
struct Ring<const STAGES: u32, FullKind = GeneralBarrier> {
    /// Contiguous stage barriers completed when producers publish their work.
    full: *mut Barrier,
    /// Contiguous stage barriers completed when consumers release their slots.
    empty: *mut Barrier,
    /// Distinguishes transaction-counted producers from ordinary arrivals.
    _full_kind: PhantomData<FullKind>,
}

impl<const STAGES: u32, K> Copy for Ring<STAGES, K> {}
impl<const STAGES: u32, K> Clone for Ring<STAGES, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const STAGES: u32, K> Ring<STAGES, K> {
    /// Bind a ring at `offset` measured in eight-byte barrier slots.
    /// The shared allocation must contain `2 * STAGES` aligned, nonoverlapping slots there.
    #[inline(always)]
    unsafe fn new(storage: *mut u8, offset: usize) -> Self {
        const { assert!(STAGES > 0) };
        let full = unsafe { storage.add(offset * 8).cast::<Barrier>() };
        Self {
            full,
            empty: unsafe { full.add(STAGES as usize) },
            _full_kind: PhantomData,
        }
    }
    /// Initialize full/empty barriers with per-stage producer/consumer arrival counts.
    /// Call once before publishing initialization to the CTA.
    #[inline(always)]
    unsafe fn init(self, producers: u32, consumers: u32) {
        let mut i = 0;
        while i < STAGES {
            unsafe {
                ManagedBarrier::<Uninit, K>::from_static(self.full.add(i as usize))
                    .init_instruction(producers);
                ManagedBarrier::<Uninit, GeneralBarrier>::from_static(self.empty.add(i as usize))
                    .init_instruction(consumers);
            }
            i += 1;
        }
    }
    /// Wait for consumers to release `count % STAGES`, using the producer’s inverted parity.
    #[inline(always)]
    unsafe fn acquire(self, count: u32) {
        let ptr = unsafe { self.empty.add((count % STAGES) as usize) };
        unsafe { wait::<GeneralBarrier>(ptr, ((count / STAGES) & 1) ^ 1) };
    }
    /// Wait for producers to fill `count % STAGES`, then order subsequent TMEM accesses.
    #[inline(always)]
    unsafe fn wait(self, count: u32) {
        unsafe { wait::<K>(self.full_ptr(count), (count / STAGES) & 1) };
        tcgen05::fence_after_thread_sync();
    }
    /// Select the full barrier for the monotonically increasing stage count.
    #[inline(always)]
    unsafe fn full_ptr(self, count: u32) -> *mut Barrier {
        unsafe { self.full.add((count % STAGES) as usize) }
    }
    /// Order prior TMEM accesses and contribute one thread arrival to the full barrier.
    #[inline(always)]
    unsafe fn commit_threads(self, count: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, K>::adopt_ready(
                self.full_ptr(count),
            ));
        }
    }
    /// Order prior TMEM accesses and contribute one thread arrival to the empty barrier.
    #[inline(always)]
    unsafe fn release_threads(self, count: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                self.empty.add((count % STAGES) as usize),
            ));
        }
    }
    /// Have one elected lane signal the full barrier after outstanding MMA work completes.
    /// All lanes of the MMA warp must participate in the election.
    #[inline(always)]
    unsafe fn commit_mma(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                tcgen05::commit(&ManagedBarrier::<Ready, K>::adopt_ready(
                    self.full_ptr(count),
                ));
            }
        }
    }
    /// Have one elected lane release the slot after outstanding MMA work completes.
    /// All lanes of the MMA warp must participate in the election.
    #[inline(always)]
    unsafe fn release_mma(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                tcgen05::commit(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                    self.empty.add((count % STAGES) as usize),
                ));
            }
        }
    }
}

impl<const STAGES: u32> Ring<STAGES, TmaBarrier> {
    /// Have one elected lane arrive and expect one FP16 tile’s TMA byte count.
    /// All lanes of the producer warp must participate in the election.
    #[inline(always)]
    unsafe fn producer_expect_tx(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                mbarrier::arrive_expect_tx(
                    &ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(self.full_ptr(count)),
                    TILE_BYTES as u32,
                );
            }
        }
    }
}

/// Spin until an initialized shared barrier completes the requested parity phase.
#[inline(always)]
unsafe fn wait<K>(ptr: *const Barrier, phase: u32) {
    let barrier = unsafe { ManagedBarrier::<Ready, K>::adopt_ready(ptr.cast_mut()) };
    while !unsafe { mbarrier::try_wait_parity(&barrier, phase) } {}
}

/// A copy of the same twelve pipeline identities exists in every participant.
#[derive(Clone, Copy)]
struct Pipelines {
    /// Two query stages transferred by TMA and released by the MMA warp.
    q: Ring<2, TmaBarrier>,
    /// Three alternating key/value stages transferred by TMA and released by MMA.
    kv: Ring<3, TmaBarrier>,
    /// QK scores from MMA to the first 128-thread softmax group.
    s0: Ring<1>,
    /// QK scores from MMA to the second 128-thread softmax group.
    s1: Ring<1>,
    /// FP16 probabilities from the first softmax group to PV MMA.
    p0: Ring<1>,
    /// FP16 probabilities from the second softmax group to PV MMA.
    p1: Ring<1>,
    /// First-group statistics rendezvous: 32 MMA plus 128 softmax arrivals, then correction.
    stats0: Ring<1>,
    /// Second-group statistics rendezvous: 32 MMA plus 128 softmax arrivals, then correction.
    stats1: Ring<1>,
    /// Two corrected output stages handed from 128 correction threads to 32 TMA-store threads.
    epi: Ring<2>,
    /// Two PV accumulator stages handed from MMA completion to 128 correction threads.
    output: Ring<2>,
    /// First softmax group’s handoff protecting aliased score/probability TMEM from the second.
    alias0: Ring<1>,
    /// Second softmax group’s handoff protecting aliased score/probability TMEM from the first.
    alias1: Ring<1>,
}

impl Pipelines {
    /// Bind all twelve rings to the first 34 aligned barrier slots in live CTA shared memory.
    #[inline(always)]
    unsafe fn new(smem: *mut u8) -> Self {
        unsafe {
            Self {
                q: Ring::new(smem, 0),
                kv: Ring::new(smem, 4),
                s0: Ring::new(smem, 10),
                s1: Ring::new(smem, 12),
                p0: Ring::new(smem, 14),
                p1: Ring::new(smem, 16),
                stats0: Ring::new(smem, 18),
                stats1: Ring::new(smem, 20),
                epi: Ring::new(smem, 22),
                output: Ring::new(smem, 26),
                alias0: Ring::new(smem, 30),
                alias1: Ring::new(smem, 32),
            }
        }
    }
    /// Initialize the schedule’s producer/consumer counts once before CTA-wide publication.
    #[inline(always)]
    unsafe fn init(self) {
        unsafe {
            self.q.init(1, 1);
            self.kv.init(1, 1);
            self.s0.init(1, 128);
            self.s1.init(1, 128);
            self.p0.init(128, 1);
            self.p1.init(128, 1);
            // Both the MMA warp and softmax group publish each statistics phase.
            self.stats0.init(160, 128);
            self.stats1.init(160, 128);
            self.epi.init(128, 32);
            self.output.init(1, 128);
            self.alias0.init(128, 128);
            self.alias1.init(128, 128);
        }
    }
}

/// Count 128-key tiles for the 256-query tile starting at `row`, clipping causal work.
#[inline(always)]
fn trip_count(row: u32, keys: u32, causal: bool) -> u32 {
    let limit = if causal && row + 256 < keys {
        row + 256
    } else {
        keys
    };
    limit.div_ceil(128)
}

/// Add the calling warp’s 32-row offset within a 128-thread group to a TMEM address.
#[inline(always)]
fn tmem_rows(base: u32, tid: u32) -> u32 {
    base + ((((tid % 128) / 32) * 32) << 16)
}

/// Evaluate the device approximate base-two exponential with subnormal flushing.
#[inline(always)]
fn exp2(x: f32) -> f32 {
    float::ex2_approx_ftz_f32(x)
}

/// Source SM100 max reduction has two independent chains per 32-score chunk.
#[inline(always)]
fn reduce_score_chunk(scores: &TmemFragment<32>, mut left: f32, mut right: f32) -> (f32, f32) {
    let mut i = 0;
    while i < 32 {
        thread::__unroll_config::<0>();
        left = left.max(scores.values[i]);
        left = left.max(scores.values[i + 1]);
        right = right.max(scores.values[i + 2]);
        right = right.max(scores.values[i + 3]);
        i += 4;
    }
    (left, right)
}

/// Preserve the masked path's balanced reduction while keeping scores for exp2.
#[inline(always)]
fn reduce_masked_scores(scores: &TmemFragment<128>) -> f32 {
    let mut maxima = [f32::NEG_INFINITY; 64];
    let mut i = 0;
    while i < 64 {
        thread::__unroll_config::<0>();
        maxima[i] = scores.values[i * 2].max(scores.values[i * 2 + 1]);
        i += 1;
    }
    reduce_max_level::<32>(&mut maxima);
    reduce_max_level::<16>(&mut maxima);
    reduce_max_level::<8>(&mut maxima);
    reduce_max_level::<4>(&mut maxima);
    reduce_max_level::<2>(&mut maxima);
    reduce_max_level::<1>(&mut maxima);
    maxima[0]
}

/// Reduce the first `2 * WIDTH` maxima in place to `WIDTH` pairwise maxima; `WIDTH <= 32`.
#[inline(always)]
fn reduce_max_level<const WIDTH: usize>(maxima: &mut [f32; 64]) {
    let mut i = 0;
    while i < WIDTH {
        thread::__unroll_config::<0>();
        maxima[i] = maxima[i * 2].max(maxima[i * 2 + 1]);
        i += 1;
    }
}

/// Multiply two FP32 values by packed FP32 factors and return their packed FP16 bits.
#[inline(always)]
fn scaled_half_pair(lo: f32, hi: f32, factor: PackedF32x2) -> u32 {
    let scaled = mul_f32x2(pack_f32x2(lo, hi), factor);
    convert::cvt_f16x2_f32(unpack_f32x2_lo(scaled), unpack_f32x2_hi(scaled))
}

/// Acquire score/statistics slots and issue QK without accumulation, then publish score completion.
/// All 32 MMA-warp lanes participate; `destination` is a live TMEM address.
#[inline(always)]
unsafe fn mma_qk(
    destination: u32,
    a: &SharedTile<QLayout, OperandA>,
    b: &SharedTile<KLayout, OperandB>,
    scores: Ring<1>,
    stats: Ring<1>,
    score_count: u32,
    stats_count: u32,
) {
    unsafe {
        scores.acquire(score_count);
        stats.acquire(stats_count);
        stats.commit_threads(stats_count);
        if warp::is_elected_sync(u32::MAX) {
            Qk::new().gemm(destination, a, b, false);
        }
        scores.commit_mma(score_count);
    }
}

/// Wait for probabilities, issue PV into an acquired output slot, and release P after MMA completion.
/// All 32 MMA-warp lanes participate; TMEM operands must remain live through completion.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // Keep the source pipeline handoffs explicit.
unsafe fn mma_pv(
    destination: u32,
    probabilities: u32,
    v: &SharedTile<VLayout, OperandB>,
    p: Ring<1>,
    output: Ring<2>,
    p_count: u32,
    output_count: u32,
    accumulate: bool,
) {
    unsafe {
        output.acquire(output_count);
        p.wait(p_count);
        if warp::is_elected_sync(u32::MAX) {
            Pv::new().gemm(destination, probabilities, v, accumulate);
        }
        p.release_mma(p_count);
        output.commit_mma(output_count);
    }
}

/// Device entry point and generated host launch bindings for the persistent SM100 schedule.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Dense BSHD FP16 attention, with top-left causal masking and GQA.
    /// Descriptor arrays are indexed by batch/head and retained by the host
    /// until the stream completes; all data matrices use D=128.
    #[launch_bounds(512, 1)]
    #[launch_contract(domain = 1, block = (512, 1, 1), dynamic_shared = 230400,
        dynamic_shared_alignment = 1024, min_compute_capability = (10, 0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sm100_fmha_prefill_cute_rs(
        qmaps: *const TensorMap<f16, QLayout>,
        kmaps: *const TensorMap<f16, KLayout>,
        vmaps: *const TensorMap<f16, VLayout>,
        omaps: *const TensorMap<f16, OLayout>,
        batch: u64,
        query: u64,
        keys: u64,
        heads: u64,
        kv_heads: u64,
        causal: u64,
    ) {
        let (batch, query, keys, heads, kv_heads) = (
            batch as u32,
            query as u32,
            keys as u32,
            heads as u32,
            kv_heads as u32,
        );
        let causal = causal != 0;
        let tid = thread::threadIdx_x();
        let warp_id = tid / 32;
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let qtiles = query.div_ceil(256);
        let total = qtiles * heads * batch;
        let stride = thread::gridDim_x();
        let mut tile = thread::blockIdx_x();
        unsafe {
            let pipes = Pipelines::new(smem);
            let token = smem.add(TMEM_TOKEN_OFFSET).cast::<u32>();
            if tid == 0 {
                pipes.init();
            }
            fence::fence_mbarrier_init();
            barrier::barrier_cta_sync(0, 512);
            if warp_id == 8 {
                TmemGuard::<TmemUninit, 512>::from_raw_slot(token).alloc_collective();
            }
            let mut tmem = 0;
            if warp_id <= 12 {
                barrier::barrier_cta_sync(2, 416);
                tmem = TmemGuard::<TmemReady, 512>::adopt_ready(token).raw_address();
            }

            if warp_id == 13 {
                thread::setmaxnreg_dec::<32>();
                let mut qcount = 0u32;
                let mut kvcount = 0u32;
                while tile < total {
                    let row = (tile % qtiles) * 256;
                    let bh = tile / qtiles;
                    let b = bh / heads;
                    let kh = (bh % heads) / (heads / kv_heads);
                    let qdesc = &*qmaps.add(bh as usize);
                    let kdesc = &*kmaps.add((b * kv_heads + kh) as usize);
                    let vdesc = &*vmaps.add((b * kv_heads + kh) as usize);
                    let steps = trip_count(row, keys, causal);
                    pipes.q.acquire(qcount);
                    pipes.q.producer_expect_tx(qcount);
                    let mut qt = SharedTile::<QLayout, OperandA>::from_raw(
                        smem.add(Q_OFFSET + (qcount % 2) as usize * TILE_BYTES)
                            .cast(),
                    );
                    TmaCopy::<QLayout, 1>::load(
                        qdesc,
                        &mut qt,
                        row as i32,
                        0,
                        pipes.q.full_ptr(qcount),
                        0,
                    );
                    qcount += 1;
                    pipes.kv.acquire(kvcount);
                    pipes.kv.producer_expect_tx(kvcount);
                    let mut kt = SharedTile::<KLayout, OperandB>::from_raw(
                        smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                            .cast(),
                    );
                    TmaCopy::<KLayout, 1>::load(
                        kdesc,
                        &mut kt,
                        0,
                        0,
                        pipes.kv.full_ptr(kvcount),
                        0,
                    );
                    kvcount += 1;
                    pipes.q.acquire(qcount);
                    pipes.q.producer_expect_tx(qcount);
                    let mut qt = SharedTile::<QLayout, OperandA>::from_raw(
                        smem.add(Q_OFFSET + (qcount % 2) as usize * TILE_BYTES)
                            .cast(),
                    );
                    TmaCopy::<QLayout, 1>::load(
                        qdesc,
                        &mut qt,
                        (row + 128) as i32,
                        0,
                        pipes.q.full_ptr(qcount),
                        0,
                    );
                    qcount += 1;
                    let mut i = 1;
                    while i < steps {
                        pipes.kv.acquire(kvcount);
                        pipes.kv.producer_expect_tx(kvcount);
                        let mut kt = SharedTile::<KLayout, OperandB>::from_raw(
                            smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                                .cast(),
                        );
                        TmaCopy::<KLayout, 1>::load(
                            kdesc,
                            &mut kt,
                            (i * 128) as i32,
                            0,
                            pipes.kv.full_ptr(kvcount),
                            0,
                        );
                        kvcount += 1;
                        pipes.kv.acquire(kvcount);
                        pipes.kv.producer_expect_tx(kvcount);
                        let mut vt = SharedTile::<VLayout, OperandB>::from_raw(
                            smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                                .cast(),
                        );
                        TmaCopy::<VLayout, 1>::load(
                            vdesc,
                            &mut vt,
                            ((i - 1) * 128) as i32,
                            0,
                            pipes.kv.full_ptr(kvcount),
                            0,
                        );
                        kvcount += 1;
                        i += 1;
                    }
                    pipes.kv.acquire(kvcount);
                    pipes.kv.producer_expect_tx(kvcount);
                    let mut vt = SharedTile::<VLayout, OperandB>::from_raw(
                        smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                            .cast(),
                    );
                    TmaCopy::<VLayout, 1>::load(
                        vdesc,
                        &mut vt,
                        ((steps - 1) * 128) as i32,
                        0,
                        pipes.kv.full_ptr(kvcount),
                        0,
                    );
                    kvcount += 1;
                    tile += stride;
                }
            } else if warp_id == 12 {
                thread::setmaxnreg_dec::<32>();
                let (mut qcount, mut kvcount, mut scount, mut pcount, mut stats_count, mut ocount) =
                    (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
                while tile < total {
                    let row = (tile % qtiles) * 256;
                    let steps = trip_count(row, keys, causal);
                    pipes.q.wait(qcount);
                    let q0 = SharedTile::<QLayout, OperandA>::from_raw(
                        smem.add(Q_OFFSET + (qcount % 2) as usize * TILE_BYTES)
                            .cast(),
                    );
                    pipes.kv.wait(kvcount);
                    let k0 = SharedTile::<KLayout, OperandB>::from_raw(
                        smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                            .cast(),
                    );
                    mma_qk(tmem, &q0, &k0, pipes.s0, pipes.stats0, scount, stats_count);
                    pipes.q.wait(qcount + 1);
                    let q1 = SharedTile::<QLayout, OperandA>::from_raw(
                        smem.add(Q_OFFSET + ((qcount + 1) % 2) as usize * TILE_BYTES)
                            .cast(),
                    );
                    mma_qk(
                        tmem + 128,
                        &q1,
                        &k0,
                        pipes.s1,
                        pipes.stats1,
                        scount,
                        stats_count,
                    );
                    scount += 1;
                    stats_count += 1;
                    pipes.kv.release_mma(kvcount);
                    kvcount += 1;
                    let mut i = 1;
                    while i < steps {
                        pipes.kv.wait(kvcount);
                        let kt = SharedTile::<KLayout, OperandB>::from_raw(
                            smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                                .cast(),
                        );
                        mma_qk(tmem, &q0, &kt, pipes.s0, pipes.stats0, scount, stats_count);
                        pipes.kv.wait(kvcount + 1);
                        let vt = SharedTile::<VLayout, OperandB>::from_raw(
                            smem.add(KV_OFFSET + ((kvcount + 1) % 3) as usize * TILE_BYTES)
                                .cast(),
                        );
                        mma_pv(
                            tmem + 256,
                            tmem + 160,
                            &vt,
                            pipes.p0,
                            pipes.output,
                            pcount,
                            ocount,
                            i > 1,
                        );
                        ocount += 1;
                        mma_qk(
                            tmem + 128,
                            &q1,
                            &kt,
                            pipes.s1,
                            pipes.stats1,
                            scount,
                            stats_count,
                        );
                        pipes.kv.release_mma(kvcount);
                        scount += 1;
                        stats_count += 1;
                        mma_pv(
                            tmem + 384,
                            tmem + 32,
                            &vt,
                            pipes.p1,
                            pipes.output,
                            pcount,
                            ocount,
                            i > 1,
                        );
                        ocount += 1;
                        pcount += 1;
                        pipes.kv.release_mma(kvcount + 1);
                        kvcount += 2;
                        i += 1;
                    }
                    pipes.q.release_mma(qcount);
                    pipes.q.release_mma(qcount + 1);
                    qcount += 2;
                    pipes.kv.wait(kvcount);
                    let vt = SharedTile::<VLayout, OperandB>::from_raw(
                        smem.add(KV_OFFSET + (kvcount % 3) as usize * TILE_BYTES)
                            .cast(),
                    );
                    mma_pv(
                        tmem + 256,
                        tmem + 160,
                        &vt,
                        pipes.p0,
                        pipes.output,
                        pcount,
                        ocount,
                        steps > 1,
                    );
                    ocount += 1;
                    mma_pv(
                        tmem + 384,
                        tmem + 32,
                        &vt,
                        pipes.p1,
                        pipes.output,
                        pcount,
                        ocount,
                        steps > 1,
                    );
                    ocount += 1;
                    pcount += 1;
                    pipes.kv.release_mma(kvcount);
                    kvcount += 1;
                    pipes.stats0.acquire(stats_count);
                    pipes.stats0.commit_threads(stats_count);
                    pipes.stats1.acquire(stats_count);
                    pipes.stats1.commit_threads(stats_count);
                    stats_count += 1;
                    tile += stride;
                }
            } else if warp_id < 8 {
                thread::setmaxnreg_inc::<192>();
                let stage = warp_id / 4;
                let (scores, probabilities, stats, alias_producer, alias_consumer) = if stage == 0 {
                    (pipes.s0, pipes.p0, pipes.stats0, pipes.alias0, pipes.alias1)
                } else {
                    (pipes.s1, pipes.p1, pipes.stats1, pipes.alias1, pipes.alias0)
                };
                let score_addr = tmem_rows(tmem + stage * 128, tid);
                let p_addr = tmem_rows(tmem + if stage == 0 { 160 } else { 32 }, tid);
                let (mut count, mut stats_count, mut alias_send, mut alias_recv) =
                    (0u32, 0u32, 0u32, 0u32);
                while tile < total {
                    let row = (tile % qtiles) * 256;
                    let steps = trip_count(row, keys, causal);
                    let query_row = row + stage * 128 + tid % 128;
                    let mut row_max = f32::NEG_INFINITY;
                    let mut row_sum = 0.0f32;
                    // P1 aliases S0 after the following S0 read. Account for
                    // the initial S0 read separately, exactly as the source.
                    if stage == 1 {
                        alias_consumer.wait(alias_recv);
                        alias_recv += 1;
                    }
                    let mut step = 0;
                    while step < steps {
                        scores.wait(count);
                        let old_max = row_max;
                        // The reference partitions whole CTA trips: causal
                        // trailing tiles start at the first query row, and a
                        // residual key tile is always masked.
                        let apply_mask = (step + 1) * 128 > keys
                            || (causal && (step >= row / 128 || step + 1 == steps));
                        let mut values;
                        if apply_mask {
                            // Retain one conservative completion wait before
                            // masked values can lose their register dependency.
                            values = TmemFragment::<128>::load(score_addr);
                            let mut col = 0usize;
                            #[unroll]
                            while col < 128 {
                                let key = step * 128 + col as u32;
                                if key >= keys || (causal && key > query_row) {
                                    values.values[col] = f32::NEG_INFINITY;
                                }
                                col += 1;
                            }
                            row_max = row_max.max(reduce_masked_scores(&values));
                            scores.release_threads(count);
                            alias_producer.commit_threads(alias_send);
                        } else {
                            let first = TmemFragment::<32>::load_deferred(score_addr);
                            let second = TmemFragment::<32>::load_deferred(score_addr + 32);
                            let (left, right) =
                                reduce_score_chunk(&first, f32::NEG_INFINITY, f32::NEG_INFINITY);
                            let third = TmemFragment::<32>::load_deferred(score_addr + 64);
                            let (left, right) = reduce_score_chunk(&second, left, right);
                            let fourth = TmemFragment::<32>::load_deferred(score_addr + 96);
                            let (left, right) = reduce_score_chunk(&third, left, right);
                            // Match the source's handoff before the last max
                            // chunk, permitting the aliased P writer to advance.
                            tmem::store_wait();
                            scores.release_threads(count);
                            alias_producer.commit_threads(alias_send);
                            let (left, right) = reduce_score_chunk(&fourth, left, right);
                            row_max = row_max.max(left.max(right));
                            values = TmemFragment::<128> { values: [0.0; 128] };
                            let mut col = 0usize;
                            #[unroll]
                            while col < 32 {
                                values.values[col] = first.values[col];
                                values.values[col + 32] = second.values[col];
                                values.values[col + 64] = third.values[col];
                                values.values[col + 96] = fourth.values[col];
                                col += 1;
                            }
                        }
                        alias_send += 1;
                        let safe_max = if row_max == f32::NEG_INFINITY {
                            0.0
                        } else {
                            row_max
                        };
                        let scale = pack_f32x2(SCALE_LOG2, SCALE_LOG2);
                        let shift = -safe_max * SCALE_LOG2;
                        let shift = pack_f32x2(shift, shift);
                        // The source primes eight values before publishing
                        // maxima, then overlaps future FMA with exp2 and CVT.
                        let mut col = 0usize;
                        #[unroll]
                        while col < 4 {
                            let scaled = fma_f32x2(
                                pack_f32x2(values.values[col * 2], values.values[col * 2 + 1]),
                                scale,
                                shift,
                            );
                            values.values[col * 2] = unpack_f32x2_lo(scaled);
                            values.values[col * 2 + 1] = unpack_f32x2_hi(scaled);
                            col += 1;
                        }
                        stats.acquire(stats_count);
                        TmemFragment::<2> {
                            values: [old_max, safe_max],
                        }
                        .store(score_addr);
                        stats.commit_threads(stats_count);
                        stats_count += 1;
                        let correction = exp2((old_max - safe_max) * SCALE_LOG2) * 0.5;
                        let initial_sum = row_sum * correction;
                        let mut sums = pack_f32x2(initial_sum, initial_sum);
                        let mut packed = [0u32; 64];
                        let mut col = 0usize;
                        #[unroll]
                        while col < 64 {
                            if col >= 2 {
                                let x = values.values[(col - 2) * 2];
                                let y = values.values[(col - 2) * 2 + 1];
                                packed[col - 2] = convert::cvt_f16x2_f32(x, y);
                                sums = add_f32x2(sums, pack_f32x2(x, y));
                            }
                            values.values[col * 2] = exp2(values.values[col * 2]);
                            if col + 4 < 64 {
                                let next = (col + 4) * 2;
                                let scaled = fma_f32x2(
                                    pack_f32x2(values.values[next], values.values[next + 1]),
                                    scale,
                                    shift,
                                );
                                values.values[next] = unpack_f32x2_lo(scaled);
                                values.values[next + 1] = unpack_f32x2_hi(scaled);
                            }
                            values.values[col * 2 + 1] = exp2(values.values[col * 2 + 1]);
                            col += 1;
                        }
                        let mut col = 62usize;
                        #[unroll]
                        while col < 64 {
                            let x = values.values[col * 2];
                            let y = values.values[col * 2 + 1];
                            packed[col] = convert::cvt_f16x2_f32(x, y);
                            sums = add_f32x2(sums, pack_f32x2(x, y));
                            col += 1;
                        }
                        row_sum = unpack_f32x2_lo(sums) + unpack_f32x2_hi(sums);
                        row_max = safe_max;
                        alias_consumer.wait(alias_recv);
                        alias_recv += 1;
                        probabilities.acquire(count);
                        TmemFragment::<64>::store_packed(p_addr, packed);
                        probabilities.commit_threads(count);
                        count += 1;
                        step += 1;
                    }
                    if stage == 0 {
                        alias_producer.commit_threads(alias_send);
                        alias_send += 1;
                    }
                    stats.acquire(stats_count);
                    TmemFragment::<2> {
                        values: [row_sum, row_max],
                    }
                    .store(score_addr);
                    stats.commit_threads(stats_count);
                    stats_count += 1;
                    tile += stride;
                }
            } else if warp_id < 12 {
                thread::setmaxnreg_dec::<96>();
                let (mut stat_count, mut ocount, mut epicount) = (0u32, 0u32, 0u32);
                while tile < total {
                    let row = (tile % qtiles) * 256;
                    let steps = trip_count(row, keys, causal);
                    pipes.stats0.wait(stat_count);
                    pipes.stats0.release_threads(stat_count);
                    pipes.stats1.wait(stat_count);
                    pipes.stats1.release_threads(stat_count);
                    stat_count += 1;
                    let mut step = 1;
                    while step < steps {
                        let mut stage = 0;
                        while stage < 2 {
                            let stats = if stage == 0 {
                                pipes.stats0
                            } else {
                                pipes.stats1
                            };
                            stats.wait(stat_count);
                            let maxima =
                                TmemFragment::<2>::load(tmem_rows(tmem + stage * 128, tid));
                            stats.release_threads(stat_count);
                            pipes.output.wait(ocount);
                            if !warp::all(maxima.values[0] == maxima.values[1]) {
                                let factor =
                                    exp2((maxima.values[0] - maxima.values[1]) * SCALE_LOG2);
                                let factor = pack_f32x2(factor, factor);
                                let mut col = 0;
                                #[unroll]
                                while col < 128 {
                                    let addr = tmem_rows(tmem + 256 + stage * 128 + col, tid);
                                    // Register dependencies order each load and its
                                    // scale/store. A terminal store wait protects
                                    // the handoff back to the MMA warp.
                                    let mut out = TmemFragment::<16>::load_deferred(addr);
                                    let mut j = 0;
                                    #[unroll]
                                    while j < 16 {
                                        let scaled = mul_f32x2(
                                            pack_f32x2(out.values[j], out.values[j + 1]),
                                            factor,
                                        );
                                        out.values[j] = unpack_f32x2_lo(scaled);
                                        out.values[j + 1] = unpack_f32x2_hi(scaled);
                                        j += 2;
                                    }
                                    out.store_deferred(addr);
                                    col += 16;
                                }
                                tmem::store_wait();
                            }
                            pipes.output.release_threads(ocount);
                            ocount += 1;
                            stage += 1;
                        }
                        stat_count += 1;
                        step += 1;
                    }
                    let mut stage = 0;
                    while stage < 2 {
                        let stats = if stage == 0 {
                            pipes.stats0
                        } else {
                            pipes.stats1
                        };
                        stats.wait(stat_count);
                        let sums = TmemFragment::<2>::load(tmem_rows(tmem + stage * 128, tid));
                        stats.release_threads(stat_count);
                        pipes.output.wait(ocount);
                        pipes.epi.acquire(epicount);
                        let factor = float::rcp_approx_ftz_f32(sums.values[0]);
                        let factor = pack_f32x2(factor, factor);
                        let dst = smem
                            .add(O_OFFSET + (epicount % 2) as usize * TILE_BYTES)
                            .cast::<f16>();
                        let mut col = 0;
                        while col < 128 {
                            // Match the reference's runtime epilogue loop; the
                            // dependent shared stores complete each TMEM read.
                            let out = TmemFragment::<16>::load_deferred(tmem_rows(
                                tmem + 256 + stage * 128 + col,
                                tid,
                            ));
                            // Eight FP16 values stay contiguous under B128
                            // swizzling, permitting aligned aggregate stores.
                            let mut j = 0usize;
                            #[unroll]
                            while j < 16 {
                                let words = U32x4([
                                    scaled_half_pair(out.values[j], out.values[j + 1], factor),
                                    scaled_half_pair(out.values[j + 2], out.values[j + 3], factor),
                                    scaled_half_pair(out.values[j + 4], out.values[j + 5], factor),
                                    scaled_half_pair(out.values[j + 6], out.values[j + 7], factor),
                                ]);
                                *dst.add(OLayout::crd2idx(tid % 128, col + j as u32))
                                    .cast::<U32x4>() = words;
                                j += 8;
                            }
                            col += 16;
                        }
                        fence::fence_proxy_async_shared_cta();
                        pipes.output.release_threads(ocount);
                        pipes.epi.commit_threads(epicount);
                        ocount += 1;
                        epicount += 1;
                        stage += 1;
                    }
                    stat_count += 1;
                    tile += stride;
                }
                barrier::barrier_cta_sync(7, 128);
                if warp_id == 8 {
                    tcgen05::relinquish_alloc_permit();
                    TmemGuard::<TmemReady, 512>::adopt_ready(token).dealloc_collective();
                }
            } else if warp_id == 14 {
                thread::setmaxnreg_dec::<32>();
                let mut count = 0u32;
                while tile < total {
                    let row = (tile % qtiles) * 256;
                    let bh = tile / qtiles;
                    let desc = &*omaps.add(bh as usize);
                    pipes.epi.wait(count);
                    let out0 = SharedTile::<OLayout, OperandA>::from_raw(smem.add(O_OFFSET).cast());
                    if warp::is_elected_sync(u32::MAX) {
                        TmaCopy::<OLayout, 1>::store(desc, &out0, row as i32, 0);
                        ctm_rs::tma::cp_async_bulk_commit_group();
                    }
                    pipes.epi.wait(count + 1);
                    let out1 = SharedTile::<OLayout, OperandA>::from_raw(
                        smem.add(O_OFFSET + TILE_BYTES).cast(),
                    );
                    if warp::is_elected_sync(u32::MAX) {
                        TmaCopy::<OLayout, 1>::store(desc, &out1, (row + 128) as i32, 0);
                        ctm_rs::tma::cp_async_bulk_commit_group();
                        ctm_rs::tma::cp_async_bulk_wait_group_read::<1>();
                    }
                    warp::bar_warp_sync(u32::MAX);
                    pipes.epi.release_threads(count);
                    if warp::is_elected_sync(u32::MAX) {
                        ctm_rs::tma::cp_async_bulk_wait_group_read::<0>();
                    }
                    warp::bar_warp_sync(u32::MAX);
                    pipes.epi.release_threads(count + 1);
                    count += 2;
                    tile += stride;
                }
            } else {
                thread::setmaxnreg_dec::<32>();
            }
        }
    }
}

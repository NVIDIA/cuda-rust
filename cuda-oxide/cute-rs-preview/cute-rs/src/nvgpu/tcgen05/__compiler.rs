/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler markers for SM100 `tcgen05` MMA, TMEM, and TMA operations.

use crate::arch::Barrier;
use crate::nvgpu::cpasync::TensorMap;
use core::marker::PhantomData;
/// Device-only marker allocating COLUMNS of group-two tensor memory.
///
/// # Safety
/// The allocating warps and shared address slot must satisfy
/// [`Sm100Tmem::alloc`](super::Sm100Tmem::alloc).
#[inline(never)]
pub unsafe fn sm100_tmem_alloc<const COLUMNS: u32>(token: *mut u32) {
    let _ = token;
    unreachable!("SM100 CuTe marker requires device compilation")
}
/// Device-only marker releasing a completed group-two tensor-memory allocation.
///
/// # Safety
/// Both allocating warps must rendezvous after every asynchronous user has
/// finished, as [`Sm100Tmem::dealloc`](super::Sm100Tmem::dealloc) requires.
#[inline(never)]
pub unsafe fn sm100_tmem_dealloc<const COLUMNS: u32>(tmem: u32) {
    let _ = tmem;
    unreachable!("SM100 CuTe marker requires device compilation")
}
/// Device-only marker accumulating the paired N partitions with shared A-collector reuse.
///
/// # Safety
/// The elected leader must supply complete live shared tiles and the matching 512-column accumulator.
#[inline(never)]
pub unsafe fn sm100_tiled_mma<AL, BL0, BL1, const N0: u32, const N1: u32, const K: u32>(
    tmem: u32,
    a: *const f16,
    b0: *const f16,
    b1: *const f16,
    accumulate: bool,
) {
    let _ = (tmem, a, b0, b1, accumulate, PhantomData::<(AL, BL0, BL1)>);
    unreachable!("SM100 CuTe marker requires device compilation")
}
/// Device-only marker loading, biasing and converting one FP32 TMEM slice to shared FP16.
///
/// # Safety
/// All epilogue lanes must supply their valid row-group address and reusable output storage.
#[inline(never)]
pub unsafe fn sm100_tmem_epilogue<L>(tmem: u32, dst: *mut f16, tid: u32, bias: f32) {
    let _ = (tmem, dst, tid, bias, PhantomData::<L>);
    unreachable!("SM100 CuTe marker requires device compilation")
}
// Every lane of the producer warp participates. CuTe selects the issuing
// lane and performs any required warp synchronization.
/// Device-only marker loading one CTA’s tile and routing completion to the leader.
///
/// # Safety
/// All producer-warp lanes participate with the matching descriptor, stage and cluster byte expectation.
#[inline(never)]
pub unsafe fn sm100_cluster_tma_load<L>(
    desc: *const TensorMap<f16, L>,
    dst: *mut f16,
    row: u32,
    col: u32,
    bar: *mut Barrier,
    rank: u32,
) {
    let _ = (desc, dst, row, col, bar, rank);
    unreachable!("SM100 CuTe marker requires device compilation")
}

/// Device-only marker storing a published shared tile to global memory.
///
/// # Safety
/// The elected issuer must retain matching source and descriptor storage through read completion.
#[inline(never)]
pub unsafe fn sm100_tma_store<L>(
    desc: *const TensorMap<f16, L>,
    src: *const f16,
    row: u32,
    col: u32,
) {
    let _ = (desc, src, row, col);
    unreachable!("SM100 CuTe marker requires device compilation")
}
/// Define an exact compiler hook for one store-group ownership operation.
macro_rules! store_marker {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[inline(never)]
        pub unsafe fn $name<const STAGES: u32>() {
            unreachable!("SM100 CuTe marker requires device compilation")
        }
    };
}
store_marker!(
    #[doc = "Device-only hook committing the issuer's pending TMA stores to one group."]
    sm100_store_commit
);
store_marker!(
    #[doc = "Device-only hook waiting until at most STAGES-1 groups still read shared storage."]
    sm100_store_acquire
);
store_marker!(
    #[doc = "Device-only hook waiting until every TMA store group stops reading shared storage."]
    sm100_store_tail
);
/// Define an exact compiler hook for one full/empty input-stage operation.
macro_rules! pipeline_marker {
    ($(#[$doc:meta])* $name:ident ($($arg:ident:$ty:ty),*)) => {
        $(#[$doc])*
        #[inline(never)]
        pub unsafe fn $name<const STAGES:u32,const TX_BYTES:u32>($($arg:$ty),*) {
            let _ = ($($arg),*); unreachable!("SM100 CuTe marker requires device compilation")
        }
    };
}
pipeline_marker!(#[doc = "Initialize full/empty barriers; one lane per CTA publishes initialization before use."] sm100_pipeline_init(full: *mut Barrier, empty: *mut Barrier));
pipeline_marker!(#[doc = "Wait for the producer's reusable stage; stage and phase must match the live ring position."] sm100_pipeline_acquire(empty: *mut Barrier, stage: u32, phase: u32));
pipeline_marker!(#[doc = "Attach TX_BYTES for both CTAs; exactly one elected leader lane calls after acquisition."] sm100_pipeline_expect(full: *mut Barrier, stage: u32));
pipeline_marker!(#[doc = "Wait for TMA completion and order later MMA; called by the leader's consuming warp."] sm100_pipeline_wait(full: *mut Barrier, stage: u32, phase: u32));
pipeline_marker!(#[doc = "Release both CTAs' stage after prior MMA completes; called by the elected leader issuer."] sm100_pipeline_release(empty: *mut Barrier, stage: u32));
/// Define an exact compiler hook for one accumulator ownership handoff.
macro_rules! accumulator_marker {
    ($(#[$doc:meta])* $name:ident ($($arg:ident:$ty:ty),*)) => {
        $(#[$doc])*
        #[inline(never)]
        pub unsafe fn $name<const CONSUMER_WARPS:u32>($($arg:$ty),*) {
            let _ = ($($arg),*); unreachable!("SM100 CuTe marker requires device compilation")
        }
    };
}
accumulator_marker!(#[doc = "Initialize accumulator handoff barriers from one lane per CTA before cluster publication."] sm100_accumulator_init(empty: *mut Barrier, full: *mut Barrier));
accumulator_marker!(#[doc = "Wait for all epilogue consumers to release TMEM; the leader's MMA warp uses empty parity."] sm100_accumulator_acquire(empty: *mut Barrier, phase: u32));
accumulator_marker!(#[doc = "Publish completed accumulation to both CTAs; one elected leader lane commits after final MMA."] sm100_accumulator_commit(full: *mut Barrier));
accumulator_marker!(#[doc = "Wait for an accumulator tile and order TMEM loads; every consumer warp uses full parity."] sm100_accumulator_wait(full: *mut Barrier, phase: u32));
accumulator_marker!(#[doc = "Fence completed epilogue reads and elect one arrival per warp; every consumer lane participates."] sm100_accumulator_release(empty: *mut Barrier));

/// Device-only shared/shared MMA marker; operand displacements count bytes from retained bases.
///
/// # Safety
/// The public `MmaSs::gemm` participation, completion and storage contracts apply.
#[inline(never)]
pub unsafe fn mma_ss<AL, BL, const M: u32, const N: u32, const K: u32, const G: u32>(
    dst: u32,
    a: *mut f16,
    b: *mut f16,
    accumulate: bool,
    a_byte_offset: u32,
    b_byte_offset: u32,
) {
    let _ = (dst, a, b, accumulate, a_byte_offset, b_byte_offset);
    unreachable!("CuTe compiler marker")
}

/// Device-only TMEM/shared MMA marker; B's displacement counts bytes from its retained base.
///
/// # Safety
/// The public `MmaTs::gemm` participation, nonoverlap and completion contracts apply.
#[inline(never)]
pub unsafe fn mma_ts<BL, const M: u32, const N: u32, const K: u32, const G: u32>(
    dst: u32,
    a: u32,
    b: *mut f16,
    accumulate: bool,
    b_byte_offset: u32,
) {
    let _ = (dst, a, b, accumulate, b_byte_offset);
    unreachable!("CuTe compiler marker")
}

/// Device-only rank-three load marker with inner, row and outer-chunk coordinates.
///
/// # Safety
/// The producer warp must supply matching live storage and a correctly initialized group completion barrier.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn tma_load<L, const G: u32>(
    desc: *const TensorMap<f16, L>,
    dst: *mut f16,
    c0: i32,
    c1: i32,
    c2: i32,
    bar: *mut Barrier,
    rank: u32,
) {
    let _ = (desc, dst, c0, c1, c2, bar, rank);
    unreachable!("CuTe compiler marker")
}

/// Device-only rank-three store marker with inner, row and outer-chunk coordinates.
///
/// # Safety
/// The elected issuer must publish shared writes and retain storage until the copy stops reading it.
#[inline(never)]
pub unsafe fn tma_store<L, const G: u32>(
    desc: *const TensorMap<f16, L>,
    src: *mut f16,
    c0: i32,
    c1: i32,
    c2: i32,
) {
    let _ = (desc, src, c0, c1, c2);
    unreachable!("CuTe compiler marker")
}

/// Device-only marker copying one 512-byte `(32, 4, 4)` scale atom from shared memory into TMEM.
///
/// # Safety
/// The elected issuer must supply a valid 128-byte-aligned shared scale pointer and 4-column TMEM destination.
#[inline(never)]
pub unsafe fn sm100_block_scaled_copy(tmem_col: u32, s_scale: *mut u32, step: u32) {
    let _ = (tmem_col, s_scale, step);
    unreachable!("SM100 CuTe marker requires device compilation")
}

/// Device-only marker issuing one `K = 64` step of a group-one block-scaled `mxf4nvf4` MMA.
///
/// # Safety
/// The elected issuer must supply live shared operand tiles `a` and `b` and populated TMEM scales.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub unsafe fn sm100_block_scaled_gemm<AL, BL, const M: u32, const K: u32>(
    tmem_d: u32,
    a: *mut f16,
    b: *mut f16,
    sfa_tmem: u32,
    sfb_tmem: u32,
    valid_n: u32,
    b_lbo_bytes: u32,
    step: u32,
    accumulate: bool,
) {
    let _ = (
        tmem_d,
        a,
        b,
        sfa_tmem,
        sfb_tmem,
        valid_n,
        b_lbo_bytes,
        step,
        accumulate,
        PhantomData::<(AL, BL)>,
    );
    unreachable!("SM100 CuTe marker requires device compilation")
}

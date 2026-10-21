/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-token attention: parallel KV tiles followed by a stable split reduction.
//!
//! CuTe copies stage 64 keys and values per CTA. Four lanes cooperate on each
//! dot product; no query padding or matrix-multiply query rows are allocated.
//! Scores, probabilities, split numerators and reduction states remain FP32.
use core::marker::PhantomData;
use ctm_rs::{
    async_copy, cuda_module, float, kernel, launch_bounds, launch_contract, mbarrier,
    memory::{DisjointSlice, DynamicSharedArray},
    thread, warp,
};
use cute_rs::markers::{C, Composed, CpAsync, L, LeadingDim, RowMajor, Swizzle, T2};
use cute_rs::{GmemMatrix, SmemTile, copy_g2s};

/// Swizzled 64-by-64 FP16 layout used for both K and V in shared memory.
type KvLayout = Composed<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>;
/// Swizzled 16-by-64 region covered by one cooperative copy.
type CopyLayout = Composed<Swizzle<3, 3, 3>, 0, RowMajor<16, 64>>;
/// Map 128 threads to 16 rows and eight 16-byte copy positions.
type CopyThreads = L<T2<C<16>, C<8>>, T2<C<8>, C<1>>>;
/// Eight adjacent FP16 values copied by each thread.
type CopyValues = L<T2<C<1>, C<8>>, T2<C<8>, C<1>>>;
/// Typed shared-memory view of one 64-row K or V tile.
type KvTile = SmemTile<f16, KvLayout>;

/// Attention scale converted to base two for the exponential instruction.
const SCALE_LOG2: f32 = 0.125 * core::f32::consts::LOG2_E;
/// FP32 elements per split: a maximum, a sum, and 64 numerator values.
const STATE_SIZE: usize = 66;

/// Zero/zero selects runtime dimensions. Every static shape is checked while
/// Rust monomorphizes the kernel, before any device code can be launched.
const fn assert_shape<const SEQ_K: u32, const HEADS: u32>() {
    assert!((SEQ_K == 0) == (HEADS == 0));
    assert!(SEQ_K as u64 * HEADS as u64 <= u32::MAX as u64 / 64);
    assert!((SEQ_K as u64).div_ceil(64) * HEADS as u64 <= u32::MAX as u64 / 66);
}

/// Expand fixed iterations with constant indices so fragments stay in registers.
macro_rules! static_for {
    ($index:ident in [$($value:expr),*] $body:block) => {
        $({ #[doc = "Compile-time index for this unrolled iteration."] const $index: usize = $value; $body })*
    };
}

/// Select the larger value; comparisons involving NaN fall back to `b`.
#[inline(always)]
fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Reduce a maximum across all 32 participating lanes.
#[inline(always)]
fn warp_max(mut value: f32) -> f32 {
    static_for!(MASK in [16, 8, 4, 2, 1] {
        value = max(value, warp::shuffle_xor_f32(value, MASK as u32));
    });
    value
}

/// Sum values across all 32 participating lanes.
#[inline(always)]
fn warp_sum(mut value: f32) -> f32 {
    static_for!(MASK in [16, 8, 4, 2, 1] {
        value += warp::shuffle_xor_f32(value, MASK as u32);
    });
    value
}

/// Map a logical K/V coordinate through the tile's shared-memory swizzle.
#[inline(always)]
fn shared_offset(row: usize, col: usize) -> usize {
    let linear = row * 64 + col;
    linear ^ ((linear & (7 << 6)) >> 3)
}

/// Wrap storage for 4096 FP16 values as a typed K/V tile.
///
/// # Safety
/// `base` must point to live shared storage with the tile's required alignment.
#[inline(always)]
unsafe fn tile(base: *mut f16) -> KvTile {
    SmemTile {
        base,
        capacity: 4096,
        layout: PhantomData,
    }
}

/// Issue cooperative copies for one KV split, zero-filling rows past the sequence.
///
/// # Safety
/// All 128 threads must participate with valid source and destination views; wait before reading the tile.
#[inline(always)]
unsafe fn stage(src: &GmemMatrix<f16, LeadingDim<8>>, split: usize, dst: &KvTile, tid: u32) {
    static_for!(GROUP in [0, 1, 2, 3] {
        let mut part = SmemTile::<f16, CopyLayout> {
            base: unsafe { dst.base.add(GROUP * 1024) }, capacity: 1024, layout: PhantomData,
        };
        // CuTe's copy predicate zero-fills rows beyond the actual KV length.
        unsafe {
            copy_g2s::<CpAsync<16>, CopyThreads, CopyValues, RowMajor<16, 64>, CopyLayout, f16>(
                src, (split * 4 + GROUP, 0), &mut part, tid,
            );
        }
    });
}

/// Device entry points for partial attention and the split reduction.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Compute one head's 64-key split and write its FP32 numerator, maximum, and sum.
    #[launch_bounds(128)]
    #[launch_contract(domain=1, block=(128,1,1), dynamic_shared=17696,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    pub fn fmha_decode_partial<const SEQ_K: u32, const HEADS: u32>(
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mut partials: DisjointSlice<f32>,
        seq_k: u32,
        heads: u32,
    ) {
        const { assert_shape::<SEQ_K, HEADS>() };
        let seq_k = if SEQ_K == 0 { seq_k } else { SEQ_K };
        let heads = if HEADS == 0 { heads } else { HEADS };
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let splits = (seq_k as usize + 63) / 64;
        let split = thread::blockIdx_x() as usize % splits;
        let head = thread::blockIdx_x() as usize / splits;
        unsafe {
            let smem = DynamicSharedArray::<f16, 1024>::get_raw().cast::<f16>();
            let sk = tile(smem);
            let sv = tile(smem.add(4096));
            let scores = smem.add(8192).cast::<f32>();
            let maxima = scores.add(64);
            let sums = maxima.add(4);
            let warp_outputs = sums.add(4);
            let global_k = GmemMatrix {
                base: k.as_ptr().add(head * 64),
                rows: seq_k as usize,
                cols: 64,
                leading_dim: LeadingDim::<8> {
                    elements: heads as usize * 64,
                },
            };
            let global_v = GmemMatrix {
                base: v.as_ptr().add(head * 64),
                rows: seq_k as usize,
                cols: 64,
                leading_dim: LeadingDim::<8> {
                    elements: heads as usize * 64,
                },
            };
            stage(&global_k, split, &sk, tid as u32);
            stage(&global_v, split, &sv, tid as u32);
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_group::<0>();
            mbarrier::barrier_cta_sync(1, 128);

            // 32 independent keys at a time, with four lanes per 64-D dot.
            static_for!(PASS in [0, 1] {
                let row = tid / 4 + PASS * 32;
                let mut dot = 0.0f32;
                static_for!(D in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                    let column = tid % 4 * 16 + D;
                    dot += *q.as_ptr().add(head * 64 + column) as f32
                        * *sk.base.add(shared_offset(row, column)) as f32;
                });
                dot += warp::shuffle_xor_f32(dot, 1);
                dot += warp::shuffle_xor_f32(dot, 2);
                if tid % 4 == 0 {
                    *scores.add(row) = if split * 64 + row < seq_k as usize {
                        dot
                    } else { f32::NEG_INFINITY };
                }
            });
            mbarrier::barrier_cta_sync(1, 128);
            let score = if tid < 64 {
                *scores.add(tid)
            } else {
                f32::NEG_INFINITY
            };
            let local_maximum = warp_max(score);
            if lane == 0 {
                *maxima.add(warp_id) = local_maximum;
            }
            mbarrier::barrier_cta_sync(1, 128);
            let maximum = max(
                max(*maxima, *maxima.add(1)),
                max(*maxima.add(2), *maxima.add(3)),
            );
            // Every launched split has at least one valid key, hence a finite maximum.
            let weight = float::ex2_approx_f32((score - maximum) * SCALE_LOG2);
            if tid < 64 {
                *scores.add(tid) = weight;
            }
            let local_sum = warp_sum(weight);
            if lane == 0 {
                *sums.add(warp_id) = local_sum;
            }
            mbarrier::barrier_cta_sync(1, 128);

            // Each warp integrates 16 keys into two output dimensions per lane.
            let mut numerator = [0.0f32; 2];
            static_for!(ROW in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                let row = warp_id * 16 + ROW;
                let probability = *scores.add(row);
                static_for!(D in [0, 1] {
                    numerator[D] += probability * *sv.base.add(shared_offset(row, lane * 2 + D)) as f32;
                });
            });
            static_for!(D in [0, 1] {
                *warp_outputs.add(warp_id * 64 + lane * 2 + D) = numerator[D];
            });
            mbarrier::barrier_cta_sync(1, 128);
            if warp_id == 0 {
                let dst = partials
                    .as_mut_ptr()
                    .add((head * splits + split) * STATE_SIZE);
                static_for!(D in [0, 1] {
                    let column = lane * 2 + D;
                    let mut sum = 0.0;
                    static_for!(WARP in [0, 1, 2, 3] {
                        sum += *warp_outputs.add(WARP * 64 + column);
                    });
                    *dst.add(2 + column) = sum;
                });
                if lane == 0 {
                    *dst = maximum;
                    *dst.add(1) = *sums + *sums.add(1) + *sums.add(2) + *sums.add(3);
                }
            }
        }
    }

    /// Stably combine all splits for one head and write its normalized FP16 output.
    #[launch_bounds(32)]
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(12,0))]
    #[kernel]
    pub fn fmha_decode_merge<const SEQ_K: u32, const HEADS: u32>(
        partials: &[f32],
        mut out: DisjointSlice<f16>,
        heads: u32,
        splits: u32,
    ) {
        const { assert_shape::<SEQ_K, HEADS>() };
        let heads = if HEADS == 0 { heads } else { HEADS };
        let splits = if SEQ_K == 0 {
            splits
        } else {
            SEQ_K.div_ceil(64)
        };
        let head = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if head < heads as usize {
            unsafe {
                let base = partials.as_ptr().add(head * splits as usize * STATE_SIZE);
                let mut maximum = f32::NEG_INFINITY;
                let mut split = 0usize;
                while split < splits as usize {
                    maximum = max(maximum, *base.add(split * STATE_SIZE));
                    split += 1;
                }
                let mut denominator = 0.0f32;
                let mut numerator = [0.0f32; 2];
                split = 0;
                while split < splits as usize {
                    let state = base.add(split * STATE_SIZE);
                    let rescale = float::ex2_approx_f32((*state - maximum) * SCALE_LOG2);
                    denominator += *state.add(1) * rescale;
                    static_for!(D in [0, 1] {
                        numerator[D] += *state.add(2 + lane * 2 + D) * rescale;
                    });
                    split += 1;
                }
                let reciprocal = float::rcp_approx_ftz_f32(denominator);
                static_for!(D in [0, 1] {
                    *out.as_mut_ptr().add(head * 64 + lane * 2 + D) = (numerator[D] * reciprocal) as f16;
                });
            }
        }
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-token attention: parallel KV tiles followed by a stable split reduction.
//!
//! CuTe copies stage 64 keys and values per CTA. Four lanes cooperate on each
//! dot product; no query padding or matrix-multiply query rows are allocated.
//! Scores, probabilities, split numerators and reduction states remain FP32.
use ctm_rs::{
    cuda_module, kernel, launch_bounds, launch_contract,
    memory::{DisjointSlice, DynamicSharedArray},
    thread,
};
use cute_rs::arch::{
    NamedBarrier, exp2_f32, quad_sum_f32, reciprocal_ftz_f32, warp_reduction_max_f32,
    warp_reduction_sum_f32,
};
use cute_rs::copy::cooperative::finish_async_copies;
use cute_rs::markers::{C, ComposedLayout, CopyG2S, Layout, LeadingDim, RowMajor, Swizzle};
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::{GmemMatrix, SmemTile, TiledCopy};

/// Swizzled 64-by-64 FP16 layout used for both K and V in shared memory.
type KvLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>;
/// Swizzled 16-by-64 region covered by one cooperative copy.
type CopyLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<16, 64>>;
/// Map 128 threads to 16 rows and eight 16-byte copy positions.
type CopyThreads = Layout<(C<16>, C<8>), (C<8>, C<1>)>;
/// Eight adjacent FP16 values copied by each thread.
type CopyValues = Layout<(C<1>, C<8>), (C<8>, C<1>)>;
/// Block-wide cooperative global-to-shared copy plan for one 16-by-64 KV group.
type GmemTiledCopy = TiledCopy<CopyG2S<16>, CopyThreads, CopyValues, RowMajor<16, 64>>;
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

/// Issue cooperative copies for one KV split, zero-filling rows past the sequence.
///
/// # Safety
/// All 128 threads must participate with valid source and destination views; wait before reading the tile.
#[inline(always)]
unsafe fn stage(src: &GmemMatrix<f16, LeadingDim<8>>, split: usize, dst: KvTile, tid: u32) {
    let thr_copy = GmemTiledCopy::new().get_slice(tid);
    static_for!(GROUP in [0, 1, 2, 3] {
        // CuTe's copy predicate zero-fills rows beyond the actual KV length.
        unsafe {
            let mut part = dst.local_tile::<CopyLayout>(GROUP);
            thr_copy.copy_g2s(src, (split * 4 + GROUP, 0), &mut part);
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
        let splits = (seq_k as usize).div_ceil(64);
        let split = thread::blockIdx_x() as usize % splits;
        let head = thread::blockIdx_x() as usize / splits;
        unsafe {
            let smem = DynamicSharedArray::<f16, 1024>::get_raw().cast::<f16>();
            let sk = KvTile::from_raw_parts(smem, 4096);
            let sv = sk.next_tile();
            let (scores, score_out) = sv.trailing_views::<f32>(0, 64);
            let (maxima, maxima_out) = sv.trailing_views::<f32>(64, 4);
            let (sums, sums_out) = sv.trailing_views::<f32>(68, 4);
            let (warp_outputs, warp_out) = sv.trailing_views::<f32>(72, 256);
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
            stage(&global_k, split, sk, tid as u32);
            stage(&global_v, split, sv, tid as u32);
            finish_async_copies::<1, 128>();

            let col_base = (tid % 4) * 16;
            let query = StridedTensor::from_slice(q);
            let q_base = head * 64 + col_base;
            // 32 independent keys at a time, with four lanes per 64-D dot.
            static_for!(PASS in [0, 1] {
                let row = tid / 4 + PASS * 32;
                let mut dot = 0.0f32;
                static_for!(D in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                    let column = col_base + D;
                    dot += query.at_unchecked(q_base + D) as f32
                        * sk.get(row, column) as f32;
                });
                let dot = quad_sum_f32(dot);
                if tid.is_multiple_of(4) {
                    score_out.write_at(
                        row,
                        if split * 64 + row < seq_k as usize {
                            dot
                        } else {
                            f32::NEG_INFINITY
                        },
                    );
                }
            });
            NamedBarrier::<1, 128>::sync();
            let score = if tid < 64 {
                scores.at_unchecked(tid)
            } else {
                f32::NEG_INFINITY
            };
            let local_maximum = warp_reduction_max_f32(score);
            if lane == 0 {
                maxima_out.write_at(warp_id, local_maximum);
            }
            NamedBarrier::<1, 128>::sync();
            let maximum = max(
                max(maxima.at_unchecked(0), maxima.at_unchecked(1)),
                max(maxima.at_unchecked(2), maxima.at_unchecked(3)),
            );
            // Every launched split has at least one valid key, hence a finite maximum.
            let weight = exp2_f32((score - maximum) * SCALE_LOG2);
            if tid < 64 {
                score_out.write_at(tid, weight);
            }
            let local_sum = warp_reduction_sum_f32(weight);
            if lane == 0 {
                sums_out.write_at(warp_id, local_sum);
            }
            NamedBarrier::<1, 128>::sync();

            // Each warp integrates 16 keys into two output dimensions per lane.
            let mut numerator = [0.0f32; 2];
            static_for!(ROW in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
                let row = warp_id * 16 + ROW;
                let probability = scores.at_unchecked(row);
                static_for!(D in [0, 1] {
                    numerator[D] += probability * sv.get(row, lane * 2 + D) as f32;
                });
            });
            static_for!(D in [0, 1] {
                warp_out.write_at(warp_id * 64 + lane * 2 + D, numerator[D]);
            });
            NamedBarrier::<1, 128>::sync();
            if warp_id == 0 {
                let dst = StridedTensorMut::from_disjoint_slice(&mut partials)
                    .offset_unchecked((head * splits + split) * STATE_SIZE, STATE_SIZE);
                static_for!(D in [0, 1] {
                    let column = lane * 2 + D;
                    let mut sum = 0.0;
                    static_for!(WARP in [0, 1, 2, 3] {
                        sum += warp_outputs.at_unchecked(WARP * 64 + column);
                    });
                    dst.write_at(2 + column, sum);
                });
                if lane == 0 {
                    dst.write_at(0, maximum);
                    dst.write_at(
                        1,
                        sums.at_unchecked(0)
                            + sums.at_unchecked(1)
                            + sums.at_unchecked(2)
                            + sums.at_unchecked(3),
                    );
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
                let states = StridedTensor::from_slice(partials).offset_unchecked(
                    head * splits as usize * STATE_SIZE,
                    splits as usize * STATE_SIZE,
                );
                let mut maximum = f32::NEG_INFINITY;
                let mut split = 0usize;
                while split < splits as usize {
                    maximum = max(maximum, states.at_unchecked(split * STATE_SIZE));
                    split += 1;
                }
                let mut denominator = 0.0f32;
                let mut numerator = [0.0f32; 2];
                split = 0;
                while split < splits as usize {
                    let state = states.offset_unchecked(split * STATE_SIZE, STATE_SIZE);
                    let rescale = exp2_f32((state.at_unchecked(0) - maximum) * SCALE_LOG2);
                    denominator += state.at_unchecked(1) * rescale;
                    static_for!(D in [0, 1] {
                        numerator[D] += state.at_unchecked(2 + lane * 2 + D) * rescale;
                    });
                    split += 1;
                }
                let reciprocal = reciprocal_ftz_f32(denominator);
                let out_tensor =
                    StridedTensorMut::from_disjoint_slice(&mut out).offset_unchecked(head * 64, 64);
                static_for!(D in [0, 1] {
                    out_tensor.write_at(lane * 2 + D, (numerator[D] * reciprocal) as f16);
                });
            }
        }
    }
}

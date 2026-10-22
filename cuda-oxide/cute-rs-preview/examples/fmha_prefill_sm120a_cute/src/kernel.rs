/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere-style FA2: typed CuTe copies, register MMA, and online softmax.
//! One CTA owns 64 queries; four warps each own 16 rows of a 64-column tile.
use core::marker::PhantomData;
use ctm_rs::{
    async_copy, barrier, convert, cuda_module, float, kernel, launch_bounds, launch_contract,
    memory::{DisjointSlice, DynamicSharedArray},
    thread, warp,
};
use cute_rs::markers::{C, ComposedLayout, CopyG2S, Layout, LeadingDim, RowMajor, Swizzle};
use cute_rs::{FragmentA, FragmentBPair, FragmentC, GmemMatrix, SmemTile, copy_g2s};

/// Swizzled 64-by-64 FP16 layout shared by Q, K, V, and the output staging tile.
type SharedLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>;
/// Swizzled 16-by-64 subview used by one cooperative copy.
type CopyLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<16, 64>>;
/// Map 128 threads to 16 rows and eight 16-byte copy positions.
type CopyThreads = Layout<(C<16>, C<8>), (C<8>, C<1>)>;
/// Eight adjacent FP16 values copied by each thread.
type CopyValues = Layout<(C<1>, C<8>), (C<8>, C<1>)>;
/// Typed view of one 64-by-64 FP16 shared tile.
type SharedTile = SmemTile<f16, SharedLayout>;

// Constant indices keep fragments in registers and allow the compiler to
// specialize each MMA, without repeating the algorithm for each fragment.
/// Expand fixed iterations with constant indices so MMA fragments stay in registers.
macro_rules! static_for {
    ($index:ident in [$($value:expr),*] $body:block) => {
        $({ #[doc = "Compile-time index for this unrolled iteration."] const $index: usize = $value; $body })*
    };
}

/// The eight 16x8 accumulators making one warp's 16x64 tile.
struct WarpTile(#[doc = "Eight adjacent 16-by-8 FP32 accumulator fragments."] [FragmentC; 8]);

impl WarpTile {
    /// Create an all-zero warp accumulator tile.
    #[inline(always)]
    fn zero() -> Self {
        Self([FragmentC::ZERO; 8])
    }

    /// Accumulate one K16 step across all eight column fragments, one atom call each.
    ///
    /// # Safety
    /// All warp lanes must participate with matching, valid operand fragments.
    #[inline(always)]
    unsafe fn gemm(&mut self, a: FragmentA, b: [FragmentBPair; 4]) {
        static_for!(PAIR in [0, 1, 2, 3] {
            self.0[2 * PAIR] =
                unsafe { self.0[2 * PAIR].mma_atom_call(a, b[PAIR].get::<0>()) };
            self.0[2 * PAIR + 1] =
                unsafe { self.0[2 * PAIR + 1].mma_atom_call(a, b[PAIR].get::<1>()) };
        });
    }

    /// Retile two adjacent C fragments into one A fragment, entirely in registers.
    #[inline(always)]
    fn probability<const K: usize>(&self) -> FragmentA {
        const { assert!(K < 4) };
        let left = self.0[2 * K].0;
        let right = self.0[2 * K + 1].0;
        FragmentA([
            pack(left[0], left[1]),
            pack(left[2], left[3]),
            pack(right[0], right[1]),
            pack(right[2], right[3]),
        ])
    }

    /// Narrow the whole probability tile before loading the value fragments.
    #[inline(always)]
    fn probabilities(&self) -> [FragmentA; 4] {
        let mut fragments = [FragmentA([0; 4]); 4];
        static_for!(K in [0, 1, 2, 3] {
            fragments[K] = self.probability::<K>();
        });
        fragments
    }

    /// Scale one of this lane's two rows across all accumulator fragments.
    #[inline(always)]
    fn rescale<const ROW: usize>(&mut self, scale: f32) {
        static_for!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
            self.0[N].0[ROW * 2] *= scale;
            self.0[N].0[ROW * 2 + 1] *= scale;
        });
    }

    /// Reduce this lane's 16 scores as a tree, as in the CuTe row reduction.
    #[inline(always)]
    fn maximum<const ROW: usize>(&self) -> f32 {
        let mut pairs = [0.0; 8];
        static_for!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
            pairs[N] = self.0[N].0[ROW * 2].max(self.0[N].0[ROW * 2 + 1]);
        });
        static_for!(N in [0, 1, 2, 3] {
            pairs[N] = pairs[2 * N].max(pairs[2 * N + 1]);
        });
        static_for!(N in [0, 1] {
            pairs[N] = pairs[2 * N].max(pairs[2 * N + 1]);
        });
        pairs[0].max(pairs[1])
    }

    /// Mask and exponentiate scores, updating the running normalization and output scale.
    #[inline(always)]
    fn softmax<const CAUSAL: bool, const FIRST: bool>(
        &mut self,
        output: &mut Self,
        state: &mut Softmax,
        query_row: u32,
        lane: u32,
    ) {
        static_for!(ROW in [0, 1] {
            let row = query_row + ROW as u32 * 8;
            static_for!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
                static_for!(COL in [0, 1] {
                    let column = N as u32 * 8 + (lane % 4) * 2 + COL as u32;
                    let value = if CAUSAL && FIRST && column > row { f32::NEG_INFINITY }
                                else { self.0[N].0[ROW * 2 + COL] };
                    self.0[N].0[ROW * 2 + COL] = value;
                });
            });
            let mut maximum = self.maximum::<ROW>();
            maximum = max(maximum, warp::shuffle_xor_f32(maximum, 2));
            maximum = max(maximum, warp::shuffle_xor_f32(maximum, 1));
            if !FIRST { maximum = max(maximum, state.maximum[ROW]); }
            // Causal traversal may begin with a fully masked row in this tile.
            if CAUSAL && maximum == f32::NEG_INFINITY { maximum = 0.0; }
            /// Attention scale converted to base two for the exponential instruction.
            const SCALE_LOG2: f32 = 0.125 * core::f32::consts::LOG2_E;
            let mut sum = 0.0;
            static_for!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
                static_for!(COL in [0, 1] {
                    let value = float::ex2_approx_ftz_f32(
                        self.0[N].0[ROW * 2 + COL] * SCALE_LOG2 - maximum * SCALE_LOG2,
                    );
                    self.0[N].0[ROW * 2 + COL] = value;
                    sum += value;
                });
            });
            if FIRST {
                state.sum[ROW] = sum;
            } else {
                let old_scale = float::ex2_approx_ftz_f32(
                    state.maximum[ROW] * SCALE_LOG2 - maximum * SCALE_LOG2,
                );
                state.sum[ROW] = sum + state.sum[ROW] * old_scale;
                output.rescale::<ROW>(old_scale);
            }
            state.maximum[ROW] = maximum;
        });
    }

    /// Normalize the output and pack it into the shared tile.
    ///
    /// # Safety
    /// All warp lanes must participate; `dst` must provide disjoint slots for each warp.
    #[inline(always)]
    unsafe fn store(&mut self, state: &Softmax, dst: &SharedTile, warp_id: usize, lane: usize) {
        static_for!(ROW in [0, 1] {
            let mut sum = state.sum[ROW];
            sum += warp::shuffle_xor_f32(sum, 2);
            sum += warp::shuffle_xor_f32(sum, 1);
            let scale = if sum == 0.0 || sum.is_nan() { 1.0 }
                        else { float::rcp_approx_ftz_f32(sum) };
            self.rescale::<ROW>(scale);
            static_for!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
                let row = warp_id * 16 + lane / 4 + ROW * 8;
                let col = N * 8 + lane % 4 * 2;
                unsafe {
                    *dst.base.add(shared_offset(row as u32, col as u32)).cast::<u32>() =
                        pack(self.0[N].0[ROW * 2], self.0[N].0[ROW * 2 + 1]);
                }
            });
        });
    }
}

/// Running softmax statistics for the two rows represented by each lane.
struct Softmax {
    /// Running maximum for each row, shared by its four cooperating lanes.
    maximum: [f32; 2],
    /// Lane-local sums to be combined across each row's four lanes at output time.
    sum: [f32; 2],
}

/// Return the floating-point maximum used by row reductions.
#[inline(always)]
fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Round two FP32 values to the packed FP16 representation used by MMA and stores.
#[inline(always)]
fn pack(a: f32, b: f32) -> u32 {
    convert::cvt_f16x2_f32(a, b)
}

/// Physical coordinate of the typed `Swizzle<3,3,3>` shared layout.
#[inline(always)]
fn shared_offset(row: u32, column: u32) -> usize {
    // Within this 64-column tile, the row and column occupy disjoint bits.
    // Keep the row stride separate so adjacent fragment rows share an offset.
    (row * 64 + (column ^ ((row & 7) << 3))) as usize
}

/// Wrap storage for 4096 FP16 values as a typed shared tile.
///
/// # Safety
/// `base` must point to live shared storage with the tile's required alignment.
#[inline(always)]
unsafe fn tile(base: *mut f16) -> SharedTile {
    SmemTile {
        base,
        capacity: 4096,
        layout: PhantomData,
    }
}

/// Issue four cooperative copies to stage one 64-row tile.
///
/// # Safety
/// All 128 threads must participate with valid views; wait before reading the destination.
#[inline(always)]
unsafe fn stage(src: &GmemMatrix<f16, LeadingDim<8>>, block: usize, dst: &SharedTile, tid: u32) {
    static_for!(GROUP in [0, 1, 2, 3] {
        let mut part = SmemTile::<f16, CopyLayout> {
            base: unsafe { dst.base.add(GROUP * 1024) }, capacity: 1024, layout: PhantomData,
        };
        unsafe {
            copy_g2s::<CopyG2S<16>, CopyThreads, CopyValues, RowMajor<16, 64>, CopyLayout, f16>(
                src, (block * 4 + GROUP, 0), &mut part, tid,
            );
        }
    });
}

/// Stage a complete KV tile. The checked host shape is a multiple of 64,
/// and reverse traversal keeps `block` within that shape.
#[inline(always)]
unsafe fn stage_full(
    src: &GmemMatrix<f16, LeadingDim<8>>,
    block: usize,
    dst: &SharedTile,
    tid: u32,
) {
    let source_tile = GmemMatrix {
        base: unsafe { src.base.add(block * 64 * src.leading_dim.elements) },
        rows: 64,
        cols: 64,
        leading_dim: LeadingDim::<8> {
            elements: src.leading_dim.elements,
        },
    };
    unsafe { stage(&source_tile, 0, dst, tid) };
}

/// Load four pairs of K-major B fragments for one K16 reduction step.
///
/// # Safety
/// The shared tile must be ready, and every warp lane must participate with its own lane ID.
#[inline(always)]
unsafe fn load_k<const K: usize>(src: &SharedTile, lane: u32) -> [FragmentBPair; 4] {
    let mut fragments = [FragmentBPair([0; 4]); 4];
    static_for!(N in [0, 1, 2, 3] {
        fragments[N] = unsafe { cute_rs::load_matrix_b_pair_k_major(src, (N, K), lane) };
    });
    fragments
}

/// Load four pairs of V fragments for one K16 reduction step.
///
/// # Safety
/// The shared tile must be ready, and every warp lane must participate with its own lane ID.
#[inline(always)]
unsafe fn load_v<const K: usize>(src: &SharedTile, lane: u32) -> [FragmentBPair; 4] {
    let mut fragments = [FragmentBPair([0; 4]); 4];
    static_for!(N in [0, 1, 2, 3] {
        fragments[N] = unsafe { cute_rs::load_matrix_b_pair(src, (K, N), lane) };
    });
    fragments
}

/// Shared-memory pipeline for one query tile. Only the first visited KV tile
/// can contain future tokens; all earlier tiles are wholly visible.
struct TilePipeline {
    /// Query tile, reused as output staging after the mainloop.
    q: SharedTile,
    /// Shared key tile, refilled while the current scores are consumed.
    k: SharedTile,
    /// Shared value tile used by the probability-times-value MMA.
    v: SharedTile,
    /// Global K matrix for the CTA's attention head.
    keys: GmemMatrix<f16, LeadingDim<8>>,
    /// Global V matrix for the CTA's attention head.
    values: GmemMatrix<f16, LeadingDim<8>>,
    /// CTA-local thread ID used for cooperative copies and warp ownership.
    tid: u32,
}

impl TilePipeline {
    /// Consume one KV tile, update softmax and output, and prefetch the preceding key tile.
    ///
    /// # Safety
    /// All 128 threads must call in lockstep with live, nonoverlapping shared tiles and valid global views.
    #[inline(always)]
    unsafe fn step<const CAUSAL: bool, const FIRST: bool>(
        &self,
        key_block: usize,
        output: &mut WarpTile,
        state: &mut Softmax,
    ) {
        let tid = self.tid;
        let lane = tid % 32;
        let warp_id = (tid / 32) as usize;
        // SAFETY: the caller supplies full KV tiles and the shared allocations
        // remain live across all steps; waits and CTA barriers order reuse.
        unsafe {
            async_copy::cp_async_wait_group::<0>();
            barrier::barrier_cta_sync(1, 128);
            stage_full(&self.values, key_block, &self.v, tid);
            async_copy::cp_async_commit_group();
            let mut scores = WarpTile::zero();
            let mut a = cute_rs::load_matrix_a(&self.q, (warp_id, 0), lane);
            let mut b = load_k::<0>(&self.k, lane);
            static_for!(K in [0, 1, 2] {
                let next_a = cute_rs::load_matrix_a(&self.q, (warp_id, K + 1), lane);
                let next_b = load_k::<{ K + 1 }>(&self.k, lane);
                scores.gemm(a, b);
                a = next_a;
                b = next_b;
            });
            scores.gemm(a, b);
            async_copy::cp_async_wait_group::<0>();
            barrier::barrier_cta_sync(1, 128);
            if key_block > 0 {
                stage_full(&self.keys, key_block - 1, &self.k, tid);
                async_copy::cp_async_commit_group();
            }
            scores.softmax::<CAUSAL, FIRST>(output, state, warp_id as u32 * 16 + lane / 4, lane);
            let probabilities = scores.probabilities();
            let mut values = load_v::<0>(&self.v, lane);
            static_for!(K in [0, 1, 2] {
                let next = load_v::<{ K + 1 }>(&self.v, lane);
                output.gemm(probabilities[K], values);
                values = next;
            });
            output.gemm(probabilities[3], values);
        }
    }
}

/// Device entry point and generated host launcher for prefill attention.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Compute one head's 64-query tile with reverse KV traversal and online softmax.
    #[launch_bounds(128)]
    #[launch_contract(domain=1, block=(128,1,1), dynamic_shared=24576,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    pub fn fmha_prefill<const CAUSAL: bool, const SQ: u32, const SK: u32, const HEADS: u32>(
        q: &[f16],
        k: &[f16],
        v: &[f16],
        mut out: DisjointSlice<f16>,
        seq_q: u32,
        seq_k: u32,
        heads: u32,
    ) {
        // Zero selects runtime geometry. Static instantiations fix all three
        // dimensions; the host checks them before dispatching this entry point.
        const { assert!((SQ == 0 && SK == 0 && HEADS == 0) || (SQ == 1024 && SK == 1024 && HEADS == 8)) };
        let seq_q = if SQ == 0 { seq_q } else { SQ };
        let seq_k = if SK == 0 { seq_k } else { SK };
        let heads = if HEADS == 0 { heads } else { HEADS };
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp_id = (tid / 32) as usize;
        let query_blocks = seq_q.div_ceil(64);
        let query_block = (thread::blockIdx_x() % query_blocks) as usize;
        let head = (thread::blockIdx_x() / query_blocks) as usize;
        let stride = heads as usize * 64;
        unsafe {
            let shared = DynamicSharedArray::<f16, 1024>::get_raw().cast::<f16>();
            let sq = tile(shared);
            let sk = tile(shared.add(4096));
            let sv = tile(shared.add(8192));
            let q_global = GmemMatrix {
                base: q.as_ptr().add(head * 64),
                rows: seq_q as usize,
                cols: 64,
                leading_dim: LeadingDim::<8> { elements: stride },
            };
            let k_global = GmemMatrix {
                base: k.as_ptr().add(head * 64),
                rows: seq_k as usize,
                cols: 64,
                leading_dim: LeadingDim::<8> { elements: stride },
            };
            let v_global = GmemMatrix {
                base: v.as_ptr().add(head * 64),
                rows: seq_k as usize,
                cols: 64,
                leading_dim: LeadingDim::<8> { elements: stride },
            };
            // Causal shapes have equal sequence lengths. The first visited
            // tile is the diagonal, so its mask uses tile-local coordinates.
            let mut remaining = if CAUSAL {
                query_block + 1
            } else {
                seq_k as usize / 64
            };
            stage(&q_global, query_block, &sq, tid);
            stage_full(&k_global, remaining - 1, &sk, tid);
            async_copy::cp_async_commit_group();
            let mut output = WarpTile::zero();
            let mut state = Softmax {
                maximum: [f32::NEG_INFINITY; 2],
                sum: [0.0; 2],
            };
            let pipeline = TilePipeline {
                q: sq,
                k: sk,
                v: sv,
                keys: k_global,
                values: v_global,
                tid,
            };
            // The first tile initializes softmax and handles the causal diagonal.
            // Keep the steady-state loop free of masks and first-tile rescaling.
            remaining -= 1;
            pipeline.step::<CAUSAL, true>(remaining, &mut output, &mut state);
            while remaining > 0 {
                remaining -= 1;
                pipeline.step::<CAUSAL, false>(remaining, &mut output, &mut state);
            }
            output.store(&state, &pipeline.q, warp_id, lane as usize);
            barrier::barrier_cta_sync(1, 128);
            static_for!(GROUP in [0, 1, 2, 3] {
                let row = tid as usize / 8 + GROUP * 16;
                let col = tid as usize % 8 * 8;
                if query_block * 64 + row < seq_q as usize {
                    let packed = *pipeline.q.base.add(shared_offset(row as u32, col as u32)).cast::<u128>();
                    *out.as_mut_ptr().add((query_block * 64 + row) * stride + head * 64 + col)
                        .cast::<u128>() = packed;
                }
            });
        }
    }
}

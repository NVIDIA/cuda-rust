/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 grouped-query attention using parallel, stable 64-key reductions.
use ctm_rs::{
    cuda_module, kernel, launch_bounds, launch_contract,
    memory::{DisjointSlice, DynamicSharedArray},
    thread,
};
use cute_rs::arch::NamedBarrier;
use cute_rs::markers::{C, ComposedLayout, CopyG2S, Layout, LeadingDim, RowMajor, Swizzle};
use cute_rs::static_layout::StaticLayout;
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::{BFloat16, GmemMatrix, RegisterTile, SmemTile, TiledCopy};

/// Shared bytes for two 64-by-128 BF16 tiles and FP32 reduction state.
pub const PARTIAL_SHARED_BYTES: u32 = 35_104;
/// FP32 values per split: maximum, denominator, and 128 numerator coordinates.
pub const STATE_SIZE: usize = 130;

/// Shared-memory storage layout for one complete BF16 key or value tile.
type KvLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 128>>;
/// Eight rows per 128-thread copy; each thread transports eight BF16 values.
type CopyLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<8, 128>>;
/// Map 128 copy threads to eight rows and sixteen contiguous 8-element vectors per row.
type CopyThreads = Layout<(C<8>, C<16>), (C<16>, C<1>)>;
/// Map each copy thread's eight contiguous BF16 carriers to a 16-byte transaction.
type CopyValues = Layout<(C<1>, C<8>), (C<8>, C<1>)>;
/// Cooperative global-to-shared copy plan for one 8-by-128 BF16 KV group.
type KvTiledCopy = TiledCopy<CopyG2S<16>, CopyThreads, CopyValues, RowMajor<8, 128>>;
/// Four neighboring threads own consecutive 32-coordinate pieces of a key row.
type DotThreads = Layout<(C<4>, C<32>), (C<32>, C<128>)>;
/// Each lane owns four adjacent numerator coordinates; warps own separate rows.
type NumeratorThreads = Layout<(C<32>, C<4>), (C<4>, C<128>)>;

/// Typed shared-memory BF16 key or value tile in the 64x128 swizzled layout.
#[derive(Clone, Copy)]
struct Bf16KvTile(
    #[doc = "Shared BF16 tile in the typed 64x128 swizzled layout."] SmemTile<BFloat16, KvLayout>,
);

/// Attention scale `1 / sqrt(128)` in base two for the exponential instruction.
const SCALE_LOG2: f32 = 0.088_388_346 * core::f32::consts::LOG2_E;

/// Expand fixed iterations so per-lane numerator coordinates remain registers.
macro_rules! static_for {
    ($index:ident in [$($value:expr),*] $body:block) => {
        $({ const $index: usize = $value; $body })*
    };
}

/// Return the greater score for the finite-score and negative-infinity reduction domain.
#[inline(always)]
fn maximum(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

impl Bf16KvTile {
    /// Wrap shared storage as a typed 64x128 BF16 tile without initializing it.
    ///
    /// # Safety
    /// `base` supplies 8192 live, suitably aligned shared-memory `BFloat16` elements.
    #[inline(always)]
    unsafe fn new(base: *mut BFloat16) -> Self {
        Self(unsafe { SmemTile::from_raw_parts(base, 8192) })
    }

    /// Return the adjacent 64x128 BF16 shared-memory tile immediately following `self`.
    ///
    /// # Safety
    /// Shared memory must contain an additional 8192 `BFloat16` elements after `self`.
    #[inline(always)]
    unsafe fn next_tile(self) -> Self {
        Self(unsafe { self.0.next_tile() })
    }

    /// Load `N` neighboring BF16 coordinates and widen them to FP32 registers.
    ///
    /// # Safety
    /// The copy must be complete and CTA-synchronized; row < 64 and the
    /// complete `column..column + N` interval must lie within 128 columns.
    #[inline(always)]
    unsafe fn row<const N: usize>(&self, row: usize, column: usize) -> RegisterTile<f32, N> {
        unsafe { self.0.load::<N>(row, column) }.map(BFloat16::to_f32)
    }
}

/// Stage only initialized and causally visible cache rows, zero-filling tails.
///
/// # Safety
/// All 128 threads participate with their unique thread IDs. `source` contains
/// at least `visible * kv_heads * 128` initialized BF16 values, `head < kv_heads`,
/// and destination storage remains live until the copy has been waited on.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn stage(
    source: &[BFloat16],
    head: usize,
    kv_heads: usize,
    visible: usize,
    split: usize,
    destination: Bf16KvTile,
    tid: u32,
) {
    let source = GmemMatrix {
        base: unsafe { source.as_ptr().add(head * 128) },
        rows: visible,
        cols: 128,
        leading_dim: LeadingDim::<8> {
            elements: kv_heads * 128,
        },
    };
    let thr_copy = KvTiledCopy::new().get_slice(tid);
    static_for!(GROUP in [0, 1, 2, 3, 4, 5, 6, 7] {
        unsafe {
            let mut part = destination.0.local_tile::<CopyLayout>(GROUP);
            thr_copy.copy_g2s(&source, (split * 8 + GROUP, 0), &mut part);
        }
    });
}

/// Fixed-launch BF16 split-attention kernels; host code proves extents and readiness.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Produce one stable FP32 attention state per query/head and 64-key split.
    ///
    /// Q is dense `[query_tokens, query_heads, 128]`; K/V use
    /// `[key_tokens, kv_heads, 128]`. Adjacent query-head groups share a KV head.
    /// Each state stores an unscaled score maximum, scaled-exponential denominator,
    /// and 128 weighted-value numerator coordinates. Causal offsets count physical
    /// cache tokens; fully masked splits publish a neutral state without reading K/V.
    ///
    /// The launch contract requires 128 threads and 35,104 shared bytes aligned to
    /// 1024 bytes. The host must validate positive integral head grouping, a nonempty
    /// key prefix for nonempty queries, every buffer/scratch extent, and readiness.
    /// All threads of each active CTA reach copies, waits, and barriers convergently;
    /// the type system does not establish that participation or output disjointness.
    #[launch_bounds(128)]
    #[launch_contract(domain=1, block=(128,1,1), dynamic_shared=35104,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn partial(
        q: &[BFloat16],
        k: &[BFloat16],
        v: &[BFloat16],
        mut partials: DisjointSlice<f32>,
        query_tokens: u32,
        key_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        query_start: u32,
        causal: bool,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let numerator_column = NumeratorThreads::crd2idx(lane);
        let warp_numerator_offset = NumeratorThreads::crd2idx(tid);
        let splits = (key_tokens as usize).div_ceil(64);
        let work = thread::blockIdx_x() as usize;
        let split = work % splits;
        let query_head = (work / splits) % query_heads as usize;
        let query_row = (work / splits) / query_heads as usize;
        // This guard is CTA-uniform. The checked host grid supplies exactly
        // the query rows, heads and splits described by these dimensions.
        if query_row >= query_tokens as usize {
            return;
        }
        let visible = if causal {
            (query_start as usize + query_row + 1).min(key_tokens as usize)
        } else {
            key_tokens as usize
        };
        let kv_head = query_head / (query_heads as usize / kv_heads as usize);

        // SAFETY: the host checks head grouping, all tensor/scratch extents,
        // initialized KV prefix, context/readiness and the exact launch shape.
        // Each CTA owns one 130-value state; each lane owns four numerator
        // coordinates. All collectives below execute convergently.
        unsafe {
            let mut destination = StridedTensorMut::from_disjoint_slice(&mut partials)
                .subview_unchecked(work * STATE_SIZE, STATE_SIZE);
            if split * 64 >= visible {
                // A causal query can have whole future splits. Publish the
                // neutral reduction state without touching any cache storage.
                if warp_id == 0 {
                    destination
                        .reborrow()
                        .subview_unchecked(2 + numerator_column, 4)
                        .store(RegisterTile::from_array([0.0; 4]));
                    if lane == 0 {
                        destination.store_at(0, f32::NEG_INFINITY);
                        destination.store_at(1, 0.0);
                    }
                }
                return;
            }

            let shared = DynamicSharedArray::<u16, 1024>::get_raw().cast::<BFloat16>();
            let keys = Bf16KvTile::new(shared);
            let values = keys.next_tile();
            let (scores, mut score_output) = values.0.trailing_views::<f32>(0, 64);
            let (maxima, mut maximum_output) = values.0.trailing_views::<f32>(64, 4);
            let (sums, mut sum_output) = values.0.trailing_views::<f32>(68, 4);
            let (warp_outputs, mut warp_output) = values.0.trailing_views::<f32>(72, 512);
            stage(
                k,
                kv_head,
                kv_heads as usize,
                visible,
                split,
                keys,
                tid as u32,
            );
            stage(
                v,
                kv_head,
                kv_heads as usize,
                visible,
                split,
                values,
                tid as u32,
            );
            cute_rs::copy::cooperative::finish_async_copies::<1, 128>();

            let query_base = (query_row * query_heads as usize + query_head) * 128;
            // Four neighboring lanes sum 32 dimensions apiece. Two passes
            // cover all 64 keys, including zero-filled tail rows.
            static_for!(PASS in [0, 1] {
                let origin = DotThreads::crd2idx(tid);
                let row = origin / 128 + PASS * 32;
                let column = origin % 128;
                let query = StridedTensor::from_slice(q)
                    .subview_unchecked(query_base + column, 32);
                let dot = keys.0.row_view::<32>(row, column)
                    .zip_fold(query, 0.0_f32, |dot, key, query| {
                        dot + query.to_f32() * key.to_f32()
                    });
                let dot = cute_rs::arch::quad_sum_f32(dot);
                if tid.is_multiple_of(4) {
                    score_output.store_at(row, if split * 64 + row < visible {
                        dot
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            });
            NamedBarrier::<1, 128>::sync();

            let score = if tid < 64 {
                scores.get_unchecked(tid)
            } else {
                f32::NEG_INFINITY
            };
            let local_maximum = cute_rs::arch::warp_reduction_max_f32(score);
            if lane == 0 {
                maximum_output.store_at(warp_id, local_maximum);
            }
            NamedBarrier::<1, 128>::sync();
            let maximum = maximum(
                maximum(maxima.get_unchecked(0), maxima.get_unchecked(1)),
                maximum(maxima.get_unchecked(2), maxima.get_unchecked(3)),
            );
            // Nonempty splits have at least one visible score. Explicitly
            // zero masked rows before they contribute to normalization/PV.
            let probability = if tid < 64 && split * 64 + tid < visible {
                cute_rs::arch::exp2_f32((score - maximum) * SCALE_LOG2)
            } else {
                0.0
            };
            if tid < 64 {
                score_output.store_at(tid, probability);
            }
            let local_sum = cute_rs::arch::warp_reduction_sum_f32(probability);
            if lane == 0 {
                sum_output.store_at(warp_id, local_sum);
            }
            NamedBarrier::<1, 128>::sync();

            // A warp integrates 16 keys into four dimensions per lane.
            let mut numerator = RegisterTile::from_array([0.0f32; 4]);
            static_for!(ROW in [0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15] {
                let row = warp_id * 16 + ROW;
                let probability = scores.get_unchecked(row);
                numerator = numerator.zip_map(values.row::<4>(row, numerator_column), |sum, value| sum + probability * value);
            });
            warp_output
                .reborrow()
                .subview_unchecked(warp_numerator_offset, 4)
                .store(numerator);
            NamedBarrier::<1, 128>::sync();
            if warp_id == 0 {
                let mut total = RegisterTile::from_array([0.0_f32; 4]);
                static_for!(WARP in [0, 1, 2, 3] {
                    total = total.zip_map(warp_outputs.subview_unchecked(WARP * 128 + numerator_column, 4)
                        .load_unchecked::<4>(), |a, b| a + b);
                });
                destination
                    .reborrow()
                    .subview_unchecked(2 + numerator_column, 4)
                    .store(total);
                if lane == 0 {
                    destination.store_at(0, maximum);
                    destination.store_at(
                        1,
                        sums.get_unchecked(0)
                            + sums.get_unchecked(1)
                            + sums.get_unchecked(2)
                            + sums.get_unchecked(3),
                    );
                }
            }
        }
    }

    /// Merge completed split states in FP32, then round the normalized output to BF16.
    ///
    /// Scratch is `[query_tokens, query_heads, splits, STATE_SIZE]`; output is dense
    /// `[query_tokens, query_heads, 128]`. One full 32-thread CTA owns each output
    /// row. The caller must provide all initialized states from a completed partial
    /// launch, positive `splits`, sufficient distinct output storage, and buffer
    /// lifetimes through completion. Empty states have zero denominator and are
    /// skipped; each lane stores its own four coordinates.
    #[launch_bounds(32)]
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(12,0))]
    #[kernel]
    pub fn merge(
        partials: &[f32],
        mut out: DisjointSlice<BFloat16>,
        query_tokens: u32,
        query_heads: u32,
        splits: u32,
    ) {
        let work = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let numerator_column = NumeratorThreads::crd2idx(lane);
        if work >= query_tokens as usize * query_heads as usize {
            return;
        }
        // SAFETY: one complete warp owns each query/head output. Host-checked
        // scratch extents include every split; the preceding launch completed
        // on this stream, and output coordinates are pairwise disjoint.
        unsafe {
            let states = StridedTensor::from_slice(partials).subview_unchecked(
                work * splits as usize * STATE_SIZE,
                splits as usize * STATE_SIZE,
            );
            let mut max_score = f32::NEG_INFINITY;
            let mut split = 0;
            while split < splits as usize {
                let state = states.subview_unchecked(split * STATE_SIZE, STATE_SIZE);
                if state.get_unchecked(1) != 0.0 {
                    max_score = maximum(max_score, state.get_unchecked(0));
                }
                split += 1;
            }
            let mut denominator = 0.0f32;
            let mut numerator = RegisterTile::from_array([0.0f32; 4]);
            split = 0;
            while split < splits as usize {
                let state = states.subview_unchecked(split * STATE_SIZE, STATE_SIZE);
                let sum = state.get_unchecked(1);
                // A fully masked split is neutral. Skipping it also prevents
                // evaluating (-infinity)-(-infinity) for an empty state.
                if sum != 0.0 {
                    let scale =
                        cute_rs::arch::exp2_f32((state.get_unchecked(0) - max_score) * SCALE_LOG2);
                    denominator += sum * scale;
                    let partial = state
                        .subview_unchecked(2 + numerator_column, 4)
                        .load_unchecked::<4>();
                    numerator = numerator.zip_map(partial, |sum, value| sum + value * scale);
                }
                split += 1;
            }
            let reciprocal = if denominator == 0.0 {
                0.0
            } else {
                cute_rs::arch::reciprocal_ftz_f32(denominator)
            };
            StridedTensorMut::from_disjoint_slice(&mut out)
                .subview_unchecked(work * 128 + numerator_column, 4)
                .store(numerator.map(|value| BFloat16::from_f32(value * reciprocal)));
        }
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 (`sm_100a`, B200) persistent two-CTA fused QKV projection + per-head `RMSNorm` + `FP8 (e4m3)`
//! quantization + contiguous `Q` store + indirect Paged `KV` cache scatter in `cute-rs`.
//!
//! # 1. How the Fused QKV + Epilogue Kernel Works
//!
//! ```text
//!    Input X                 Stacked Weights W_qkv                GEMM Output (in TMEM)
//! [ M tokens x K ]     x     [ K x (N_q + N_k + N_v) ]     =      [ M tokens x (N_q + N_k + N_v) ]
//!
//! ┌──────────────┐           ┌───────────┬─────┬─────┐            ┌───────────┬─────┬─────┐
//! │   Token 0    │           │           │     │     │            │     Q     │  K  │  V  │
//! │   Token 1    │     x     │    W_Q    │ W_K │ W_V │     =      │  columns  │cols │cols │
//! │     ...      │           │           │     │     │            │           │     │     │
//! │  Token M-1   │           └───────────┴─────┴─────┘            └─────┬─────┴──┬──┴──┬──┘
//! └──────────────┘                                                      │        │     │
//!                                                                       ▼        ▼     ▼
//!                                                       Epilogue:    RMSNorm  RMSNorm  (skip)
//!                                                                       │        │     │
//!                                                                    FP8 Quant  FP8   FP8
//!                                                                       │        │     │
//!                                                                    Store Q   Scatter into
//!                                                                              Paged KV Cache
//! ```
//!
//! # 2. How `cute-rs` Overlaps the Complex Epilogue on B200 (`sm_100a`)
//!
//! Each two-CTA cluster computes a `128 × 256` output tile (`128` tokens × `2` heads of `D = 128`)
//! using an 8-stage cluster TMA ring (`MainloopRing`) and a 2-stage TMEM ping-pong ring
//! (`AccumulatorRing`: `TMEM_0` at columns `0..128`, `TMEM_1` at columns `128..256`):
//!
//! ```text
//! OVERLAPPED EPILOGUE ON B200 (`sm_100a`)
//! ────────────────────────────────────────────────────────────────────────────────────────────
//! Warp 5 (TMA):    [ Stream X (64x64) & W_qkv (128x64) into 8-stage SMEM Ring                ]
//! Warp 4 (MMA):    [ Tile 0 -> TMEM_0 ] [ Tile 1 -> TMEM_1 ] [ Tile 2 -> TMEM_0 ] [ Tile 3 -> ]
//! Warps 0..3:                           [ Tile 0: ld(TMEM_0) -> Release TMEM_0 Immediately!   ]
//!                                       [         -> Register RMSNorm -> FP8 -> Q/KV Scatter  ]
//! ```

#![allow(unsafe_op_in_unsafe_fn)]

use cute_rs::arch::{
    NamedBarrier, block_idx_in_cluster, cluster_arrive_relaxed, cluster_wait, cvt_e4m3x2_f32,
    elect_one_sync, make_warp_uniform, mbarrier_init_fence, rsqrt_f32, setmaxregister_decrease,
    setmaxregister_increase, tcgen05_fence_before_thread_sync, thread_idx, warp_idx,
};
use cute_rs::nvgpu::cpasync::TmaCopy;
use cute_rs::nvgpu::tcgen05::{MmaSs, SharedTile, SharedTileArray, TmemFragment, tmem_row_address};
use cute_rs::pipeline::sm100::{
    AsyncCompletion, AsyncCopyReuse, Consumer, Producer, RingCursor, TensorMemory, TwoCtaRing,
};
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::utils::tmem_allocator::{Sm100DeallocBarrier, Sm100Tmem};
use cute_rs::{
    ColMajor, ComposedLayout, GlobalStoreTile, OperandA, OperandB, PersistentTileSchedulerParams,
    RowMajor, Swizzle, TensorMap, crd2idx_col_major_2d, crd2idx_row_major_2d, crd2idx_row_major_3d,
    size_in_bytes, store_global_no_allocate,
};
use cutlass_primitives as prims;
use prims::mbarrier::Barrier;
use prims::memory::DynamicSharedArray;
use prims::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract};

/// Per-CTA `64 × 64` FP16 K-major `X` activation tile with 128-byte swizzle (`8,192` bytes per stage).
/// Across the two-CTA cluster, the pair covers `M = 128` tokens (`64` rows per CTA) × `K = 64`.
pub type XLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>;

/// Per-CTA `128 × 64` FP16 K-major `W_qkv` weight tile with 128-byte swizzle (`16,384` bytes per stage).
/// Across the two-CTA cluster, the pair covers `N = 256` columns (`2` heads of `D = 128`) × `K = 64`.
pub type WLayout = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;

/// Group-two `128 × 256 × 64` Tensor Core (`tcgen05.mma.cta_group::2.kind::f16`) tiled MMA.
/// Verifies at compile time that `XLayout` (`64×64`) and `WLayout` (`128×64`) match `(128, 256, 64)`.
pub type QkvMma = MmaSs<XLayout, WLayout, 128, 256, 64, 2>;

/// Persistent two-CTA `128 × 256` tile scheduler with `G = 8` M-tile (`1,024`-token) L2 swizzling.
/// Keeps `1,024 × K` of `X` (`8 MB` at `K = 4,096`) hot in B200's L2 cache across all `W_qkv` heads.
pub type TileScheduler = PersistentTileSchedulerParams<128, 256, 8, 2>;

/// Column-major `(64, 2) : (1, 64)` thread layout partitioning the `128` epilogue threads (`Warps 0..3`).
/// Maps flat thread index `tid` (`0..128`) to `(lane_row 0..64, lane_head 0..2)`.
pub type EpilogueThreadLayout = ColMajor<64, 2>;

/// Eight-stage two-CTA TMA-to-MMA input ring (`full[0..8]` and `empty[0..8]` mbarriers).
/// Overlaps 2-CTA cluster TMA loads of `X` and `W_qkv` with Tensor Core MMA consumption.
pub type MainloopRing = TwoCtaRing<8, AsyncCompletion, AsyncCopyReuse>;

/// Single-stage two-CTA TMA-to-MMA input ring used only by the unpipelined ablation kernel.
/// Models an unpipelined schedule where TMA and MMA cannot overlap across stages.
pub type UnpipelinedMainloopRing = TwoCtaRing<1, AsyncCompletion, AsyncCopyReuse>;

/// Two-stage TMEM accumulator ring (`TMEM_0` at `+0`, `TMEM_1` at `+128`) for epilogue ping-pong.
/// Allows `Warp 4` to accumulate Tile `t + 1` into `TMEM_1` while `Warps 0..3` drain `TMEM_0`.
pub type AccumulatorRing = TwoCtaRing<2, TensorMemory>;

/// Single-stage TMEM accumulator ring used by the exposed-epilogue ablation kernel.
/// Forces `Warp 4` to wait for the entire epilogue of Tile `t` before starting Tile `t + 1`.
pub type ExposedAccumulatorRing = TwoCtaRing<1, TensorMemory>;

/// Threads per CTA (`192` threads = `4` epilogue warps + `1` MMA warp + `1` TMA producer warp).
/// Matches the warp-specialized launch contract of the Python `CuTeDSL` reference.
pub const THREADS: u32 = 192;

/// Number of shared-memory stages in the overlapped mainloop ring (`8` stages).
/// Each stage holds `8 KB` of `X` and `16 KB` of `W_qkv` per CTA (`24 KB` × `8` = `192 KB`).
const STAGES: u32 = 8;

/// Bytes occupied by one CTA's `X` tile stage (`8,192` bytes), derived from [`XLayout`] via [`size_in_bytes`].
/// Transferred by [`TmaCopy::<XLayout, 2>::load`] into the `X` shared-memory ring.
const A_BYTES: usize = size_in_bytes::<f16, XLayout>() as usize;

/// Bytes occupied by one CTA's `W_qkv` tile stage (`16,384` bytes), derived from [`WLayout`] via [`size_in_bytes`].
/// Transferred by [`TmaCopy::<WLayout, 2>::load`] into the `W_qkv` shared-memory ring.
const B_BYTES: usize = size_in_bytes::<f16, WLayout>() as usize;

/// Total cluster-wide TMA transaction bytes expected per stage across both CTAs (`49,152` bytes).
/// Announced to CTA 0's `full` barrier via `producer_expect_tx_cursor`.
const CLUSTER_STAGE_TX_BYTES: u32 = (2 * (A_BYTES + B_BYTES)) as u32;

/// Shared-memory byte offset of the 8-stage `W_qkv` ring buffer (`8 × 8,192 = 65,536`).
/// Aligned to `1,024` bytes as required by the 128-byte swizzle period.
const B_OFFSET: usize = (STAGES as usize) * A_BYTES;

/// Shared-memory byte offset of the pre-scaled `Q` RMSNorm gamma weights (`128 × f32 = 512` bytes).
/// Placed immediately after the `192 KB` `X` and `W_qkv` stage rings.
const GAMMA_Q_OFFSET: usize = B_OFFSET + (STAGES as usize) * B_BYTES;

/// Shared-memory byte offset of the pre-scaled `K` RMSNorm gamma weights (`128 × f32 = 512` bytes).
/// Contiguous with `GAMMA_Q_OFFSET` so `0` or `128` selects `Q` or `K` gamma weights.
const GAMMA_K_OFFSET: usize = GAMMA_Q_OFFSET + 512;

/// Shared-memory byte offset of the pipeline mbarriers and TMEM allocation token (`197,632`).
/// Holds `16` mainloop barriers, `4` accumulator barriers, `1` dealloc barrier, and `tmem_token`.
const CONTROL_OFFSET: usize = GAMMA_K_OFFSET + 512;

/// Total dynamic shared memory reserved per CTA (`197,888` bytes, `1,024`-byte aligned).
/// Identical to `SMEM_BYTES` in `reference/qkv_cutedsl.py`.
pub const DYNAMIC_SMEM_BYTES: u32 = (CONTROL_OFFSET + 256) as u32;

const _: () = assert!(DYNAMIC_SMEM_BYTES == 197_888);

/// Shared implementation of the persistent two-CTA fused QKV + Epilogue schedule, parameterized by
/// `OVERLAP_EPILOGUE` (`true` = 8-stage TMA + 2-stage TMEM ping-pong; `false` = 1-stage exposed epilogue).
///
/// # Safety
/// Must be launched with `block = (192, 1, 1)`, `cluster = (2, 1, 1)`, `dynamic_shared = 197888`,
/// and valid 16-byte/32-byte aligned device pointers and tensor maps.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn run_qkv_epilogue_schedule<const OVERLAP_EPILOGUE: bool>(
    x_tma: &TensorMap<f16, XLayout>,
    w_tma: &TensorMap<f16, WLayout>,
    gamma_q: *const f32,
    gamma_k: *const f32,
    slot_mapping: *const i32,
    q_out: *mut u8,
    k_cache: *mut u8,
    v_cache: *mut u8,
    m: u32,
    n_total: u32,
    k: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    rms_eps: f32,
    inv_scale_q: f32,
    inv_scale_k: f32,
    inv_scale_v: f32,
) {
    let (tid, _, _) = thread_idx();
    let warp_id = warp_idx();
    let rank = make_warp_uniform(block_idx_in_cluster());
    let smem = DynamicSharedArray::<u8, 1024>::get_raw();
    let smem_gq = smem.add(GAMMA_Q_OFFSET).cast::<f32>();
    let smem_gk = smem.add(GAMMA_K_OFFSET).cast::<f32>();
    let bars = smem.add(CONTROL_OFFSET).cast::<Barrier>();
    let dealloc = Sm100DeallocBarrier::from_raw(bars.add(20));
    let token = bars.add(21).cast::<u32>();

    let mainloop = MainloopRing::from_raw(bars, 0);
    let mainloop_1 = UnpipelinedMainloopRing::from_raw(bars, 0);
    let acc_ring = AccumulatorRing::from_raw(bars, 16);
    let acc_ring_1 = ExposedAccumulatorRing::from_raw(bars, 16);

    if tid < 128 {
        let gq_view = StridedTensor::<f32>::from_raw_parts(gamma_q, 128, 1);
        let gk_view = StridedTensor::<f32>::from_raw_parts(gamma_k, 128, 1);
        let smem_gq_mut = StridedTensorMut::<f32>::from_raw_parts(smem_gq, 128, 1);
        let smem_gk_mut = StridedTensorMut::<f32>::from_raw_parts(smem_gk, 128, 1);
        smem_gq_mut.write_at(
            tid as usize,
            gq_view.get_unchecked(tid as usize) * inv_scale_q,
        );
        smem_gk_mut.write_at(
            tid as usize,
            gk_view.get_unchecked(tid as usize) * inv_scale_k,
        );
    }
    if tid == 0 {
        x_tma.prefetch_descriptor();
        w_tma.prefetch_descriptor();
        if OVERLAP_EPILOGUE {
            mainloop.init(1, 1);
            acc_ring.init(1, 2);
        } else {
            mainloop_1.init(1, 1);
            acc_ring_1.init(1, 2);
        }
        dealloc.init();
    }
    mbarrier_init_fence();
    cluster_arrive_relaxed();
    cluster_wait();

    let sched = TileScheduler::new_2d(m, n_total);
    let first = sched.first_tile();
    let stride = sched.stride();
    let tile_count = sched.tile_count();
    let ktiles = k / 64;

    if warp_id == 5 {
        setmaxregister_decrease::<40>();
        let x_tiles = SharedTileArray::<XLayout, OperandA, STAGES>::from_raw(smem.cast());
        let w_tiles =
            SharedTileArray::<WLayout, OperandB, STAGES>::from_raw(smem.add(B_OFFSET).cast());
        if OVERLAP_EPILOGUE {
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
                    let mut xt = x_tiles.tile_unchecked(cursor.index());
                    let mut wt = w_tiles.tile_unchecked(cursor.index());
                    TmaCopy::<XLayout, 2>::load(x_tma, &mut xt, row, (kt * 64) as i32, bar, rank);
                    TmaCopy::<WLayout, 2>::load(w_tma, &mut wt, col, (kt * 64) as i32, bar, rank);
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
        } else {
            let mut cursor = RingCursor::<1, Producer>::new();
            let mut tile = first;
            while tile < tile_count {
                let (m_tile, n_tile) = sched.tile_idx(tile);
                let (row, col) = sched.cta_tile_coord(m_tile, n_tile, rank);
                let mut kt = 0u32;
                while kt < ktiles {
                    mainloop_1.acquire_cursor(cursor);
                    if rank == 0 && elect_one_sync() {
                        mainloop_1.producer_expect_tx_cursor(cursor, CLUSTER_STAGE_TX_BYTES);
                    }
                    let bar = mainloop_1.full_at_cursor(cursor);
                    let mut xt = SharedTile::<XLayout, OperandA>::from_raw(smem.cast());
                    let mut wt =
                        SharedTile::<WLayout, OperandB>::from_raw(smem.add(B_OFFSET).cast());
                    TmaCopy::<XLayout, 2>::load(x_tma, &mut xt, row, (kt * 64) as i32, bar, rank);
                    TmaCopy::<WLayout, 2>::load(w_tma, &mut wt, col, (kt * 64) as i32, bar, rank);
                    cursor.advance();
                    kt += 1;
                }
                tile += stride;
            }
            mainloop_1.acquire_cursor(cursor);
        }
    } else if warp_id == 4 {
        setmaxregister_decrease::<40>();
        NamedBarrier::<2, 160>::sync();
        let tmem = *token;
        let mma = QkvMma::new();
        let x_tiles = SharedTileArray::<XLayout, OperandA, STAGES>::from_raw(smem.cast());
        let w_tiles =
            SharedTileArray::<WLayout, OperandB, STAGES>::from_raw(smem.add(B_OFFSET).cast());
        if rank == 0 {
            if OVERLAP_EPILOGUE {
                let mut smem_cursor = RingCursor::<STAGES, Consumer>::new();
                let mut acc_cursor = RingCursor::<2, Producer>::new();
                let mut tile = first;
                while tile < tile_count {
                    acc_ring.acquire_cursor(acc_cursor);
                    let dst_tmem = tmem + acc_cursor.index() * 128;
                    let mut kt = 0u32;
                    while kt < ktiles {
                        mainloop.wait_cursor(smem_cursor);
                        let xt = x_tiles.tile_unchecked(smem_cursor.index());
                        let wt = w_tiles.tile_unchecked(smem_cursor.index());
                        if elect_one_sync() {
                            mma.gemm(dst_tmem, &xt, &wt, kt != 0);
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
            } else {
                let mut smem_cursor = RingCursor::<1, Consumer>::new();
                let mut acc_cursor = RingCursor::<1, Producer>::new();
                let mut tile = first;
                while tile < tile_count {
                    acc_ring_1.acquire_cursor(acc_cursor);
                    let mut kt = 0u32;
                    while kt < ktiles {
                        mainloop_1.wait_cursor(smem_cursor);
                        let xt = SharedTile::<XLayout, OperandA>::from_raw(smem.cast());
                        let wt =
                            SharedTile::<WLayout, OperandB>::from_raw(smem.add(B_OFFSET).cast());
                        if elect_one_sync() {
                            mma.gemm(tmem, &xt, &wt, kt != 0);
                            mainloop_1.mma_empty_cursor(smem_cursor);
                        }
                        smem_cursor.advance();
                        kt += 1;
                    }
                    if elect_one_sync() {
                        acc_ring_1.mma_full_cursor(acc_cursor);
                    }
                    acc_cursor.advance();
                    tile += stride;
                }
                acc_ring_1.acquire_cursor(acc_cursor);
            }
        }
    } else {
        setmaxregister_increase::<232>();
        if warp_id == 0 {
            Sm100Tmem::<512>::alloc(token);
        }
        NamedBarrier::<2, 160>::sync();
        let tmem = *token;
        let qk_heads = num_q_heads + num_kv_heads;
        let slots_view = StridedTensor::<i32>::from_raw_parts(slot_mapping, m as usize, 1);
        let smem_gq_view = StridedTensor::<f32>::from_raw_parts(smem_gq.cast_const(), 128, 1);
        let smem_gk_view = StridedTensor::<f32>::from_raw_parts(smem_gk.cast_const(), 128, 1);
        let q_view = StridedTensorMut::<u8>::from_raw_parts(q_out, (m * num_q_heads) as usize, 128);
        let k_view = StridedTensorMut::<u8>::from_raw_parts(k_cache, usize::MAX / 256, 128);
        let v_view = StridedTensorMut::<u8>::from_raw_parts(v_cache, usize::MAX / 256, 128);

        let mut acc_cursor = RingCursor::<2, Consumer>::new();
        let mut acc_cursor_1 = RingCursor::<1, Consumer>::new();
        let mut tile = first;
        while tile < tile_count {
            let (m_tile, n_tile) = sched.tile_idx(tile);
            let (row, head_idx) =
                sched.epilogue_thread_coord::<EpilogueThreadLayout>(m_tile, n_tile, rank, tid);
            let acc_idx = if OVERLAP_EPILOGUE {
                acc_ring.wait_cursor(acc_cursor);
                acc_cursor.index()
            } else {
                acc_ring_1.wait_cursor(acc_cursor_1);
                0
            };
            let tmem_row = tmem_row_address(tmem, warp_id, acc_idx * 128);
            let fragment = TmemFragment::<128>::load(tmem_row);
            if OVERLAP_EPILOGUE {
                tcgen05_fence_before_thread_sync();
                NamedBarrier::<1, 128>::sync();
                if tid == 0 {
                    acc_ring.thread_empty_cursor(acc_cursor);
                }
            }

            let (scale, gamma_ptr) = if head_idx < qk_heads {
                let mut sum_sq = 0.0f32;
                let mut j = 0u32;
                while j < 32 {
                    let v0 = fragment.values[crd2idx_col_major_2d(j, 0, 32) as usize];
                    let v1 = fragment.values[crd2idx_col_major_2d(j, 1, 32) as usize];
                    let v2 = fragment.values[crd2idx_col_major_2d(j, 2, 32) as usize];
                    let v3 = fragment.values[crd2idx_col_major_2d(j, 3, 32) as usize];
                    sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
                    j += 1;
                }
                let inv_rms = rsqrt_f32(sum_sq * (1.0f32 / 128.0f32) + rms_eps);
                let g_ptr = if head_idx < num_q_heads {
                    smem_gq_view.ptr_at_unchecked(0)
                } else {
                    smem_gk_view.ptr_at_unchecked(0)
                };
                (inv_rms, Some(g_ptr))
            } else {
                (inv_scale_v, None)
            };

            let dst_ptr = if head_idx < num_q_heads {
                q_view.ptr_at_2d_unchecked(row as usize, head_idx as usize, num_q_heads as usize)
            } else if head_idx < qk_heads {
                let slot_idx = slots_view.get_unchecked(row as usize) as usize;
                let kv_head = (head_idx - num_q_heads) as usize;
                k_view.ptr_at_2d_unchecked(slot_idx, kv_head, num_kv_heads as usize)
            } else {
                let slot_idx = slots_view.get_unchecked(row as usize) as usize;
                let kv_head = (head_idx - qk_heads) as usize;
                v_view.ptr_at_2d_unchecked(slot_idx, kv_head, num_kv_heads as usize)
            };

            let gamma_view = gamma_ptr.map(|ptr| StridedTensor::from_raw_parts(ptr, 128, 1));
            let dst_u32 = dst_ptr.cast::<u32>();
            let mut chunk = 0usize;
            while chunk < 4 {
                let mut packed = [0u32; 8];
                let mut w = 0usize;
                while w < 8 {
                    let idx = crd2idx_row_major_3d(chunk, w, 0, 8, 4);
                    let (v0, v1, v2, v3) = if let Some(g_view) = gamma_view {
                        (
                            fragment.values[idx] * scale * g_view.get_unchecked(idx),
                            fragment.values[idx + 1] * scale * g_view.get_unchecked(idx + 1),
                            fragment.values[idx + 2] * scale * g_view.get_unchecked(idx + 2),
                            fragment.values[idx + 3] * scale * g_view.get_unchecked(idx + 3),
                        )
                    } else {
                        (
                            fragment.values[idx] * scale,
                            fragment.values[idx + 1] * scale,
                            fragment.values[idx + 2] * scale,
                            fragment.values[idx + 3] * scale,
                        )
                    };
                    let lo16 = cvt_e4m3x2_f32(v0, v1) as u32;
                    let hi16 = cvt_e4m3x2_f32(v2, v3) as u32;
                    packed[w] = lo16 | (hi16 << 16);
                    w += 1;
                }
                let chunk_offset = crd2idx_row_major_2d(chunk, 0, 8);
                store_global_no_allocate(dst_u32.add(chunk_offset), &GlobalStoreTile(packed));
                chunk += 1;
            }

            if !OVERLAP_EPILOGUE {
                tcgen05_fence_before_thread_sync();
                NamedBarrier::<1, 128>::sync();
                if tid == 0 {
                    acc_ring_1.thread_empty_cursor(acc_cursor_1);
                }
                acc_cursor_1.advance();
            } else {
                acc_cursor.advance();
            }
            tile += stride;
        }

        if warp_id == 0 {
            dealloc.sync_cluster(rank);
            Sm100Tmem::<512>::dealloc(tmem);
        }
    }
}

/// Device module exporting the overlapped (2-stage TMEM) and exposed-epilogue (1-stage) kernels.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Overlapped 2-CTA persistent fused QKV + RMSNorm + FP8 (e4m3) + Q store + Paged KV Scatter.
    /// Uses 8-stage cluster TMA and 2-stage TMEM ping-pong (`TMEM_0`, `TMEM_1`) with immediate
    /// TMEM release after `TmemFragment::<128>::load` to hide the epilogue behind Tensor Core MMA.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 197888,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sm100_qkv_epilogue_cute(
        #[grid_constant] x_tma: &TensorMap<f16, XLayout>,
        #[grid_constant] w_tma: &TensorMap<f16, WLayout>,
        gamma_q: *const f32,
        gamma_k: *const f32,
        slot_mapping: *const i32,
        q_out: *mut u8,
        k_cache: *mut u8,
        v_cache: *mut u8,
        m: u64,
        n_total: u64,
        k: u64,
        num_q_heads: u64,
        num_kv_heads: u64,
        rms_eps_bits: u64,
        inv_scale_q_bits: u64,
        inv_scale_k_bits: u64,
        inv_scale_v_bits: u64,
    ) {
        unsafe {
            run_qkv_epilogue_schedule::<true>(
                x_tma,
                w_tma,
                gamma_q,
                gamma_k,
                slot_mapping,
                q_out,
                k_cache,
                v_cache,
                m as u32,
                n_total as u32,
                k as u32,
                num_q_heads as u32,
                num_kv_heads as u32,
                f32::from_bits(rms_eps_bits as u32),
                f32::from_bits(inv_scale_q_bits as u32),
                f32::from_bits(inv_scale_k_bits as u32),
                f32::from_bits(inv_scale_v_bits as u32),
            );
        }
    }

    /// Unpipelined / exposed-epilogue baseline (`1`-stage SMEM, `1`-stage TMEM, late TMEM release).
    /// Demonstrates the performance penalty at high token counts `M` when the epilogue is not overlapped.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 197888,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sm100_qkv_epilogue_exposed_cute(
        #[grid_constant] x_tma: &TensorMap<f16, XLayout>,
        #[grid_constant] w_tma: &TensorMap<f16, WLayout>,
        gamma_q: *const f32,
        gamma_k: *const f32,
        slot_mapping: *const i32,
        q_out: *mut u8,
        k_cache: *mut u8,
        v_cache: *mut u8,
        m: u64,
        n_total: u64,
        k: u64,
        num_q_heads: u64,
        num_kv_heads: u64,
        rms_eps_bits: u64,
        inv_scale_q_bits: u64,
        inv_scale_k_bits: u64,
        inv_scale_v_bits: u64,
    ) {
        unsafe {
            run_qkv_epilogue_schedule::<false>(
                x_tma,
                w_tma,
                gamma_q,
                gamma_k,
                slot_mapping,
                q_out,
                k_cache,
                v_cache,
                m as u32,
                n_total as u32,
                k as u32,
                num_q_heads as u32,
                num_kv_heads as u32,
                f32::from_bits(rms_eps_bits as u32),
                f32::from_bits(inv_scale_q_bits as u32),
                f32::from_bits(inv_scale_k_bits as u32),
                f32::from_bits(inv_scale_v_bits as u32),
            );
        }
    }
}

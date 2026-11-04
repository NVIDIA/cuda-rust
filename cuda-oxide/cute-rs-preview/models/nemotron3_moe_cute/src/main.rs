// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nemotron-3 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` scale, `SF_VEC = 16`) non-gated `ReLU²`
//! Mixture-of-Experts (MoE) layer in `cute-rs` for NVIDIA Blackwell (`sm_100a`).
//!
//! # 1. How the 4-Stage MoE Layer Works
//!
//! Every token is routed to `top_k = 4` of `8` experts. Each expert runs two matrix
//! multiplications (`FC1` and `FC2`) with `ReLU²` in between:
//!
//! ```text
//! THE 4 STAGES OF THE MOE LAYER:
//! [K1: Quantize BF16->FP4] ──► [K2: Route] ──► [K3: FC1 + ReLU²] ──► [K4: FC2 + Finalize]
//!
//!   Token [B, 1024] (BF16)
//!      │
//!      ▼  K1 (`quantize_activations_nvfp4`)
//!   Packed FP4 E2M1 + Canonical (32,4,4) E4M3 Scales
//!      │
//!      ▼  K2 (Token-to-Expert Routing & Grouping)
//!   Expert Buckets (`16 / 32 / 64 / 256` tokens for `B <= 1024` | `128` tokens for `B >= 2048`)
//!      │
//!      ▼  K3 (`moe_fc1_relu2_cute` | `moe_fc1_relu2_tcgen05_cute` | `moe_fc1_relu2_tcgen05_128_cute`)
//!   W_up [1024, 1024] × Tokens ──► ReLU²(x) ──► Requantize to NVFP4
//!      │
//!      ▼  K4 (`moe_fc2_cute` | `moe_fc2_tcgen05_cute` | `moe_fc2_tcgen05_128_cute` + `moe_finalize_cute`)
//!   W_down [1024, 1024] × Activated ──► Weighted Top-4 Sum ──► Output [B, 1024] (BF16)
//! ```
//!
//! # 2. How CuTe Is Encoded in Rust's Type System
//!
//! Layouts, TMA descriptors, producer/consumer pipelines, operand roles, and 5th-gen Tensor Core
//! (`tcgen05`) MMA operations are all checked at compile time by Rust's type system:
//!
//! ```text
//! Host `create_tensor_map_tiled`                        Device `#[grid_constant]`
//! ──────────────────────────────                        ─────────────────────────
//! TensorMap<u8,  WeightTileSmem>      ───────────────►  &TensorMap<u8,  WeightTileSmem>
//! TensorMap<u8,  ActTile128Smem>      ───────────────►  &TensorMap<u8,  ActTile128Smem>
//! TensorMap<u16, ScaleTmaSmem>        ───────────────►  &TensorMap<u16, ScaleTmaSmem>
//! TensorMap<u16, GroupedOutTileSmem>  ───────────────►  &TensorMap<u16, GroupedOutTileSmem>
//!                                                             │
//!   ┌─────────────────────────────────────────────────────────┘
//!   ▼
//! Warp 5 (TMA Producer, `PipelineState<Producer, 4>`, `Act128Pipeline`)
//!   • `w_pipeline` (`Weight128Pipeline`) stages stationary `W` (`64 KB`) + `SFA` (`8 KB`) once
//!   • `b_pipeline.producer_acquire` ──► `producer_expect_tx` (`36,864` bytes)
//!   • `copy_tma_2d(b_tma)`   ──► `SMEM_B[0..3]`   (`32 KB` raw 4-bit tokens per stage)
//!   • `copy_tma_2d(sfb_tma)` ──► `SMEM_SFB[0..3]` (`4 KB` canonical 8-bit scales per stage)
//!   │
//!   ▼  `Act128Pipeline` (`PipelineTmaAsync<4, 1, 36_864>`) + 2-Stage `TMEM[0..1]`
//! Warp 4 (MMA Consumer/Producer) & Warps 0..3 (Epilogue Consumer)
//!   • `TiledMma128::{copy, gemm}` (`Cp4x32x128bOp` + `MmaMXF4NVF4Op`)
//!   • `TmemFragment::<64>::load` ──► `RegisterTile` (`TensorSSA`) `ReLU²` + `Float4E2M1FN` quant
//!   • `TmemFragment::<64>::load` ──► `RegisterTile` (`TensorSSA`) `BFloat16` + `Sm100TmemEpilogue::store_async`
//! ```
//!
//! # 3. CUDA Graph Capture & 1,000× Replay Timer
//!
//! The benchmark records the entire `BF16 -> FP4 -> FC1 -> FC2 -> BF16` DAG between
//! `cuStreamBeginCapture` and `cuStreamEndCapture`, then replays the graph `1,000×` back-to-back
//! between CUDA events (`mean_us = total_us / 1000`).
//!
//! Build and run on B200 (`sm_100a`):
//! ```bash
//! CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run nemotron3_moe_cute --arch sm_100a
//! ```

#![feature(f16)]

use ctm_rs::memory::{DisjointSlice, DynamicSharedArray, SharedArray};
use ctm_rs::{cuda_module, kernel, launch_bounds, launch_contract, thread};
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cute_rs::arch::{
    NamedBarrier, fence_view_async_shared, fence_view_async_tmem_load,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
};
use cute_rs::nvgpu::cpasync::{TmaEncodeOptions, create_tensor_map_tiled_with_options};
use cute_rs::nvgpu::tcgen05::{TmemFragment, tmem_row_address};
use cute_rs::pipeline::sm100::SingleCtaRing;
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::utils::blockscaled::{
    CANONICAL_SCALE_ATOM_BYTES, CANONICAL_SCALE_K_GROUPS, CANONICAL_SCALE_M, Sm1xxBlockScaleKMajor,
};
use cute_rs::utils::tmem_allocator::TmemAllocator;
use cute_rs::{
    BFloat16, BlockScaledTensor, ComposedLayout, Consumer, Float4E2M1FN, Float4E2M1FNx8,
    Float8E4M3FN, KMajor, Mkl, Nkl, OperandA, OperandB, PipelineState, PipelineTmaAsync, Producer,
    ReductionOp, RowMajor, Sm100BlockScaledTiledMma, Sm100ScaleTile, Sm100SharedTile, SmemTile,
    Swizzle, TensorMap, TensorSSA, copy_tma_2d, idx2crd_2d, idx2crd_col_major_2d, size_in_bytes,
    sm1xx_block_scale_crd2idx, sm100_noswizzle_burst_u64_crd2idx,
};
use std::mem::MaybeUninit;
use std::sync::Arc;

pub mod moe_192e;

/// Hidden dimension `H` of each input and output token vector (`1,024` BF16 elements).
/// Divided into `64` scale groups of `16` elements (`512` packed FP4 `E2M1` bytes per row).
const HIDDEN: usize = 1024;

/// Intermediate FFN dimension `I` inside each expert (`1,024` elements).
/// Divided into `8` tiles of `128` rows for the 5th-gen Tensor Core (`tcgen05`) MMA.
const INTERMEDIATE: usize = 1024;

/// Total number of routed experts in the MoE layer (`E = 8`).
/// Each expert owns an independent `W_up` (`[I, H]`) and `W_down` (`[H, I]`) weight matrix.
const NUM_EXPERTS: usize = 8;

/// Maximum token batch size `B` evaluated in the latency sweep (`16,384` tokens).
/// At `TOP_K = 4`, this produces up to `65,536` routed token-expert pairs per layer.
const MAX_NUM_TOKENS: usize = 16384;

/// Token batch sizes `B` swept from single-token decode (`B = 1`) to large prefill (`B = 16,384`).
/// Every batch size is checked against the exact analytical reference before timing.
const BATCH_SIZES: [usize; 15] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    MAX_NUM_TOKENS,
];

/// Number of experts activated per token (`top_k = 4`).
/// Each token's final output is the weighted sum of its `4` selected experts.
const TOP_K: usize = 4;

/// Untimed CUDA Graph replay warmups executed before each timing sample.
/// Warms up the GPU clock and L2 cache before recording the start event.
const TIMING_WARMUPS: usize = 50;

/// Number of back-to-back CUDA Graph replays timed between each start/stop event pair.
/// Dividing total elapsed time by `1,000` removes sub-microsecond event jitter.
const TIMING_REPLAYS: usize = 1000;

/// Number of independent `1,000×` replay samples averaged for each batch size report.
/// Produces stable steady-state latency measurements on Blackwell (`sm_100a`).
const TIMING_SAMPLES: usize = 20;

/// Deterministic top-4 routing weights summing to `1.0` (`1/8 + 1/4 + 1/4 + 3/8`).
/// Chosen as exact binary fractions so the CPU reference has zero rounding ambiguity.
const ROUTING_PATTERN: [f32; TOP_K] = [0.125, 0.25, 0.25, 0.375];

/// Per-expert `FC1` (`W_up`) scale factors for active experts `0..4`.
/// Multiplied with `1 / HIDDEN` to form each expert's global `w_up_global` dequant scale.
const UP_FACTORS: [f32; TOP_K] = [0.5, 1.0, 2.0, 1.0];

/// Per-expert `FC2` (`W_down`) scale factors for active experts `0..4`.
/// Multiplied with `1 / INTERMEDIATE` to form each expert's global `w_down_global` scale.
const DOWN_FACTORS: [f32; TOP_K] = [1.0, 2.0, 0.5, 1.0];

// =========================================================================
// COMPILE-TIME CUTE LAYOUTS, TMA TILES, PIPELINES & TILED MMA TYPES
// =========================================================================

/// Packed `128 × 256` byte (`8 × K=64` sub-tiles = `K = 512` FP4 `E2M1`) weight stage (`32,768` bytes).
/// Encodes the shared-memory box transferred by one 2D TMA weight load (`w_up_tma` / `w_down_tma`).
type WeightTileSmem = RowMajor<128, 256>;

/// Packed `256 × 256` byte (`8 × K=64` sub-tiles = `K = 512` FP4 `E2M1`) activation stage (`65,536` bytes).
/// Transferred per burst by `b_up_tma` / `b_down_tma` into the 2-stage TMA ring buffer for `B <= 1024`!
type ActTileSmem = RowMajor<256, 256>;

/// Packed `128 × 256` byte (`8 × K=64` sub-tiles = `K = 512` FP4 `E2M1`) activation stage (`32,768` bytes).
/// Transferred per burst by `b_up_128_tma` / `b_down_128_tma` into the 4-stage TMA ring buffer for `B >= 2048`!
type ActTile128Smem = RowMajor<128, 256>;

/// Contiguous `4,096`-byte stage (`8` canonical `512`-byte `(32, 4, 4)` A scale atoms) carried as `8 × 256` `u16`s.
/// Allows a single 2D TMA copy (`sfa_up_tma` / `sfa_down_tma`) to stage all 8 A scale atoms per burst.
type ScaleTmaSmem = RowMajor<8, 256>;

/// Contiguous `8,192`-byte stage (`16` canonical `512`-byte `(32, 4, 4)` B scale atoms) carried as `16 × 256` `u16`s.
/// Allows a single 2D TMA copy (`sfb_up_tma` / `sfb_down_tma`) to stage all 16 B scale atoms per burst.
type ScaleBTmaSmem = RowMajor<16, 256>;

/// Shared `128 × 32` `u16` (`8,192` bytes) `B64`-swizzled epilogue tile for 2D S2G TMA Store (`copy_tma_s2g_2d`).
/// Maps `u ^ ((tid >> 1) & 3)` bank-conflict-free shared writes to unswizzled `[128, 32]` `bf16` global tiles.
type GroupedOutTileSmem = ComposedLayout<Swizzle<2, 3, 3>, 0, RowMajor<128, 32>>;

/// Role-checked shared-memory A operand tile (`OperandA` role + `WeightTileSmem` layout).
/// Prevents swapping the weight tile `A` and activation tile `B` at compile time.
type SharedWeightTile = Sm100SharedTile<WeightTileSmem, OperandA>;

/// Role-checked shared-memory B operand tile (`OperandB` role + `ActTileSmem` layout).
/// Paired with [`SharedWeightTile`] inside [`Sm100BlockScaledTiledMma::gemm`].
type SharedActTile = Sm100SharedTile<ActTileSmem, OperandB>;

/// Role-checked shared-memory B operand tile for `128`-token groups (`OperandB` role + `ActTile128Smem` layout).
/// Paired with [`SharedWeightTile`] inside [`Sm100BlockScaledTiledMma::gemm`].
type SharedAct128Tile = Sm100SharedTile<ActTile128Smem, OperandA>;

/// Role-checked shared-memory A scale tile (`Mkl` role + canonical `Sm120ScaleAtom` layout).
/// Stages `W_up` / `W_down` `Float8E4M3FN` scales before `tcgen05.cp` copies them into TMEM.
type SharedWeightScales = Sm100ScaleTile<Mkl>;

/// Role-checked shared-memory B scale tile (`Nkl` role + canonical `Sm120ScaleAtom` layout).
/// Stages routed token `Float8E4M3FN` scales before `tcgen05.cp` copies them into TMEM.
type SharedActScales = Sm100ScaleTile<Nkl>;

/// Group-one Blackwell block-scaled NVFP4 (`mxf4nvf4.block_scale.block16`) tiled MMA.
/// Verifies at compile time that `WeightTileSmem` and `ActTileSmem` match `M = 128, K = 64`.
type TiledMma = Sm100BlockScaledTiledMma<WeightTileSmem, ActTileSmem, 128, 64>;

/// Group-one Blackwell block-scaled NVFP4 (`mxf4nvf4.block_scale.block16`) tiled MMA for `128`-token groups.
/// Verifies at compile time that `WeightTileSmem` and `ActTile128Smem` match `M = 128, K = 64`.
type TiledMma128 = Sm100BlockScaledTiledMma<WeightTileSmem, ActTile128Smem, 128, 64>;

/// Bytes copied by TMA per `K = 512` burst (`32,768` A + `65,536` B + `4,096` SFA + `8,192` SFB = `110,592` bytes).
/// Statically verified by `dialect-cute` against the 4 `copy_tma_2d` calls in the producer warp.
const TMA_BURST_TX_BYTES: u32 = size_in_bytes::<u8, WeightTileSmem>()
    + size_in_bytes::<u8, ActTileSmem>()
    + size_in_bytes::<u16, ScaleTmaSmem>()
    + size_in_bytes::<u16, ScaleBTmaSmem>();

/// Bytes copied by TMA per `128`-row `K = 512` burst (`32,768` B + `4,096` SFB = `36,864` bytes).
/// Statically verified by `dialect-cute` for `Act128Pipeline` and `Weight128Pipeline`.
const TMA_BURST_128_TX_BYTES: u32 =
    size_in_bytes::<u8, ActTile128Smem>() + size_in_bytes::<u16, ScaleTmaSmem>();

/// Number of compute warps (`16` warps = `512` threads) that run the epilogue for `B <= 1024`.
/// Only `Warp 0` (`1` consumer warp) participates in `MainloopPipeline`!
const COMPUTE_WARPS: u32 = 16;

/// Dedicated warp index (`Warp 16`) that issues asynchronous 2D TMA loads for `A`, `B`, `SFA`, and `SFB`.
/// Runs concurrently with `Warp 0` via `MainloopPipeline`.
const PRODUCER_WARP: u32 = COMPUTE_WARPS;

/// Total threads per CTA (`544` threads = `16` compute warps + `1` TMA producer warp) for `B <= 1024`.
/// Matches the warp-specialized producer/consumer contract of `MainloopPipeline`.
const TCGEN05_THREADS: u32 = (COMPUTE_WARPS + 1) * 32;

/// Total threads per CTA (`192` threads = `4` epilogue warps + `1` MMA warp + `1` TMA warp) for `B >= 2048`.
/// Gives each thread up to `255` registers for `tcgen05.ld.32x32b.x64` without spills.
const TCGEN05_128_THREADS: u32 = 192;

/// 2-Stage TMA load pipeline (`PipelineTmaAsync<2, 1, 110_592>`) with `1` MMA consumer warp (`Warp 0`).
/// Streams `Burst 0` (`slot 0`) and `Burst 1` (`slot 1`) of `A`, `B`, `SFA`, and `SFB` concurrently
/// with `0` `barrier_cta_sync` calls in the mainloop and `1` single `mbarrier` wait after all 16 MMA steps!
type MainloopPipeline = PipelineTmaAsync<2, 1, TMA_BURST_TX_BYTES>;

/// 2-Stage stationary weight TMA pipeline (`PipelineTmaAsync<2, 1, 36_864>`) for `B >= 2048`.
/// Loads the 2 stationary `K = 512` weight bursts (`A` + `SFA`) once per CTA!
type Weight128Pipeline = PipelineTmaAsync<2, 1, TMA_BURST_128_TX_BYTES>;

/// 4-Stage streaming token TMA pipeline (`PipelineTmaAsync<4, 1, 36_864>`) for `B >= 2048`.
/// Streams `128`-token `B` (`SMEM_B[0..3]`) and `SFB` (`SMEM_SFB[0..3]`) bursts concurrently with 2-stage TMEM MMA!
type Act128Pipeline = PipelineTmaAsync<4, 1, TMA_BURST_128_TX_BYTES>;

/// Byte offset of the 64 KB 2-stage `A` weight buffer (`2 × 32 KB`) in dynamic shared memory.
/// Also serves as the base of the 136 KB `(256, 136)` single-pass epilogue buffer during the epilogue.
const SMEM_A_OFFSET: usize = 0;

/// Byte offset of the 128 KB 2-stage `B` activation buffer (`2 × 64 KB`) in dynamic shared memory.
/// Contiguous with `SMEM_A_OFFSET` (`SMEM_A + SMEM_B = 192 KB`).
const SMEM_B_OFFSET: usize = 65_536;

/// Byte offset of the 8 KB 2-stage `SFA` weight scale buffer (`2 × 4 KB`) in dynamic shared memory.
/// Aligned to `1,024` bytes for 2D TMA scale copies and `tcgen05.cp`.
const SMEM_SFA_OFFSET: usize = 196_608;

/// Byte offset of the 16 KB 2-stage `SFB` activation scale buffer (`2 × 8 KB`) in dynamic shared memory.
/// Aligned to `1,024` bytes for 2D TMA scale copies and `tcgen05.cp`.
const SMEM_SFB_OFFSET: usize = 204_800;

/// Byte offset of the `MainloopPipeline` barriers (`32` bytes), `MMA_BAR` (`8` bytes), and `TMEM_SLOT` (`4` bytes).
/// Placed immediately after the 216 KB value and scale buffers.
const PIPELINE_OFFSET: usize = 221_184;

/// Total dynamic shared memory reserved per CTA (`222,208` bytes = `217 KB`, `1,024`-byte aligned).
/// Fits within NVIDIA B200's `227 KB` (`232,448` bytes) max dynamic SMEM per CTA!
const DYNAMIC_SMEM_BYTES: u32 = 222_208;

const _: () = {
    assert!(size_in_bytes::<u8, WeightTileSmem>() == 32_768);
    assert!(size_in_bytes::<u8, ActTileSmem>() == 65_536);
    assert!(size_in_bytes::<u16, ScaleTmaSmem>() == 4_096);
    assert!(size_in_bytes::<u16, ScaleBTmaSmem>() == 8_192);
    assert!(TMA_BURST_TX_BYTES == 110_592);
    assert!(MainloopPipeline::STORAGE_BYTES == 32);
    assert!(TCGEN05_THREADS == 544);
    assert!(SMEM_B_OFFSET == 65_536);
    assert!(SMEM_SFA_OFFSET == 196_608);
    assert!(SMEM_SFB_OFFSET == 204_800);
    assert!(PIPELINE_OFFSET == 221_184);
};

/// Device kernels for the `cute-rs` Nemotron-3 NVFP4 `ReLU²` MoE pipeline (`K1`..`K4`).
/// Compiled through `dialect-cute` and `cutlass-mlir` into a Blackwell (`sm_100a`) cubin.
#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;

    /// Quantize a 16-element `TensorSSA<f32, 16>` into a `TensorSSA<Float4E2M1FNx8, 2>` vector
    /// and its `Float8E4M3FN` block scale using CuTeDSL `TensorSSA::{abs, reduce, to}`,
    /// `Float8E4M3FN::{from_f32, to_f32}`, and `cute_rs::arch::rcp_approx`.
    #[inline(always)]
    fn quantize_tile16_nvfp4(
        vec: TensorSSA<f32, 16>,
    ) -> (TensorSSA<Float4E2M1FNx8, 2>, Float8E4M3FN) {
        let amax = vec.abs().reduce(ReductionOp::Max, 0.0f32);
        let scale = Float8E4M3FN::from_f32(amax * (1.0f32 / 6.0f32));
        let inv_scale = cute_rs::arch::rcp_approx(scale.to_f32()).min(f32::MAX);
        let q_vec = (vec * inv_scale).to::<Float4E2M1FN>();
        (q_vec, scale)
    }

    /// Apply `ReLU²` to a 16-element `TensorSSA<f32, 16>` accumulator slice and quantize it into
    /// a `TensorSSA<Float4E2M1FNx8, 2>` vector and its `Float8E4M3FN` block scale.
    #[inline(always)]
    fn quantize_relu2_tile16_nvfp4(
        acc_vec: TensorSSA<f32, 16>,
        scale_sq_div6: f32,
        g_scale_sq: f32,
    ) -> (TensorSSA<Float4E2M1FNx8, 2>, Float8E4M3FN) {
        let relu = acc_vec.fmax(0.0f32);
        let max_r = relu.reduce(ReductionOp::Max, 0.0f32);
        let scale = Float8E4M3FN::from_f32(max_r * max_r * scale_sq_div6);
        let inv_scale = cute_rs::arch::rcp_approx(scale.to_f32()).min(f32::MAX) * g_scale_sq;
        let q_vec = (relu * relu * inv_scale).to::<Float4E2M1FN>();
        (q_vec, scale)
    }

    /// Quantize `[rows, k]` BF16 values into packed `Float4E2M1FN` bytes and canonical `(32, 4)` `Float8E4M3FN` scales
    /// using `Tensor::local_tile::<8>` and `RegisterTile::to::<Float4E2M1FN>()`, writing TMA-ready `[256, 256]` burst tiles
    /// (`b_up_vals` / `sfb_up_scales`) for `B >= 4`, while zero-initializing `out_bf16`.
    #[kernel]
    pub fn quantize_activations_nvfp4(
        input: &[BFloat16],
        tok_to_perm: &[i32],
        mut out_values: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut b_up_vals: DisjointSlice<u8>,
        mut sfb_up_scales: DisjointSlice<u8>,
        mut out_bf16: DisjointSlice<BFloat16>,
        rows: usize,
        k: usize,
        zero_out: usize,
    ) {
        let tid = thread::index_1d().get();
        let _ = k;
        let total_groups = rows * 64;
        if tid >= total_groups {
            return;
        }
        let (row, grp) = idx2crd_2d::<RowMajor<16384, 64>>(tid);
        let val_base = row * 512 + grp * 8;
        let tile8_base = tid * 2;
        if zero_out != 0 {
            let zeros = cute_rs::RegisterTile::from_array([BFloat16::from_bits(0); 8]);
            unsafe {
                cute_rs::TensorMut::from_disjoint_slice(&mut out_bf16)
                    .local_tile::<8>(tile8_base)
                    .store(zeros);
                cute_rs::TensorMut::from_disjoint_slice(&mut out_bf16)
                    .local_tile::<8>(tile8_base + 1)
                    .store(zeros);
            }
        }

        let vec = unsafe {
            let in_t = cute_rs::Tensor::from_slice(input);
            let h0 = in_t.local_tile::<8>(tile8_base).load().to::<f32>();
            let h1 = in_t.local_tile::<8>(tile8_base + 1).load().to::<f32>();
            h0.concat(h1)
        };
        let (q_vec, scale) = quantize_tile16_nvfp4(vec);
        let packed_u64 = q_vec.bitcast::<u64, 1>()[0];
        let scale_code = scale.to_bits();

        if rows == 1 {
            let scale_layout = Sm1xxBlockScaleKMajor::<16>::new(rows, k);
            let scale_idx = scale_layout.crd2idx(0, row, grp);
            let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut out_scales);
            let mut vals_u64_out =
                unsafe { StridedTensorMut::<u64>::from_disjoint_slice_as(&mut out_values) };
            unsafe {
                scales_out.store_at(scale_idx, scale_code);
                vals_u64_out.store_at(val_base / 8, packed_u64);
            }
        } else if zero_out != 0 {
            let mut b_u64_out =
                unsafe { StridedTensorMut::<u64>::from_disjoint_slice_as(&mut b_up_vals) };
            let mut sfb_out = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
            let perms = unsafe {
                cute_rs::Tensor::from_slice(tok_to_perm)
                    .local_tile::<4>(row)
                    .load()
                    .into_array()
            };
            let u64_grp_base = sm100_noswizzle_burst_u64_crd2idx::<256>(0, 0, grp, 2);
            let s_grp_base = sm1xx_block_scale_crd2idx(grp / 4, 0, grp % 4, 2, 1);
            macro_rules! scatter_256 {
                ($perm:expr) => {{
                    let (r_u32, g_u32) = idx2crd_col_major_2d($perm as u32, 256);
                    let (g, r) = (g_u32 as usize, r_u32 as usize);
                    let u64_idx = g * 16384 + u64_grp_base + r * 2;
                    let s_idx = g * 16384 + s_grp_base + sm1xx_block_scale_crd2idx(0, r, 0, 2, 1);
                    b_u64_out.store_at(u64_idx, packed_u64);
                    sfb_out.store_at(s_idx, scale_code);
                }};
            }
            unsafe {
                scatter_256!(perms[0]);
                scatter_256!(perms[1]);
                scatter_256!(perms[2]);
                scatter_256!(perms[3]);
            }
        } else {
            let mut b_u64_out =
                unsafe { StridedTensorMut::<u64>::from_disjoint_slice_as(&mut b_up_vals) };
            let mut sfb_out = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
            let perms = unsafe {
                cute_rs::Tensor::from_slice(tok_to_perm)
                    .local_tile::<4>(row)
                    .load()
                    .into_array()
            };
            let u64_grp_base = sm100_noswizzle_burst_u64_crd2idx::<128>(0, 0, grp, 2);
            let s_grp_base = sm1xx_block_scale_crd2idx(0, 0, grp, 1, 16);
            macro_rules! scatter_128 {
                ($perm:expr) => {{
                    let (r_u32, g_u32) = idx2crd_col_major_2d($perm as u32, 128);
                    let (g, r) = (g_u32 as usize, r_u32 as usize);
                    let u64_idx = g * 8192 + u64_grp_base + r * 2;
                    let s_idx = g * 8192 + s_grp_base + sm1xx_block_scale_crd2idx(0, r, 0, 1, 16);
                    b_u64_out.store_at(u64_idx, packed_u64);
                    sfb_out.store_at(s_idx, scale_code);
                }};
            }
            unsafe {
                scatter_128!(perms[0]);
                scatter_128!(perms[1]);
                scatter_128!(perms[2]);
                scatter_128!(perms[3]);
            }
        }
    }

    /// Routed FC1 + `ReLU²` using `cute_rs::BlockScaledTensor` and 8-lane cooperative
    /// `cute_rs::arch::warp_reduction_sum_f32::<8, 1>`.
    /// Routed FC1 + `ReLU²` + fused NVFP4 requantization using `cute_rs::BlockScaledTensor` and
    /// `cute_rs::arch::warp_reduction_sum_f32`.
    ///
    /// Each 512-thread CTA (`16` warps) computes 16 output columns (`1` NVFP4 scale group) and
    /// writes `q_inter_vals` + `q_inter_scales` directly (`0` intermediate `inter_bf16` round-trip!).
    #[kernel]
    pub fn moe_fc1_relu2_cute(
        w_up_values: &[u8],
        w_up_scales: &[u8],
        w_up_global: &[f32],
        input_values: &[u8],
        input_scales: &[u8],
        expert_ids: &[i32],
        mut q_inter_vals: DisjointSlice<u8>,
        mut q_inter_scales: DisjointSlice<u8>,
        num_tokens: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
    ) {
        /// Shared-memory buffer (`16` `f32` elements) staging the 16 `ReLU²` values computed by
        /// the 16 warps in the CTA before thread 0 quantizes them into one NVFP4 scale group.
        static mut SMEM_ACT: SharedArray<f32, 16, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let warp_in_block = tid / 32; // 0..15
        let lane = tid % 32; // 0..31
        let grp = thread::blockIdx_x() as usize; // 0..63
        let out_col = grp * 16 + warp_in_block;
        let route = thread::blockIdx_y() as usize;
        let token = route / top_k;
        let total_w_rows = num_experts * intermediate_size;

        let weights = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mkl>>::from_slices(
            w_up_values,
            w_up_scales,
            total_w_rows,
            hidden_size,
        );
        let inputs = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Nkl>>::from_slices(
            input_values,
            input_scales,
            num_tokens,
            hidden_size,
        );
        let expert_ids_t = StridedTensor::from_slice(expert_ids);
        let w_up_global_t = StridedTensor::from_slice(w_up_global);
        let expert = unsafe { expert_ids_t.get_unchecked(route) } as usize;
        let w_row_idx = expert * intermediate_size + out_col;
        let w_row = weights.thread_row(0, w_row_idx);
        let x_row = inputs.thread_row(0, token);

        let tile = lane & 15;
        let (w_frag, x_frag) = unsafe { (w_row.k_tile(tile).load(), x_row.k_tile(tile).load()) };
        let acc = w_frag.dot_accumulate(x_frag, 0.0f32);
        let sum = unsafe { cute_rs::arch::warp_reduction_sum_f32(acc) } * 0.5f32;

        let g_scale = unsafe { w_up_global_t.get_unchecked(expert) };
        let scaled = sum * g_scale;
        let relu = if scaled > 0.0 { scaled } else { 0.0 };
        let activated = relu * relu;

        #[allow(static_mut_refs)]
        let mut smem_act =
            unsafe { StridedTensorMut::<f32>::from_raw_parts(SMEM_ACT.as_mut_ptr(), 16, 1) };
        if lane == 0 {
            unsafe {
                smem_act.store_at(warp_in_block, activated);
            }
        }
        unsafe {
            NamedBarrier::<0, 512>::sync();
        }

        if tid == 0 {
            #[allow(static_mut_refs)]
            let smem_act_ro =
                unsafe { StridedTensor::<f32>::from_raw_parts(SMEM_ACT.as_ptr(), 16, 1) };
            let h0 = unsafe {
                cute_rs::RegisterTile::from_array([
                    smem_act_ro.get_unchecked(0),
                    smem_act_ro.get_unchecked(1),
                    smem_act_ro.get_unchecked(2),
                    smem_act_ro.get_unchecked(3),
                    smem_act_ro.get_unchecked(4),
                    smem_act_ro.get_unchecked(5),
                    smem_act_ro.get_unchecked(6),
                    smem_act_ro.get_unchecked(7),
                ])
            };
            let h1 = unsafe {
                cute_rs::RegisterTile::from_array([
                    smem_act_ro.get_unchecked(8),
                    smem_act_ro.get_unchecked(9),
                    smem_act_ro.get_unchecked(10),
                    smem_act_ro.get_unchecked(11),
                    smem_act_ro.get_unchecked(12),
                    smem_act_ro.get_unchecked(13),
                    smem_act_ro.get_unchecked(14),
                    smem_act_ro.get_unchecked(15),
                ])
            };
            let (q_vec, scale) = quantize_tile16_nvfp4(h0.concat(h1));

            let packed_k = intermediate_size / 2;
            let scale_layout =
                Sm1xxBlockScaleKMajor::<16>::new(num_tokens * top_k, intermediate_size);
            let scale_idx = scale_layout.crd2idx(0, route, grp);
            let val_base = route * packed_k + grp * 8;

            let mut vals_u64_out =
                unsafe { StridedTensorMut::<u64>::from_disjoint_slice_as(&mut q_inter_vals) };
            let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut q_inter_scales);
            unsafe {
                scales_out.store_at(scale_idx, scale.to_bits());
                vals_u64_out.store_at(val_base / 8, q_vec.bitcast::<u64, 1>()[0]);
            }
        }
    }

    /// Low-batch (`B <= 2`) routed FC2 + top-4 weighted reduction (`K4`) using
    /// `cute_rs::BlockScaledTensor` (`cute.scaled_view_*` + `cute.dot`) and
    /// `cute_rs::arch::warp_reduction_sum_f32`.
    ///
    /// Half-warp 0 (`lanes 0..15`) reduces routed expert slots `0..1` while half-warp 1
    /// (`lanes 16..31`) reduces routed expert slots `2..3`, combining all 4 expert outputs
    /// in a single 32-lane warp reduction.
    #[kernel]
    pub fn moe_fc2_cute(
        w_down_values: &[u8],
        w_down_scales: &[u8],
        w_down_global: &[f32],
        inter_values: &[u8],
        inter_scales: &[u8],
        expert_ids: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        total_routes: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let warp_in_block = tid / 32;
        let lane = tid % 32;
        let out_col = (thread::blockIdx_x() as usize) * 4 + warp_in_block;
        let token = thread::blockIdx_y() as usize;
        let total_w_rows = num_experts * hidden_size;

        let (tile_u32, half_u32) = idx2crd_col_major_2d(lane as u32, 16);
        let (tile, half) = (tile_u32 as usize, half_u32 as usize);
        let r0 = token * 4 + half * 2;
        let r1 = r0 + 1;
        let expert_ids_t = StridedTensor::from_slice(expert_ids);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);
        let exp0 = unsafe { expert_ids_t.get_unchecked(r0) } as usize;
        let exp1 = unsafe { expert_ids_t.get_unchecked(r1) } as usize;
        let s0 = routing_weights[r0].to_f32() * unsafe { w_down_global_t.get_unchecked(exp0) };
        let s1 = routing_weights[r1].to_f32() * unsafe { w_down_global_t.get_unchecked(exp1) };

        let weights0 =
            BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mkl>>::from_slices(
                w_down_values,
                w_down_scales,
                total_w_rows,
                intermediate_size,
            );
        let inputs0 = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Nkl>>::from_slices(
            inter_values,
            inter_scales,
            total_routes,
            intermediate_size,
        );
        let w_row0 = weights0.thread_row(0, exp0 * hidden_size + out_col);
        let x_row0 = inputs0.thread_row(0, r0);
        let (wf0, xf0) = unsafe { (w_row0.k_tile(tile).load(), x_row0.k_tile(tile).load()) };
        let d0 = wf0.dot_accumulate(xf0, 0.0f32);

        let weights1 =
            BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Mkl>>::from_slices(
                w_down_values,
                w_down_scales,
                total_w_rows,
                intermediate_size,
            );
        let inputs1 = BlockScaledTensor::<Float4E2M1FN, Float8E4M3FN, 16, KMajor<Nkl>>::from_slices(
            inter_values,
            inter_scales,
            total_routes,
            intermediate_size,
        );
        let w_row1 = weights1.thread_row(0, exp1 * hidden_size + out_col);
        let x_row1 = inputs1.thread_row(0, r1);
        let (wf1, xf1) = unsafe { (w_row1.k_tile(tile).load(), x_row1.k_tile(tile).load()) };
        let d1 = wf1.dot_accumulate(xf1, 0.0f32);

        let total = unsafe { cute_rs::arch::warp_reduction_sum_f32(d0 * s0 + d1 * s1) };
        let mut out_tensor = StridedTensorMut::from_disjoint_slice(&mut output);
        if lane == 0 {
            unsafe {
                out_tensor.store_at(token * hidden_size + out_col, BFloat16::from_f32(total));
            }
        }
    }

    /// Issue all 8 `K = 64` steps of one `K = 512` `256`-row (`ActTileSmem`) burst via
    /// `TiledMma::copy` (`Cp4x32x128bOp`) and `TiledMma::gemm` (`MmaMXF4NVF4Op`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn issue_burst_256(
        tiled_mma: &TiledMma,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        s_a: &SharedWeightTile,
        s_b: &SharedActTile,
        s_sfa: &SharedWeightScales,
        s_sfb: &SharedActScales,
        copy_sfa: bool,
        first_burst: bool,
    ) {
        let mma = *tiled_mma;
        let a = *s_a;
        let b = *s_b;
        let sfa = *s_sfa;
        let sfb = *s_sfb;
        macro_rules! step_256 {
            ($s:expr, $acc:expr) => {{
                let sa = sfa_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 8;
                if copy_sfa {
                    TiledMma::copy(sa, &sfa, $s);
                }
                TiledMma::copy(sb, &sfb, $s * 2);
                TiledMma::copy(sb + 4, &sfb, $s * 2 + 1);
                mma.gemm(tmem_d, &a, &b, sa, sb, $s, $acc);
            }};
        }
        unsafe {
            step_256!(0, !first_burst);
            step_256!(1, true);
            step_256!(2, true);
            step_256!(3, true);
            step_256!(4, true);
            step_256!(5, true);
            step_256!(6, true);
            step_256!(7, true);
        }
    }

    /// Issue all 8 `K = 64` steps of one `K = 512` `128`-row (`ActTile128Smem`) burst via
    /// `TiledMma128::copy` (`Cp4x32x128bOp`) and `TiledMma128::gemm` (`MmaMXF4NVF4Op`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn issue_burst_128(
        tiled_mma: &TiledMma128,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        s_a: &Sm100SharedTile<WeightTileSmem, OperandB>,
        s_b: &SharedAct128Tile,
        s_sfb: &SharedActScales,
        first_burst: bool,
    ) {
        let mma = *tiled_mma;
        let a = *s_a;
        let b = *s_b;
        let sfb = *s_sfb;
        macro_rules! step_128 {
            ($s:expr, $acc:expr) => {{
                let sa = sfa_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                TiledMma128::copy(sb, &sfb, $s);
                mma.gemm(tmem_d, &b, &a, sb, sa, $s, $acc);
            }};
        }
        unsafe {
            step_128!(0, !first_burst);
            step_128!(1, true);
            step_128!(2, true);
            step_128!(3, true);
            step_128!(4, true);
            step_128!(5, true);
            step_128!(6, true);
            step_128!(7, true);
        }
    }

    /// Copy all 8 `(32, 4, 4)` A scale atoms of one `K = 512` burst into `sfa_tmem_base` via `TiledMma128::copy`.
    #[inline(always)]
    unsafe fn copy_a_scales_burst_128(sfa_tmem_base: u32, s_sfa: &SharedWeightScales) {
        let sfa = *s_sfa;
        unsafe {
            TiledMma128::copy(sfa_tmem_base, &sfa, 0);
            TiledMma128::copy(sfa_tmem_base + 4, &sfa, 1);
            TiledMma128::copy(sfa_tmem_base + 8, &sfa, 2);
            TiledMma128::copy(sfa_tmem_base + 12, &sfa, 3);
            TiledMma128::copy(sfa_tmem_base + 16, &sfa, 4);
            TiledMma128::copy(sfa_tmem_base + 20, &sfa, 5);
            TiledMma128::copy(sfa_tmem_base + 24, &sfa, 6);
            TiledMma128::copy(sfa_tmem_base + 28, &sfa, 7);
        }
    }

    /// 5th-Gen Tensor Core (`TiledMma`: `nvvm.tcgen05.mma.block_scale` + `nvvm.tcgen05.cp` +
    /// 2-Stage Full-TMA `MainloopPipeline` (`PipelineTmaAsync<2, 1, 73_728>`) + `copy_tma_2d` for `A, B, SFA, SFB`)
    /// Grouped `FC1 + ReLU²` + fused NVFP4 requantization (`K3`) for `B >= 4`.
    ///
    /// # Safety
    /// Must be launched with `block_dim = (544, 1, 1)` (`16` compute warps + `1` TMA producer warp),
    /// `shared_mem_bytes = 222208`, and valid host-encoded TMA descriptors.
    #[launch_bounds(544)]
    #[launch_contract(
        domain = 1,
        block = (544, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc1_relu2_tcgen05_cute(
        #[grid_constant] w_up_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_up_tma: &TensorMap<u8, ActTileSmem>,
        #[grid_constant] sfa_up_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_up_tma: &TensorMap<u16, ScaleBTmaSmem>,
        w_up_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        mut q_inter_vals: DisjointSlice<u8>,
        mut q_inter_scales: DisjointSlice<u8>,
        intermediate_size: usize,
        groups_per_cta: usize,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let (tile_m128_u32, worker_u32) = idx2crd_col_major_2d(cta_id as u32, 8);
        let tile_m128 = tile_m128_u32 as usize;
        let group_start = (worker_u32 as usize) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let w_up_global_t = StridedTensor::from_slice(w_up_global);
        let expert = unsafe { group_experts_t.get_unchecked(group_start) } as usize;
        let limit = unsafe { group_limits_t.get_unchecked(group_start) } as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let w_outer = expert * (intermediate_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring =
            unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 4) };
        let mma_done_ring =
            unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 8) };
        let slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, PIPELINE_OFFSET + 80) };

        unsafe {
            let pipeline = MainloopPipeline::from_shared_bytes(dyn_smem, PIPELINE_OFFSET);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(slot_ptr);
                TmemAllocator::<512, 1>::relinquish_alloc_permit();
            }
            pipeline.init(0);

            if warp_id < COMPUTE_WARPS {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma::new_with_lbo(valid_n as u32, 4096);
                let g_scale = w_up_global_t.get_unchecked(expert);
                let mut vals_u64_out =
                    StridedTensorMut::<u64>::from_disjoint_slice_as(&mut q_inter_vals);
                let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut q_inter_scales);

                let mut consumer_state = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    if warp_id == 0 {
                        let mut burst = 0usize;
                        while burst < 2 {
                            pipeline.consumer_wait(&consumer_state);
                            let slot = consumer_state.index();
                            if tid == 0 {
                                let s_a = SharedWeightTile::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_A_OFFSET,
                                    slot,
                                );
                                let s_b =
                                    SharedActTile::from_stage_bytes(dyn_smem, SMEM_B_OFFSET, slot);
                                let s_sfa = SharedWeightScales::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_SFA_OFFSET,
                                    4096,
                                    slot,
                                );
                                let s_sfb = SharedActScales::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_SFB_OFFSET,
                                    8192,
                                    slot,
                                );
                                let sfa_burst_tmem = sfa_tmem_base + (slot as u32) * 32;
                                let sfb_burst_tmem = sfb_tmem_base + (slot as u32) * 64;
                                issue_burst_256(
                                    &tiled_mma,
                                    tmem_addr,
                                    sfa_burst_tmem,
                                    sfb_burst_tmem,
                                    &s_a,
                                    &s_b,
                                    &s_sfa,
                                    &s_sfb,
                                    g_idx == 0,
                                    burst == 0,
                                );
                            }
                            if burst == 1 {
                                mma_done_ring.commit_mma(g_idx as u32);
                                if tid == 0 {
                                    mma_done_ring.wait(g_idx as u32);
                                }
                            }
                            pipeline.consumer_release(&consumer_state);
                            consumer_state.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 512>::sync();
                    tcgen05_fence_after_thread_sync();

                    let mut smem_epi = StridedTensorMut::<f32>::from_shared_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        256 * 128,
                    );
                    if tid < 128 {
                        let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                        let (c_lo4_u32, c_hi32_u32) = idx2crd_col_major_2d(tid as u32, 4);
                        let (c_mid4_u32, c_top8_u32) = idx2crd_col_major_2d(c_hi32_u32, 4);
                        let col_perm = (c_mid4_u32 * 32 + c_top8_u32 * 4 + c_lo4_u32) as usize;
                        let mut n_base = 0usize;
                        while n_base + 32 <= valid_n {
                            let r0 =
                                TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                    .to_tensor_ssa();
                            let r1 = TmemFragment::<16>::load_deferred(
                                tmem_row_base + ((n_base + 16) as u32),
                            )
                            .to_tensor_ssa();
                            let u0 = (r0 * g_scale).fmax(0.0);
                            let u1 = (r1 * g_scale).fmax(0.0);
                            let act0 = u0 * u0;
                            let act1 = u1 * u1;
                            let mut c = 0usize;
                            while c < 16 {
                                smem_epi.store_at((n_base + c) * 128 + col_perm, act0[c]);
                                smem_epi.store_at((n_base + 16 + c) * 128 + col_perm, act1[c]);
                                c += 1;
                            }
                            n_base += 32;
                        }
                        if n_base < valid_n {
                            let r0 =
                                TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                    .to_tensor_ssa();
                            let u0 = (r0 * g_scale).fmax(0.0);
                            let act0 = u0 * u0;
                            let mut c = 0usize;
                            while c < 16 {
                                smem_epi.store_at((n_base + c) * 128 + col_perm, act0[c]);
                                c += 1;
                            }
                        }
                        fence_view_async_tmem_load();
                    }
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 512>::sync();

                    let (sub_tile16_u32, row_off64_u32) = idx2crd_col_major_2d(tid as u32, 8);
                    let (sub_tile16, row_off64) = (sub_tile16_u32 as usize, row_off64_u32 as usize);
                    let grp = tile_m128 * 8 + sub_tile16;
                    let smem_epi_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, SMEM_B_OFFSET, 256 * 32);
                    let mut n = row_off64;
                    while n < limit {
                        let base_u128 = n * 32 + sub_tile16;
                        let q0 = smem_epi_u128.load_f32x4(base_u128);
                        let q1 = smem_epi_u128.load_f32x4(base_u128 + 8);
                        let q2 = smem_epi_u128.load_f32x4(base_u128 + 16);
                        let q3 = smem_epi_u128.load_f32x4(base_u128 + 24);
                        let vec = q0.concat(q1).concat(q2.concat(q3));
                        let (q_vec, scale) = quantize_tile16_nvfp4(vec);
                        let s_idx =
                            sm1xx_block_scale_crd2idx(group * 16 + grp / 4, n, grp % 4, 2, 1);
                        let u64_idx = sm100_noswizzle_burst_u64_crd2idx::<256>(group, n, grp, 2);
                        scales_out.store_at(s_idx, scale.to_bits());
                        vals_u64_out.store_at(u64_idx, q_vec.bitcast::<u64, 1>()[0]);
                        n += 64;
                    }
                    if g_idx + 1 < groups_per_cta && tid == 0 {
                        b_free_ring.commit_threads((g_idx * 2) as u32);
                        b_free_ring.commit_threads((g_idx * 2 + 1) as u32);
                    }
                    fence_view_async_shared();
                    NamedBarrier::<1, 512>::sync();
                    g_idx += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(slot_ptr);
                }
            } else if warp_id == PRODUCER_WARP && (tid & 31) == 0 {
                let mut producer_state = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let (burst_u32, g_idx_u32) = idx2crd_col_major_2d(step as u32, 2);
                    let (burst, g_idx) = (burst_u32 as usize, g_idx_u32 as usize);
                    let group = group_start + g_idx;
                    if g_idx > 0 {
                        b_free_ring.wait(((g_idx - 1) * 2 + burst) as u32);
                    }
                    let slot = producer_state.index();
                    pipeline.producer_acquire(&producer_state);
                    let full_barrier = pipeline.producer_expect_tx(&producer_state);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_A_OFFSET,
                        slot,
                    );
                    let mut b_tile = SmemTile::<u8, ActTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        slot,
                    );
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        slot,
                    );
                    let mut sfb_tile = SmemTile::<u16, ScaleBTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFB_OFFSET,
                        slot,
                    );
                    copy_tma_2d(w_up_tma, (w_outer, burst), &mut a_tile, full_barrier);
                    copy_tma_2d(b_up_tma, (group * 2 + burst, 0), &mut b_tile, full_barrier);
                    copy_tma_2d(
                        sfa_up_tma,
                        (w_outer * 2 + burst, 0),
                        &mut sfa_tile,
                        full_barrier,
                    );
                    copy_tma_2d(
                        sfb_up_tma,
                        (group * 2 + burst, 0),
                        &mut sfb_tile,
                        full_barrier,
                    );
                    producer_state.advance();
                    step += 1;
                }
                pipeline.producer_tail(producer_state);
            }
        }
    }

    /// 5th-Gen Tensor Core (`TiledMma`: `nvvm.tcgen05.mma.block_scale` + `nvvm.tcgen05.cp` +
    /// Persistent 2-Stage Full-TMA `MainloopPipeline` (`PipelineTmaAsync<2, 1, 110_592>`) + `copy_tma_2d` for `A, B, SFA, SFB`)
    /// Grouped `FC2 + Finalize` (`K4` fused) for `B >= 2`.
    ///
    /// # Safety
    /// Must be launched with `block_dim = (544, 1, 1)` (`16` compute warps + `1` TMA producer warp),
    /// `shared_mem_bytes = 222208`, and valid host-encoded TMA descriptors.
    #[launch_bounds(544)]
    #[launch_contract(
        domain = 1,
        block = (544, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc2_tcgen05_cute(
        #[grid_constant] w_down_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_down_tma: &TensorMap<u8, ActTileSmem>,
        #[grid_constant] sfa_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_down_tma: &TensorMap<u16, ScaleBTmaSmem>,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_routes: &[i32],
        group_limits: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        hidden_size: usize,
        groups_per_cta: usize,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let (tile_m128_u32, worker_u32) = idx2crd_col_major_2d(cta_id as u32, 8);
        let tile_m128 = tile_m128_u32 as usize;
        let group_start = (worker_u32 as usize) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let group_routes_t = StridedTensor::from_slice(group_routes);
        let routing_weights_t = StridedTensor::from_slice(routing_weights);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);
        let expert = unsafe { group_experts_t.get_unchecked(group_start) } as usize;
        let limit = unsafe { group_limits_t.get_unchecked(group_start) } as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let w_outer = expert * (hidden_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring =
            unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 4) };
        let mma_done_ring =
            unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 8) };
        let slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, PIPELINE_OFFSET + 80) };

        unsafe {
            let pipeline = MainloopPipeline::from_shared_bytes(dyn_smem, PIPELINE_OFFSET);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(slot_ptr);
                TmemAllocator::<512, 1>::relinquish_alloc_permit();
            }
            pipeline.init(0);

            if warp_id < COMPUTE_WARPS {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma::new_with_lbo(valid_n as u32, 4096);
                let g_scale = w_down_global_t.get_unchecked(expert);

                let mut consumer_state = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    if warp_id == 0 {
                        let mut burst = 0usize;
                        while burst < 2 {
                            pipeline.consumer_wait(&consumer_state);
                            let slot = consumer_state.index();
                            if tid == 0 {
                                let s_a = SharedWeightTile::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_A_OFFSET,
                                    slot,
                                );
                                let s_b =
                                    SharedActTile::from_stage_bytes(dyn_smem, SMEM_B_OFFSET, slot);
                                let s_sfa = SharedWeightScales::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_SFA_OFFSET,
                                    4096,
                                    slot,
                                );
                                let s_sfb = SharedActScales::from_stage_bytes(
                                    dyn_smem,
                                    SMEM_SFB_OFFSET,
                                    8192,
                                    slot,
                                );
                                let sfa_burst_tmem = sfa_tmem_base + (slot as u32) * 32;
                                let sfb_burst_tmem = sfb_tmem_base + (slot as u32) * 64;
                                issue_burst_256(
                                    &tiled_mma,
                                    tmem_addr,
                                    sfa_burst_tmem,
                                    sfb_burst_tmem,
                                    &s_a,
                                    &s_b,
                                    &s_sfa,
                                    &s_sfb,
                                    g_idx == 0,
                                    burst == 0,
                                );
                            }
                            if burst == 1 {
                                mma_done_ring.commit_mma(g_idx as u32);
                                if tid == 0 {
                                    mma_done_ring.wait(g_idx as u32);
                                }
                            }
                            pipeline.consumer_release(&consumer_state);
                            consumer_state.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 512>::sync();
                    tcgen05_fence_after_thread_sync();

                    let mut smem_epi = StridedTensorMut::<f32>::from_shared_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        256 * 128,
                    );
                    if tid < 128 {
                        let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                        let mut n_base = 0usize;
                        while n_base + 32 <= valid_n {
                            let r0 =
                                TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                    .to_tensor_ssa();
                            let r1 = TmemFragment::<16>::load_deferred(
                                tmem_row_base + ((n_base + 16) as u32),
                            )
                            .to_tensor_ssa();
                            let mut c = 0usize;
                            while c < 16 {
                                smem_epi.store_at((n_base + c) * 128 + tid, r0[c]);
                                smem_epi.store_at((n_base + 16 + c) * 128 + tid, r1[c]);
                                c += 1;
                            }
                            n_base += 32;
                        }
                        if n_base < valid_n {
                            let r0 =
                                TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                    .to_tensor_ssa();
                            let mut c = 0usize;
                            while c < 16 {
                                smem_epi.store_at((n_base + c) * 128 + tid, r0[c]);
                                c += 1;
                            }
                        }
                        fence_view_async_tmem_load();
                    }
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 512>::sync();

                    let (col4_u32, row_off128_u32) = idx2crd_col_major_2d(tid as u32, 4);
                    let (col4, row_off128) = (col4_u32 as usize, row_off128_u32 as usize);
                    let output_t = StridedTensorMut::from_disjoint_slice(&mut output);
                    let col_bf16_base = tile_m128 * 128 + col4 * 4;
                    let smem_epi_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, SMEM_B_OFFSET, 256 * 32);

                    let half_n = valid_n.min(128);
                    let half_mask = half_n - 1;
                    let n_rot = (expert * (half_n / 4)) & half_mask;
                    let mut n_step = row_off128;
                    while n_step < valid_n {
                        let n = (n_step & 128) | ((n_step + n_rot) & half_mask);
                        if n < limit {
                            let route = group_routes_t.get_unchecked(group * 256 + n) as usize;
                            let token = route / 4;
                            let sw = g_scale * routing_weights_t.get_unchecked(route).to_f32();
                            let base_u128 = n * 32 + col4;
                            macro_rules! add_quad {
                                ($k:expr) => {{
                                    let q =
                                        smem_epi_u128.load_f32x4(base_u128 + $k * 4).into_array();
                                    let c = col_bf16_base + $k * 16;
                                    output_t.atomic_add_bf16x2_at_2d(
                                        token,
                                        c,
                                        hidden_size,
                                        BFloat16::from_f32(q[0] * sw),
                                        BFloat16::from_f32(q[1] * sw),
                                    );
                                    output_t.atomic_add_bf16x2_at_2d(
                                        token,
                                        c + 2,
                                        hidden_size,
                                        BFloat16::from_f32(q[2] * sw),
                                        BFloat16::from_f32(q[3] * sw),
                                    );
                                }};
                            }
                            add_quad!(0);
                            add_quad!(1);
                            add_quad!(2);
                            add_quad!(3);
                            add_quad!(4);
                            add_quad!(5);
                            add_quad!(6);
                            add_quad!(7);
                        }
                        if g_idx + 1 < groups_per_cta && n_step < 128 {
                            fence_view_async_shared();
                            NamedBarrier::<1, 512>::sync();
                            if tid == 0 {
                                b_free_ring.commit_threads((g_idx * 2) as u32);
                            }
                        }
                        n_step += 128;
                    }
                    if g_idx + 1 < groups_per_cta {
                        fence_view_async_shared();
                        NamedBarrier::<1, 512>::sync();
                        if tid == 0 {
                            b_free_ring.commit_threads((g_idx * 2 + 1) as u32);
                        }
                    }
                    g_idx += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(slot_ptr);
                }
            } else if warp_id == PRODUCER_WARP && (tid & 31) == 0 {
                let mut producer_state = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let (burst_u32, g_idx_u32) = idx2crd_col_major_2d(step as u32, 2);
                    let (burst, g_idx) = (burst_u32 as usize, g_idx_u32 as usize);
                    let group = group_start + g_idx;
                    if g_idx > 0 {
                        b_free_ring.wait(((g_idx - 1) * 2 + burst) as u32);
                    }
                    let slot = producer_state.index();
                    pipeline.producer_acquire(&producer_state);
                    let full_barrier = pipeline.producer_expect_tx(&producer_state);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_A_OFFSET,
                        slot,
                    );
                    let mut b_tile = SmemTile::<u8, ActTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        slot,
                    );
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        slot,
                    );
                    let mut sfb_tile = SmemTile::<u16, ScaleBTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFB_OFFSET,
                        slot,
                    );
                    copy_tma_2d(w_down_tma, (w_outer, burst), &mut a_tile, full_barrier);
                    copy_tma_2d(
                        b_down_tma,
                        (group * 2 + burst, 0),
                        &mut b_tile,
                        full_barrier,
                    );
                    copy_tma_2d(
                        sfa_down_tma,
                        (w_outer * 2 + burst, 0),
                        &mut sfa_tile,
                        full_barrier,
                    );
                    copy_tma_2d(
                        sfb_down_tma,
                        (group * 2 + burst, 0),
                        &mut sfb_tile,
                        full_barrier,
                    );
                    producer_state.advance();
                    step += 1;
                }
                pipeline.producer_tail(producer_state);
            }
        }
    }

    /// 6-Warp (`192`-Thread) Weight-Stationary 4-Stage TMA (`Act128Pipeline`) + 2-Stage TMEM (`TMEM[0..1]`)
    /// Grouped `FC1 + ReLU²` (`K3`) for `B >= 2048`.
    ///
    /// # Safety
    /// Must be launched with `block_dim = (192, 1, 1)` (`4` epilogue warps + `1` MMA warp + `1` TMA producer warp),
    /// `shared_mem_bytes = 222208`, and valid host-encoded TMA descriptors.
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc1_relu2_tcgen05_128_cute(
        #[grid_constant] w_up_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_up_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_up_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_up_tma: &TensorMap<u16, ScaleTmaSmem>,
        w_up_global: &[f32],
        group_experts: &[i32],
        mut q_inter_vals: DisjointSlice<u8>,
        mut q_inter_scales: DisjointSlice<u8>,
        intermediate_size: usize,
        groups_per_cta: usize,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let (tile_m128_u32, worker_u32) = idx2crd_col_major_2d(cta_id as u32, 8);
        let tile_m128 = tile_m128_u32 as usize;
        let group_start = (worker_u32 as usize) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let group_experts_t = StridedTensor::from_slice(group_experts);
        let w_up_global_t = StridedTensor::from_slice(w_up_global);
        let expert = unsafe { group_experts_t.get_unchecked(group_start) } as usize;
        let w_outer = expert * (intermediate_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring =
            unsafe { SingleCtaRing::<4>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 8) };
        let mma_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 16) };
        let tmem_free_ring =
            unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, PIPELINE_OFFSET / 8 + 20) };
        let slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, PIPELINE_OFFSET + 224) };

        unsafe {
            let b_pipeline = Act128Pipeline::from_shared_bytes(dyn_smem, PIPELINE_OFFSET);
            let w_pipeline = Weight128Pipeline::from_shared_bytes(dyn_smem, PIPELINE_OFFSET + 256);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_ring.init(1, 1);
                tmem_free_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(slot_ptr);
                TmemAllocator::<512, 1>::relinquish_alloc_permit();
            }
            w_pipeline.init(0);
            b_pipeline.init(0);

            if warp_id == 5 && (tid & 31) == 0 {
                let mut w_prod = PipelineState::<Producer, 2>::new();
                let mut w_burst = 0usize;
                while w_burst < 2 {
                    let w_slot = w_prod.index();
                    w_pipeline.producer_acquire(&w_prod);
                    let w_bar = w_pipeline.producer_expect_tx(&w_prod);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_A_OFFSET,
                        w_slot,
                    );
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        w_slot,
                    );
                    copy_tma_2d(w_up_tma, (w_outer, w_burst), &mut a_tile, w_bar);
                    copy_tma_2d(sfa_up_tma, (w_outer * 2 + w_burst, 0), &mut sfa_tile, w_bar);
                    w_prod.advance();
                    w_burst += 1;
                }

                let mut b_prod = PipelineState::<Producer, 4>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let slot = b_prod.index();
                    if step >= 4 {
                        b_free_ring.wait((step - 4) as u32);
                    }
                    b_pipeline.producer_acquire(&b_prod);
                    let b_bar = b_pipeline.producer_expect_tx(&b_prod);
                    let mut b_tile = SmemTile::<u8, ActTile128Smem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        slot,
                    );
                    let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFB_OFFSET,
                        slot,
                    );
                    copy_tma_2d(b_up_tma, (group_start * 2 + step, 0), &mut b_tile, b_bar);
                    copy_tma_2d(
                        sfb_up_tma,
                        (group_start * 2 + step, 0),
                        &mut sfb_tile,
                        b_bar,
                    );
                    b_prod.advance();
                    step += 1;
                }
                w_pipeline.producer_tail(w_prod);
                b_pipeline.producer_tail(b_prod);
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma128::new_with_lbo(128, 2048);

                let mut w_cons = PipelineState::<Consumer, 2>::new();
                let mut w_burst = 0usize;
                while w_burst < 2 {
                    let w_slot = w_cons.index();
                    w_pipeline.consumer_wait(&w_cons);
                    let s_sfa = SharedWeightScales::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        4096,
                        w_slot,
                    );
                    copy_a_scales_burst_128(sfa_tmem_base + (w_burst as u32) * 32, &s_sfa);
                    w_pipeline.consumer_release(&w_cons);
                    w_cons.advance();
                    w_burst += 1;
                }

                let mut b_cons = PipelineState::<Consumer, 4>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let tmem_slot = g_idx & 1;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if g_idx >= 2 {
                        tmem_free_ring.wait((g_idx - 2) as u32);
                    }
                    let mut burst = 0usize;
                    while burst < 2 {
                        let step = g_idx * 2 + burst;
                        b_pipeline.consumer_wait(&b_cons);
                        let slot = b_cons.index();
                        let s_a = Sm100SharedTile::<WeightTileSmem, OperandB>::from_stage_bytes(
                            dyn_smem,
                            SMEM_A_OFFSET,
                            burst,
                        );
                        let s_b = SharedAct128Tile::from_stage_bytes(dyn_smem, SMEM_B_OFFSET, slot);
                        let s_sfb = SharedActScales::from_stage_bytes(
                            dyn_smem,
                            SMEM_SFB_OFFSET,
                            4096,
                            slot,
                        );
                        issue_burst_128(
                            &tiled_mma,
                            tmem_d,
                            sfa_tmem_base + (burst as u32) * 32,
                            sfb_tmem_base + (slot as u32) * 32,
                            &s_a,
                            &s_b,
                            &s_sfb,
                            burst == 0,
                        );
                        b_pipeline.consumer_release(&b_cons);
                        b_cons.advance();
                        if step + 4 < total_steps {
                            b_free_ring.producer_commit(step as u32);
                        }
                        if burst == 1 {
                            mma_ring.producer_commit(g_idx as u32);
                        }
                        burst += 1;
                    }
                    g_idx += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let g_scale = w_up_global_t.get_unchecked(expert);
                let scale_sq_div6 = g_scale * g_scale * (1.0f32 / 6.0f32);
                let g_scale_sq = g_scale * g_scale;
                let mut vals_u128_out =
                    StridedTensorMut::<u128>::from_disjoint_slice_as(&mut q_inter_vals);
                let mut scales_u32_out =
                    StridedTensorMut::<u32>::from_disjoint_slice_as(&mut q_inter_scales);
                let grp_m0 = tile_m128 * 8;
                let grp_m1 = tile_m128 * 8 + 4;
                let s_base_m0 = sm1xx_block_scale_crd2idx(0, tid, grp_m0, 1, 16) / 4;
                let s_base_m1 = sm1xx_block_scale_crd2idx(0, tid, grp_m1, 1, 16) / 4;
                let u128_base_m0_0 =
                    sm100_noswizzle_burst_u64_crd2idx::<128>(0, tid, grp_m0, 2) / 2;
                let u128_base_m0_2 =
                    sm100_noswizzle_burst_u64_crd2idx::<128>(0, tid, grp_m0 + 2, 2) / 2;
                let u128_base_m1_0 =
                    sm100_noswizzle_burst_u64_crd2idx::<128>(0, tid, grp_m1, 2) / 2;
                let u128_base_m1_2 =
                    sm100_noswizzle_burst_u64_crd2idx::<128>(0, tid, grp_m1 + 2, 2) / 2;

                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        mma_ring.wait(g_idx as u32);
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let tmem_row_base = tmem_row_address(tmem_d, warp_id, 0);
                    let group_s_off = group * 2048;
                    let group_u128_off = group * 4096;

                    let frag0 = TmemFragment::<64>::load(tmem_row_base);
                    let (q0, s0) = quantize_relu2_tile16_nvfp4(
                        frag0.slice::<0, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q1, s1) = quantize_relu2_tile16_nvfp4(
                        frag0.slice::<16, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q2, s2) = quantize_relu2_tile16_nvfp4(
                        frag0.slice::<32, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q3, s3) = quantize_relu2_tile16_nvfp4(
                        frag0.slice::<48, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let sfc0 = TensorSSA::from_array([s0, s1, s2, s3]);
                    scales_u32_out.store_at(group_s_off + s_base_m0, sfc0.bitcast::<u32, 1>()[0]);
                    vals_u128_out.store_at(
                        group_u128_off + u128_base_m0_0,
                        q0.concat(q1).bitcast::<u128, 1>()[0],
                    );
                    vals_u128_out.store_at(
                        group_u128_off + u128_base_m0_2,
                        q2.concat(q3).bitcast::<u128, 1>()[0],
                    );

                    let frag1 = TmemFragment::<64>::load(tmem_row_base + 64);
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 128>::sync();
                    if g_idx + 2 < groups_per_cta && tid == 0 {
                        tmem_free_ring.commit_threads(g_idx as u32);
                    }
                    let (q4, s4) = quantize_relu2_tile16_nvfp4(
                        frag1.slice::<0, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q5, s5) = quantize_relu2_tile16_nvfp4(
                        frag1.slice::<16, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q6, s6) = quantize_relu2_tile16_nvfp4(
                        frag1.slice::<32, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let (q7, s7) = quantize_relu2_tile16_nvfp4(
                        frag1.slice::<48, 16>(),
                        scale_sq_div6,
                        g_scale_sq,
                    );
                    let sfc1 = TensorSSA::from_array([s4, s5, s6, s7]);
                    scales_u32_out.store_at(group_s_off + s_base_m1, sfc1.bitcast::<u32, 1>()[0]);
                    vals_u128_out.store_at(
                        group_u128_off + u128_base_m1_0,
                        q4.concat(q5).bitcast::<u128, 1>()[0],
                    );
                    vals_u128_out.store_at(
                        group_u128_off + u128_base_m1_2,
                        q6.concat(q7).bitcast::<u128, 1>()[0],
                    );

                    g_idx += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(slot_ptr);
                }
            }
        }
    }

    /// 6-Warp (`192`-Thread) Weight-Stationary 4-Stage TMA (`Act128Pipeline`) + 2-Stage TMEM (`TMEM[0..1]`) +
    /// 2-Stage `B64`-Swizzled S2G TMA Store (`copy_tma_s2g_2d`) Grouped `FC2` (`K4`) for `B >= 2048`.
    ///
    /// # Safety
    /// Must be launched with `block_dim = (192, 1, 1)`, `shared_mem_bytes = 230400`, and valid TMA descriptors.
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 230400,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc2_tcgen05_128_cute(
        #[grid_constant] w_down_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_down_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] grouped_out_tma: &TensorMap<u16, GroupedOutTileSmem>,
        w_down_global: &[f32],
        group_experts: &[i32],
        mut grouped_out: DisjointSlice<BFloat16>,
        hidden_size: usize,
        groups_per_cta: usize,
    ) {
        let _ = (grouped_out.as_mut_ptr(), hidden_size);
        let cta_id = thread::blockIdx_x() as usize;
        let (tile_m128_u32, worker_u32) = idx2crd_col_major_2d(cta_id as u32, 8);
        let tile_m128 = tile_m128_u32 as usize;
        let group_start = (worker_u32 as usize) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let group_experts_t = StridedTensor::from_slice(group_experts);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);
        let expert = unsafe { group_experts_t.get_unchecked(group_start) } as usize;
        let w_outer = expert * (hidden_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring = unsafe { SingleCtaRing::<4>::from_raw(dyn_smem, 229376 / 8 + 8) };
        let mma_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 229376 / 8 + 16) };
        let tmem_free_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 229376 / 8 + 20) };
        let slot_ptr = unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, 229376 + 224) };

        unsafe {
            let b_pipeline = Act128Pipeline::from_shared_bytes(dyn_smem, 229376);
            let w_pipeline = Weight128Pipeline::from_shared_bytes(dyn_smem, 229376 + 256);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                grouped_out_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_ring.init(1, 1);
                tmem_free_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(slot_ptr);
                TmemAllocator::<512, 1>::relinquish_alloc_permit();
            }
            w_pipeline.init(0);
            b_pipeline.init(0);

            if warp_id == 5 && (tid & 31) == 0 {
                let mut w_prod = PipelineState::<Producer, 2>::new();
                let mut w_burst = 0usize;
                while w_burst < 2 {
                    let w_slot = w_prod.index();
                    w_pipeline.producer_acquire(&w_prod);
                    let w_bar = w_pipeline.producer_expect_tx(&w_prod);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_A_OFFSET,
                        w_slot,
                    );
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        w_slot,
                    );
                    copy_tma_2d(w_down_tma, (w_outer, w_burst), &mut a_tile, w_bar);
                    copy_tma_2d(
                        sfa_down_tma,
                        (w_outer * 2 + w_burst, 0),
                        &mut sfa_tile,
                        w_bar,
                    );
                    w_prod.advance();
                    w_burst += 1;
                }

                let mut b_prod = PipelineState::<Producer, 4>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let slot = b_prod.index();
                    if step >= 4 {
                        b_free_ring.wait((step - 4) as u32);
                    }
                    b_pipeline.producer_acquire(&b_prod);
                    let b_bar = b_pipeline.producer_expect_tx(&b_prod);
                    let mut b_tile = SmemTile::<u8, ActTile128Smem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_B_OFFSET,
                        slot,
                    );
                    let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFB_OFFSET,
                        slot,
                    );
                    copy_tma_2d(b_down_tma, (group_start * 2 + step, 0), &mut b_tile, b_bar);
                    copy_tma_2d(
                        sfb_down_tma,
                        (group_start * 2 + step, 0),
                        &mut sfb_tile,
                        b_bar,
                    );
                    b_prod.advance();
                    step += 1;
                }
                w_pipeline.producer_tail(w_prod);
                b_pipeline.producer_tail(b_prod);
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma128::new_with_lbo(128, 2048);

                let mut w_cons = PipelineState::<Consumer, 2>::new();
                let mut w_burst = 0usize;
                while w_burst < 2 {
                    let w_slot = w_cons.index();
                    w_pipeline.consumer_wait(&w_cons);
                    let s_sfa = SharedWeightScales::from_stage_bytes(
                        dyn_smem,
                        SMEM_SFA_OFFSET,
                        4096,
                        w_slot,
                    );
                    copy_a_scales_burst_128(sfa_tmem_base + (w_burst as u32) * 32, &s_sfa);
                    w_pipeline.consumer_release(&w_cons);
                    w_cons.advance();
                    w_burst += 1;
                }

                let mut b_cons = PipelineState::<Consumer, 4>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let tmem_slot = g_idx & 1;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if g_idx >= 2 {
                        tmem_free_ring.wait((g_idx - 2) as u32);
                    }
                    let mut burst = 0usize;
                    while burst < 2 {
                        let step = g_idx * 2 + burst;
                        b_pipeline.consumer_wait(&b_cons);
                        let slot = b_cons.index();
                        let s_a = Sm100SharedTile::<WeightTileSmem, OperandB>::from_stage_bytes(
                            dyn_smem,
                            SMEM_A_OFFSET,
                            burst,
                        );
                        let s_b = SharedAct128Tile::from_stage_bytes(dyn_smem, SMEM_B_OFFSET, slot);
                        let s_sfb = SharedActScales::from_stage_bytes(
                            dyn_smem,
                            SMEM_SFB_OFFSET,
                            4096,
                            slot,
                        );
                        issue_burst_128(
                            &tiled_mma,
                            tmem_d,
                            sfa_tmem_base + (burst as u32) * 32,
                            sfb_tmem_base + (slot as u32) * 32,
                            &s_a,
                            &s_b,
                            &s_sfb,
                            burst == 0,
                        );
                        b_pipeline.consumer_release(&b_cons);
                        b_cons.advance();
                        if step + 4 < total_steps {
                            b_free_ring.producer_commit(step as u32);
                        }
                        if burst == 1 {
                            mma_ring.producer_commit(g_idx as u32);
                        }
                        burst += 1;
                    }
                    g_idx += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(slot_ptr);
                let g_scale = w_down_global_t.get_unchecked(expert);
                let mut smem_epi0 =
                    StridedTensorMut::<u128>::from_shared_bytes(dyn_smem, SMEM_SFA_OFFSET, 512);
                let mut smem_epi1 =
                    StridedTensorMut::<u128>::from_shared_bytes(dyn_smem, 221184, 512);
                let src0 = SmemTile::<u16, GroupedOutTileSmem>::from_stage_bytes(
                    dyn_smem,
                    SMEM_SFA_OFFSET,
                    0,
                );
                let src1 =
                    SmemTile::<u16, GroupedOutTileSmem>::from_stage_bytes(dyn_smem, 221184, 0);
                let idx0 = GroupedOutTileSmem::crd2idx(tid as u32, 0) / 8;
                let idx1 = GroupedOutTileSmem::crd2idx(tid as u32, 8) / 8;
                let idx2 = GroupedOutTileSmem::crd2idx(tid as u32, 16) / 8;
                let idx3 = GroupedOutTileSmem::crd2idx(tid as u32, 24) / 8;

                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        mma_ring.wait(g_idx as u32);
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let tmem_row_base = tmem_row_address(tmem_d, warp_id, 0);
                    macro_rules! store_half {
                        ($half_m:expr, $is_second:expr) => {{
                            let frag = TmemFragment::<64>::load(
                                tmem_row_base + (($half_m * 64) as u32),
                            );
                            if $is_second {
                                tcgen05_fence_before_thread_sync();
                                NamedBarrier::<1, 128>::sync();
                                if g_idx + 2 < groups_per_cta && tid == 0 {
                                    tmem_free_ring.commit_threads(g_idx as u32);
                                }
                            }
                            if tid == 0 {
                                cute_rs::arch::cp_async_bulk_wait_group_read::<1>();
                            }
                            NamedBarrier::<1, 128>::sync();
                            smem_epi0.store_at(
                                idx0,
                                (frag.slice::<0, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi0.store_at(
                                idx1,
                                (frag.slice::<8, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi0.store_at(
                                idx2,
                                (frag.slice::<16, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi0.store_at(
                                idx3,
                                (frag.slice::<24, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            fence_view_async_shared();
                            NamedBarrier::<1, 128>::sync();
                            if tid == 0 {
                                cute_rs::nvgpu::tcgen05::Sm100TmemEpilogue::<GroupedOutTileSmem>::store_async(
                                    grouped_out_tma,
                                    src0.base,
                                    (group * 128) as u32,
                                    (tile_m128 * 128 + $half_m * 64) as u32,
                                );
                                cute_rs::arch::cp_async_bulk_commit_group();
                            }

                            if tid == 0 {
                                cute_rs::arch::cp_async_bulk_wait_group_read::<1>();
                            }
                            NamedBarrier::<1, 128>::sync();
                            smem_epi1.store_at(
                                idx0,
                                (frag.slice::<32, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi1.store_at(
                                idx1,
                                (frag.slice::<40, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi1.store_at(
                                idx2,
                                (frag.slice::<48, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            smem_epi1.store_at(
                                idx3,
                                (frag.slice::<56, 8>() * g_scale)
                                    .to::<BFloat16>()
                                    .bitcast::<u128, 1>()[0],
                            );
                            fence_view_async_shared();
                            NamedBarrier::<1, 128>::sync();
                            if tid == 0 {
                                cute_rs::nvgpu::tcgen05::Sm100TmemEpilogue::<GroupedOutTileSmem>::store_async(
                                    grouped_out_tma,
                                    src1.base,
                                    (group * 128) as u32,
                                    (tile_m128 * 128 + $half_m * 64 + 32) as u32,
                                );
                                cute_rs::arch::cp_async_bulk_commit_group();
                            }
                        }};
                    }
                    store_half!(0, false);
                    store_half!(1, true);
                    g_idx += 1;
                }

                if tid == 0 {
                    cute_rs::arch::cp_async_bulk_wait_group_read::<0>();
                }
                NamedBarrier::<1, 128>::sync();
                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(slot_ptr);
                }
            }
        }
    }

    /// Weighted top-4 expert combination (`K5` finalize step for `B >= 2048`) using
    /// `1` warp (`32` threads, `4 × RegisterTile<BFloat16, 8>` = `32` `BFloat16`s per thread) per token
    /// over `Tensor::local_tile::<8>` and `TensorMut::local_tile::<8>`.
    #[kernel]
    pub fn moe_finalize_cute(
        grouped_out: &[BFloat16],
        tok_to_perm: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        num_tokens: usize,
        hidden_size: usize,
    ) {
        let tid_global = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + (thread::threadIdx_x() as usize);
        let (lane_u32, token_u32) = idx2crd_col_major_2d(tid_global as u32, 32);
        let (token, lane) = (token_u32 as usize, lane_u32 as usize);
        if token >= num_tokens {
            return;
        }
        let h_tiles = hidden_size / 8; // 128
        let perms = unsafe {
            cute_rs::Tensor::from_slice(tok_to_perm)
                .local_tile::<4>(token)
                .load()
                .into_array()
        };
        let rws = unsafe {
            cute_rs::Tensor::from_slice(routing_weights)
                .local_tile::<4>(token)
                .load()
                .to::<f32>()
                .into_array()
        };
        let r0 = (perms[0] as usize) * h_tiles + lane;
        let r1 = (perms[1] as usize) * h_tiles + lane;
        let r2 = (perms[2] as usize) * h_tiles + lane;
        let r3 = (perms[3] as usize) * h_tiles + lane;
        let (w0, w1, w2, w3) = (rws[0], rws[1], rws[2], rws[3]);
        let grouped_t = cute_rs::Tensor::from_slice(grouped_out);
        let out_base = token * h_tiles + lane;

        /// Coalesced 32-lane warp load/combine/store of 8 `BFloat16` elements at tile offset `$off`.
        macro_rules! combine_step {
            ($off:expr) => {
                unsafe {
                    let g0 = grouped_t.local_tile::<8>(r0 + $off).load().to::<f32>();
                    let g1 = grouped_t.local_tile::<8>(r1 + $off).load().to::<f32>();
                    let g2 = grouped_t.local_tile::<8>(r2 + $off).load().to::<f32>();
                    let g3 = grouped_t.local_tile::<8>(r3 + $off).load().to::<f32>();
                    let combined = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3).to::<BFloat16>();
                    cute_rs::TensorMut::from_disjoint_slice(&mut output)
                        .local_tile::<8>(out_base + $off)
                        .store(combined);
                }
            };
        }
        combine_step!(0);
        combine_step!(32);
        combine_step!(64);
        combine_step!(96);
    }
}

/// Synthetic input value assigned to every element of `token`'s hidden vector.
/// Alternates sign by token parity and cycles through `{0.375, 0.75, 1.5}` so every
/// FP4 product and `ReLU²` activation is exact in `f32`.
fn hidden_value(token: usize) -> f32 {
    let magnitude = 0.375 * [1.0, 2.0, 4.0][token % 3];
    if token.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

/// Deterministic expert index (`0..TOP_K`) selected by `token` at routing slot `slot`.
/// Rotates across the first `TOP_K = 4` experts so every active expert receives an equal share of tokens.
fn expert_id(token: usize, slot: usize) -> usize {
    (token + slot) % TOP_K
}

/// Routing weight assigned to `token`'s `slot`-th selected expert.
/// Sum of the `TOP_K = 4` weights for any token is `1.0`.
fn routing_weight(token: usize, slot: usize) -> f32 {
    ROUTING_PATTERN[(token + slot) % TOP_K]
}

/// Exact analytical `BFloat16` output reference for any element of `token`'s output row.
/// Evaluates `FC1 -> ReLU² -> FC2` across all `TOP_K = 4` routed experts and combines them
/// with [`routing_weight`].
fn expected_output(token: usize) -> BFloat16 {
    let hidden = hidden_value(token);
    let mut total = 0.0f32;
    for slot in 0..TOP_K {
        let expert = expert_id(token, slot);
        let pre_activation = hidden * UP_FACTORS[expert];
        let relu = pre_activation.max(0.0);
        let activated = relu * relu;
        let fc2 = activated * DOWN_FACTORS[expert];
        total += routing_weight(token, slot) * fc2;
    }
    BFloat16::from_f32(total)
}

/// Device buffers and typed TMA descriptors owned by one batch-size benchmark instance.
/// Holds the input tokens, routing tables, quantized workspace buffers, expert weights,
/// and the four host-encoded [`TensorMap`] descriptors passed to the `tcgen05` kernels.
struct CuteMoeBuffers {
    /// Input token matrix `[num_tokens, HIDDEN]` in `BFloat16`.
    /// Consumed by `K1` (`quantize_activations_nvfp4`).
    hidden: DeviceBuffer<BFloat16>,
    /// Selected expert IDs `[num_tokens * TOP_K]` (`i32` in `0..NUM_EXPERTS`).
    /// Used by the low-batch (`B <= 2`) kernels to index expert rows directly.
    expert_ids: DeviceBuffer<i32>,
    /// Per-token routing weights `[num_tokens * TOP_K]` in `BFloat16`.
    /// Applied during `K4` (`moe_fc2_cute` / `moe_finalize_cute`).
    routing_weights: DeviceBuffer<BFloat16>,
    /// Expert ID (`0..NUM_EXPERTS`) for each bucketed token group (`[num_groups8]`).
    /// Read by `moe_fc1_relu2_tcgen05_cute` and `moe_fc2_tcgen05_cute`.
    group_experts: DeviceBuffer<i32>,
    /// Padded source token indices `[num_groups8 * 128]` for each expert group.
    /// Maps each column `0..limit` of the `B` tile back to its input token row `0..num_tokens`.
    group_tokens: DeviceBuffer<i32>,
    /// Padded expanded route indices `[num_groups8 * 128]` (`token * TOP_K + slot`).
    /// Maps each column `0..limit` of the `B` tile to its intermediate/output route slot.
    group_routes: DeviceBuffer<i32>,
    /// Packed `(group << 8) | row_in_group` mapping `[num_tokens * TOP_K]` from `(token, slot)`
    /// to its TMA-ready `[128, 256]` burst tile position in `b_up_vals` / `sfb_up_scales`.
    tok_to_perm: DeviceBuffer<i32>,
    /// Valid token count `limit` (`1..=128`) in each bucketed expert group (`[num_groups8]`).
    /// Used to select the hardware `valid_n` (`16, 32, 64, 128`) and guard out-of-bounds lanes.
    group_limits: DeviceBuffer<i32>,
    /// Quantized input token values `[num_tokens, HIDDEN / 2]` (packed `Float4E2M1FN` bytes).
    /// Produced by `K1` for the low-batch (`B <= 2`) path.
    q_hidden_vals: DeviceBuffer<u8>,
    /// Quantized input token scales in canonical `(32, 4, 4)` `Sm120ScaleAtom` layout.
    /// Produced by `K1` for the low-batch (`B <= 2`) path.
    q_hidden_scales: DeviceBuffer<u8>,
    /// TMA-ready group-permuted `FC1` activation burst tiles `[num_groups8 * 2, 32768]` bytes.
    /// Written by `K1` and copied into the 2-stage `SMEM_B` ring buffer by [`Self::b_up_tma`].
    b_up_vals: DeviceBuffer<u8>,
    /// TMA-ready group-permuted `FC1` activation scale atoms `[num_groups8 * 2, 4096]` bytes.
    /// Written by `K1` and copied into the 2-stage `SMEM_SFB` ring buffer by [`Self::sfb_up_tma`].
    sfb_up_scales: DeviceBuffer<u8>,
    /// Typed 2D TMA descriptor for `FC1` `B` tiles encoding [`ActTileSmem`] (`RowMajor<128, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_cute`.
    b_up_tma: TensorMap<u8, ActTileSmem>,
    /// Typed 2D TMA descriptor for `FC1` `SFB` scales encoding [`ScaleBTmaSmem`] (`RowMajor<16, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_cute`.
    sfb_up_tma: TensorMap<u16, ScaleBTmaSmem>,
    /// Typed 2D TMA descriptor for `FC1` `B` tiles encoding [`ActTile128Smem`] (`RowMajor<128, 256>`) for `B >= 2048`.
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_128_cute`.
    b_up_128_tma: TensorMap<u8, ActTile128Smem>,
    /// Typed 2D TMA descriptor for `FC1` `SFB` scales encoding [`ScaleTmaSmem`] (`RowMajor<8, 256>`) for `B >= 2048`.
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_128_cute`.
    sfb_up_128_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Quantized `FC1 + ReLU²` intermediate values `[num_groups8 * 2, 65536]` bytes in TMA burst layout.
    /// Written by the `FC1` epilogue and copied into `FC2` `SMEM_B` by [`Self::b_down_tma`].
    q_inter_vals: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate scales `[num_groups8 * 2, 8192]` bytes in TMA layout.
    /// Written by the `FC1` epilogue and copied into `FC2` `SMEM_SFB` by [`Self::sfb_down_tma`].
    q_inter_scales: DeviceBuffer<u8>,
    /// Typed 2D TMA descriptor for `FC2` `B` tiles encoding [`ActTileSmem`] (`RowMajor<256, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_cute`.
    b_down_tma: TensorMap<u8, ActTileSmem>,
    /// Typed 2D TMA descriptor for `FC2` `SFB` scales encoding [`ScaleBTmaSmem`] (`RowMajor<16, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_cute`.
    sfb_down_tma: TensorMap<u16, ScaleBTmaSmem>,
    /// Typed 2D TMA descriptor for `FC2` `B` tiles encoding [`ActTile128Smem`] (`RowMajor<128, 256>`) for `B >= 2048`.
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_128_cute`.
    b_down_128_tma: TensorMap<u8, ActTile128Smem>,
    /// Typed 2D TMA descriptor for `FC2` `SFB` scales encoding [`ScaleTmaSmem`] (`RowMajor<8, 256>`) for `B >= 2048`.
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_128_cute`.
    sfb_down_128_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Typed 2D TMA descriptor for `grouped_out` encoding [`GroupedOutTileSmem`] (`B64`-swizzled `RowMajor<128, 32>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_128_cute` for S2G TMA stores.
    grouped_out_tma: TensorMap<u16, GroupedOutTileSmem>,
    /// Packed `FC1` expert weights `W_up` (`[NUM_EXPERTS * INTERMEDIATE, HIDDEN / 2]` bytes).
    /// Backed by [`Self::w_up_tma`] for 2D TMA loads.
    w_up: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_up`.
    /// Backed by [`Self::sfa_up_tma`] for 2D TMA loads.
    w_up_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization multiplier for `W_up` (`[NUM_EXPERTS]`).
    /// Applied in the `FC1` epilogue before `ReLU²`.
    w_up_global: DeviceBuffer<f32>,
    /// Typed 2D TMA descriptor for `W_up` encoding [`WeightTileSmem`] (`RowMajor<128, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_cute`.
    w_up_tma: TensorMap<u8, WeightTileSmem>,
    /// Typed 2D TMA descriptor for `W_up` scales encoding [`ScaleTmaSmem`] (`RowMajor<8, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc1_relu2_tcgen05_cute`.
    sfa_up_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Packed `FC2` expert weights `W_down` (`[NUM_EXPERTS * HIDDEN, INTERMEDIATE / 2]` bytes).
    /// Backed by [`Self::w_down_tma`] for 2D TMA loads.
    w_down: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_down`.
    /// Backed by [`Self::sfa_down_tma`] for 2D TMA loads.
    w_down_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization multiplier for `W_down` (`[NUM_EXPERTS]`).
    /// Applied in the `FC2` epilogue before converting to `BFloat16`.
    w_down_global: DeviceBuffer<f32>,
    /// Typed 2D TMA descriptor for `W_down` encoding [`WeightTileSmem`] (`RowMajor<128, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_cute`.
    w_down_tma: TensorMap<u8, WeightTileSmem>,
    /// Typed 2D TMA descriptor for `W_down` scales encoding [`ScaleTmaSmem`] (`RowMajor<8, 256>`).
    /// Passed as a `#[grid_constant]` kernel parameter to `moe_fc2_tcgen05_cute`.
    sfa_down_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Contiguous per-group `FC2` `BFloat16` output `[num_groups8 * 256, HIDDEN]` for `B >= 2048`.
    /// Written with 128-bit (`u128`) vector stores in `moe_fc2_tcgen05_cute` and combined in `moe_finalize_cute`.
    grouped_out: DeviceBuffer<BFloat16>,
    /// Final MoE layer output `[num_tokens, HIDDEN]` in `BFloat16`.
    /// Verified against [`expected_output`] after execution.
    output: DeviceBuffer<BFloat16>,
    /// Number of input tokens `B` in this workload instance.
    /// Selects between the low-batch (`B <= 2`) and `tcgen05` (`B >= 4`) schedules.
    num_tokens: usize,
    /// Number of bucketed expert groups launched along `blockIdx.y` in `FC1` and `FC2`.
    /// Each group covers up to `tile_n` (`16, 32, or 64`) tokens routed to one expert.
    num_groups8: usize,
}

/// Compute the byte size of a canonical `(32, 4, 4)` scale buffer for a `[rows, k]` matrix
/// with 16-element NVFP4 scale blocks (`512` bytes per `128 × 4` scale atom).
fn scale_buf_bytes(rows: usize, k: usize) -> usize {
    let rest_m = rows.div_ceil(CANONICAL_SCALE_M);
    let rest_k = (k / 16).div_ceil(CANONICAL_SCALE_K_GROUPS);
    rest_m * rest_k * CANONICAL_SCALE_ATOM_BYTES
}

/// Allocate and initialize all device buffers and host-encoded [`TensorMap`] descriptors
/// for a batch of `num_tokens` tokens.
fn allocate_buffers(
    stream: &Arc<CudaStream>,
    num_tokens: usize,
) -> Result<CuteMoeBuffers, Box<dyn std::error::Error>> {
    let routed_rows = num_tokens * TOP_K;
    let hidden_host: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|token| std::iter::repeat_n(BFloat16::from_f32(hidden_value(token)), HIDDEN))
        .collect();
    let expert_ids_host: Vec<i32> = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| expert_id(token, slot) as i32))
        .collect();
    let routing_host: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|token| {
            (0..TOP_K).map(move |slot| BFloat16::from_f32(routing_weight(token, slot)))
        })
        .collect();

    let w_up_host = vec![0x22u8; NUM_EXPERTS * INTERMEDIATE * (HIDDEN / 2)];
    let w_up_scales_host = vec![0x38u8; scale_buf_bytes(NUM_EXPERTS * INTERMEDIATE, HIDDEN)];
    let mut w_up_global_host = vec![1.0f32 / HIDDEN as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_up_global_host[expert] = UP_FACTORS[expert] / HIDDEN as f32;
    }

    let w_down_host = vec![0x22u8; NUM_EXPERTS * HIDDEN * (INTERMEDIATE / 2)];
    let w_down_scales_host = vec![0x38u8; scale_buf_bytes(NUM_EXPERTS * HIDDEN, INTERMEDIATE)];
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_down_global_host[expert] = DOWN_FACTORS[expert] / INTERMEDIATE as f32;
    }

    let tile_n = if num_tokens <= 64 {
        16
    } else if num_tokens <= 128 {
        32
    } else if num_tokens <= 256 {
        64
    } else if num_tokens == 1024 {
        256
    } else {
        128
    };
    let mut group_experts_host = Vec::new();
    let mut group_tokens_host = Vec::new();
    let mut group_routes_host = Vec::new();
    let mut group_limits_host = Vec::new();
    let mut tok_to_perm_host = vec![0i32; routed_rows];
    for expert in 0..NUM_EXPERTS {
        let mut matches = Vec::new();
        for (r, &exp_id) in expert_ids_host.iter().enumerate().take(routed_rows) {
            if exp_id as usize == expert {
                matches.push(((r / TOP_K) as i32, r as i32));
            }
        }
        for chunk in matches.chunks(tile_n) {
            let g_idx = group_experts_host.len();
            group_experts_host.push(expert as i32);
            group_limits_host.push(chunk.len() as i32);
            let g_shift = if num_tokens >= 2048 { 7 } else { 8 };
            for (r_in_g, &(_tok, route)) in chunk.iter().enumerate() {
                tok_to_perm_host[route as usize] = ((g_idx << g_shift) | r_in_g) as i32;
            }
            for i in 0..256 {
                let (tok, route) = chunk[i.min(chunk.len() - 1)];
                group_tokens_host.push(tok);
                group_routes_host.push(route);
            }
        }
    }
    let num_groups8 = group_experts_host.len();
    let b_tma_bytes = num_groups8 * 2 * 65536;
    let sfb_tma_bytes = num_groups8 * 2 * 8192;

    let w_up = DeviceBuffer::from_host(stream, &w_up_host)?;
    let w_up_scales = DeviceBuffer::from_host(stream, &w_up_scales_host)?;
    let w_down = DeviceBuffer::from_host(stream, &w_down_host)?;
    let w_down_scales = DeviceBuffer::from_host(stream, &w_down_scales_host)?;
    let b_up_vals = DeviceBuffer::from_host(stream, &vec![0u8; b_tma_bytes])?;
    let sfb_up_scales = DeviceBuffer::from_host(stream, &vec![0u8; sfb_tma_bytes])?;
    let q_inter_vals = DeviceBuffer::from_host(
        stream,
        &vec![0u8; b_tma_bytes.max(routed_rows * (INTERMEDIATE / 2))],
    )?;
    let q_inter_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; sfb_tma_bytes.max(scale_buf_bytes(routed_rows, INTERMEDIATE))],
    )?;

    let w_up_tma = create_tensor_map_tiled_with_options::<u8, WeightTileSmem>(
        w_up.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS * INTERMEDIATE) as u64,
        (HIDDEN / 2) as u64,
        (HIDDEN / 2) as u64,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_up_tma = create_tensor_map_tiled_with_options::<u8, ActTileSmem>(
        b_up_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 256) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfa_up_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        w_up_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS * (INTERMEDIATE / 128) * (HIDDEN / 64)) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_up_tma = create_tensor_map_tiled_with_options::<u16, ScaleBTmaSmem>(
        sfb_up_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 16) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let w_down_tma = create_tensor_map_tiled_with_options::<u8, WeightTileSmem>(
        w_down.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS * HIDDEN) as u64,
        (INTERMEDIATE / 2) as u64,
        (INTERMEDIATE / 2) as u64,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_down_tma = create_tensor_map_tiled_with_options::<u8, ActTileSmem>(
        q_inter_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 256) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfa_down_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        w_down_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS * (HIDDEN / 128) * (INTERMEDIATE / 64)) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_down_tma = create_tensor_map_tiled_with_options::<u16, ScaleBTmaSmem>(
        q_inter_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 16) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_up_128_tma = create_tensor_map_tiled_with_options::<u8, ActTile128Smem>(
        b_up_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 128) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_up_128_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        sfb_up_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_down_128_tma = create_tensor_map_tiled_with_options::<u8, ActTile128Smem>(
        q_inter_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 128) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_down_128_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        q_inter_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 2 * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let grouped_out = DeviceBuffer::from_host(
        stream,
        &vec![BFloat16::from_f32(0.0); num_groups8 * 256 * HIDDEN],
    )?;
    let grouped_out_tma = create_tensor_map_tiled_with_options::<u16, GroupedOutTileSmem>(
        grouped_out.cu_deviceptr() as *mut core::ffi::c_void,
        (num_groups8 * 256) as u64,
        HIDDEN as u64,
        HIDDEN as u64,
        TmaEncodeOptions::L2_128B,
    )?;

    Ok(CuteMoeBuffers {
        hidden: DeviceBuffer::from_host(stream, &hidden_host)?,
        expert_ids: DeviceBuffer::from_host(stream, &expert_ids_host)?,
        routing_weights: DeviceBuffer::from_host(stream, &routing_host)?,
        group_experts: DeviceBuffer::from_host(stream, &group_experts_host)?,
        group_tokens: DeviceBuffer::from_host(stream, &group_tokens_host)?,
        group_routes: DeviceBuffer::from_host(stream, &group_routes_host)?,
        tok_to_perm: DeviceBuffer::from_host(stream, &tok_to_perm_host)?,
        group_limits: DeviceBuffer::from_host(stream, &group_limits_host)?,
        q_hidden_vals: DeviceBuffer::from_host(stream, &vec![0u8; num_tokens * (HIDDEN / 2)])?,
        q_hidden_scales: DeviceBuffer::from_host(
            stream,
            &vec![0u8; scale_buf_bytes(num_tokens, HIDDEN)],
        )?,
        b_up_vals,
        sfb_up_scales,
        b_up_tma,
        sfb_up_tma,
        b_up_128_tma,
        sfb_up_128_tma,
        q_inter_vals,
        q_inter_scales,
        b_down_tma,
        sfb_down_tma,
        b_down_128_tma,
        sfb_down_128_tma,
        grouped_out_tma,
        w_up,
        w_up_scales,
        w_up_global: DeviceBuffer::from_host(stream, &w_up_global_host)?,
        w_up_tma,
        sfa_up_tma,
        w_down,
        w_down_scales,
        w_down_global: DeviceBuffer::from_host(stream, &w_down_global_host)?,
        w_down_tma,
        sfa_down_tma,
        grouped_out,
        output: DeviceBuffer::from_host(
            stream,
            &vec![BFloat16::from_f32(0.0); num_tokens * HIDDEN],
        )?,
        num_tokens,
        num_groups8,
    })
}

/// Launch the complete `BF16 -> NVFP4 -> FC1 + ReLU² -> FC2 + Finalize -> BF16` MoE pipeline
/// on `stream`. Used both for eager numerical verification and for CUDA Graph stream capture.
fn launch_pipeline(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    bufs: &mut CuteMoeBuffers,
) -> Result<(), Box<dyn std::error::Error>> {
    let routed_rows = bufs.num_tokens * TOP_K;

    let q1_groups = bufs.num_tokens * (HIDDEN / 16);
    let q1_cfg = LaunchConfig {
        grid_dim: (q1_groups.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let zero_out = usize::from(bufs.num_tokens > 1 && bufs.num_groups8 <= 32);
    let _ = bufs.group_tokens.len();
    unsafe {
        module.quantize_activations_nvfp4(
            stream,
            q1_cfg,
            &bufs.hidden,
            &bufs.tok_to_perm,
            &mut bufs.q_hidden_vals,
            &mut bufs.q_hidden_scales,
            &mut bufs.b_up_vals,
            &mut bufs.sfb_up_scales,
            &mut bufs.output,
            bufs.num_tokens,
            HIDDEN,
            zero_out,
        )?;
    }

    if bufs.num_tokens <= 1 {
        let fc1_cfg = LaunchConfig {
            grid_dim: ((INTERMEDIATE / 16) as u32, routed_rows as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.moe_fc1_relu2_cute(
                stream,
                fc1_cfg,
                &bufs.w_up,
                &bufs.w_up_scales,
                &bufs.w_up_global,
                &bufs.q_hidden_vals,
                &bufs.q_hidden_scales,
                &bufs.expert_ids,
                &mut bufs.q_inter_vals,
                &mut bufs.q_inter_scales,
                bufs.num_tokens,
                HIDDEN,
                INTERMEDIATE,
                NUM_EXPERTS,
                TOP_K,
            )?;
        }
        let fc2_cfg = LaunchConfig {
            grid_dim: ((HIDDEN / 4) as u32, bufs.num_tokens as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.moe_fc2_cute(
                stream,
                fc2_cfg,
                &bufs.w_down,
                &bufs.w_down_scales,
                &bufs.w_down_global,
                &bufs.q_inter_vals,
                &bufs.q_inter_scales,
                &bufs.expert_ids,
                &bufs.routing_weights,
                &mut bufs.output,
                routed_rows,
                HIDDEN,
                INTERMEDIATE,
                NUM_EXPERTS,
            )?;
        }
    } else {
        let groups_per_cta = if bufs.num_groups8 > 16 {
            bufs.num_groups8 / 16
        } else {
            1
        };
        let total_ctas = ((INTERMEDIATE / 128) * (bufs.num_groups8 / groups_per_cta)) as u32;
        if groups_per_cta <= 2 {
            let fc1_cfg = LaunchConfig1D::new(total_ctas, TCGEN05_THREADS, DYNAMIC_SMEM_BYTES);
            let fc1_launch = module.prepare_moe_fc1_relu2_tcgen05_cute(fc1_cfg)?;
            unsafe {
                module.moe_fc1_relu2_tcgen05_cute(
                    stream,
                    &fc1_launch,
                    bufs.w_up_tma,
                    bufs.b_up_tma,
                    bufs.sfa_up_tma,
                    bufs.sfb_up_tma,
                    &bufs.w_up_global,
                    &bufs.group_experts,
                    &bufs.group_limits,
                    &mut bufs.q_inter_vals,
                    &mut bufs.q_inter_scales,
                    INTERMEDIATE,
                    groups_per_cta,
                )?;
            }
            let fc2_cfg = LaunchConfig1D::new(total_ctas, TCGEN05_THREADS, DYNAMIC_SMEM_BYTES);
            let fc2_launch = module.prepare_moe_fc2_tcgen05_cute(fc2_cfg)?;
            unsafe {
                module.moe_fc2_tcgen05_cute(
                    stream,
                    &fc2_launch,
                    bufs.w_down_tma,
                    bufs.b_down_tma,
                    bufs.sfa_down_tma,
                    bufs.sfb_down_tma,
                    &bufs.w_down_global,
                    &bufs.group_experts,
                    &bufs.group_routes,
                    &bufs.group_limits,
                    &bufs.routing_weights,
                    &mut bufs.output,
                    HIDDEN,
                    groups_per_cta,
                )?;
            }
        } else {
            let fc1_cfg = LaunchConfig1D::new(total_ctas, TCGEN05_128_THREADS, DYNAMIC_SMEM_BYTES);
            let fc1_launch = module.prepare_moe_fc1_relu2_tcgen05_128_cute(fc1_cfg)?;
            unsafe {
                module.moe_fc1_relu2_tcgen05_128_cute(
                    stream,
                    &fc1_launch,
                    bufs.w_up_tma,
                    bufs.b_up_128_tma,
                    bufs.sfa_up_tma,
                    bufs.sfb_up_128_tma,
                    &bufs.w_up_global,
                    &bufs.group_experts,
                    &mut bufs.q_inter_vals,
                    &mut bufs.q_inter_scales,
                    INTERMEDIATE,
                    groups_per_cta,
                )?;
            }
            let fc2_cfg = LaunchConfig1D::new(total_ctas, TCGEN05_128_THREADS, 230_400);
            let fc2_launch = module.prepare_moe_fc2_tcgen05_128_cute(fc2_cfg)?;
            unsafe {
                module.moe_fc2_tcgen05_128_cute(
                    stream,
                    &fc2_launch,
                    bufs.w_down_tma,
                    bufs.b_down_128_tma,
                    bufs.sfa_down_tma,
                    bufs.sfb_down_128_tma,
                    bufs.grouped_out_tma,
                    &bufs.w_down_global,
                    &bufs.group_experts,
                    &mut bufs.grouped_out,
                    HIDDEN,
                    groups_per_cta,
                )?;
            }
            let fin_threads = (bufs.num_tokens * 32) as u32;
            let fin_cfg = LaunchConfig {
                grid_dim: (fin_threads.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                module.moe_finalize_cute(
                    stream,
                    fin_cfg,
                    &bufs.grouped_out,
                    &bufs.tok_to_perm,
                    &bufs.routing_weights,
                    &mut bufs.output,
                    bufs.num_tokens,
                    HIDDEN,
                )?;
            }
        }
    }
    Ok(())
}

/// Copy `bufs.output` back to the host and verify every element against [`expected_output`].
/// Fails if any value is non-finite or if the normalized maximum error exceeds `0.01`.
fn check_output(
    stream: &Arc<CudaStream>,
    bufs: &CuteMoeBuffers,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.synchronize()?;
    let output_vals = bufs.output.to_host_vec(stream)?;
    let mut max_abs_error = 0.0f32;
    let mut max_reference = 1.0f32;
    for (index, &val) in output_vals.iter().enumerate() {
        let token = index / HIDDEN;
        let actual = val.to_f32();
        if !actual.is_finite() {
            return Err(format!("non-finite output at index {index}: {actual}").into());
        }
        let reference = expected_output(token).to_f32();
        max_abs_error = max_abs_error.max((actual - reference).abs());
        max_reference = max_reference.max(reference.abs());
    }
    let normalized_error = max_abs_error / max_reference;
    if normalized_error > 0.01 {
        return Err(format!(
            "output check failed: normalized max error {normalized_error:.6}, max abs error {max_abs_error:.6}"
        )
        .into());
    }
    println!(
        "Output check passed: normalized max error={normalized_error:.6}, max_abs_error={max_abs_error:.6}"
    );
    Ok(())
}

/// Verify numerical correctness and measure `1,000×` CUDA Graph replay latency across all
/// 9 batch sizes (`B = 1..256`) on NVIDIA Blackwell (`sm_100a`).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    println!("=== Nemotron-3 NVFP4 ReLU² MoE (cute-rs) ===");
    println!("Target GPU: sm_{major}{minor}");
    println!("H={HIDDEN}, I={INTERMEDIATE}, experts={NUM_EXPERTS}, top-k={TOP_K}");

    let stream = ctx.new_stream()?;
    if std::env::var("MOE_ONLY_192E").is_ok() {
        return moe_192e::run_benchmark_192e(&ctx, &stream);
    }
    let module = unsafe { kernels::load(&ctx) }?;

    let mut results = Vec::with_capacity(BATCH_SIZES.len());
    for num_tokens in BATCH_SIZES {
        let mut bufs = allocate_buffers(&stream, num_tokens)?;
        launch_pipeline(&module, &stream, &mut bufs)?;
        print!("Batch {num_tokens:>3}: ");
        check_output(&stream, &bufs)?;

        let cu_graph = unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .result()?;
            launch_pipeline(&module, &stream, &mut bufs)?;
            let mut graph = MaybeUninit::<sys::CUgraph>::uninit();
            sys::cuStreamEndCapture(stream.cu_stream(), graph.as_mut_ptr()).result()?;
            graph.assume_init()
        };
        let cu_graph_exec = unsafe {
            let mut exec = MaybeUninit::<sys::CUgraphExec>::uninit();
            sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), cu_graph, 0).result()?;
            let exec = exec.assume_init();
            sys::cuGraphUpload(exec, stream.cu_stream()).result()?;
            exec
        };
        stream.synchronize()?;

        let start = ctx.new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let stop = ctx.new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let mut samples_us = Vec::with_capacity(TIMING_SAMPLES);
        for _ in 0..TIMING_SAMPLES {
            for _ in 0..TIMING_WARMUPS {
                unsafe {
                    sys::cuGraphLaunch(cu_graph_exec, stream.cu_stream()).result()?;
                }
            }
            start.record(&stream)?;
            for _ in 0..TIMING_REPLAYS {
                unsafe {
                    sys::cuGraphLaunch(cu_graph_exec, stream.cu_stream()).result()?;
                }
            }
            stop.record(&stream)?;
            stop.synchronize()?;
            let elapsed_ms = start.elapsed_ms(&stop)?;
            samples_us.push((elapsed_ms as f64) * 1000.0 / TIMING_REPLAYS as f64);
        }
        unsafe {
            let _ = sys::cuGraphExecDestroy(cu_graph_exec);
            let _ = sys::cuGraphDestroy(cu_graph);
        }
        let mean_us = samples_us.iter().sum::<f64>() / TIMING_SAMPLES as f64;
        results.push((num_tokens, mean_us));
    }

    /// SOTA `FlashInfer CuTeDSL` CUDA Graph latency (`µs`) across `BATCH_SIZES` (`B = 1..16,384`) on B200 (`sm_100a`).
    /// Measured with identical `1,000×` CUDA Graph replay (`K1 + Route + FC1 + FC2 + Finalize`).
    const SOTA_FLASHINFER_US: [f64; 15] = [
        18.430, 18.420, 18.420, 18.440, 20.390, 19.800, 20.490, 22.520, 23.240, 24.570, 28.750,
        38.890, 51.250, 92.610, 179.350,
    ];
    /// Reference Native `cuda-oxide` SIMT (`nemotron3_moe`) CUDA Graph latency (`µs`) across `BATCH_SIZES` on B200.
    /// Included so a single `cargo oxide run` prints the complete 3-way comparison table.
    const NATIVE_SIMT_REF_US: [f64; 15] = [
        12.282, 14.329, 14.345, 14.345, 14.344, 14.345, 14.509, 14.345, 16.387, 18.583, 24.590,
        32.183, 49.144, 88.325, 174.825,
    ];

    println!(
        "\nBatch size | FlashInfer CuTeDSL (us) |  Native SIMT [ref] (us) |    cute-rs [live] (us) | cute-rs vs. SOTA"
    );
    println!(
        "-----------+-------------------------+-------------------------+------------------------+-----------------"
    );
    for (idx, &(b, cute_us)) in results.iter().enumerate() {
        let sota_us = SOTA_FLASHINFER_US[idx];
        let native_us = NATIVE_SIMT_REF_US[idx];
        let speedup = sota_us / cute_us;
        println!(
            "{b:>10} | {sota_us:>23.3} | {native_us:>23.3} | {cute_us:>22.3} | {speedup:>10.2}x SOTA"
        );
    }
    moe_192e::run_benchmark_192e(&ctx, &stream)?;
    Ok(())
}

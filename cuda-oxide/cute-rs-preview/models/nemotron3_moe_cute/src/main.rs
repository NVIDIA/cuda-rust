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
//!   • `TiledMma128::load_stationary_weights_2d` stages `W` (`64 KB`) + `SFA` (`8 KB`) once
//!   • `b_pipeline.producer_acquire` ──► `producer_expect_tx` (`36,864` bytes)
//!   • `copy_tma_2d(b_tma)`   ──► `SMEM_B[0..3]`   (`32 KB` raw 4-bit tokens per stage)
//!   • `copy_tma_2d(sfb_tma)` ──► `SMEM_SFB[0..3]` (`4 KB` canonical 8-bit scales per stage)
//!   │
//!   ▼  `Act128Pipeline` (`PipelineTmaAsync<4, 1, 36_864>`) + 2-Stage `TMEM[0..1]`
//! Warp 4 (MMA Consumer/Producer) & Warps 0..3 (Epilogue Consumer)
//!   • `TiledMma128::copy_b_scales_and_gemm_128_burst` (`tcgen05.cp` + `tcgen05.mma...mxf4nvf4`)
//!   • `StagedEpilogue::unload_relu2_and_quantize_128` (`tcgen05.ld.32x32b.x64` ──► `ReLU²` ──► NVFP4)
//!   • `StagedEpilogue::unload_and_tma_store_bf16_128` (`tcgen05.ld.32x32b.x64` ──► 2D TMA Store)
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

use core::marker::PhantomData;
use ctm_rs::mbarrier::Barrier;
use ctm_rs::memory::{DisjointSlice, DynamicSharedArray, SharedArray};
use ctm_rs::{cuda_module, kernel, launch_bounds, launch_contract, thread};
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cute_rs::arch::{NamedBarrier, cp_async_bulk_wait_group_read};
use cute_rs::block_scaled::{
    CANONICAL_SCALE_ATOM_BYTES, CANONICAL_SCALE_K_GROUPS, CANONICAL_SCALE_M, Sm1xxBlockScaleKMajor,
};
use cute_rs::copy::tma::{TmaEncodeOptions, create_tensor_map_tiled_with_options};
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::{
    BFloat16, BlockScaledTensor, ComposedLayout, Consumer, Float4E2M1FN, Float8E4M3FN, KMajor, Mkl,
    Nkl, OperandA, OperandB, PipelineState, PipelineTmaAsync, Producer, RowMajor,
    Sm100BlockScaledTiledMma, Sm100ScaleTile, Sm100SharedTile, Sm100TmemStagedEpilogue, SmemTile,
    Swizzle, TensorMap, copy_tma_2d, size_in_bytes,
};
use std::mem::MaybeUninit;
use std::sync::Arc;

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

/// Shared `256 × 128` FP32 epilogue tile (`131,072` bytes, overlaying `SMEM_A + SMEM_B`)
/// used by [`Sm100TmemStagedEpilogue`] so all 256 rows unload in 1 single pass!
type EpilogueSmem = RowMajor<256, 128>;

/// Role-checked shared-memory A operand tile (`OperandA` role + `WeightTileSmem` layout).
/// Prevents swapping the weight tile `A` and activation tile `B` at compile time.
type SharedWeightTile = Sm100SharedTile<WeightTileSmem, OperandA>;

/// Role-checked shared-memory B operand tile (`OperandB` role + `ActTileSmem` layout).
/// Paired with [`SharedWeightTile`] inside [`Sm100BlockScaledTiledMma::copy_scales_and_gemm`].
type SharedActTile = Sm100SharedTile<ActTileSmem, OperandB>;

/// Role-checked shared-memory B operand tile for `128`-token groups (`OperandB` role + `ActTile128Smem` layout).
/// Paired with [`SharedWeightTile`] inside [`Sm100BlockScaledTiledMma::copy_b_scales_and_gemm_128_burst`].
type SharedAct128Tile = Sm100SharedTile<ActTile128Smem, OperandB>;

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

/// Typed TMEM epilogue staging `[256, 128]` FP32 accumulators through `SMEM_A + SMEM_B` (`136 KB`).
/// Unloads TMEM (`tcgen05.ld`), applies `ReLU²`, and exposes 16-element scale groups for NVFP4 quantization.
type StagedEpilogue = Sm100TmemStagedEpilogue<EpilogueSmem, 128>;

/// Bytes copied by TMA per `K = 512` burst (`32,768` A + `65,536` B + `4,096` SFA + `8,192` SFB = `110,592` bytes).
/// Statically verified by `dialect-cute` against the 4 `copy_tma_2d` calls in the producer warp.
const TMA_BURST_TX_BYTES: u32 = size_in_bytes::<u8, WeightTileSmem>()
    + size_in_bytes::<u8, ActTileSmem>()
    + size_in_bytes::<u16, ScaleTmaSmem>()
    + size_in_bytes::<u16, ScaleBTmaSmem>();

/// Bytes copied by TMA per `128`-row `K = 512` burst (`32,768` + `4,096` = `36,864` bytes).
/// Statically verified by `dialect-cute` for both `WeightPipeline` (`A + SFA`) and `Act128Pipeline` (`B + SFB`).
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

/// Encode one signed `f32` value into its nearest 4-bit `Float4E2M1FN` nibble (`bits 3..0`).
/// Maps magnitude thresholds `{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}` to `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`
/// and sets bit `3` (`0x08`) for negative values.
#[must_use]
#[inline(always)]
pub fn f32_to_e2m1(value: f32) -> u8 {
    let sign = if value < 0.0 { 0x08u8 } else { 0x00u8 };
    let abs = if value < 0.0 { -value } else { value };
    let mag = (abs >= 0.25) as u8
        + (abs >= 0.75) as u8
        + (abs >= 1.25) as u8
        + (abs >= 1.75) as u8
        + (abs >= 2.50) as u8
        + (abs >= 3.50) as u8
        + (abs >= 5.00) as u8;
    sign | mag
}

/// Device kernels for the `cute-rs` Nemotron-3 NVFP4 `ReLU²` MoE pipeline (`K1`..`K4`).
/// Compiled through `dialect-cute` and `cutlass-mlir` into a Blackwell (`sm_100a`) cubin.
#[allow(clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;

    /// Decode one `Float8E4M3FN` scale byte into `f32` in 2 branchless IEEE 754 instructions!
    /// Shifts the 7-bit `E4M3` exponent+mantissa into `f32` bits `26..20` (`subnormal/normal`) and multiplies by `2^120`.
    #[inline(always)]
    fn dev_e4m3_to_f32(byte: u8) -> f32 {
        f32::from_bits(((byte & 0x7f) as u32) << 20) * f32::from_bits(247 << 23)
    }

    /// Encode one non-negative `f32` scale factor (`max_abs / 6.0`) into `Float8E4M3FN` in 3 branchless instructions!
    /// Multiplies by `2^-120` (`f32::from_bits(7 << 23)`) to align the `E4M3` exponent bias (`7`) and subnormals with `f32` bits `26..20`.
    #[inline(always)]
    fn dev_f32_to_e4m3(value: f32) -> u8 {
        let clamped = if value > 448.0 {
            448.0
        } else if value > 0.0 {
            value
        } else {
            0.0
        };
        let scaled = clamped * f32::from_bits(7 << 23);
        let code = ((scaled.to_bits() + 0x80000) >> 20) as u8;
        if code > 0x7e { 0x7e } else { code }
    }

    /// Compute `2^-126 / e4m3_to_f32(scale_code)` in 3 branchless instructions (`0` `div.rn.f32`!).
    /// Uses 128-bit packed `bf16` reciprocal LUTs for the 3-bit `E4M3` mantissa (`0..7`) and `2^(7 - exp - 63)`.
    #[inline(always)]
    fn dev_e4m3_inv_e2m1(scale_code: u8) -> f32 {
        let exp = (scale_code >> 3) as u32;
        let mant = (scale_code & 7) as u32;
        if exp == 0 {
            let sub =
                ((0x0392_03AB_03CD_0400_042B_0480_0500_0000u128 >> (mant * 16)) & 0xFFFF) as u32;
            f32::from_bits(sub << 16)
        } else {
            let lut =
                ((0x1F89_1F92_1F9E_1FAB_1FBA_1FCD_1FE4_2000u128 >> (mant * 16)) & 0xFFFF) as u32;
            f32::from_bits(lut << 16) * f32::from_bits((71 - exp) << 23)
        }
    }

    /// Encode one non-negative `f32` value already pre-multiplied by `inv_scale * 2^-126` (`f32::from_bits(1 << 23)`)
    /// into its 4-bit `Float4E2M1FN` nibble (`0..7`) in 2 branchless bitwise instructions (`0` float comparisons!).
    #[inline(always)]
    fn fast_pos_e2m1(scaled_v: f32) -> u64 {
        let bits = scaled_v.to_bits().min(0x01C0_0000);
        ((bits + 0x0020_0000) >> 22) as u64
    }

    /// Quantize `[rows, k]` BF16 values into packed `Float4E2M1FN` bytes and canonical `(32, 4)` `Float8E4M3FN` scales
    /// using `2 × u128` vector loads (`StridedTensor<u128>`), writing TMA-ready `[256, 256]` burst tiles
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
        let groups_per_row = k / 16;
        let total_groups = rows * groups_per_row;
        if tid >= total_groups {
            return;
        }
        let row = tid / groups_per_row;
        let grp = tid % groups_per_row;
        let packed_k = k / 2;
        let val_base = row * packed_k + grp * 8;

        let u128_base = (row * k + grp * 16) / 8;
        if zero_out != 0 {
            let out_u128 = out_bf16.as_mut_ptr().cast::<u128>();
            unsafe {
                *out_u128.add(u128_base) = 0u128;
                *out_u128.add(u128_base + 1) = 0u128;
            }
        }

        let (h0, h1) = unsafe {
            let in_u128 = StridedTensor::<u128>::from_raw_parts(
                input.as_ptr().cast::<u128>(),
                input.len() / 8,
                1,
            );
            (
                in_u128.get_unchecked(u128_base),
                in_u128.get_unchecked(u128_base + 1),
            )
        };
        let mut max_abs = 0.0f32;
        let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
        /// Unpack one `bf16` element from `h0` or `h1` into `f32` and update `max_abs`.
        macro_rules! unpack_bf16 {
            ($vi:ident, $src:ident, $idx:expr) => {
                let b = (($src >> ($idx * 16)) & 0xFFFF) as u32;
                $vi = f32::from_bits(b << 16);
                let a = f32::from_bits((b << 16) & 0x7FFF_FFFF);
                if a > max_abs {
                    max_abs = a;
                }
            };
        }
        unpack_bf16!(v0, h0, 0);
        unpack_bf16!(v1, h0, 1);
        unpack_bf16!(v2, h0, 2);
        unpack_bf16!(v3, h0, 3);
        unpack_bf16!(v4, h0, 4);
        unpack_bf16!(v5, h0, 5);
        unpack_bf16!(v6, h0, 6);
        unpack_bf16!(v7, h0, 7);
        unpack_bf16!(v8, h1, 0);
        unpack_bf16!(v9, h1, 1);
        unpack_bf16!(v10, h1, 2);
        unpack_bf16!(v11, h1, 3);
        unpack_bf16!(v12, h1, 4);
        unpack_bf16!(v13, h1, 5);
        unpack_bf16!(v14, h1, 6);
        unpack_bf16!(v15, h1, 7);

        let scale_code = dev_f32_to_e4m3(max_abs * (1.0 / 6.0));
        let inv_scale_e2m1 = dev_e4m3_inv_e2m1(scale_code);
        /// Quantize one signed `f32` into its 4-bit `E2M1` nibble at bit shift `$sh`.
        macro_rules! q_nib {
            ($v:ident, $sh:expr) => {{
                let sign = (($v.to_bits() >> 28) & 8) as u64;
                let abs_v = f32::from_bits($v.to_bits() & 0x7FFF_FFFF);
                (sign | fast_pos_e2m1(abs_v * inv_scale_e2m1)) << $sh
            }};
        }
        let packed_u64 = q_nib!(v0, 0)
            | q_nib!(v1, 4)
            | q_nib!(v2, 8)
            | q_nib!(v3, 12)
            | q_nib!(v4, 16)
            | q_nib!(v5, 20)
            | q_nib!(v6, 24)
            | q_nib!(v7, 28)
            | q_nib!(v8, 32)
            | q_nib!(v9, 36)
            | q_nib!(v10, 40)
            | q_nib!(v11, 44)
            | q_nib!(v12, 48)
            | q_nib!(v13, 52)
            | q_nib!(v14, 56)
            | q_nib!(v15, 60);

        if rows == 1 {
            let scale_layout = Sm1xxBlockScaleKMajor::<16>::new(rows, k);
            let scale_idx = scale_layout.crd2idx(0, row, grp);
            let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut out_scales);
            unsafe {
                scales_out.store_at(scale_idx, scale_code);
                *out_values.as_mut_ptr().cast::<u64>().add(val_base / 8) = packed_u64;
            }
        } else if zero_out != 0 {
            let b_u64_ptr = b_up_vals.as_mut_ptr().cast::<u64>();
            let mut sfb_out = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
            let gp128 = unsafe { *tok_to_perm.as_ptr().cast::<u128>().add(row) };
            unsafe {
                TiledMma::store_routed_token_group_256(
                    b_u64_ptr,
                    &mut sfb_out,
                    gp128,
                    grp,
                    packed_u64,
                    scale_code,
                );
            }
        } else {
            let b_u64_ptr = b_up_vals.as_mut_ptr().cast::<u64>();
            let mut sfb_out = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
            let gp128 = unsafe { *tok_to_perm.as_ptr().cast::<u128>().add(row) };
            unsafe {
                TiledMma128::store_routed_token_group_128(
                    b_u64_ptr,
                    &mut sfb_out,
                    gp128,
                    grp,
                    packed_u64,
                    scale_code,
                );
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
        let expert = unsafe { *expert_ids.as_ptr().add(route) } as usize;
        let w_row_idx = expert * intermediate_size + out_col;
        let w_row = weights.thread_row(0, w_row_idx);
        let x_row = inputs.thread_row(0, token);

        let tile = lane & 15;
        let (w_frag, x_frag) = unsafe { (w_row.k_tile(tile).load(), x_row.k_tile(tile).load()) };
        let acc = w_frag.dot_accumulate(x_frag, 0.0f32);
        let sum = unsafe { cute_rs::arch::warp_reduction_sum_f32(acc) } * 0.5f32;

        let g_scale = unsafe { *w_up_global.as_ptr().add(expert) };
        let scaled = sum * g_scale;
        let relu = if scaled > 0.0 { scaled } else { 0.0 };
        let activated = relu * relu;

        let smem_act_ptr = (&raw mut SMEM_ACT).cast::<f32>();
        if lane == 0 {
            unsafe {
                *smem_act_ptr.add(warp_in_block) = activated;
            }
        }
        unsafe {
            NamedBarrier::<0, 512>::sync();
        }

        if tid == 0 {
            let mut max_abs = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let v = unsafe { *smem_act_ptr.add(i) };
                if v > max_abs {
                    max_abs = v;
                }
                i += 1;
            }
            let scale_code = dev_f32_to_e4m3(max_abs * (1.0 / 6.0));
            let scale_val = dev_e4m3_to_f32(scale_code);
            let inv_scale = if scale_val > 0.0 {
                1.0 / scale_val
            } else {
                0.0
            };

            let packed_k = intermediate_size / 2;
            let scale_layout =
                Sm1xxBlockScaleKMajor::<16>::new(num_tokens * top_k, intermediate_size);
            let scale_idx = scale_layout.crd2idx(0, route, grp);
            let val_base = route * packed_k + grp * 8;

            let mut vals_out = StridedTensorMut::from_disjoint_slice(&mut q_inter_vals);
            let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut q_inter_scales);
            unsafe {
                scales_out.store_at(scale_idx, scale_code);
            }
            let mut pair = 0usize;
            while pair < 8 {
                let x0 = unsafe { *smem_act_ptr.add(pair * 2) } * inv_scale;
                let x1 = unsafe { *smem_act_ptr.add(pair * 2 + 1) } * inv_scale;
                let packed = f32_to_e2m1(x0) | (f32_to_e2m1(x1) << 4);
                unsafe {
                    vals_out.store_at(val_base + pair, packed);
                }
                pair += 1;
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

        let half = lane >> 4; // 0 for lanes 0..15 (slots 0, 1), 1 for lanes 16..31 (slots 2, 3)
        let tile = lane & 15; // 0..15 (16 K-tiles of K=64)
        let r0 = token * 4 + half * 2;
        let r1 = r0 + 1;
        let exp0 = unsafe { *expert_ids.as_ptr().add(r0) } as usize;
        let exp1 = unsafe { *expert_ids.as_ptr().add(r1) } as usize;
        let s0 = routing_weights[r0].to_f32() * unsafe { *w_down_global.as_ptr().add(exp0) };
        let s1 = routing_weights[r1].to_f32() * unsafe { *w_down_global.as_ptr().add(exp1) };

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

    /// 5th-Gen Tensor Core (`TiledMma`: `nvvm.tcgen05.mma.block_scale` + `nvvm.tcgen05.cp` +
    /// 2-Stage Full-TMA `MainloopPipeline` (`PipelineTmaAsync<2, 1, 73_728>`) + `copy_tma_2d` for `A, B, SFA, SFB`
    /// + `StagedEpilogue`) Grouped `FC1 + ReLU²` + fused NVFP4 requantization (`K3`) for `B >= 4`.
    ///
    /// # Safety
    /// Must be launched with `block_dim = (544, 1, 1)` (`16` compute warps + `1` TMA producer warp),
    /// `shared_mem_bytes = 148480`, and valid host-encoded TMA descriptors.
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
        let tile_m128 = cta_id & 7;
        let group_start = (cta_id >> 3) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = unsafe { *group_experts.as_ptr().add(group_start) } as usize;
        let limit = unsafe { *group_limits.as_ptr().add(group_start) } as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let w_outer = expert * (intermediate_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = unsafe { dyn_smem.add(SMEM_A_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_u32 = unsafe { dyn_smem.add(SMEM_SFA_OFFSET).cast::<u32>() };
        let smem_sfb_u32 = unsafe { dyn_smem.add(SMEM_SFB_OFFSET).cast::<u32>() };
        let pipe_bar_ptr = unsafe { dyn_smem.add(PIPELINE_OFFSET).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            let pipeline = MainloopPipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                TiledMma::init_barriers::<3>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma::alloc_tmem_cg1(slot_ptr);
            }
            pipeline.init(0);

            if warp_id < COMPUTE_WARPS {
                let tmem_addr = *slot_ptr.cast_const();
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma::new_with_lbo(valid_n as u32, 4096);
                let epilogue = StagedEpilogue::from_raw(smem_b_ptr.cast::<f32>());
                let g_scale = *w_up_global.as_ptr().add(expert);
                let vals_u64_ptr = q_inter_vals.as_mut_ptr().cast::<u64>();
                let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut q_inter_scales);

                let mut consumer_state = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let parity = (g_idx & 1) as u32;
                    if warp_id == 0 {
                        let mut burst = 0usize;
                        while burst < 2 {
                            pipeline.consumer_wait(&consumer_state);
                            let slot = consumer_state.index();
                            if tid == 0 {
                                let s_a =
                                    SharedWeightTile::from_raw_bytes(smem_a_ptr.add(slot * 32768));
                                let s_b =
                                    SharedActTile::from_raw_bytes(smem_b_ptr.add(slot * 65536));
                                let s_sfa =
                                    SharedWeightScales::from_raw(smem_sfa_u32.add(slot * 1024));
                                let s_sfb =
                                    SharedActScales::from_raw(smem_sfb_u32.add(slot * 2048));
                                let sfa_burst_tmem = sfa_tmem_base + (slot as u32) * 32;
                                let sfb_burst_tmem = sfb_tmem_base + (slot as u32) * 64;
                                if g_idx == 0 {
                                    tiled_mma.copy_scales_and_gemm_burst(
                                        tmem_addr,
                                        sfa_burst_tmem,
                                        sfb_burst_tmem,
                                        &s_a,
                                        &s_b,
                                        &s_sfa,
                                        &s_sfb,
                                        burst == 0,
                                    );
                                } else {
                                    tiled_mma.copy_b_scales_and_gemm_burst(
                                        tmem_addr,
                                        sfa_burst_tmem,
                                        sfb_burst_tmem,
                                        &s_a,
                                        &s_b,
                                        &s_sfb,
                                        burst == 0,
                                    );
                                }
                                if burst == 1 {
                                    TiledMma::commit_barrier(mma_bar_ptr);
                                    TiledMma::wait_barrier(mma_bar_ptr, parity);
                                }
                            }
                            pipeline.consumer_release(&consumer_state);
                            consumer_state.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 512>::sync();

                    epilogue.unload_relu2_and_quantize_256(
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        tile_m128,
                        group,
                        g_scale,
                        vals_u64_ptr,
                        &mut scales_out,
                    );
                    g_idx += 1;
                }

                if warp_id == 0 {
                    TiledMma::dealloc_tmem_cg1(slot_ptr);
                }
            } else if warp_id == PRODUCER_WARP && (tid & 31) == 0 {
                let mut producer_state = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let burst = step & 1;
                    let g_idx = step >> 1;
                    let group = group_start + g_idx;
                    if g_idx > 0 {
                        let wait_parity = ((g_idx - 1) & 1) as u32;
                        TiledMma::wait_barrier(b_free_bar_ptr.add(burst), wait_parity);
                    }
                    let slot = producer_state.index();
                    pipeline.producer_acquire(&producer_state);
                    let full_barrier = pipeline.producer_expect_tx(&producer_state);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem> {
                        base: smem_a_ptr.add(slot * 32768),
                        capacity: 32768,
                        layout: PhantomData,
                    };
                    let mut b_tile = SmemTile::<u8, ActTileSmem> {
                        base: smem_b_ptr.add(slot * 65536),
                        capacity: 65536,
                        layout: PhantomData,
                    };
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem> {
                        base: smem_sfa_u32.add(slot * 1024).cast::<u16>(),
                        capacity: 2048,
                        layout: PhantomData,
                    };
                    let mut sfb_tile = SmemTile::<u16, ScaleBTmaSmem> {
                        base: smem_sfb_u32.add(slot * 2048).cast::<u16>(),
                        capacity: 4096,
                        layout: PhantomData,
                    };
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
    /// Persistent 2-Stage Full-TMA `MainloopPipeline` (`PipelineTmaAsync<2, 1, 110_592>`) + `copy_tma_2d` for `A, B, SFA, SFB`
    /// + `StagedEpilogue::unload_and_atomic_add_bf16x2`) Grouped `FC2 + Finalize` (`K4` fused) for `B >= 2`.
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
        let tile_m128 = cta_id & 7;
        let group_start = (cta_id >> 3) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = unsafe { *group_experts.as_ptr().add(group_start) } as usize;
        let limit = unsafe { *group_limits.as_ptr().add(group_start) } as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let w_outer = expert * (hidden_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = unsafe { dyn_smem.add(SMEM_A_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_u32 = unsafe { dyn_smem.add(SMEM_SFA_OFFSET).cast::<u32>() };
        let smem_sfb_u32 = unsafe { dyn_smem.add(SMEM_SFB_OFFSET).cast::<u32>() };
        let pipe_bar_ptr = unsafe { dyn_smem.add(PIPELINE_OFFSET).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            let pipeline = MainloopPipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                TiledMma::init_barriers::<3>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma::alloc_tmem_cg1(slot_ptr);
            }
            pipeline.init(0);

            if warp_id < COMPUTE_WARPS {
                let tmem_addr = *slot_ptr.cast_const();
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma::new_with_lbo(valid_n as u32, 4096);
                let epilogue = StagedEpilogue::from_raw(smem_b_ptr.cast::<f32>());
                let g_scale = *w_down_global.as_ptr().add(expert);

                let mut consumer_state = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let parity = (g_idx & 1) as u32;
                    if warp_id == 0 {
                        let mut burst = 0usize;
                        while burst < 2 {
                            pipeline.consumer_wait(&consumer_state);
                            let slot = consumer_state.index();
                            if tid == 0 {
                                let s_a =
                                    SharedWeightTile::from_raw_bytes(smem_a_ptr.add(slot * 32768));
                                let s_b =
                                    SharedActTile::from_raw_bytes(smem_b_ptr.add(slot * 65536));
                                let s_sfa =
                                    SharedWeightScales::from_raw(smem_sfa_u32.add(slot * 1024));
                                let s_sfb =
                                    SharedActScales::from_raw(smem_sfb_u32.add(slot * 2048));
                                let sfa_burst_tmem = sfa_tmem_base + (slot as u32) * 32;
                                let sfb_burst_tmem = sfb_tmem_base + (slot as u32) * 64;
                                if g_idx == 0 {
                                    tiled_mma.copy_scales_and_gemm_burst(
                                        tmem_addr,
                                        sfa_burst_tmem,
                                        sfb_burst_tmem,
                                        &s_a,
                                        &s_b,
                                        &s_sfa,
                                        &s_sfb,
                                        burst == 0,
                                    );
                                } else {
                                    tiled_mma.copy_b_scales_and_gemm_burst(
                                        tmem_addr,
                                        sfa_burst_tmem,
                                        sfb_burst_tmem,
                                        &s_a,
                                        &s_b,
                                        &s_sfb,
                                        burst == 0,
                                    );
                                }
                                if burst == 1 {
                                    TiledMma::commit_barrier(mma_bar_ptr);
                                    TiledMma::wait_barrier(mma_bar_ptr, parity);
                                }
                            }
                            pipeline.consumer_release(&consumer_state);
                            consumer_state.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 512>::sync();

                    epilogue.unload_and_atomic_add_bf16x2(
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        valid_n,
                        tile_m128,
                        group,
                        expert,
                        g_scale,
                        group_routes.as_ptr(),
                        routing_weights.as_ptr(),
                        output.as_mut_ptr().cast::<u32>(),
                        hidden_size,
                        b_free_bar_ptr,
                        g_idx + 1 < groups_per_cta,
                        groups_per_cta > 2,
                    );
                    g_idx += 1;
                }

                if warp_id == 0 {
                    TiledMma::dealloc_tmem_cg1(slot_ptr);
                }
            } else if warp_id == PRODUCER_WARP && (tid & 31) == 0 {
                let mut producer_state = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let burst = step & 1;
                    let g_idx = step >> 1;
                    let group = group_start + g_idx;
                    if g_idx > 0 {
                        let wait_parity = ((g_idx - 1) & 1) as u32;
                        TiledMma::wait_barrier(b_free_bar_ptr.add(burst), wait_parity);
                    }
                    let slot = producer_state.index();
                    pipeline.producer_acquire(&producer_state);
                    let full_barrier = pipeline.producer_expect_tx(&producer_state);
                    let mut a_tile = SmemTile::<u8, WeightTileSmem> {
                        base: smem_a_ptr.add(slot * 32768),
                        capacity: 32768,
                        layout: PhantomData,
                    };
                    let mut b_tile = SmemTile::<u8, ActTileSmem> {
                        base: smem_b_ptr.add(slot * 65536),
                        capacity: 65536,
                        layout: PhantomData,
                    };
                    let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem> {
                        base: smem_sfa_u32.add(slot * 1024).cast::<u16>(),
                        capacity: 2048,
                        layout: PhantomData,
                    };
                    let mut sfb_tile = SmemTile::<u16, ScaleBTmaSmem> {
                        base: smem_sfb_u32.add(slot * 2048).cast::<u16>(),
                        capacity: 4096,
                        layout: PhantomData,
                    };
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

    /// 6-Warp (`192`-Thread) Weight-Stationary 4-Stage TMA (`Act128Pipeline`) + 2-Stage TMEM (`TMEM[0..1]`) +
    /// Register-Direct Epilogue (`StagedEpilogue::unload_relu2_and_quantize_128`) Grouped `FC1 + ReLU²` (`K3`) for `B >= 2048`.
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
        let tile_m128 = cta_id & 7;
        let group_start = (cta_id >> 3) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = unsafe { *group_experts.as_ptr().add(group_start) } as usize;
        let w_outer = expert * (intermediate_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = unsafe { dyn_smem.add(SMEM_A_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_u32 = unsafe { dyn_smem.add(SMEM_SFA_OFFSET).cast::<u32>() };
        let smem_sfb_u32 = unsafe { dyn_smem.add(SMEM_SFB_OFFSET).cast::<u32>() };
        let b_pipe_bar_ptr = unsafe { dyn_smem.add(PIPELINE_OFFSET).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { b_pipe_bar_ptr.add(8) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(4) };
        let tmem_free_bar_ptr = unsafe { mma_bar_ptr.add(2) };
        let w_bar_ptr = unsafe { tmem_free_bar_ptr.add(2) };
        let slot_ptr = unsafe { w_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            let b_pipeline = Act128Pipeline::from_raw_base(b_pipe_bar_ptr);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<9>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1(slot_ptr);
            }
            b_pipeline.init(0);

            if warp_id == 5 && (tid & 31) == 0 {
                TiledMma128::load_stationary_weights_2d(
                    w_up_tma,
                    sfa_up_tma,
                    smem_a_ptr,
                    smem_sfa_u32.cast::<u8>(),
                    w_outer,
                    w_bar_ptr,
                );
                let mut b_prod = PipelineState::<Producer, 4>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let slot = b_prod.index();
                    if step >= 4 {
                        let wait_parity = (((step >> 2) & 1) ^ 1) as u32;
                        TiledMma128::wait_barrier(b_free_bar_ptr.add(slot), wait_parity);
                    }
                    b_pipeline.producer_acquire(&b_prod);
                    let b_bar = b_pipeline.producer_expect_tx(&b_prod);
                    let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                        base: smem_b_ptr.add(slot * 32768),
                        capacity: 32768,
                        layout: PhantomData,
                    };
                    let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                        base: smem_sfb_u32.add(slot * 1024).cast::<u16>(),
                        capacity: 2048,
                        layout: PhantomData,
                    };
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
                b_pipeline.producer_tail(b_prod);
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = *slot_ptr.cast_const();
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma128::new_with_lbo(128, 2048);

                TiledMma128::wait_barrier(w_bar_ptr, 0);
                let s_sfa0 = SharedWeightScales::from_raw(smem_sfa_u32);
                let s_sfa1 = SharedWeightScales::from_raw(smem_sfa_u32.add(1024));
                tiled_mma.copy_a_scales_burst(sfa_tmem_base, &s_sfa0);
                tiled_mma.copy_a_scales_burst(sfa_tmem_base + 32, &s_sfa1);

                let mut b_cons = PipelineState::<Consumer, 4>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if g_idx >= 2 {
                        TiledMma128::wait_barrier(
                            tmem_free_bar_ptr.add(tmem_slot),
                            tmem_parity ^ 1,
                        );
                    }
                    let mut burst = 0usize;
                    while burst < 2 {
                        let step = g_idx * 2 + burst;
                        b_pipeline.consumer_wait(&b_cons);
                        let slot = b_cons.index();
                        let s_a = SharedWeightTile::from_raw_bytes(smem_a_ptr.add(burst * 32768));
                        let s_b = SharedAct128Tile::from_raw_bytes(smem_b_ptr.add(slot * 32768));
                        let s_sfb = SharedActScales::from_raw(smem_sfb_u32.add(slot * 1024));
                        tiled_mma.copy_b_scales_and_gemm_128_burst(
                            tmem_d,
                            sfa_tmem_base + (burst as u32) * 32,
                            sfb_tmem_base + (slot as u32) * 32,
                            &s_a,
                            &s_b,
                            &s_sfb,
                            burst == 0,
                        );
                        if step + 4 < total_steps {
                            TiledMma128::commit_barrier(b_free_bar_ptr.add(slot));
                        }
                        if burst == 1 {
                            TiledMma128::commit_barrier(mma_bar_ptr.add(tmem_slot));
                        }
                        b_pipeline.consumer_release(&b_cons);
                        b_cons.advance();
                        burst += 1;
                    }
                    g_idx += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *slot_ptr.cast_const();
                let epilogue = StagedEpilogue::from_raw(smem_b_ptr.cast::<f32>());
                let g_scale = *w_up_global.as_ptr().add(expert);
                let vals_u128_ptr = q_inter_vals.as_mut_ptr().cast::<u128>();
                let scales_u32_ptr = q_inter_scales.as_mut_ptr().cast::<u32>();

                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        TiledMma128::wait_barrier(mma_bar_ptr.add(tmem_slot), tmem_parity);
                    }
                    NamedBarrier::<1, 128>::sync();
                    epilogue.unload_relu2_and_quantize_128(
                        tmem_d,
                        warp_id,
                        tid,
                        tile_m128,
                        group,
                        g_scale,
                        vals_u128_ptr,
                        scales_u32_ptr,
                        tmem_free_bar_ptr.add(tmem_slot),
                        g_idx + 2 < groups_per_cta,
                    );
                    g_idx += 1;
                }

                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1(slot_ptr);
                }
            }
        }
    }

    /// 6-Warp (`192`-Thread) Weight-Stationary 4-Stage TMA (`Act128Pipeline`) + 2-Stage TMEM (`TMEM[0..1]`) +
    /// 2-Stage `B64`-Swizzled S2G TMA Store (`StagedEpilogue::unload_and_tma_store_bf16_128`) Grouped `FC2` (`K4`) for `B >= 2048`.
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
        let _ = grouped_out.as_mut_ptr();
        let cta_id = thread::blockIdx_x() as usize;
        let tile_m128 = cta_id & 7;
        let group_start = (cta_id >> 3) * groups_per_cta;
        let total_steps = groups_per_cta * 2;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = unsafe { *group_experts.as_ptr().add(group_start) } as usize;
        let w_outer = expert * (hidden_size / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = unsafe { dyn_smem.add(SMEM_A_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_u32 = unsafe { dyn_smem.add(SMEM_SFA_OFFSET).cast::<u32>() };
        let smem_sfb_u32 = unsafe { dyn_smem.add(SMEM_SFB_OFFSET).cast::<u32>() };
        let smem_epi1_u128 = unsafe { dyn_smem.add(221184).cast::<u128>() };
        let b_pipe_bar_ptr = unsafe { dyn_smem.add(229376).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { b_pipe_bar_ptr.add(8) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(4) };
        let tmem_free_bar_ptr = unsafe { mma_bar_ptr.add(2) };
        let w_bar_ptr = unsafe { tmem_free_bar_ptr.add(2) };
        let slot_ptr = unsafe { w_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            let b_pipeline = Act128Pipeline::from_raw_base(b_pipe_bar_ptr);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                grouped_out_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<9>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1(slot_ptr);
            }
            b_pipeline.init(0);

            if warp_id == 5 && (tid & 31) == 0 {
                TiledMma128::load_stationary_weights_2d(
                    w_down_tma,
                    sfa_down_tma,
                    smem_a_ptr,
                    smem_sfa_u32.cast::<u8>(),
                    w_outer,
                    w_bar_ptr,
                );
                let mut b_prod = PipelineState::<Producer, 4>::new();
                let mut step = 0usize;
                while step < total_steps {
                    let slot = b_prod.index();
                    if step >= 4 {
                        let wait_parity = (((step >> 2) & 1) ^ 1) as u32;
                        TiledMma128::wait_barrier(b_free_bar_ptr.add(slot), wait_parity);
                    }
                    b_pipeline.producer_acquire(&b_prod);
                    let b_bar = b_pipeline.producer_expect_tx(&b_prod);
                    let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                        base: smem_b_ptr.add(slot * 32768),
                        capacity: 32768,
                        layout: PhantomData,
                    };
                    let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                        base: smem_sfb_u32.add(slot * 1024).cast::<u16>(),
                        capacity: 2048,
                        layout: PhantomData,
                    };
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
                b_pipeline.producer_tail(b_prod);
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = *slot_ptr.cast_const();
                let sfa_tmem_base = tmem_addr + 256;
                let sfb_tmem_base = tmem_addr + 320;
                let tiled_mma = TiledMma128::new_with_lbo(128, 2048);

                TiledMma128::wait_barrier(w_bar_ptr, 0);
                let s_sfa0 = SharedWeightScales::from_raw(smem_sfa_u32);
                let s_sfa1 = SharedWeightScales::from_raw(smem_sfa_u32.add(1024));
                tiled_mma.copy_a_scales_burst(sfa_tmem_base, &s_sfa0);
                tiled_mma.copy_a_scales_burst(sfa_tmem_base + 32, &s_sfa1);

                let mut b_cons = PipelineState::<Consumer, 4>::new();
                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if g_idx >= 2 {
                        TiledMma128::wait_barrier(
                            tmem_free_bar_ptr.add(tmem_slot),
                            tmem_parity ^ 1,
                        );
                    }
                    let mut burst = 0usize;
                    while burst < 2 {
                        let step = g_idx * 2 + burst;
                        b_pipeline.consumer_wait(&b_cons);
                        let slot = b_cons.index();
                        let s_a = SharedWeightTile::from_raw_bytes(smem_a_ptr.add(burst * 32768));
                        let s_b = SharedAct128Tile::from_raw_bytes(smem_b_ptr.add(slot * 32768));
                        let s_sfb = SharedActScales::from_raw(smem_sfb_u32.add(slot * 1024));
                        tiled_mma.copy_b_scales_and_gemm_128_burst(
                            tmem_d,
                            sfa_tmem_base + (burst as u32) * 32,
                            sfb_tmem_base + (slot as u32) * 32,
                            &s_a,
                            &s_b,
                            &s_sfb,
                            burst == 0,
                        );
                        if step + 4 < total_steps {
                            TiledMma128::commit_barrier(b_free_bar_ptr.add(slot));
                        }
                        if burst == 1 {
                            TiledMma128::commit_barrier(mma_bar_ptr.add(tmem_slot));
                        }
                        b_pipeline.consumer_release(&b_cons);
                        b_cons.advance();
                        burst += 1;
                    }
                    g_idx += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *slot_ptr.cast_const();
                let epilogue = StagedEpilogue::from_raw(smem_b_ptr.cast::<f32>());
                let g_scale = *w_down_global.as_ptr().add(expert);
                let smem_epi0_u128 = smem_sfa_u32.cast::<u128>();

                let mut g_idx = 0usize;
                while g_idx < groups_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        TiledMma128::wait_barrier(mma_bar_ptr.add(tmem_slot), tmem_parity);
                    }
                    NamedBarrier::<1, 128>::sync();
                    epilogue.unload_and_tma_store_bf16_128(
                        tmem_d,
                        warp_id,
                        tid,
                        tile_m128,
                        group,
                        g_scale,
                        smem_epi0_u128,
                        smem_epi1_u128,
                        grouped_out_tma,
                        tmem_free_bar_ptr.add(tmem_slot),
                        g_idx + 2 < groups_per_cta,
                    );
                    g_idx += 1;
                }

                if tid == 0 {
                    cp_async_bulk_wait_group_read::<0>();
                }
                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1(slot_ptr);
                }
            }
        }
    }

    /// Weighted top-4 expert combination (`K5` finalize step for `B >= 2048`) using
    /// `1` warp (`32` threads, `4 × u128` = `32` `BFloat16`s per thread) per token over
    /// `StridedTensor<u128>` and `StridedTensorMut<u128>`.
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
        let token = tid_global >> 5;
        let lane = tid_global & 31;
        if token >= num_tokens {
            return;
        }
        let h_u128 = hidden_size >> 3;
        let grouped = unsafe {
            StridedTensor::<u128>::from_raw_parts(
                grouped_out.as_ptr().cast::<u128>(),
                grouped_out.len() >> 3,
                1,
            )
        };
        let (gp128, w64) = unsafe {
            (
                *(tok_to_perm.as_ptr().cast::<u128>()).add(token),
                *(routing_weights.as_ptr().cast::<u64>()).add(token),
            )
        };
        let r0 = ((gp128 & 0xFFFF_FFFF) as usize) * h_u128 + lane;
        let r1 = (((gp128 >> 32) & 0xFFFF_FFFF) as usize) * h_u128 + lane;
        let r2 = (((gp128 >> 64) & 0xFFFF_FFFF) as usize) * h_u128 + lane;
        let r3 = ((gp128 >> 96) as usize) * h_u128 + lane;
        let w0 = f32::from_bits(((w64 & 0xFFFF) as u32) << 16);
        let w1 = f32::from_bits((((w64 >> 16) & 0xFFFF) as u32) << 16);
        let w2 = f32::from_bits((((w64 >> 32) & 0xFFFF) as u32) << 16);
        let w3 = f32::from_bits((((w64 >> 48) & 0xFFFF) as u32) << 16);
        let out_u128 = unsafe { (output.as_mut_ptr().cast::<u128>()).add(token * h_u128 + lane) };

        /// Combine one `u32` (`bf16x2`) pair across the 4 routed experts at bit shift `$sh`.
        macro_rules! combine_pair {
            ($q0:ident, $q1:ident, $q2:ident, $q3:ident, $sh:expr) => {{
                let p0 = (($q0 >> $sh) & 0xFFFF_FFFF) as u32;
                let p1 = (($q1 >> $sh) & 0xFFFF_FFFF) as u32;
                let p2 = (($q2 >> $sh) & 0xFFFF_FFFF) as u32;
                let p3 = (($q3 >> $sh) & 0xFFFF_FFFF) as u32;
                let acc_lo = w0 * f32::from_bits(p0 << 16)
                    + w1 * f32::from_bits(p1 << 16)
                    + w2 * f32::from_bits(p2 << 16)
                    + w3 * f32::from_bits(p3 << 16);
                let acc_hi = w0 * f32::from_bits(p0 & 0xFFFF_0000)
                    + w1 * f32::from_bits(p1 & 0xFFFF_0000)
                    + w2 * f32::from_bits(p2 & 0xFFFF_0000)
                    + w3 * f32::from_bits(p3 & 0xFFFF_0000);
                let b_lo = BFloat16::from_f32(acc_lo).to_bits() as u32;
                let b_hi = BFloat16::from_f32(acc_hi).to_bits() as u32;
                ((b_lo | (b_hi << 16)) as u128) << $sh
            }};
        }
        /// Coalesced 32-lane warp load/combine/store at column offset `$off`.
        macro_rules! combine_step {
            ($off:expr) => {
                unsafe {
                    let q0 = grouped.get_unchecked(r0 + $off);
                    let q1 = grouped.get_unchecked(r1 + $off);
                    let q2 = grouped.get_unchecked(r2 + $off);
                    let q3 = grouped.get_unchecked(r3 + $off);
                    *out_u128.add($off) = combine_pair!(q0, q1, q2, q3, 0)
                        | combine_pair!(q0, q1, q2, q3, 32)
                        | combine_pair!(q0, q1, q2, q3, 64)
                        | combine_pair!(q0, q1, q2, q3, 96);
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
    Ok(())
}

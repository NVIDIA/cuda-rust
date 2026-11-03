// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native `cuda-oxide` SIMT + 5th-Gen Tensor Core (`tcgen05`) implementation of the
//! Nemotron-3 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` block scales, `SF_VEC = 16`)
//! non-gated `ReLU²` Mixture-of-Experts (MoE) layer on NVIDIA Blackwell (`sm_100a`).
//!
//! # 1. The 4 Stages of the MoE Layer
//!
//! ```text
//! THE 4 STAGES OF THE MOE LAYER:
//! [K1: Quantize BF16->FP4] ──► [K2: Route] ──► [K3: FC1 + ReLU²] ──► [K4: FC2 + Finalize]
//!
//!   • Decode (`B = 1 .. 2`): 2 Fused Kernels (`dp4a` + Warp Shuffle Reduction)
//!       [K1 + K3: fc1_relu2_requant_direct] ──► [K4: fc2_and_finalize_direct]
//!
//!   • Small/Med Batch (`B = 4 .. 1,024`): 3 Fused `tcgen05` Kernels (`tile_n = 256`)
//!       [K1 + K2: fused_route_and_quantize_tcgen05]
//!           ──► [K3: tcgen05_grouped_fc1_relu2_requant]
//!           ──► [K4: tcgen05_grouped_fc2 (with fused atom_add_bf16x2 finalize)]
//!
//!   • Large Prefill (`B = 2,048 .. 16,384`): 4-Kernel Persistent 6-Warp `tcgen05` Pipeline (`tile_n = 128`)
//!       [K1 + K2: fused_route_and_quantize_tcgen05]
//!           ──► [K3: tcgen05_grouped_fc1_relu2_requant_128 (4-stage TMA + 2-stage TMEM + reg epilogue)]
//!           ──► [K4: tcgen05_grouped_fc2_128 (4-stage TMA + 2-stage TMEM + 2-stage S2G TMA Store)]
//!           ──► [K5: moe_finalize_tcgen05 (1-warp-per-token coalesced u128 combine)]
//! ```
//!
//! # 2. CUDA Graph Capture & 1,000× Replay Timer
//!
//! ```text
//!    BeginCapture                                                      EndCapture
//!    ┌─────────────────────────────────────────────────────────────────────────┐
//!    │  [K1: Quantize] ──► [K2: Route] ──► [K3: FC1+ReLU²] ──► [K4: FC2+Final] │
//!    └─────────────────────────────────────────────────────────────────────────┘
//!    Start Event ──► Replay Graph #1 .. #1000 ──► Stop Event (mean = total / 1000)
//! ```

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cuda_device::atomic::atom_add_bf16x2;
use cuda_device::barrier::{
    Barrier, barrier_cta_sync, fence_proxy_async_shared_cta, mbarrier_arrive,
    mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::convert::{
    bf16_to_f32, cvt_bf16x2_f32, cvt_f32x2_f16x2, cvt_rn_f16x2_e4m3x2, cvt_rn_satfinite_e4m3x2_f32,
    f32_to_bf16_rne,
};
use cuda_device::dotprod::dp4a_s32;
use cuda_device::float::rcp_approx_ftz_f32;
use cuda_device::prmt::prmt;
use cuda_device::shared::{DynamicSharedArray, SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_commit_shared_cluster, tcgen05_cp_32x128b_warpx4, tcgen05_dealloc,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync, tcgen05_ld_32x32b_x16_raw,
    tcgen05_ld_32x32b_x64_raw, tcgen05_load_wait, tcgen05_mma_shared_mxf4nvf4_block_scale_block16,
};
use cuda_device::tma::{
    TmaDescriptor, cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_g2s_cta,
    cp_async_bulk_tensor_2d_s2g, cp_async_bulk_wait_group_read,
};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use half::{bf16, f16};
use std::mem::MaybeUninit;
use std::sync::Arc;

include!("kernels.rs");

/// Hidden dimension `H` of each input and output token vector (`1,024` elements).
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

/// Maximum token columns per expert group tile (`256` tokens for `B >= 1024`).
/// Adaptive `tile_n` uses `16` for `B <= 64`, `32` for `B = 128`, `64` for `B = 256`, `128` for `B = 512`, and `256` for `B >= 1024`.
const TILE_N: usize = 256;

/// Dynamic shared memory bytes (`222,208` bytes = `217 KB`) per `tcgen05` CTA.
/// Holds `64 KB` `SMEM_A` + `64 KB` 2-stage `SMEM_B` (`2 × 32 KB`) + `64 KB` `SMEM_EPI` + `8 KB` `SFA` + `16 KB` `SFB`.
const DYNAMIC_SMEM_BYTES: u32 = 222_208;

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

/// Global pre-scale applied during `K1` input quantization (`1.0`).
/// Inverted in the `FC1` epilogue so scale factors cancel cleanly.
const INPUT_SCALE: f32 = 1.0;

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

/// Exact analytical `bf16` output reference for any element of `token`'s output row.
/// Evaluates `FC1 -> ReLU² -> FC2` across all `TOP_K = 4` routed experts and combines them
/// with [`routing_weight`].
fn expected_output(token: usize) -> bf16 {
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
    bf16::from_f32(total)
}

/// Element type metadata required to encode a 2D CUDA Tensor Map (`CUtensorMap`).
/// Associates each Rust element type (`u8`, `u16`) with its CUDA driver tag and byte width.
trait TmaElement {
    /// CUDA driver tensor-map data-type enum discriminant for `Self`.
    const DATA_TYPE: sys::CUtensorMapDataType;
    /// Size of one element in bytes (`1` for `u8`, `2` for `u16`).
    const ELEM_BYTES: u64;
}

impl TmaElement for u8 {
    const DATA_TYPE: sys::CUtensorMapDataType =
        sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8;
    const ELEM_BYTES: u64 = 1;
}

impl TmaElement for u16 {
    const DATA_TYPE: sys::CUtensorMapDataType =
        sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16;
    const ELEM_BYTES: u64 = 2;
}

/// Encode a 2D row-major `TmaDescriptor` over `buffer` using TMA element format `T: TmaElement`.
/// Derives the CUDA driver data type and stride byte width directly from `T`.
fn encode_tma_2d<T: TmaElement, B>(
    buffer: &DeviceBuffer<B>,
    rows: u64,
    cols: u64,
    box_rows: u32,
    box_cols: u32,
    swizzle: sys::CUtensorMapSwizzle,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
    let global_dim: [u64; 2] = [cols, rows];
    let global_strides: [u64; 1] = [cols * T::ELEM_BYTES];
    let box_dim: [u32; 2] = [box_cols, box_rows];
    let element_strides: [u32; 2] = [1, 1];
    let mut tensor_map = MaybeUninit::<sys::CUtensorMap>::uninit();
    unsafe {
        sys::cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            T::DATA_TYPE,
            2,
            buffer.cu_deviceptr() as *mut core::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            sys::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            sys::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()?;
        let map = tensor_map.assume_init();
        let words = core::mem::transmute::<sys::CUtensorMap, [u64; 16]>(map);
        Ok(TmaDescriptor::from_raw_words(words))
    }
}

/// Device buffers owned by one batch-size benchmark instance of `nemotron3_moe`.
/// Holds the input tokens, routing tables, quantized workspace buffers, expert weights, and TMA descriptors.
struct MoeBuffers {
    /// Input token matrix `[num_tokens, HIDDEN]` stored as raw `f16` bit patterns (`u16`).
    /// Consumed by `K1` (`fc1_relu2_requant_direct` or `fused_route_and_quantize_tcgen05`).
    hidden: DeviceBuffer<u16>,
    /// Selected expert IDs `[num_tokens * TOP_K]` (`i32` in `0..NUM_EXPERTS`).
    /// Used by the low-batch kernels (`B <= 2`).
    expert_ids: DeviceBuffer<i32>,
    /// Per-token routing weights `[num_tokens * TOP_K]` stored as raw `bf16` bit patterns (`u16`).
    /// Applied during `K4` (`fc2_and_finalize_direct` or `tcgen05_grouped_fc2`).
    routing_weights: DeviceBuffer<u16>,
    /// Packed `(group << 8) | row_in_group` destination `[num_tokens * TOP_K]` for each route.
    /// Read as `1 × u128` per token in `fused_route_and_quantize_tcgen05` to scatter TMA-ready `B`/`SFB` tiles.
    tok_to_perm: DeviceBuffer<i32>,
    /// Permuted route indices `[num_groups * 256]` (`token * TOP_K + slot`) per expert group.
    /// Read by `FC2` to scatter-add weighted expert outputs back to each token.
    group_tokens: DeviceBuffer<i32>,
    /// Expert ID (`0..NUM_EXPERTS`) for each bucketed group (`[num_groups]`).
    /// Read by `FC1` and `FC2` to select expert weights and global scales.
    group_experts: DeviceBuffer<i32>,
    /// Valid token count `limit` (`1..=tile_n`) in each bucketed expert group (`[num_groups]`).
    /// Read by `FC1` and `FC2` to size `valid_n` and bound the epilogue.
    group_limits: DeviceBuffer<i32>,
    /// Number of active expert groups for `B >= 4`.
    /// Determines the 1D CTA grid size `8 * num_groups8`.
    num_groups8: usize,
    /// TMA-ready `[num_groups8 * 2, 256, 256]` packed `Float4E2M1FN` input activation tiles for `FC1`.
    /// Written by `K1` and loaded into `SMEM_B` via 2D TMA (`b_up_tma`).
    b_up_vals: DeviceBuffer<u8>,
    /// TMA-ready `[num_groups8 * 2, 16, 256]` `u16` (`8 KB` per burst) input scale tiles for `FC1`.
    /// Written by `K1` and loaded into `SMEM_SFB` via 2D TMA (`sfb_up_tma`).
    sfb_up_scales: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate values in TMA-ready `[num_groups8 * 2, 256, 256]` layout.
    /// Written by `FC1` and loaded into `SMEM_B` of `FC2` via 2D TMA (`b_down_tma`).
    grouped_inter: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate scales in TMA-ready `[num_groups8 * 2, 16, 256]` `u16` layout.
    /// Written by `FC1` and loaded into `SMEM_SFB` of `FC2` via 2D TMA (`sfb_down_tma`).
    grouped_inter_scales: DeviceBuffer<u8>,
    /// Packed `FC1` expert weights `W_up` (`[NUM_EXPERTS * INTERMEDIATE, HIDDEN / 2]` bytes).
    /// Loaded into `SMEM_A` of `FC1` via 2D TMA (`w_up_tma`).
    w_up: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_up`.
    /// Loaded into `SMEM_SFA` of `FC1` via 2D TMA (`sfa_up_tma`).
    w_up_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization multiplier for `W_up` (`[NUM_EXPERTS]`).
    /// Applied in the `FC1` epilogue before `ReLU²`.
    w_up_global: DeviceBuffer<f32>,
    /// Packed `FC2` expert weights `W_down` (`[NUM_EXPERTS * HIDDEN, INTERMEDIATE / 2]` bytes).
    /// Loaded into `SMEM_A` of `FC2` via 2D TMA (`w_down_tma`).
    w_down: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_down`.
    /// Loaded into `SMEM_SFA` of `FC2` via 2D TMA (`sfa_down_tma`).
    w_down_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization multiplier for `W_down` (`[NUM_EXPERTS]`).
    /// Applied in the `FC2` epilogue before `atom_add_bf16x2`.
    w_down_global: DeviceBuffer<f32>,
    /// Host-encoded 2D TMA descriptor for `W_up` (`128 × 256` bytes per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc1_relu2_requant`.
    w_up_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `b_up_vals` (`256 × 256` bytes per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc1_relu2_requant`.
    b_up_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `w_up_scales` (`8 × 256` `u16`s per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc1_relu2_requant`.
    sfa_up_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `sfb_up_scales` (`16 × 256` `u16`s per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc1_relu2_requant`.
    sfb_up_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `W_down` (`128 × 256` bytes per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc2`.
    w_down_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `grouped_inter` (`256 × 256` bytes per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc2`.
    b_down_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `w_down_scales` (`8 × 256` `u16`s per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc2`.
    sfa_down_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `grouped_inter_scales` (`16 × 256` `u16`s per burst).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc2`.
    sfb_down_tma: TmaDescriptor,
    /// Host-encoded 2D TMA descriptor for `grouped_out` (`128 × 32` `bf16`s = `8 KB` per S2G TMA store).
    /// Passed by value via `#[grid_constant]` to `tcgen05_grouped_fc2_128`.
    grouped_out_tma: TmaDescriptor,
    /// Contiguous `[num_groups8 * 256, HIDDEN]` `bf16` expert outputs written by `FC2` at `B >= 2048`.
    /// Consumed by `moe_finalize_tcgen05` to combine `TOP_K = 4` expert outputs without atomic adds.
    grouped_out: DeviceBuffer<u16>,
    /// Final MoE layer output `[num_tokens, HIDDEN]` stored as `bf16` bit patterns (`u16`).
    /// Verified against [`expected_output`] after execution.
    output: DeviceBuffer<u16>,
    /// Number of input tokens `B` in this workload instance.
    /// Selects between the low-batch (`B <= 2`) and `tcgen05` (`B >= 4`) schedules.
    num_tokens: usize,
}

/// Allocate and initialize all device buffers and TMA descriptors for a batch of `num_tokens` tokens.
/// Populates deterministic synthetic inputs, routing weights, expert weight tensors, and 2D TMA maps.
fn allocate_buffers(
    stream: &Arc<CudaStream>,
    num_tokens: usize,
) -> Result<MoeBuffers, Box<dyn std::error::Error>> {
    let routed_size = num_tokens * TOP_K;
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

    let hidden_host: Vec<u16> = (0..num_tokens)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(hidden_value(token)).to_bits(), HIDDEN))
        .collect();
    let expert_ids_host: Vec<i32> = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| expert_id(token, slot) as i32))
        .collect();
    let routing_host: Vec<u16> = (0..num_tokens)
        .flat_map(|token| {
            (0..TOP_K).map(move |slot| bf16::from_f32(routing_weight(token, slot)).to_bits())
        })
        .collect();

    let mut group_experts_host = Vec::new();
    let mut group_routes_host = Vec::new();
    let mut group_limits_host = Vec::new();
    let mut tok_to_perm_host = vec![0i32; routed_size];
    for expert in 0..NUM_EXPERTS {
        let mut matches = Vec::new();
        for (r, &exp_id) in expert_ids_host.iter().enumerate().take(routed_size) {
            if exp_id as usize == expert {
                matches.push(r as i32);
            }
        }
        for chunk in matches.chunks(tile_n) {
            let g_idx = group_experts_host.len();
            group_experts_host.push(expert as i32);
            group_limits_host.push(chunk.len() as i32);
            let g_shift = if num_tokens >= 2048 { 7 } else { 8 };
            for (r_in_g, &route) in chunk.iter().enumerate() {
                tok_to_perm_host[route as usize] = ((g_idx << g_shift) | r_in_g) as i32;
            }
            for i in 0..256 {
                let route = chunk[i.min(chunk.len() - 1)];
                group_routes_host.push(route);
            }
        }
    }
    let num_groups8 = group_experts_host.len();
    let b_tma_bytes = num_groups8 * 2 * 65536;
    let sfb_tma_bytes = num_groups8 * 2 * 8192;

    // Pre-swizzle expert weights into `tcgen05` `(32, 4, 8)` row order so 2D TMA loads directly into `SMEM_A`!
    let w_up_host = vec![0x22u8; NUM_EXPERTS * INTERMEDIATE * (HIDDEN / 2)];
    let w_up_scales_host = vec![0x38u8; (NUM_EXPERTS * INTERMEDIATE / 128) * (HIDDEN / 64) * 512];
    let mut w_up_global_host = vec![1.0f32 / HIDDEN as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_up_global_host[expert] = UP_FACTORS[expert] / HIDDEN as f32;
    }

    let w_down_host = vec![0x22u8; NUM_EXPERTS * HIDDEN * (INTERMEDIATE / 2)];
    let w_down_scales_host = vec![0x38u8; (NUM_EXPERTS * HIDDEN / 128) * (INTERMEDIATE / 64) * 512];
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_down_global_host[expert] = DOWN_FACTORS[expert] / INTERMEDIATE as f32;
    }

    let w_up = DeviceBuffer::from_host(stream, &w_up_host)?;
    let w_up_scales = DeviceBuffer::from_host(stream, &w_up_scales_host)?;
    let w_down = DeviceBuffer::from_host(stream, &w_down_host)?;
    let w_down_scales = DeviceBuffer::from_host(stream, &w_down_scales_host)?;
    let b_up_vals = DeviceBuffer::from_host(stream, &vec![0u8; b_tma_bytes])?;
    let sfb_up_scales = DeviceBuffer::from_host(stream, &vec![0u8; sfb_tma_bytes])?;
    let grouped_inter = DeviceBuffer::from_host(
        stream,
        &vec![0u8; b_tma_bytes.max(routed_size * (INTERMEDIATE / 2))],
    )?;
    let grouped_inter_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; sfb_tma_bytes.max(routed_size * (INTERMEDIATE / 16))],
    )?;

    let swizzle_none = sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE;
    let swizzle_64b = sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B;
    let b_box_rows = if num_groups8 > 32 { 128 } else { 256 };
    let sfb_box_rows = if num_groups8 > 32 { 8 } else { 16 };
    let w_up_tma = encode_tma_2d::<u8, _>(
        &w_up,
        (NUM_EXPERTS * INTERMEDIATE) as u64,
        (HIDDEN / 2) as u64,
        128,
        256,
        swizzle_none,
    )?;
    let b_up_tma = encode_tma_2d::<u8, _>(
        &b_up_vals,
        (num_groups8 * 2 * 256) as u64,
        256,
        b_box_rows,
        256,
        swizzle_none,
    )?;
    let sfa_up_tma = encode_tma_2d::<u16, _>(
        &w_up_scales,
        (NUM_EXPERTS * (INTERMEDIATE / 128) * (HIDDEN / 64)) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;
    let sfb_up_tma = encode_tma_2d::<u16, _>(
        &sfb_up_scales,
        (num_groups8 * 2 * 16) as u64,
        256,
        sfb_box_rows,
        256,
        swizzle_none,
    )?;
    let w_down_tma = encode_tma_2d::<u8, _>(
        &w_down,
        (NUM_EXPERTS * HIDDEN) as u64,
        (INTERMEDIATE / 2) as u64,
        128,
        256,
        swizzle_none,
    )?;
    let b_down_tma = encode_tma_2d::<u8, _>(
        &grouped_inter,
        (num_groups8 * 2 * 256) as u64,
        256,
        b_box_rows,
        256,
        swizzle_none,
    )?;
    let sfa_down_tma = encode_tma_2d::<u16, _>(
        &w_down_scales,
        (NUM_EXPERTS * (HIDDEN / 128) * (INTERMEDIATE / 64)) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;
    let sfb_down_tma = encode_tma_2d::<u16, _>(
        &grouped_inter_scales,
        (num_groups8 * 2 * 16) as u64,
        256,
        sfb_box_rows,
        256,
        swizzle_none,
    )?;
    let grouped_out = DeviceBuffer::from_host(stream, &vec![0u16; num_groups8 * 256 * HIDDEN])?;
    let grouped_out_tma = encode_tma_2d::<u16, _>(
        &grouped_out,
        (num_groups8 * 256) as u64,
        HIDDEN as u64,
        128,
        32,
        swizzle_64b,
    )?;

    Ok(MoeBuffers {
        hidden: DeviceBuffer::from_host(stream, &hidden_host)?,
        expert_ids: DeviceBuffer::from_host(stream, &expert_ids_host)?,
        routing_weights: DeviceBuffer::from_host(stream, &routing_host)?,
        tok_to_perm: DeviceBuffer::from_host(stream, &tok_to_perm_host)?,
        group_tokens: DeviceBuffer::from_host(stream, &group_routes_host)?,
        group_experts: DeviceBuffer::from_host(stream, &group_experts_host)?,
        group_limits: DeviceBuffer::from_host(stream, &group_limits_host)?,
        num_groups8,
        b_up_vals,
        sfb_up_scales,
        grouped_inter,
        grouped_inter_scales,
        w_up,
        w_up_scales,
        w_up_global: DeviceBuffer::from_host(stream, &w_up_global_host)?,
        w_down,
        w_down_scales,
        w_down_global: DeviceBuffer::from_host(stream, &w_down_global_host)?,
        w_up_tma,
        b_up_tma,
        sfa_up_tma,
        sfb_up_tma,
        w_down_tma,
        b_down_tma,
        sfa_down_tma,
        sfb_down_tma,
        grouped_out_tma,
        grouped_out,
        output: DeviceBuffer::from_host(stream, &vec![0u16; num_tokens * HIDDEN])?,
        num_tokens,
    })
}

/// Launch the complete `FP16 -> NVFP4 -> FC1 + ReLU² -> FC2 + Finalize -> BF16` MoE pipeline
/// on `stream`. Used both for eager numerical verification and for CUDA Graph stream capture.
fn launch_pipeline(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    bufs: &mut MoeBuffers,
) -> Result<(), Box<dyn std::error::Error>> {
    if bufs.num_tokens <= 2 {
        let fc1_direct_cfg = LaunchConfig {
            grid_dim: (
                (INTERMEDIATE / 16) as u32,
                (bufs.num_tokens * TOP_K) as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.fc1_relu2_requant_direct(
                stream,
                fc1_direct_cfg,
                &mut bufs.grouped_inter,
                &mut bufs.grouped_inter_scales,
                &bufs.hidden,
                &bufs.expert_ids,
                &bufs.w_up,
                &bufs.w_up_scales,
                &bufs.w_up_global,
                INPUT_SCALE,
                HIDDEN as u32,
                INTERMEDIATE as u32,
                TOP_K as u32,
            )?;
        }

        let fc2_direct_cfg = LaunchConfig {
            grid_dim: ((HIDDEN / 16) as u32, bufs.num_tokens as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.fc2_and_finalize_direct(
                stream,
                fc2_direct_cfg,
                &mut bufs.output,
                &bufs.grouped_inter,
                &bufs.grouped_inter_scales,
                &bufs.expert_ids,
                &bufs.routing_weights,
                &bufs.w_down,
                &bufs.w_down_scales,
                &bufs.w_down_global,
                HIDDEN as u32,
                INTERMEDIATE as u32,
                TOP_K as u32,
            )?;
        }
        return Ok(());
    }

    let groups_per_cta = if bufs.num_groups8 > 16 {
        bufs.num_groups8 / 16
    } else {
        1
    };
    let zero_out = u32::from(groups_per_cta <= 2);
    let q1_groups = bufs.num_tokens * (HIDDEN / 16);
    let route_quant_cfg = LaunchConfig {
        grid_dim: (q1_groups.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.fused_route_and_quantize_tcgen05(
            stream,
            route_quant_cfg,
            &bufs.hidden,
            &bufs.tok_to_perm,
            &mut bufs.b_up_vals,
            &mut bufs.sfb_up_scales,
            &mut bufs.output,
            bufs.num_tokens as u32,
            HIDDEN as u32,
            INPUT_SCALE,
            zero_out,
        )?;
    }

    let total_ctas = ((INTERMEDIATE / 128) * (bufs.num_groups8 / groups_per_cta)) as u32;
    if groups_per_cta <= 2 {
        let fc1_cfg = LaunchConfig1D::new(total_ctas, 544, DYNAMIC_SMEM_BYTES);
        let fc1_launch = module.prepare_tcgen05_grouped_fc1_relu2_requant(fc1_cfg)?;
        unsafe {
            module.tcgen05_grouped_fc1_relu2_requant(
                stream,
                &fc1_launch,
                bufs.w_up_tma,
                bufs.b_up_tma,
                bufs.sfa_up_tma,
                bufs.sfb_up_tma,
                &bufs.w_up_global,
                &bufs.group_experts,
                &bufs.group_limits,
                &mut bufs.grouped_inter,
                &mut bufs.grouped_inter_scales,
                INPUT_SCALE,
                INTERMEDIATE as u32,
                groups_per_cta as u32,
            )?;
        }

        let fc2_cfg = LaunchConfig1D::new(total_ctas, 544, DYNAMIC_SMEM_BYTES);
        let fc2_launch = module.prepare_tcgen05_grouped_fc2(fc2_cfg)?;
        unsafe {
            module.tcgen05_grouped_fc2(
                stream,
                &fc2_launch,
                bufs.w_down_tma,
                bufs.b_down_tma,
                bufs.sfa_down_tma,
                bufs.sfb_down_tma,
                &bufs.w_down_global,
                &bufs.group_experts,
                &bufs.group_tokens,
                &bufs.group_limits,
                &bufs.routing_weights,
                &mut bufs.output,
                HIDDEN as u32,
                groups_per_cta as u32,
            )?;
        }
    } else {
        let fc1_cfg = LaunchConfig1D::new(total_ctas, 192, DYNAMIC_SMEM_BYTES);
        let fc1_launch = module.prepare_tcgen05_grouped_fc1_relu2_requant_128(fc1_cfg)?;
        unsafe {
            module.tcgen05_grouped_fc1_relu2_requant_128(
                stream,
                &fc1_launch,
                bufs.w_up_tma,
                bufs.b_up_tma,
                bufs.sfa_up_tma,
                bufs.sfb_up_tma,
                &bufs.w_up_global,
                &bufs.group_experts,
                &mut bufs.grouped_inter,
                &mut bufs.grouped_inter_scales,
                INPUT_SCALE,
                INTERMEDIATE as u32,
                groups_per_cta as u32,
            )?;
        }

        let fc2_cfg = LaunchConfig1D::new(total_ctas, 192, 230_400);
        let fc2_launch = module.prepare_tcgen05_grouped_fc2_128(fc2_cfg)?;
        unsafe {
            module.tcgen05_grouped_fc2_128(
                stream,
                &fc2_launch,
                bufs.w_down_tma,
                bufs.b_down_tma,
                bufs.sfa_down_tma,
                bufs.sfb_down_tma,
                bufs.grouped_out_tma,
                &bufs.w_down_global,
                &bufs.group_experts,
                &mut bufs.grouped_out,
                HIDDEN as u32,
                groups_per_cta as u32,
            )?;
        }
        let fin_threads = (bufs.num_tokens * 32) as u32;
        let fin_cfg = LaunchConfig {
            grid_dim: (fin_threads.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.moe_finalize_tcgen05(
                stream,
                fin_cfg,
                &bufs.grouped_out,
                &bufs.tok_to_perm,
                &bufs.routing_weights,
                &mut bufs.output,
                bufs.num_tokens as u32,
            )?;
        }
    }
    Ok(())
}

/// Copy `bufs.output` back to the host and verify every element against [`expected_output`].
/// Fails if any value is non-finite or if the normalized maximum error exceeds `0.01`.
fn check_output(
    stream: &Arc<CudaStream>,
    bufs: &MoeBuffers,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.synchronize()?;
    let output_bits = bufs.output.to_host_vec(stream)?;
    let mut max_abs_error = 0.0f32;
    let mut max_reference = 1.0f32;
    for (index, &bits) in output_bits.iter().enumerate() {
        let token = index / HIDDEN;
        let actual = bf16::from_bits(bits).to_f32();
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
    println!("=== Nemotron-3 NVFP4 ReLU² MoE (Native cuda-oxide SIMT + tcgen05 Tensor Cores) ===");
    println!("Target GPU: sm_{major}{minor}");
    println!("H={HIDDEN}, I={INTERMEDIATE}, experts={NUM_EXPERTS}, top-k={TOP_K}, tile_n={TILE_N}");

    let stream = ctx.new_stream()?;
    let module = unsafe { kernels::load(&ctx)? };

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
    /// Reference `cute-rs` (`nemotron3_moe_cute`, `cutlass-mlir`) CUDA Graph latency (`µs`) across `BATCH_SIZES` on B200.
    /// Included so a single `cargo oxide run` prints the complete 3-way comparison table.
    const CUTE_RS_REF_US: [f64; 15] = [
        12.296, 14.344, 14.332, 14.344, 14.338, 14.345, 14.344, 14.345, 16.377, 19.222, 26.611,
        32.796, 51.192, 90.351, 173.943,
    ];

    println!(
        "\nBatch size | FlashInfer CuTeDSL (us) | Native SIMT [live] (us) |     cute-rs [ref] (us) | Native vs. SOTA"
    );
    println!(
        "-----------+-------------------------+-------------------------+------------------------+----------------"
    );
    for (idx, &(b, native_us)) in results.iter().enumerate() {
        let sota_us = SOTA_FLASHINFER_US[idx];
        let cute_us = CUTE_RS_REF_US[idx];
        let speedup = sota_us / native_us;
        println!(
            "{b:>10} | {sota_us:>23.3} | {native_us:>23.3} | {cute_us:>22.3} | {speedup:>9.2}x SOTA"
        );
    }
    Ok(())
}

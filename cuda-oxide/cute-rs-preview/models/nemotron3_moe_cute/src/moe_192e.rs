/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Wide 192-Expert NVFP4 ReLU² MoE (`H = 6,144, I = 1,536, E = 192, top_k = 4, B = 1 .. 256`)
//! implemented with `cute-rs` layouts, typed TMA descriptors, `PipelineTmaAsync`, and
//! `Sm100BlockScaledTiledMma` / `Sm100TmemStagedEpilogue`.
//!
//! # End-to-End Captured CUDA Graph Pipeline
//!
//! ```text
//! BeginCapture                                                                                         EndCapture
//! ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
//! │  [K1: Route & Group] ──► [K2: FP16->NVFP4 Quant] ──► [K3: Persistent FC1+ReLU²] ──► [K4: Persistent FC2]    │
//! │   (1 CTA, SMEM scan,      (Vectorized FP16 load,      (144 CTAs, 12x K=512 TMA      (Dual-M M=256 / Triple- │
//! │    bit-shift prefix,       RN-even E4M3 + E2M1,        bursts, PipelineTmaAsync,     M M=384, 1-wave grid)  │
//! │    192 experts)            4-way 128-row scatter)      Sm100BlockScaledTiledMma)                  │         │
//! │                                                                                                   ▼         │
//! │                                                                                    [K5: FP32 Top-4 Combine] │
//! └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
//! ```

use core::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use ctm_rs::mbarrier::Barrier;
use ctm_rs::memory::{DisjointSlice, DynamicSharedArray};
use ctm_rs::{cuda_module, kernel, launch_bounds, launch_contract, thread};
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cute_rs::arch::NamedBarrier;
use cute_rs::copy::tma::{TmaEncodeOptions, create_tensor_map_tiled_with_options};
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};
use cute_rs::{
    BFloat16, Consumer, Float4E2M1FN, Float8E4M3FN, Mkl, Nkl, OperandA, OperandB, PipelineState,
    PipelineTmaAsync, Producer, RowMajor, Sm100BlockScaledTiledMma, Sm100ScaleTile,
    Sm100SharedTile, Sm100TmemStagedEpilogue, SmemTile, TensorMap, copy_tma_2d, size_in_bytes,
};

/// Hidden dimension `H = 6,144` (`384` groups of `16` = `12` bursts of `K = 512`).
/// Matches the wide 192-expert benchmark configuration.
pub const HIDDEN_192E: usize = 6_144;

/// Intermediate dimension `I = 1,536` (`96` groups of `16` = `3` bursts of `K = 512`).
/// Matches the wide 192-expert benchmark configuration.
pub const INTERMEDIATE_192E: usize = 1_536;

/// Total routed experts `E = 192` allocated in device memory.
/// Every expert's `W_up` and `W_down` weights and scales reside in global memory.
pub const NUM_EXPERTS_192E: usize = 192;

/// Number of experts selected per token (`top_k = 4`).
/// Each token's output is the weighted sum of 4 routed experts.
pub const TOP_K_192E: usize = 4;

/// Batch sizes benchmarked for the 192-expert MoE layer (`B = 1 .. 256`).
/// Evaluated with `50` graph warmups and `1,000` timed graph replays per sample.
pub const BATCH_SIZES_192E: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Fixed physical token row capacity per TMA group tile (`128` rows).
/// Covers `tile_n = 16, 32, 64, 128` with a single 512-byte `(32, 4, 4)` scale atom per `K = 64` step.
const GROUP_TILE_ROWS: usize = 128;

/// Compile-time 2D shared-memory layout for one `128 × 256` (`K = 512`) packed FP4 weight burst (`32,768` B).
/// Encoded into `TensorMap<u8, WeightTileSmem>` and `Sm100SharedTile<WeightTileSmem, OperandA>`.
pub type WeightTileSmem = RowMajor<128, 256>;

/// Compile-time 2D shared-memory layout for one `128 × 256` (`K = 512`) packed FP4 activation burst (`32,768` B).
/// Encoded into `TensorMap<u8, ActTile128Smem>` and `Sm100SharedTile<ActTile128Smem, OperandB>`.
pub type ActTile128Smem = RowMajor<128, 256>;

/// Compile-time 2D shared-memory layout for 8 canonical `(32, 4, 4)` scale atoms (`8 × 256` `u16` = `4,096` B).
/// Encoded into `TensorMap<u16, ScaleTmaSmem>` for both `SFA` and `SFB`.
pub type ScaleTmaSmem = RowMajor<8, 256>;

/// Compile-time 2D shared-memory layout for the `[128, 128]` `f32` / `bf16` TMEM staged epilogue (`64 KB` / `32 KB`).
/// Encoded into `Sm100TmemStagedEpilogue<Epilogue128Smem, 128>`.
pub type Epilogue128Smem = RowMajor<128, 128>;

/// Compile-time verified TMA transaction bytes per `FC1` stage (`32 KB A + 32 KB B + 4 KB SFA + 4 KB SFB = 73,728` B).
/// Checked against the 4 `copy_tma_2d` calls by `dialect-cute`.
const FC1_STAGE_TX_BYTES: u32 = size_in_bytes::<u8, WeightTileSmem>()
    + size_in_bytes::<u8, ActTile128Smem>()
    + size_in_bytes::<u16, ScaleTmaSmem>()
    + size_in_bytes::<u16, ScaleTmaSmem>();

/// Compile-time verified TMA transaction bytes per `FC2` Dual-M (`M = 256`) stage (`110,592` B).
/// Covers `2 × 32 KB A + 32 KB B + 2 × 4 KB SFA + 4 KB SFB`.
const FC2_M256_STAGE_TX_BYTES: u32 = 2 * size_in_bytes::<u8, WeightTileSmem>()
    + size_in_bytes::<u8, ActTile128Smem>()
    + 2 * size_in_bytes::<u16, ScaleTmaSmem>()
    + size_in_bytes::<u16, ScaleTmaSmem>();

/// Compile-time verified TMA transaction bytes per `FC2` Triple-M (`M = 384`) stage (`147,456` B).
/// Covers `3 × 32 KB A + 32 KB B + 3 × 4 KB SFA + 4 KB SFB`.
const FC2_M384_STAGE_TX_BYTES: u32 = 3 * size_in_bytes::<u8, WeightTileSmem>()
    + size_in_bytes::<u8, ActTile128Smem>()
    + 3 * size_in_bytes::<u16, ScaleTmaSmem>()
    + size_in_bytes::<u16, ScaleTmaSmem>();

/// Byte size of one `WeightTileSmem` burst (`32,768` bytes).
/// Used to index 2-stage shared-memory weight buffers.
const WEIGHT_BURST_BYTES: usize = size_in_bytes::<u8, WeightTileSmem>() as usize;

/// Byte size of one `ActTile128Smem` burst (`32,768` bytes).
/// Used to index 2-stage shared-memory activation buffers.
const ACT128_BURST_BYTES: usize = size_in_bytes::<u8, ActTile128Smem>() as usize;

/// Byte size of one `ScaleTmaSmem` burst (`4,096` bytes).
/// Used to index 2-stage shared-memory scale buffers.
const SCALE_BURST_BYTES: usize = size_in_bytes::<u16, ScaleTmaSmem>() as usize;

/// Dynamic shared memory bytes per `FC1` CTA (`148,480` bytes = `145 KB`).
/// Holds `2 × 32 KB` `SMEM_A`, `2 × 32 KB` `SMEM_B`, `2 × 4 KB` `SMEM_SFA`, `2 × 4 KB` `SMEM_SFB`, and barriers.
const DYNAMIC_SMEM_FC1_192E: u32 = 148_480;

/// Dynamic shared memory bytes per `FC2` CTA (`222,208` bytes = `217 KB`).
/// Holds Dual-M (`M = 256`) or Triple-M (`M = 384`) staged weight, activation, and scale buffers.
const DYNAMIC_SMEM_FC2_192E: u32 = 222_208;

/// Synthetic `FC1` per-expert scale factors for the 4 active experts in the uniform benchmark.
/// Matches the `nemotron3_moe` reference fixture.
const UP_FACTORS_192E: [f32; TOP_K_192E] = [0.75, 1.25, -0.5, 1.0];

/// Synthetic `FC2` per-expert scale factors for the 4 active experts in the uniform benchmark.
/// Matches the `nemotron3_moe` reference fixture.
const DOWN_FACTORS_192E: [f32; TOP_K_192E] = [1.5, -0.75, 0.5, 1.25];

/// Number of untimed warmup graph replays before each timed sample (`50`).
/// Matches the `qualify.py` benchmark protocol.
const WARMUPS_192E: usize = 50;

/// Number of timed CUDA Graph replays per sample (`1,000`).
/// Matches the `qualify.py` benchmark protocol.
const REPLAYS_192E: usize = 1_000;

/// Number of timing samples collected per batch size (`20`).
/// Matches the `qualify.py` benchmark protocol.
const SAMPLES_192E: usize = 20;

/// Device kernels for the 192-expert wide NVFP4 ReLU² MoE (`cute-rs` + `tcgen05`).
#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub mod kernels_192e {
    use super::*;

    /// Role-checked shared-memory weight tile (`RowMajor<128, 256>`, `OperandA`).
    /// Prevents accidentally swapping weight and activation tiles in `Sm100BlockScaledTiledMma`.
    type SharedWeightTile = Sm100SharedTile<WeightTileSmem, OperandA>;

    /// Role-checked shared-memory activation tile (`RowMajor<128, 256>`, `OperandB`).
    /// Prevents accidentally swapping weight and activation tiles in `Sm100BlockScaledTiledMma`.
    type SharedActTile = Sm100SharedTile<ActTile128Smem, OperandB>;

    /// Role-checked shared-memory weight scale tile (`Mkl`).
    /// Prevents accidentally swapping A-side and B-side scale tiles.
    type SharedWeightScales = Sm100ScaleTile<Mkl>;

    /// Role-checked shared-memory activation scale tile (`Nkl`).
    /// Prevents accidentally swapping A-side and B-side scale tiles.
    type SharedActScales = Sm100ScaleTile<Nkl>;

    /// Typed SM100 group-one block-scaled `mxf4nvf4` tiled MMA over `WeightTileSmem` and `ActTile128Smem`.
    /// Stages scales via `tcgen05.cp` and issues 8 `K = 64` MMAs per `K = 512` burst.
    type TiledMma128 = Sm100BlockScaledTiledMma<WeightTileSmem, ActTile128Smem, 128, 64>;

    /// Typed SM100 TMEM staged epilogue over `Epilogue128Smem` (`[128, 128]`).
    /// Unloads active rows `0..limit` from TMEM and performs `ReLU²` + NVFP4 requantization or `BF16` output.
    type StagedEpilogue128 = Sm100TmemStagedEpilogue<Epilogue128Smem, 128>;

    /// Typed 2-stage TMA load pipeline for `FC1` (`73,728` bytes per stage).
    /// Coordinates Warp 4 (Producer) and Warp 0 (MMA Consumer).
    type Fc1Pipeline = PipelineTmaAsync<2, 1, FC1_STAGE_TX_BYTES>;

    /// Typed 2-stage TMA load pipeline for `FC2` Dual-M (`110,592` bytes per stage).
    /// Coordinates Warp 4 (Producer) and Warp 0 (MMA Consumer).
    type Fc2M256Pipeline = PipelineTmaAsync<2, 1, FC2_M256_STAGE_TX_BYTES>;

    /// Decode one IEEE 754 `binary16` (`FP16`) 16-bit pattern to `f32` on device.
    #[inline(always)]
    fn fp16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) as u32) << 31;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x03FF) as u32;
        if exp == 0 {
            if mant == 0 {
                f32::from_bits(sign)
            } else {
                let mag = (mant as f32) * f32::from_bits((127 - 24) << 23);
                if sign != 0 { -mag } else { mag }
            }
        } else if exp == 31 {
            f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
        } else {
            f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
        }
    }

    /// Stage 1 (`K1` for `B >= 32`): Single-CTA (`256`-thread) shared-memory route histogram,
    /// parallel power-of-two bit-shift prefix scan, and group/permutation table builder.
    #[launch_bounds(256)]
    #[kernel]
    pub fn moe_route_192e_cute(
        expert_ids: &[i32],
        mut group_experts: DisjointSlice<i32>,
        mut group_limits: DisjointSlice<i32>,
        mut group_routes: DisjointSlice<i32>,
        mut tok_to_perm: DisjointSlice<i32>,
        mut group_count: DisjointSlice<i32>,
        routed_rows: u32,
        tile_n: u32,
        tile_n_log2: u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let dyn_smem = DynamicSharedArray::<u8, 16>::get_raw();
        unsafe {
            TiledMma128::route_histogram_and_scan_shared(
                dyn_smem,
                expert_ids.as_ptr(),
                group_experts.as_mut_ptr(),
                group_limits.as_mut_ptr(),
                group_routes.as_mut_ptr(),
                tok_to_perm.as_mut_ptr(),
                group_count.as_mut_ptr(),
                tid,
                routed_rows as usize,
                NUM_EXPERTS_192E,
                tile_n,
                tile_n_log2,
                GROUP_TILE_ROWS,
            );
        }
    }

    /// Stage 2 (`K2`): Vectorized `FP16 -> NVFP4` input activation quantizer and 4-way `128`-row
    /// `K = 512` TMA burst scatter (`H = 6,144`, `12` bursts).
    ///
    /// For `B <= 16` (`fuse_route != 0`), fuses route counting and group permutation directly in
    /// shared memory, eliminating the separate `moe_route_192e_cute` kernel launch.
    #[launch_bounds(128)]
    #[kernel]
    pub fn quantize_fp16_192e_cute(
        hidden: &[u16],
        expert_ids: &[i32],
        mut group_experts: DisjointSlice<i32>,
        mut group_limits: DisjointSlice<i32>,
        mut group_routes: DisjointSlice<i32>,
        mut group_count: DisjointSlice<i32>,
        mut tok_to_perm: DisjointSlice<i32>,
        mut b_up_vals: DisjointSlice<u8>,
        mut sfb_up_scales: DisjointSlice<u8>,
        mut output: DisjointSlice<BFloat16>,
        num_tokens: u32,
        fuse_route: u32,
        zero_out: u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let cta_id = thread::blockIdx_x() as usize;
        let idx = cta_id * 128 + tid;
        let scale_groups_per_row = HIDDEN_192E / 16; // 384
        let total_groups = (num_tokens as usize) * scale_groups_per_row;
        let row = idx / scale_groups_per_row;
        let grp = idx % scale_groups_per_row;

        let dyn_smem = DynamicSharedArray::<u8, 16>::get_raw();
        let gp128 = unsafe {
            TiledMma128::route_and_load_gp128_shared(
                dyn_smem,
                expert_ids.as_ptr(),
                group_experts.as_mut_ptr(),
                group_limits.as_mut_ptr(),
                group_routes.as_mut_ptr(),
                group_count.as_mut_ptr(),
                tok_to_perm.as_mut_ptr(),
                output.as_mut_ptr().cast::<u128>(),
                tid,
                cta_id,
                idx,
                row,
                total_groups,
                num_tokens as usize,
                GROUP_TILE_ROWS,
                fuse_route,
                zero_out,
            )
        };
        if idx >= total_groups {
            return;
        }

        let in_u128 = unsafe { hidden.as_ptr().cast::<u128>().add(idx * 2) };
        let (q0, q1) = unsafe { (*in_u128, *in_u128.add(1)) };
        let v0 = fp16_bits_to_f32((q0 & 0xFFFF) as u16);
        let v1 = fp16_bits_to_f32(((q0 >> 16) & 0xFFFF) as u16);
        let v2 = fp16_bits_to_f32(((q0 >> 32) & 0xFFFF) as u16);
        let v3 = fp16_bits_to_f32(((q0 >> 48) & 0xFFFF) as u16);
        let v4 = fp16_bits_to_f32(((q0 >> 64) & 0xFFFF) as u16);
        let v5 = fp16_bits_to_f32(((q0 >> 80) & 0xFFFF) as u16);
        let v6 = fp16_bits_to_f32(((q0 >> 96) & 0xFFFF) as u16);
        let v7 = fp16_bits_to_f32(((q0 >> 112) & 0xFFFF) as u16);
        let v8 = fp16_bits_to_f32((q1 & 0xFFFF) as u16);
        let v9 = fp16_bits_to_f32(((q1 >> 16) & 0xFFFF) as u16);
        let v10 = fp16_bits_to_f32(((q1 >> 32) & 0xFFFF) as u16);
        let v11 = fp16_bits_to_f32(((q1 >> 48) & 0xFFFF) as u16);
        let v12 = fp16_bits_to_f32(((q1 >> 64) & 0xFFFF) as u16);
        let v13 = fp16_bits_to_f32(((q1 >> 80) & 0xFFFF) as u16);
        let v14 = fp16_bits_to_f32(((q1 >> 96) & 0xFFFF) as u16);
        let v15 = fp16_bits_to_f32(((q1 >> 112) & 0xFFFF) as u16);

        let mut amax = 0.0f32;
        macro_rules! max_abs_v {
            ($v:ident) => {{
                let a = f32::from_bits($v.to_bits() & 0x7FFF_FFFF);
                if a > amax {
                    amax = a;
                }
            }};
        }
        max_abs_v!(v0);
        max_abs_v!(v1);
        max_abs_v!(v2);
        max_abs_v!(v3);
        max_abs_v!(v4);
        max_abs_v!(v5);
        max_abs_v!(v6);
        max_abs_v!(v7);
        max_abs_v!(v8);
        max_abs_v!(v9);
        max_abs_v!(v10);
        max_abs_v!(v11);
        max_abs_v!(v12);
        max_abs_v!(v13);
        max_abs_v!(v14);
        max_abs_v!(v15);
        let scale_code = Float8E4M3FN::encode_rne(amax * (1.0f32 / 6.0f32));
        let inv_scale = Float4E2M1FN::exact_inv_scale(scale_code);
        macro_rules! enc_nib {
            ($v:ident, $sh:expr) => {
                (Float4E2M1FN::encode_scaled($v * inv_scale) as u64) << $sh
            };
        }
        let packed_u64 = enc_nib!(v0, 0)
            | enc_nib!(v1, 4)
            | enc_nib!(v2, 8)
            | enc_nib!(v3, 12)
            | enc_nib!(v4, 16)
            | enc_nib!(v5, 20)
            | enc_nib!(v6, 24)
            | enc_nib!(v7, 28)
            | enc_nib!(v8, 32)
            | enc_nib!(v9, 36)
            | enc_nib!(v10, 40)
            | enc_nib!(v11, 44)
            | enc_nib!(v12, 48)
            | enc_nib!(v13, 52)
            | enc_nib!(v14, 56)
            | enc_nib!(v15, 60);

        let b_u64_ptr = b_up_vals.as_mut_ptr().cast::<u64>();
        let mut sfb_tensor = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
        unsafe {
            TiledMma128::store_routed_token_group_k(
                b_u64_ptr,
                &mut sfb_tensor,
                gp128,
                grp,
                packed_u64,
                scale_code,
                HIDDEN_192E / 512,
            );
        }
    }

    /// Stage 3 (`K3`): `5`-Warp (`160`-thread) Persistent K-Streaming `tcgen05` `FC1 + ReLU² + NVFP4 Requantize`
    /// (`I = 1,536, H = 6,144`, `12` bursts of `K = 512`) using `PipelineTmaAsync`, `Sm100BlockScaledTiledMma`,
    /// and `Sm100TmemStagedEpilogue`.
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 148480,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc1_relu2_tcgen05_192e_cute(
        #[grid_constant] w_up_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_up_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_up_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_up_tma: &TensorMap<u16, ScaleTmaSmem>,
        w_up_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_count: &[i32],
        mut inter_vals: DisjointSlice<u8>,
        mut inter_scales: DisjointSlice<u8>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m_tiles = INTERMEDIATE_192E / 128; // 12
        let tile_m128 = cta_id % m_tiles;
        let worker_id = cta_id / m_tiles;
        let active_groups = unsafe { *group_count.as_ptr() } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let base_g = active_groups / num_workers;
        let rem_g = active_groups % num_workers;
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let k_bursts = HIDDEN_192E / 512; // 12
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = dyn_smem;
        let smem_b_ptr = unsafe { dyn_smem.add(65_536) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(131_072) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(139_264) };
        let pipe_bar_ptr = unsafe { dyn_smem.add(147_456).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        let w_up_global_t = StridedTensor::from_slice(w_up_global);

        unsafe {
            let pipeline = Fc1Pipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<3>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1_256(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer = expert * m_tiles + tile_m128;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            let wait_parity = (((step >> 1) & 1) ^ 1) as u32;
                            TiledMma128::wait_barrier(b_free_bar_ptr.add(slot), wait_parity);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                            base: smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            capacity: ACT128_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        copy_tma_2d(w_up_tma, (w_outer * k_bursts + burst, 0), &mut a_tile, bar);
                        copy_tma_2d(b_up_tma, (group * k_bursts + burst, 0), &mut b_tile, bar);
                        copy_tma_2d(
                            sfa_up_tma,
                            (w_outer * k_bursts + burst, 0),
                            &mut sfa_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfb_up_tma,
                            (group * k_bursts + burst, 0),
                            &mut sfb_tile,
                            bar,
                        );
                        prod.advance();
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
                pipeline.producer_tail(prod);
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let epilogue = StagedEpilogue128::from_raw(smem_a_ptr.cast::<f32>());
                let inter_u64_ptr = inter_vals.as_mut_ptr().cast::<u64>();
                let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut inter_scales);
                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let tiled_mma = TiledMma128::new_with_lbo(valid_n as u32, 2048);
                    let global_scale = w_up_global_t.get_unchecked(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            pipeline.consumer_wait(&cons);
                            let slot = cons.index();
                            let sfa_tmem_b = tmem_addr + 128 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 192 + (slot as u32) * 32;
                            let a_tile = SharedWeightTile::from_raw_bytes(
                                smem_a_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let b_tile = SharedActTile::from_raw_bytes(
                                smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            );
                            let sfa_tile = SharedWeightScales::from_raw(
                                smem_sfa_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfb_tile = SharedActScales::from_raw(
                                smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            tiled_mma.copy_scales_and_gemm_burst_128(
                                tmem_addr,
                                sfa_tmem_b,
                                sfb_tmem_b,
                                &a_tile,
                                &b_tile,
                                &sfa_tile,
                                &sfb_tile,
                                burst == 0,
                            );
                            if burst + 2 < k_bursts {
                                TiledMma128::commit_barrier(b_free_bar_ptr.add(slot));
                            }
                            if burst + 1 == k_bursts {
                                TiledMma128::commit_barrier(mma_bar_ptr);
                                TiledMma128::wait_barrier(mma_bar_const, (g_idx & 1) as u32);
                            }
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    epilogue.unload_relu2_and_quantize_active_128(
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        tile_m128,
                        group,
                        global_scale,
                        INTERMEDIATE_192E / 512,
                        inter_u64_ptr,
                        &mut scales_out,
                        b_free_bar_ptr,
                        group + 1 < group_end,
                    );
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1_256(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4a (`K4` for `B <= 16`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` with Fused Top-4 `BF16x2` Atomic Combine (`limit <= 16`).
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc2_m256_atomic_192e_cute(
        #[grid_constant] w_down_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_down_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_routes: &[i32],
        group_count: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m256_tiles = HIDDEN_192E / 256; // 24
        let tile_m256 = cta_id % m256_tiles;
        let worker_id = cta_id / m256_tiles;
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let active_groups = unsafe { *group_count.as_ptr() } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let base_g = active_groups / num_workers;
        let rem_g = active_groups % num_workers;
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a0_ptr = dyn_smem;
        let smem_a1_ptr = unsafe { dyn_smem.add(65_536) };
        let smem_b_ptr = unsafe { dyn_smem.add(131_072) };
        let smem_sfa0_ptr = unsafe { dyn_smem.add(196_608) };
        let smem_sfa1_ptr = unsafe { dyn_smem.add(204_800) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(212_992) };
        let pipe_bar_ptr = unsafe { dyn_smem.add(221_184).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        unsafe {
            let pipeline = Fc2M256Pipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<3>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            let wait_parity = (((step >> 1) & 1) ^ 1) as u32;
                            TiledMma128::wait_barrier(b_free_bar_ptr.add(slot), wait_parity);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                            base: smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            capacity: ACT128_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut a0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut a1_tile,
                            bar,
                        );
                        copy_tma_2d(
                            b_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut b_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut sfa0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut sfa1_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfb_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut sfb_tile,
                            bar,
                        );
                        prod.advance();
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
                pipeline.producer_tail(prod);
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let epi0 = StagedEpilogue128::from_raw(smem_a0_ptr.cast::<f32>());
                let epi1 = StagedEpilogue128::from_raw(smem_a1_ptr.cast::<f32>());
                let tiled_mma = TiledMma128::new_with_lbo(16, 2048);
                let out_u32_ptr = output.as_mut_ptr().cast::<u32>();
                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let global_scale = w_down_global_t.get_unchecked(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            pipeline.consumer_wait(&cons);
                            let slot = cons.index();
                            let sfa0_tmem_b = tmem_addr + 256 + (slot as u32) * 32;
                            let sfa1_tmem_b = tmem_addr + 320 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 384 + (slot as u32) * 32;
                            let a0_tile = SharedWeightTile::from_raw_bytes(
                                smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let a1_tile = SharedWeightTile::from_raw_bytes(
                                smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let b_tile = SharedActTile::from_raw_bytes(
                                smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            );
                            let sfa0_tile = SharedWeightScales::from_raw(
                                smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfa1_tile = SharedWeightScales::from_raw(
                                smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfb_tile = SharedActScales::from_raw(
                                smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            tiled_mma.copy_scales_and_gemm_dual_m_burst(
                                tmem_addr,
                                sfa0_tmem_b,
                                sfa1_tmem_b,
                                sfb_tmem_b,
                                &a0_tile,
                                &a1_tile,
                                &b_tile,
                                &sfa0_tile,
                                &sfa1_tile,
                                &sfb_tile,
                                burst == 0,
                            );
                            if burst + 2 < k_bursts {
                                TiledMma128::commit_barrier(b_free_bar_ptr.add(slot));
                            }
                            if burst + 1 == k_bursts {
                                TiledMma128::commit_barrier(mma_bar_ptr);
                                TiledMma128::wait_barrier(mma_bar_const, (g_idx & 1) as u32);
                            }
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    epi0.unload_dual_m_and_atomic_add_bf16x2_16(
                        &epi1,
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        tile_m0,
                        tile_m1,
                        group,
                        GROUP_TILE_ROWS,
                        global_scale,
                        group_routes.as_ptr(),
                        routing_weights.as_ptr(),
                        out_u32_ptr,
                        HIDDEN_192E,
                        b_free_bar_ptr,
                        group + 1 < group_end,
                    );
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4b (`K4` for `B = 32, 64, 128`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` writing `grouped_out` (`BFloat16`) for deterministic `FP32` combine.
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc2_m256_192e_cute(
        #[grid_constant] w_down_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_down_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_count: &[i32],
        mut grouped_out: DisjointSlice<BFloat16>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m256_tiles = HIDDEN_192E / 256; // 24
        let tile_m256 = cta_id % m256_tiles;
        let worker_id = cta_id / m256_tiles;
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let active_groups = unsafe { *group_count.as_ptr() } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let base_g = active_groups / num_workers;
        let rem_g = active_groups % num_workers;
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a0_ptr = dyn_smem;
        let smem_a1_ptr = unsafe { dyn_smem.add(65_536) };
        let smem_b_ptr = unsafe { dyn_smem.add(131_072) };
        let smem_sfa0_ptr = unsafe { dyn_smem.add(196_608) };
        let smem_sfa1_ptr = unsafe { dyn_smem.add(204_800) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(212_992) };
        let pipe_bar_ptr = unsafe { dyn_smem.add(221_184).cast::<Barrier>() };
        let b_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        unsafe {
            let pipeline = Fc2M256Pipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<3>(b_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            let wait_parity = (((step >> 1) & 1) ^ 1) as u32;
                            TiledMma128::wait_barrier(b_free_bar_ptr.add(slot), wait_parity);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                            base: smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            capacity: ACT128_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut a0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut a1_tile,
                            bar,
                        );
                        copy_tma_2d(
                            b_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut b_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut sfa0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut sfa1_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfb_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut sfb_tile,
                            bar,
                        );
                        prod.advance();
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
                pipeline.producer_tail(prod);
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let epi0 = StagedEpilogue128::from_raw(smem_a0_ptr.cast::<f32>());
                let epi1 = StagedEpilogue128::from_raw(smem_a1_ptr.cast::<f32>());
                let grouped_u128_ptr = grouped_out.as_mut_ptr().cast::<u128>();
                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let tiled_mma = TiledMma128::new_with_lbo(valid_n as u32, 2048);
                    let global_scale = w_down_global_t.get_unchecked(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            pipeline.consumer_wait(&cons);
                            let slot = cons.index();
                            let sfa0_tmem_b = tmem_addr + 256 + (slot as u32) * 32;
                            let sfa1_tmem_b = tmem_addr + 320 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 384 + (slot as u32) * 32;
                            let a0_tile = SharedWeightTile::from_raw_bytes(
                                smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let a1_tile = SharedWeightTile::from_raw_bytes(
                                smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let b_tile = SharedActTile::from_raw_bytes(
                                smem_b_ptr.add(slot * ACT128_BURST_BYTES),
                            );
                            let sfa0_tile = SharedWeightScales::from_raw(
                                smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfa1_tile = SharedWeightScales::from_raw(
                                smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfb_tile = SharedActScales::from_raw(
                                smem_sfb_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            tiled_mma.copy_scales_and_gemm_dual_m_burst(
                                tmem_addr,
                                sfa0_tmem_b,
                                sfa1_tmem_b,
                                sfb_tmem_b,
                                &a0_tile,
                                &a1_tile,
                                &b_tile,
                                &sfa0_tile,
                                &sfa1_tile,
                                &sfb_tile,
                                burst == 0,
                            );
                            if burst + 2 < k_bursts {
                                TiledMma128::commit_barrier(b_free_bar_ptr.add(slot));
                            }
                            if burst + 1 == k_bursts {
                                TiledMma128::commit_barrier(mma_bar_ptr);
                                TiledMma128::wait_barrier(mma_bar_const, (g_idx & 1) as u32);
                            }
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    epi0.unload_dual_m_to_grouped_bf16_128(
                        &epi1,
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        tile_m0,
                        tile_m1,
                        group,
                        GROUP_TILE_ROWS,
                        global_scale,
                        grouped_u128_ptr,
                        HIDDEN_192E,
                        b_free_bar_ptr,
                        group + 1 < group_end,
                    );
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4c (`K4` for `B = 256`): `5`-Warp (`160`-thread) Triple-M-Tile (`M = 384` = `3 × 128`)
    /// K-Streaming `tcgen05` `FC2` (`16` M384-tiles × `8` groups = `128 <= 148` CTAs in 1 single wave!).
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn moe_fc2_m384_192e_cute(
        #[grid_constant] w_down_tma: &TensorMap<u8, WeightTileSmem>,
        #[grid_constant] b_down_tma: &TensorMap<u8, ActTile128Smem>,
        #[grid_constant] sfa_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        #[grid_constant] sfb_down_tma: &TensorMap<u16, ScaleTmaSmem>,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_count: &[i32],
        mut grouped_out: DisjointSlice<BFloat16>,
        group_step: u32,
    ) {
        /// Typed 2-stage TMA load pipeline for `FC2` Triple-M (`147,456` bytes per stage).
        /// Coordinates Warp 4 (Producer) and Warp 0 (MMA Consumer).
        type Fc2M384Pipeline = PipelineTmaAsync<2, 1, FC2_M384_STAGE_TX_BYTES>;

        let cta_id = thread::blockIdx_x() as usize;
        let m384_tiles = HIDDEN_192E / 384; // 16
        let tile_m384 = cta_id % m384_tiles;
        let worker_id = cta_id / m384_tiles;
        let tile_m0 = tile_m384 * 3;
        let tile_m1 = tile_m0 + 1;
        let tile_m2 = tile_m0 + 2;
        let active_groups = unsafe { *group_count.as_ptr() } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let base_g = active_groups / num_workers;
        let rem_g = active_groups % num_workers;
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a0_ptr = dyn_smem; // 0..64KB (2 stages)
        let smem_a1_ptr = unsafe { dyn_smem.add(65_536) }; // 64..128KB (2 stages)
        let smem_a2_ptr = unsafe { dyn_smem.add(131_072) }; // 128..160KB (1 stage)
        let smem_b_ptr = unsafe { dyn_smem.add(163_840) }; // 160..192KB (1 stage)
        let smem_sfa0_ptr = unsafe { dyn_smem.add(196_608) }; // 192..200KB (2 stages)
        let smem_sfa1_ptr = unsafe { dyn_smem.add(204_800) }; // 200..208KB (2 stages)
        let smem_sfa2_ptr = unsafe { dyn_smem.add(212_992) }; // 208..212KB (1 stage)
        let smem_sfb_ptr = unsafe { dyn_smem.add(217_088) }; // 212..216KB (1 stage)
        let pipe_bar_ptr = unsafe { dyn_smem.add(221_184).cast::<Barrier>() };
        let a_free_bar_ptr = unsafe { pipe_bar_ptr.add(4) };
        let b_free_bar_ptr = unsafe { a_free_bar_ptr.add(2) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(1) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        unsafe {
            let pipeline = Fc2M384Pipeline::from_raw_base(pipe_bar_ptr);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                TiledMma128::init_barriers::<4>(a_free_bar_ptr);
            }
            if warp_id == 0 {
                TiledMma128::alloc_tmem_cg1(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let w_outer2 = expert * m128_tiles + tile_m2;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            let wait_parity = (((step >> 1) & 1) ^ 1) as u32;
                            TiledMma128::wait_barrier(a_free_bar_ptr.add(slot), wait_parity);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut a0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut a1_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer0 * k_bursts + burst, 0),
                            &mut sfa0_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer1 * k_bursts + burst, 0),
                            &mut sfa1_tile,
                            bar,
                        );
                        if step >= 1 {
                            TiledMma128::wait_barrier(
                                b_free_bar_const,
                                ((step - 1) & 1) as u32,
                            );
                        }
                        let mut a2_tile = SmemTile::<u8, WeightTileSmem> {
                            base: smem_a2_ptr,
                            capacity: WEIGHT_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut b_tile = SmemTile::<u8, ActTile128Smem> {
                            base: smem_b_ptr,
                            capacity: ACT128_BURST_BYTES,
                            layout: PhantomData,
                        };
                        let mut sfa2_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfa2_ptr.cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem> {
                            base: smem_sfb_ptr.cast::<u16>(),
                            capacity: SCALE_BURST_BYTES / 2,
                            layout: PhantomData,
                        };
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer2 * k_bursts + burst, 0),
                            &mut a2_tile,
                            bar,
                        );
                        copy_tma_2d(
                            b_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut b_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfa_down_tma,
                            (w_outer2 * k_bursts + burst, 0),
                            &mut sfa2_tile,
                            bar,
                        );
                        copy_tma_2d(
                            sfb_down_tma,
                            (group * k_bursts + burst, 0),
                            &mut sfb_tile,
                            bar,
                        );
                        prod.advance();
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
                pipeline.producer_tail(prod);
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let epi0 = StagedEpilogue128::from_raw(smem_a0_ptr.cast::<f32>());
                let epi1 = StagedEpilogue128::from_raw(smem_a1_ptr.cast::<f32>());
                let epi2 = StagedEpilogue128::from_raw(smem_a2_ptr.cast::<f32>());
                let a2_tile = SharedWeightTile::from_raw_bytes(smem_a2_ptr);
                let b_tile = SharedActTile::from_raw_bytes(smem_b_ptr);
                let sfa2_tile = SharedWeightScales::from_raw(smem_sfa2_ptr.cast::<u32>());
                let sfb_tile = SharedActScales::from_raw(smem_sfb_ptr.cast::<u32>());
                let grouped_u128_ptr = grouped_out.as_mut_ptr().cast::<u128>();
                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let tiled_mma = TiledMma128::new_with_lbo(valid_n as u32, 2048);
                    let global_scale = w_down_global_t.get_unchecked(expert);

                    if tid == 0 {
                        let sfa0_tmem_b = tmem_addr + 384;
                        let sfa1_tmem_b = tmem_addr + 416;
                        let sfa2_tmem_b = tmem_addr + 448;
                        let sfb_tmem_b = tmem_addr + 480;
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            pipeline.consumer_wait(&cons);
                            let slot = cons.index();
                            let a0_tile = SharedWeightTile::from_raw_bytes(
                                smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let a1_tile = SharedWeightTile::from_raw_bytes(
                                smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            );
                            let sfa0_tile = SharedWeightScales::from_raw(
                                smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            let sfa1_tile = SharedWeightScales::from_raw(
                                smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES).cast::<u32>(),
                            );
                            tiled_mma.copy_scales_and_gemm_triple_m_burst(
                                tmem_addr,
                                sfa0_tmem_b,
                                sfa1_tmem_b,
                                sfa2_tmem_b,
                                sfb_tmem_b,
                                &a0_tile,
                                &a1_tile,
                                &a2_tile,
                                &b_tile,
                                &sfa0_tile,
                                &sfa1_tile,
                                &sfa2_tile,
                                &sfb_tile,
                                burst == 0,
                            );
                            if burst + 1 < k_bursts {
                                TiledMma128::commit_barrier(b_free_bar_ptr);
                            }
                            if burst == 0 {
                                TiledMma128::commit_barrier(a_free_bar_ptr.add(slot));
                            }
                            if burst + 1 == k_bursts {
                                TiledMma128::commit_barrier(mma_bar_ptr);
                                TiledMma128::wait_barrier(mma_bar_const, (g_idx & 1) as u32);
                            }
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    epi0.unload_triple_m_to_grouped_bf16_128(
                        &epi1,
                        &epi2,
                        tmem_addr,
                        warp_id,
                        tid,
                        limit,
                        tile_m0,
                        tile_m1,
                        tile_m2,
                        group,
                        GROUP_TILE_ROWS,
                        global_scale,
                        grouped_u128_ptr,
                        HIDDEN_192E,
                        a_free_bar_ptr,
                        b_free_bar_ptr,
                        group + 1 < group_end,
                    );
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TiledMma128::dealloc_tmem_cg1(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 5 (`K5` for `B >= 32`): Coalesced `u128` (`8 × bf16`) top-4 `FP32` weighted combine (`[B, 6,144]`).
    #[kernel]
    pub fn moe_finalize_192e_cute(
        grouped_out: &[BFloat16],
        tok_to_perm: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        num_tokens: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let h_u128 = HIDDEN_192E / 8; // 768
        let total_u128 = (num_tokens as usize) * h_u128;
        if idx >= total_u128 {
            return;
        }
        let token = idx / h_u128;
        let col_u128 = idx % h_u128;
        unsafe {
            let perm_u128 = StridedTensor::<u128>::from_raw_parts(
                tok_to_perm.as_ptr().cast::<u128>(),
                num_tokens as usize,
                1,
            );
            let rw_u64 = StridedTensor::<u64>::from_raw_parts(
                routing_weights.as_ptr().cast::<u64>(),
                num_tokens as usize,
                1,
            );
            let grouped_u128 = StridedTensor::<u128>::from_raw_parts(
                grouped_out.as_ptr().cast::<u128>(),
                grouped_out.len() / 8,
                1,
            );
            let mut out_u128 = StridedTensorMut::<u128>::from_raw_parts(
                output.as_mut_ptr().cast::<u128>(),
                total_u128,
                1,
            );

            let gp128 = perm_u128.get_unchecked(token);
            let w64 = rw_u64.get_unchecked(token);

            let decode_row = |gp: u32| -> usize {
                let g = (gp >> 8) as usize;
                let r = (gp & 255) as usize;
                (g * GROUP_TILE_ROWS + r) * h_u128 + col_u128
            };
            let r0 = decode_row((gp128 & 0xFFFF_FFFF) as u32);
            let r1 = decode_row(((gp128 >> 32) & 0xFFFF_FFFF) as u32);
            let r2 = decode_row(((gp128 >> 64) & 0xFFFF_FFFF) as u32);
            let r3 = decode_row((gp128 >> 96) as u32);

            let s0 = f32::from_bits(((w64 & 0xFFFF) as u32) << 16);
            let s1 = f32::from_bits((((w64 >> 16) & 0xFFFF) as u32) << 16);
            let s2 = f32::from_bits((((w64 >> 32) & 0xFFFF) as u32) << 16);
            let s3 = f32::from_bits((((w64 >> 48) & 0xFFFF) as u32) << 16);

            let q0 = grouped_u128.get_unchecked(r0);
            let q1 = grouped_u128.get_unchecked(r1);
            let q2 = grouped_u128.get_unchecked(r2);
            let q3 = grouped_u128.get_unchecked(r3);

            let mut a0 = 0.0f32;
            let mut a1 = 0.0f32;
            let mut a2 = 0.0f32;
            let mut a3 = 0.0f32;
            let mut a4 = 0.0f32;
            let mut a5 = 0.0f32;
            let mut a6 = 0.0f32;
            let mut a7 = 0.0f32;
            macro_rules! acc_slot {
                ($q:ident, $w:ident) => {{
                    let u0 = ($q & 0xFFFF_FFFF) as u32;
                    let u1 = (($q >> 32) & 0xFFFF_FFFF) as u32;
                    let u2 = (($q >> 64) & 0xFFFF_FFFF) as u32;
                    let u3 = ($q >> 96) as u32;
                    a0 += f32::from_bits(u0 << 16) * $w;
                    a1 += f32::from_bits(u0 & 0xFFFF_0000) * $w;
                    a2 += f32::from_bits(u1 << 16) * $w;
                    a3 += f32::from_bits(u1 & 0xFFFF_0000) * $w;
                    a4 += f32::from_bits(u2 << 16) * $w;
                    a5 += f32::from_bits(u2 & 0xFFFF_0000) * $w;
                    a6 += f32::from_bits(u3 << 16) * $w;
                    a7 += f32::from_bits(u3 & 0xFFFF_0000) * $w;
                }};
            }
            acc_slot!(q0, s0);
            acc_slot!(q1, s1);
            acc_slot!(q2, s2);
            acc_slot!(q3, s3);

            let pack_pair = |a: f32, b: f32| -> u32 {
                let l = BFloat16::from_f32(a).to_bits() as u32;
                let h = BFloat16::from_f32(b).to_bits() as u32;
                l | (h << 16)
            };
            let p0 = pack_pair(a0, a1) as u128;
            let p1 = pack_pair(a2, a3) as u128;
            let p2 = pack_pair(a4, a5) as u128;
            let p3 = pack_pair(a6, a7) as u128;
            out_u128.store_at(idx, p0 | (p1 << 32) | (p2 << 64) | (p3 << 96));
        }
    }
}

/// Device buffers and typed TMA descriptors for the 192-expert `cute-rs` MoE pipeline.
struct CuteMoe192Buffers {
    /// Input `FP16` activations stored as raw 16-bit IEEE 754 `binary16` words (`[num_tokens, 6144]`).
    hidden: DeviceBuffer<u16>,
    /// Per-token top-4 routed expert IDs (`[num_tokens * 4]`, in `0..192`).
    expert_ids: DeviceBuffer<i32>,
    /// Per-token top-4 `BFloat16` routing weights (`[num_tokens * 4]`).
    routing_weights: DeviceBuffer<BFloat16>,
    /// GPU-computed expert ID per active group (`[max_groups]`).
    group_experts: DeviceBuffer<i32>,
    /// GPU-computed valid token count per active group (`[max_groups]`).
    group_limits: DeviceBuffer<i32>,
    /// GPU-computed route index (`token * 4 + slot`) per group row (`[max_groups * 128]`).
    group_routes: DeviceBuffer<i32>,
    /// GPU-computed `(group << 8) | row` permutation entry per route (`[num_tokens * 4]`).
    tok_to_perm: DeviceBuffer<i32>,
    /// GPU-computed total active group count (`[1]`).
    group_count: DeviceBuffer<i32>,
    /// Routed & quantized `FC1` input tokens (`[max_groups * 12, 128 * 256]` B).
    b_up_vals: DeviceBuffer<u8>,
    /// Routed `FC1` token block scales (`[max_groups * 96, 512]` B).
    sfb_up_scales: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate activations (`[max_groups * 3, 128 * 256]` B).
    inter_vals: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate scales (`[max_groups * 24, 512]` B).
    inter_scales: DeviceBuffer<u8>,
    /// Packed `FC1` expert weights `W_up` (`[192 * 1536, 3072]` B).
    w_up: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_up`.
    w_up_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization scale for `W_up` (`[192]`).
    w_up_global: DeviceBuffer<f32>,
    /// Packed `FC2` expert weights `W_down` (`[192 * 6144, 768]` B).
    w_down: DeviceBuffer<u8>,
    /// Canonical `(32, 4, 4)` `Float8E4M3FN` block scales for `W_down`.
    w_down_scales: DeviceBuffer<u8>,
    /// Per-expert global FP32 dequantization scale for `W_down` (`[192]`).
    w_down_global: DeviceBuffer<f32>,
    /// Typed 2D TMA descriptor for `W_up` (`TensorMap<u8, WeightTileSmem>`).
    w_up_tma: TensorMap<u8, WeightTileSmem>,
    /// Typed 2D TMA descriptor for `b_up_vals` (`TensorMap<u8, ActTile128Smem>`).
    b_up_tma: TensorMap<u8, ActTile128Smem>,
    /// Typed 2D TMA descriptor for `w_up_scales` (`TensorMap<u16, ScaleTmaSmem>`).
    sfa_up_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Typed 2D TMA descriptor for `sfb_up_scales` (`TensorMap<u16, ScaleTmaSmem>`).
    sfb_up_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Typed 2D TMA descriptor for `W_down` (`TensorMap<u8, WeightTileSmem>`).
    w_down_tma: TensorMap<u8, WeightTileSmem>,
    /// Typed 2D TMA descriptor for `inter_vals` (`TensorMap<u8, ActTile128Smem>`).
    b_down_tma: TensorMap<u8, ActTile128Smem>,
    /// Typed 2D TMA descriptor for `w_down_scales` (`TensorMap<u16, ScaleTmaSmem>`).
    sfa_down_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Typed 2D TMA descriptor for `inter_scales` (`TensorMap<u16, ScaleTmaSmem>`).
    sfb_down_tma: TensorMap<u16, ScaleTmaSmem>,
    /// Grouped `BFloat16` expert outputs (`[max_groups * 128, 6144]`).
    grouped_out: DeviceBuffer<BFloat16>,
    /// Final `BFloat16` layer output (`[num_tokens, 6144]`).
    output: DeviceBuffer<BFloat16>,
    /// Token batch size `B`.
    num_tokens: usize,
    /// Tuned logical group capacity (`16, 32, 64, 128`).
    tile_n: u32,
}

/// Convert `f32` to IEEE 754 `binary16` (`FP16`) 16-bit pattern on host (round-to-nearest-even).
fn host_f32_to_fp16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | ((mant != 0) as u16);
    }
    let h_exp = exp - 127 + 15;
    if h_exp >= 31 {
        return sign | 0x7C00;
    }
    if h_exp <= 0 {
        if h_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - h_exp) as u32;
        let half_bit = 1u32 << (shift - 1);
        let lsb = (m >> shift) & 1;
        let rounded = (m + half_bit - 1 + lsb) >> shift;
        return sign | (rounded as u16);
    }
    let lsb = (mant >> 13) & 1;
    let rounded = mant + 0x0FFF + lsb;
    if rounded >= 0x0080_0000 {
        if h_exp + 1 >= 31 {
            sign | 0x7C00
        } else {
            sign | (((h_exp + 1) as u16) << 10)
        }
    } else {
        sign | ((h_exp as u16) << 10) | ((rounded >> 13) as u16)
    }
}

/// Convert IEEE 754 `binary16` (`FP16`) 16-bit pattern to `f32` on host.
fn host_fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mag = (mant as f32) * 2.0f32.powi(-24);
            if sign != 0 { -mag } else { mag }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Select the group capacity `tile_n` (`16, 32, 64, 128`) for a given token batch size `num_tokens`.
fn select_tile_n_192e(num_tokens: usize) -> u32 {
    if num_tokens <= 16 {
        16
    } else if num_tokens <= 32 {
        32
    } else if num_tokens <= 64 {
        64
    } else {
        128
    }
}

/// Upper bound on the number of non-empty expert groups across `NUM_EXPERTS_192E` experts
/// for `routed_rows = num_tokens * TOP_K_192E` and group capacity `tile_n`.
fn max_groups_192e(num_tokens: usize, tile_n: usize) -> usize {
    let routed = num_tokens * TOP_K_192E;
    let filled = NUM_EXPERTS_192E.min(routed);
    filled + (routed - filled) / tile_n
}

/// Allocate device buffers and typed TMA descriptors for the 192-expert `cute-rs` MoE layer.
fn allocate_192e(
    stream: &Arc<CudaStream>,
    num_tokens: usize,
) -> Result<CuteMoe192Buffers, Box<dyn std::error::Error>> {
    let tile_n = select_tile_n_192e(num_tokens);
    let max_groups = max_groups_192e(num_tokens, tile_n as usize);
    let routed_rows = num_tokens * TOP_K_192E;

    let hidden_host: Vec<u16> = (0..num_tokens)
        .flat_map(|t| {
            std::iter::repeat_n(host_f32_to_fp16_bits(crate::hidden_value(t)), HIDDEN_192E)
        })
        .collect();
    let expert_ids_host: Vec<i32> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| crate::expert_id(t, s) as i32))
        .collect();
    let routing_host: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| BFloat16::from_f32(crate::routing_weight(t, s))))
        .collect();

    let w_up_bytes = NUM_EXPERTS_192E * INTERMEDIATE_192E * (HIDDEN_192E / 2);
    let w_up_scale_bytes =
        NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * (HIDDEN_192E / 64) * 512;
    let w_down_bytes = NUM_EXPERTS_192E * HIDDEN_192E * (INTERMEDIATE_192E / 2);
    let w_down_scale_bytes =
        NUM_EXPERTS_192E * (HIDDEN_192E / 128) * (INTERMEDIATE_192E / 64) * 512;

    let mut w_up_global_host = vec![1.0f32 / HIDDEN_192E as f32; NUM_EXPERTS_192E];
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE_192E as f32; NUM_EXPERTS_192E];
    for e in 0..TOP_K_192E {
        w_up_global_host[e] = UP_FACTORS_192E[e] / HIDDEN_192E as f32;
        w_down_global_host[e] = DOWN_FACTORS_192E[e] / INTERMEDIATE_192E as f32;
    }

    let w_up = DeviceBuffer::from_host(stream, &vec![0x22u8; w_up_bytes])?;
    let w_up_scales = DeviceBuffer::from_host(stream, &vec![0x38u8; w_up_scale_bytes])?;
    let w_down = DeviceBuffer::from_host(stream, &vec![0x22u8; w_down_bytes])?;
    let w_down_scales = DeviceBuffer::from_host(stream, &vec![0x38u8; w_down_scale_bytes])?;

    let k_bursts_up = HIDDEN_192E / 512; // 12
    let k_bursts_down = INTERMEDIATE_192E / 512; // 3
    let b_up_vals = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_up * ACT128_BURST_BYTES],
    )?;
    let sfb_up_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_up * SCALE_BURST_BYTES],
    )?;
    let inter_vals = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_down * ACT128_BURST_BYTES],
    )?;
    let inter_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_down * SCALE_BURST_BYTES],
    )?;

    let w_up_tma = create_tensor_map_tiled_with_options::<u8, WeightTileSmem>(
        w_up.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS_192E * INTERMEDIATE_192E * k_bursts_up) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_up_tma = create_tensor_map_tiled_with_options::<u8, ActTile128Smem>(
        b_up_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (max_groups * k_bursts_up * GROUP_TILE_ROWS) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfa_up_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        w_up_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * k_bursts_up * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_up_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        sfb_up_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (max_groups * k_bursts_up * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let w_down_tma = create_tensor_map_tiled_with_options::<u8, WeightTileSmem>(
        w_down.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS_192E * HIDDEN_192E * k_bursts_down) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let b_down_tma = create_tensor_map_tiled_with_options::<u8, ActTile128Smem>(
        inter_vals.cu_deviceptr() as *mut core::ffi::c_void,
        (max_groups * k_bursts_down * GROUP_TILE_ROWS) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfa_down_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        w_down_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (NUM_EXPERTS_192E * (HIDDEN_192E / 128) * k_bursts_down * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_down_tma = create_tensor_map_tiled_with_options::<u16, ScaleTmaSmem>(
        inter_scales.cu_deviceptr() as *mut core::ffi::c_void,
        (max_groups * k_bursts_down * 8) as u64,
        256,
        256,
        TmaEncodeOptions::L2_128B,
    )?;

    Ok(CuteMoe192Buffers {
        hidden: DeviceBuffer::from_host(stream, &hidden_host)?,
        expert_ids: DeviceBuffer::from_host(stream, &expert_ids_host)?,
        routing_weights: DeviceBuffer::from_host(stream, &routing_host)?,
        group_experts: DeviceBuffer::from_host(stream, &vec![0i32; max_groups])?,
        group_limits: DeviceBuffer::from_host(stream, &vec![0i32; max_groups])?,
        group_routes: DeviceBuffer::from_host(stream, &vec![0i32; max_groups * GROUP_TILE_ROWS])?,
        tok_to_perm: DeviceBuffer::from_host(stream, &vec![0i32; routed_rows])?,
        group_count: DeviceBuffer::from_host(stream, &[0i32])?,
        b_up_vals,
        sfb_up_scales,
        inter_vals,
        inter_scales,
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
        grouped_out: DeviceBuffer::from_host(
            stream,
            &vec![BFloat16::from_f32(0.0); max_groups * GROUP_TILE_ROWS * HIDDEN_192E],
        )?,
        output: DeviceBuffer::from_host(
            stream,
            &vec![BFloat16::from_f32(0.0); num_tokens * HIDDEN_192E],
        )?,
        num_tokens,
        tile_n,
    })
}

/// Enqueue the complete 192-expert `cute-rs` MoE pipeline on `stream`.
/// Used both for eager numerical validation and for CUDA Graph capture.
fn launch_pipeline_192e(
    module: &kernels_192e::LoadedModule,
    stream: &Arc<CudaStream>,
    bufs: &mut CuteMoe192Buffers,
) -> Result<(), Box<dyn std::error::Error>> {
    let fuse_route = u32::from(bufs.num_tokens <= 16);
    let fuse_finalize = u32::from(bufs.num_tokens <= 16);
    if fuse_route == 0 {
        let routed_rows = (bufs.num_tokens * TOP_K_192E) as u32;
        let route_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 9472,
        };
        unsafe {
            module.moe_route_192e_cute(
                stream,
                route_cfg,
                &bufs.expert_ids,
                &mut bufs.group_experts,
                &mut bufs.group_limits,
                &mut bufs.group_routes,
                &mut bufs.tok_to_perm,
                &mut bufs.group_count,
                routed_rows,
                bufs.tile_n,
                bufs.tile_n.trailing_zeros(),
            )?;
        }
    }

    let q1_groups = (bufs.num_tokens * (HIDDEN_192E / 16)) as u32;
    let quant_cfg = LaunchConfig {
        grid_dim: (q1_groups.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: if fuse_route != 0 { 768 } else { 0 },
    };
    unsafe {
        module.quantize_fp16_192e_cute(
            stream,
            quant_cfg,
            &bufs.hidden,
            &bufs.expert_ids,
            &mut bufs.group_experts,
            &mut bufs.group_limits,
            &mut bufs.group_routes,
            &mut bufs.group_count,
            &mut bufs.tok_to_perm,
            &mut bufs.b_up_vals,
            &mut bufs.sfb_up_scales,
            &mut bufs.output,
            bufs.num_tokens as u32,
            fuse_route,
            fuse_finalize,
        )?;
    }

    let max_groups = max_groups_192e(bufs.num_tokens, bufs.tile_n as usize);
    let fc1_m_tiles = INTERMEDIATE_192E / 128; // 12
    let fc1_group_workers = max_groups.min(12);
    let fc1_ctas = (fc1_m_tiles * fc1_group_workers) as u32;
    let fc1_cfg = LaunchConfig1D::new(fc1_ctas, 160, DYNAMIC_SMEM_FC1_192E);
    let fc1_launch = module.prepare_moe_fc1_relu2_tcgen05_192e_cute(fc1_cfg)?;
    unsafe {
        module.moe_fc1_relu2_tcgen05_192e_cute(
            stream,
            &fc1_launch,
            bufs.w_up_tma,
            bufs.b_up_tma,
            bufs.sfa_up_tma,
            bufs.sfb_up_tma,
            &bufs.w_up_global,
            &bufs.group_experts,
            &bufs.group_limits,
            &bufs.group_count,
            &mut bufs.inter_vals,
            &mut bufs.inter_scales,
            fc1_group_workers as u32,
        )?;
    }

    if bufs.num_tokens >= 256 {
        let fc2_m_tiles = HIDDEN_192E / 384; // 16
        let fc2_group_workers = max_groups.min(9);
        let fc2_ctas = (fc2_m_tiles * fc2_group_workers) as u32;
        let fc2_cfg = LaunchConfig1D::new(fc2_ctas, 160, DYNAMIC_SMEM_FC2_192E);
        let fc2_launch = module.prepare_moe_fc2_m384_192e_cute(fc2_cfg)?;
        unsafe {
            module.moe_fc2_m384_192e_cute(
                stream,
                &fc2_launch,
                bufs.w_down_tma,
                bufs.b_down_tma,
                bufs.sfa_down_tma,
                bufs.sfb_down_tma,
                &bufs.w_down_global,
                &bufs.group_experts,
                &bufs.group_limits,
                &bufs.group_count,
                &mut bufs.grouped_out,
                fc2_group_workers as u32,
            )?;
        }
    } else if fuse_finalize != 0 {
        let fc2_m_tiles = HIDDEN_192E / 256; // 24
        let fc2_group_workers = max_groups.min(6);
        let fc2_ctas = (fc2_m_tiles * fc2_group_workers) as u32;
        let fc2_cfg = LaunchConfig1D::new(fc2_ctas, 160, DYNAMIC_SMEM_FC2_192E);
        let fc2_launch = module.prepare_moe_fc2_m256_atomic_192e_cute(fc2_cfg)?;
        unsafe {
            module.moe_fc2_m256_atomic_192e_cute(
                stream,
                &fc2_launch,
                bufs.w_down_tma,
                bufs.b_down_tma,
                bufs.sfa_down_tma,
                bufs.sfb_down_tma,
                &bufs.w_down_global,
                &bufs.group_experts,
                &bufs.group_limits,
                &bufs.group_routes,
                &bufs.group_count,
                &bufs.routing_weights,
                &mut bufs.output,
                fc2_group_workers as u32,
            )?;
        }
    } else {
        let fc2_m_tiles = HIDDEN_192E / 256; // 24
        let fc2_group_workers = max_groups.min(6);
        let fc2_ctas = (fc2_m_tiles * fc2_group_workers) as u32;
        let fc2_cfg = LaunchConfig1D::new(fc2_ctas, 160, DYNAMIC_SMEM_FC2_192E);
        let fc2_launch = module.prepare_moe_fc2_m256_192e_cute(fc2_cfg)?;
        unsafe {
            module.moe_fc2_m256_192e_cute(
                stream,
                &fc2_launch,
                bufs.w_down_tma,
                bufs.b_down_tma,
                bufs.sfa_down_tma,
                bufs.sfb_down_tma,
                &bufs.w_down_global,
                &bufs.group_experts,
                &bufs.group_limits,
                &bufs.group_count,
                &mut bufs.grouped_out,
                fc2_group_workers as u32,
            )?;
        }
    }

    if fuse_finalize == 0 {
        let total_u128 = (bufs.num_tokens * (HIDDEN_192E / 8)) as u32;
        let fin_cfg = LaunchConfig {
            grid_dim: (total_u128.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.moe_finalize_192e_cute(
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

/// Verify `bufs.output` against `expected` (`normalized_max_error <= 0.01`, zero non-finites).
fn check_output_192e(
    stream: &Arc<CudaStream>,
    bufs: &CuteMoe192Buffers,
    expected: &[f32],
) -> Result<f32, Box<dyn std::error::Error>> {
    stream.synchronize()?;
    let actual_vals = bufs.output.to_host_vec(stream)?;
    let mut max_abs = 0.0f32;
    let mut max_ref = 1.0f32;
    for (idx, (&val, &exp)) in actual_vals.iter().zip(expected.iter()).enumerate() {
        let act = val.to_f32();
        if !act.is_finite() || !exp.is_finite() {
            return Err(format!("non-finite element at {idx}: actual={act}, expected={exp}").into());
        }
        max_abs = max_abs.max((act - exp).abs());
        max_ref = max_ref.max(exp.abs());
    }
    let norm_err = max_abs / max_ref;
    if norm_err > 0.01 {
        return Err(
            format!("output check failed: norm_err={norm_err:.6}, max_abs={max_abs:.6}").into(),
        );
    }
    Ok(norm_err)
}

/// Host reference helper: decode one `Float8E4M3FN` byte to `f32`.
fn host_e4m3_to_f32(code: u8) -> f32 {
    if code == 0 || code == 0x80 {
        return 0.0;
    }
    let sign = if (code & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let mant = (code & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2.0f32.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// Host reference helper: encode one non-negative `f32` to `Float8E4M3FN` (round-to-nearest-even).
fn host_f32_to_e4m3(v: f32) -> u8 {
    let c = v.clamp(0.0, 448.0);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..=0x7E {
        let cand = host_e4m3_to_f32(code);
        let err = (c - cand).abs();
        if err < best_err || (err == best_err && (code & 1) == 0) {
            best_err = err;
            best = code;
        }
    }
    best
}

/// Host reference helper: decode one 4-bit `Float4E2M1FN` nibble (`0..15`) to `f32`.
fn host_e2m1_to_f32(nib: u8) -> f32 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mag = LUT[(nib & 7) as usize];
    if (nib & 8) != 0 { -mag } else { mag }
}

/// Host reference helper: encode one `f32` to 4-bit `Float4E2M1FN` (round-to-nearest-even).
fn host_f32_to_e2m1(v: f32) -> u8 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if v < 0.0 { 8u8 } else { 0u8 };
    let a = v.abs().min(6.0);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (idx, &cand) in LUT.iter().enumerate() {
        let err = (a - cand).abs();
        if err < best_err || (err == best_err && (idx & 1) == 0) {
            best_err = err;
            best = idx as u8;
        }
    }
    sign | best
}

/// Exact CPU reference for the uniform 192-expert fixture at token `t`.
fn uniform_expected_row_192e(t: usize) -> f32 {
    let x = host_fp16_bits_to_f32(host_f32_to_fp16_bits(crate::hidden_value(t)));
    let sc = host_f32_to_e4m3(x.abs() / 6.0);
    let s = host_e4m3_to_f32(sc);
    let q = if s == 0.0 {
        0.0
    } else {
        host_e2m1_to_f32(host_f32_to_e2m1(x / s)) * s
    };

    let mut total = 0.0f32;
    for slot in 0..TOP_K_192E {
        let expert = crate::expert_id(t, slot);
        let rw = BFloat16::from_f32(crate::routing_weight(t, slot)).to_f32();
        let fc1 = q * UP_FACTORS_192E[expert];
        let u = fc1.max(0.0);
        let relu2 = u * u;
        let isc = host_f32_to_e4m3(relu2 / 6.0);
        let is = host_e4m3_to_f32(isc);
        let iq = if is == 0.0 {
            0.0
        } else {
            host_e2m1_to_f32(host_f32_to_e2m1(relu2 / is)) * is
        };
        let fc2 = BFloat16::from_f32(iq * DOWN_FACTORS_192E[expert]).to_f32();
        total += rw * fc2;
    }
    BFloat16::from_f32(total).to_f32()
}

/// Run the `nonuniform` and `changed-routes-after-capture` validation suite at `num_tokens`.
fn validate_192e(
    module: &kernels_192e::LoadedModule,
    stream: &Arc<CudaStream>,
    num_tokens: usize,
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let mut bufs = allocate_192e(stream, num_tokens)?;

    let w_up_bytes = NUM_EXPERTS_192E * INTERMEDIATE_192E * (HIDDEN_192E / 2);
    let w_up_scale_bytes =
        NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * (HIDDEN_192E / 64) * 512;
    let w_down_bytes = NUM_EXPERTS_192E * HIDDEN_192E * (INTERMEDIATE_192E / 2);
    let w_down_scale_bytes =
        NUM_EXPERTS_192E * (HIDDEN_192E / 128) * (INTERMEDIATE_192E / 64) * 512;

    const NIB_PAIR: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
    const SCALE_CODES: [u8; 3] = [0x34, 0x38, 0x3C];

    let mut w_up_host = vec![0x22u8; w_up_bytes];
    let mut w_up_scales_host = vec![0x38u8; w_up_scale_bytes];
    let mut w_down_host = vec![0x22u8; w_down_bytes];
    let mut w_down_scales_host = vec![0x38u8; w_down_scale_bytes];
    let mut w_up_global_host = vec![1.0f32 / HIDDEN_192E as f32; NUM_EXPERTS_192E];
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE_192E as f32; NUM_EXPERTS_192E];

    let active_test_experts = [0usize, 1, 2, 3, 4, 5, 190, 191];
    for &e in &active_test_experts {
        w_up_global_host[e] = (0.65 + 0.1 * ((e % 5) as f32)) / HIDDEN_192E as f32;
        w_down_global_host[e] = (0.80 - 0.12 * ((e % 4) as f32)) / INTERMEDIATE_192E as f32;

        let up_k_chunks = HIDDEN_192E / 64; // 96
        let up_m_tiles = INTERMEDIATE_192E / 128; // 12
        for mt in 0..up_m_tiles {
            for kc in 0..up_k_chunks {
                let nib = NIB_PAIR[(e + mt + kc) & 3];
                let sc = SCALE_CODES[(e + mt + kc) % 3];
                let atom_idx = (e * up_m_tiles + mt) * up_k_chunks + kc;
                w_up_scales_host[atom_idx * 512..(atom_idx + 1) * 512].fill(sc);
                let burst = kc >> 3;
                let s_in_b = kc & 7;
                let sub_base =
                    (((e * up_m_tiles + mt) * (HIDDEN_192E / 512) + burst) * 8 + s_in_b) * 4096;
                w_up_host[sub_base..sub_base + 4096].fill(nib);
            }
        }

        let down_k_chunks = INTERMEDIATE_192E / 64; // 24
        let down_m_tiles = HIDDEN_192E / 128; // 48
        for mt in 0..down_m_tiles {
            for kc in 0..down_k_chunks {
                let nib = NIB_PAIR[(e + mt * 3 + kc) & 3];
                let sc = SCALE_CODES[(e + mt + kc * 2) % 3];
                let atom_idx = (e * down_m_tiles + mt) * down_k_chunks + kc;
                w_down_scales_host[atom_idx * 512..(atom_idx + 1) * 512].fill(sc);
                let burst = kc >> 3;
                let s_in_b = kc & 7;
                let sub_base =
                    (((e * down_m_tiles + mt) * (INTERMEDIATE_192E / 512) + burst) * 8 + s_in_b)
                        * 4096;
                w_down_host[sub_base..sub_base + 4096].fill(nib);
            }
        }
    }

    bufs.w_up.copy_from_host(stream, &w_up_host)?;
    bufs.w_up_scales.copy_from_host(stream, &w_up_scales_host)?;
    bufs.w_down.copy_from_host(stream, &w_down_host)?;
    bufs.w_down_scales
        .copy_from_host(stream, &w_down_scales_host)?;
    bufs.w_up_global.copy_from_host(stream, &w_up_global_host)?;
    bufs.w_down_global
        .copy_from_host(stream, &w_down_global_host)?;

    let eval_cpu_reference = |hidden_h: &[u16],
                              exp_h: &[i32],
                              rw_h: &[BFloat16]|
     -> Vec<f32> {
        let mut out = vec![0.0f32; num_tokens * HIDDEN_192E];
        for t in 0..num_tokens {
            let mut q_in = [0.0f32; HIDDEN_192E / 64];
            for kc in 0..(HIDDEN_192E / 64) {
                let x = host_fp16_bits_to_f32(hidden_h[t * HIDDEN_192E + kc * 64]);
                let sc = host_f32_to_e4m3(x.abs() / 6.0);
                let s = host_e4m3_to_f32(sc);
                q_in[kc] = if s == 0.0 {
                    0.0
                } else {
                    host_e2m1_to_f32(host_f32_to_e2m1(x / s)) * s
                };
            }
            for slot in 0..TOP_K_192E {
                let e = exp_h[t * TOP_K_192E + slot] as usize;
                let rw = rw_h[t * TOP_K_192E + slot].to_f32();
                let up_g = w_up_global_host[e];
                let down_g = w_down_global_host[e];
                let mut q_inter = [0.0f32; INTERMEDIATE_192E / 128];
                for mt in 0..(INTERMEDIATE_192E / 128) {
                    let mut acc = 0.0f32;
                    for kc in 0..(HIDDEN_192E / 64) {
                        let w_val = host_e2m1_to_f32(NIB_PAIR[(e + mt + kc) & 3] & 0x0F);
                        let w_sc = host_e4m3_to_f32(SCALE_CODES[(e + mt + kc) % 3]);
                        acc += 64.0 * q_in[kc] * w_val * w_sc;
                    }
                    let u = (acc * up_g).max(0.0);
                    let relu2 = u * u;
                    let isc = host_f32_to_e4m3(relu2 / 6.0);
                    let is = host_e4m3_to_f32(isc);
                    q_inter[mt] = if is == 0.0 {
                        0.0
                    } else {
                        host_e2m1_to_f32(host_f32_to_e2m1(relu2 / is)) * is
                    };
                }
                for mt in 0..(HIDDEN_192E / 128) {
                    let mut acc = 0.0f32;
                    for kc in 0..(INTERMEDIATE_192E / 64) {
                        let w_val = host_e2m1_to_f32(NIB_PAIR[(e + mt * 3 + kc) & 3] & 0x0F);
                        let w_sc = host_e4m3_to_f32(SCALE_CODES[(e + mt + kc * 2) % 3]);
                        acc += 64.0 * q_inter[kc >> 1] * w_val * w_sc;
                    }
                    let fc2_bf16 = BFloat16::from_f32(acc * down_g).to_f32();
                    for c in 0..128 {
                        out[t * HIDDEN_192E + mt * 128 + c] += rw * fc2_bf16;
                    }
                }
            }
            for c in 0..HIDDEN_192E {
                out[t * HIDDEN_192E + c] =
                    BFloat16::from_f32(out[t * HIDDEN_192E + c]).to_f32();
            }
        }
        out
    };

    let mut hidden_nu = vec![0u16; num_tokens * HIDDEN_192E];
    for t in 0..num_tokens {
        for kc in 0..(HIDDEN_192E / 64) {
            let v = crate::hidden_value(t) * (0.75 + 0.15 * (((t + kc) % 4) as f32));
            let base = t * HIDDEN_192E + kc * 64;
            hidden_nu[base..base + 64].fill(host_f32_to_fp16_bits(v));
        }
    }
    let exp_nu: Vec<i32> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| crate::expert_id(t, s) as i32))
        .collect();
    let rw_nu: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| BFloat16::from_f32(crate::routing_weight(t, s))))
        .collect();
    bufs.hidden.copy_from_host(stream, &hidden_nu)?;

    let graph = unsafe {
        sys::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        )
        .result()?;
        launch_pipeline_192e(module, stream, &mut bufs)?;
        let mut g = MaybeUninit::<sys::CUgraph>::uninit();
        sys::cuStreamEndCapture(stream.cu_stream(), g.as_mut_ptr()).result()?;
        g.assume_init()
    };
    let exec = unsafe {
        let mut e = MaybeUninit::<sys::CUgraphExec>::uninit();
        sys::cuGraphInstantiateWithFlags(e.as_mut_ptr(), graph, 0).result()?;
        let e = e.assume_init();
        sys::cuGraphUpload(e, stream.cu_stream()).result()?;
        e
    };

    for _ in 0..7 {
        unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
    }
    let expected_nu = eval_cpu_reference(&hidden_nu, &exp_nu, &rw_nu);
    let nu_err = check_output_192e(stream, &bufs, &expected_nu)?;

    const CHANGED_EXPERTS: [i32; 6] = [0, 2, 4, 5, 190, 191];
    let mut hidden_cr = vec![0u16; num_tokens * HIDDEN_192E];
    for t in 0..num_tokens {
        for kc in 0..(HIDDEN_192E / 64) {
            let v = crate::hidden_value(t + 5) * (0.85 + 0.1 * (((t + kc + 1) % 3) as f32));
            let base = t * HIDDEN_192E + kc * 64;
            hidden_cr[base..base + 64].fill(host_f32_to_fp16_bits(v));
        }
    }
    let exp_cr: Vec<i32> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| CHANGED_EXPERTS[(t + s) % 6]))
        .collect();
    let rw_cr: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|t| {
            (0..TOP_K_192E)
                .map(move |s| BFloat16::from_f32(crate::routing_weight(t + 3, (s + 1) & 3)))
        })
        .collect();

    bufs.hidden.copy_from_host(stream, &hidden_cr)?;
    bufs.expert_ids.copy_from_host(stream, &exp_cr)?;
    bufs.routing_weights.copy_from_host(stream, &rw_cr)?;
    for _ in 0..7 {
        unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
    }
    let expected_cr = eval_cpu_reference(&hidden_cr, &exp_cr, &rw_cr);
    let cr_err = check_output_192e(stream, &bufs, &expected_cr)?;

    unsafe {
        let _ = sys::cuGraphExecDestroy(exec);
        let _ = sys::cuGraphDestroy(graph);
    }
    Ok((nu_err, cr_err))
}

/// Run the full 192-expert (`H = 6,144, I = 1,536, E = 192`) `cute-rs` validation and 1,000-replay
/// CUDA Graph benchmark across `B = 1 .. 256`, printing the 5-way comparison table.
pub fn run_benchmark_192e(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Wide 192-Expert NVFP4 ReLU² MoE (cute-rs: H=6144, I=1536, E=192, top-k=4) ===");
    let module = unsafe { kernels_192e::load(ctx) }?;

    for &val_b in &[1usize, 8, 256] {
        let (nu_err, cr_err) = validate_192e(&module, stream, val_b)?;
        println!(
            "Validation B={val_b:>3}: nonuniform_err={nu_err:.6}, changed_routes_after_capture_err={cr_err:.6} (PASSED)"
        );
    }

    let mut results = Vec::with_capacity(BATCH_SIZES_192E.len());
    for &b in &BATCH_SIZES_192E {
        let mut bufs = allocate_192e(stream, b)?;
        let expected: Vec<f32> = (0..b)
            .flat_map(|t| std::iter::repeat_n(uniform_expected_row_192e(t), HIDDEN_192E))
            .collect();
        launch_pipeline_192e(&module, stream, &mut bufs)?;
        let err = check_output_192e(stream, &bufs, &expected)?;

        let graph = unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .result()?;
            launch_pipeline_192e(&module, stream, &mut bufs)?;
            let mut g = MaybeUninit::<sys::CUgraph>::uninit();
            sys::cuStreamEndCapture(stream.cu_stream(), g.as_mut_ptr()).result()?;
            g.assume_init()
        };
        let exec = unsafe {
            let mut e = MaybeUninit::<sys::CUgraphExec>::uninit();
            sys::cuGraphInstantiateWithFlags(e.as_mut_ptr(), graph, 0).result()?;
            let e = e.assume_init();
            sys::cuGraphUpload(e, stream.cu_stream()).result()?;
            e
        };
        stream.synchronize()?;

        let start = ctx.new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let stop = ctx.new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let mut samples = Vec::with_capacity(SAMPLES_192E);
        for _ in 0..SAMPLES_192E {
            for _ in 0..WARMUPS_192E {
                unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
            }
            start.record(stream)?;
            for _ in 0..REPLAYS_192E {
                unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
            }
            stop.record(stream)?;
            stop.synchronize()?;
            samples.push((start.elapsed_ms(&stop)? as f64) * 1000.0 / REPLAYS_192E as f64);
        }
        let _ = check_output_192e(stream, &bufs, &expected)?;
        unsafe {
            let _ = sys::cuGraphExecDestroy(exec);
            let _ = sys::cuGraphDestroy(graph);
        }
        let mean_us = samples.iter().sum::<f64>() / SAMPLES_192E as f64;
        println!("Batch {b:>3}: mean={mean_us:>7.3} us (norm_err={err:.6})");
        results.push((b, mean_us));
    }

    const TRTLLM_GEN_US: [f64; 9] = [
        24.570, 24.570, 24.583, 24.569, 24.704, 28.689, 31.209, 38.436, 51.196,
    ];
    const FLASHINFER_CUTEDSL_US: [f64; 9] = [
        24.591, 24.685, 24.991, 26.636, 28.660, 28.661, 28.761, 30.710, 34.800,
    ];
    const CUTILE_US: [f64; 9] = [
        22.547, 22.541, 22.541, 24.590, 26.614, 31.077, 30.502, 33.321, 49.252,
    ];
    const NATIVE_SIMT_REF_US: [f64; 9] = [
        22.505, 22.518, 22.517, 22.530, 23.540, 26.611, 30.456, 36.846, 45.062,
    ];

    println!(
        "\nBatch | TRTLLM-Gen (us) | FlashInfer CuTeDSL (us) | cuTile NVFP4 (us) | Native SIMT [ref] (us) | cute-rs [live] (us) | vs. TRTLLM-Gen"
    );
    println!(
        "------+-----------------+-------------------------+-------------------+------------------------+---------------------+---------------"
    );
    for (idx, &(b, cute_us)) in results.iter().enumerate() {
        let trt = TRTLLM_GEN_US[idx];
        let fi = FLASHINFER_CUTEDSL_US[idx];
        let ct = CUTILE_US[idx];
        let nat = NATIVE_SIMT_REF_US[idx];
        let speedup = trt / cute_us;
        println!(
            "{b:>5} | {trt:>15.3} | {fi:>23.3} | {ct:>17.3} | {nat:>22.3} | {cute_us:>19.3} | {speedup:>9.2}x"
        );
    }
    Ok(())
}

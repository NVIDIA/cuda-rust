/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Wide 192-Expert NVFP4 ReLU² MoE (`H = 6,144, I = 1,536, E = 192, top_k = 4, B = 1 .. 256`)
//! implemented with `cute-rs` `Tensor` / `RegisterTile` partitions, `cute-layout` algebra
//! (`sm100_noswizzle_burst_u64_crd2idx`, `sm1xx_block_scale_crd2idx`), typed TMA descriptors,
//! `PipelineTmaAsync`, `SingleCtaRing` (`PipelineUmmaAsync`), `TmemAllocator`, `TmemFragment`,
//! and `Sm100BlockScaledTiledMma::{copy, gemm}`.

use std::mem::MaybeUninit;
use std::sync::Arc;

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
use cute_rs::utils::tmem_allocator::TmemAllocator;
use cute_rs::{
    BFloat16, C, Consumer, Float4E2M1FN, Float4E2M1FNx8, Float8E4M3FN, Layout, Mkl, Nkl, OperandA,
    OperandB, PipelineState, PipelineTmaAsync, Producer, ReductionOp, RowMajor,
    Sm100BlockScaledTiledMma, Sm100ScaleTile, Sm100SharedTile, SmemTile, Tensor, TensorMap,
    TensorMut, TensorSSA, copy_tma_2d, crd2idx_2d, crd2idx_col_major_2d, crd2idx_row_major_2d,
    idx2crd_2d, idx2crd_col_major_2d, idx2crd_row_major_2d, size_in_bytes,
    sm1xx_block_scale_crd2idx, sm100_noswizzle_burst_u64_crd2idx,
};
use cutlass_primitives as prims;
use prims::memory::{DisjointSlice, DynamicSharedArray};
use prims::{cuda_module, kernel, launch_bounds, launch_contract, thread};

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

/// Byte size of one `ScaleTmaSmem` burst (`4,096` bytes).
/// Used to index 2-stage shared-memory scale buffers.
const SCALE_BURST_BYTES: usize = size_in_bytes::<u16, ScaleTmaSmem>() as usize;

/// Dynamic shared memory bytes per `FC1` CTA (`148,480` bytes = `145 KB`).
/// Holds `2 × 32 KB` `SMEM_A`, `2 × 32 KB` `SMEM_B`, `2 × 4 KB` `SMEM_SFA`, `2 × 4 KB` `SMEM_SFB`, and barriers.
const DYNAMIC_SMEM_FC1_192E: u32 = 148_480;

/// Dynamic shared memory bytes per `FC2` CTA (`222,208` bytes = `217 KB`).
/// Holds Dual-M (`M = 256`) or Triple-M (`M = 384`) staged weight, activation, and scale buffers.
const DYNAMIC_SMEM_FC2_192E: u32 = 222_208;

/// Compile-time layout and transaction byte invariants for `moe_192e`.
const _: () = {
    assert!(SCALE_BURST_BYTES == 4_096);
    assert!(FC1_STAGE_TX_BYTES == 73_728);
    assert!(FC2_M256_STAGE_TX_BYTES == 110_592);
    assert!(FC2_M384_STAGE_TX_BYTES == 147_456);
    assert!(DYNAMIC_SMEM_FC1_192E == 148_480);
    assert!(DYNAMIC_SMEM_FC2_192E == 222_208);
};

/// Synthetic `FC1` per-expert scale factors for the 4 active experts in the uniform benchmark.
/// Matches the `nemotron3_moe` reference fixture.
const UP_FACTORS_192E: [f32; TOP_K_192E] = [0.5, 1.0, 2.0, 1.0];

/// Synthetic `FC2` per-expert scale factors for the 4 active experts in the uniform benchmark.
/// Matches the `nemotron3_moe` reference fixture.
const DOWN_FACTORS_192E: [f32; TOP_K_192E] = [1.0, 2.0, 0.5, 1.0];

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
    /// Stages scales via `Sm100BlockScaledTiledMma::copy` and issues `K = 64` MMAs via `gemm`.
    type TiledMma128 = Sm100BlockScaledTiledMma<WeightTileSmem, ActTile128Smem, 128, 64>;

    /// Typed 2-stage TMA load pipeline for `FC1` (`73,728` bytes per stage).
    /// Coordinates Warp 4 (Producer) and Warp 0 (MMA Consumer).
    type Fc1Pipeline = PipelineTmaAsync<2, 1, FC1_STAGE_TX_BYTES>;

    /// Typed 2-stage TMA load pipeline for `FC2` Dual-M (`110,592` bytes per stage).
    /// Coordinates Warp 4 (Producer) and Warp 0 (MMA Consumer).
    type Fc2M256Pipeline = PipelineTmaAsync<2, 1, FC2_M256_STAGE_TX_BYTES>;

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

    /// Issue all 8 `K = 64` steps of one `K = 512` single-M-tile (`M = 128`) burst via
    /// `TiledMma128::copy` (`Cp4x32x128bOp`) and `TiledMma128::gemm` (`MmaMXF4NVF4Op`).
    #[inline(always)]
    unsafe fn issue_single_m_burst(
        tiled_mma: &TiledMma128,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a_tile: &SharedWeightTile,
        b_tile: &SharedActTile,
        sfa_tile: &SharedWeightScales,
        sfb_tile: &SharedActScales,
        first_burst: bool,
    ) {
        let mma = *tiled_mma;
        let a = *a_tile;
        let b = *b_tile;
        let sfa = *sfa_tile;
        let sfb = *sfb_tile;
        macro_rules! step_single {
            ($s:expr, $acc:expr) => {{
                let sa = sfa_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                TiledMma128::copy(sa, &sfa, $s);
                TiledMma128::copy(sb, &sfb, $s);
                mma.gemm(tmem_d, &a, &b, sa, sb, $s, $acc);
            }};
        }
        unsafe {
            step_single!(0, !first_burst);
            step_single!(1, true);
            step_single!(2, true);
            step_single!(3, true);
            step_single!(4, true);
            step_single!(5, true);
            step_single!(6, true);
            step_single!(7, true);
        }
    }

    /// Issue all 8 `K = 64` steps of one `K = 512` Dual-M-tile (`M = 256` = `2 × 128`) burst via
    /// `TiledMma128::copy` and `TiledMma128::gemm`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn issue_dual_m_burst(
        tiled_mma: &TiledMma128,
        tmem_d: u32,
        sfa0_tmem_base: u32,
        sfa1_tmem_base: u32,
        sfb_tmem_base: u32,
        a0_tile: &SharedWeightTile,
        a1_tile: &SharedWeightTile,
        b_tile: &SharedActTile,
        sfa0_tile: &SharedWeightScales,
        sfa1_tile: &SharedWeightScales,
        sfb_tile: &SharedActScales,
        first_burst: bool,
    ) {
        let mma = *tiled_mma;
        let a0 = *a0_tile;
        let a1 = *a1_tile;
        let b = *b_tile;
        let sfa0 = *sfa0_tile;
        let sfa1 = *sfa1_tile;
        let sfb = *sfb_tile;
        macro_rules! step_dual {
            ($s:expr, $acc:expr) => {{
                let sa0 = sfa0_tmem_base + ($s as u32) * 4;
                let sa1 = sfa1_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                TiledMma128::copy(sa0, &sfa0, $s);
                TiledMma128::copy(sa1, &sfa1, $s);
                TiledMma128::copy(sb, &sfb, $s);
                mma.gemm(tmem_d, &a0, &b, sa0, sb, $s, $acc);
                mma.gemm(tmem_d + 128, &a1, &b, sa1, sb, $s, $acc);
            }};
        }
        unsafe {
            step_dual!(0, !first_burst);
            step_dual!(1, true);
            step_dual!(2, true);
            step_dual!(3, true);
            step_dual!(4, true);
            step_dual!(5, true);
            step_dual!(6, true);
            step_dual!(7, true);
        }
    }

    /// Issue all 8 `K = 64` steps of one `K = 512` Triple-M-tile (`M = 384` = `3 × 128`) burst via
    /// `TiledMma128::copy` and `TiledMma128::gemm`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn issue_triple_m_burst(
        tiled_mma: &TiledMma128,
        tmem_d: u32,
        sfa0_tmem_base: u32,
        sfa1_tmem_base: u32,
        sfa2_tmem_base: u32,
        sfb_tmem_base: u32,
        a0_tile: &SharedWeightTile,
        a1_tile: &SharedWeightTile,
        a2_tile: &SharedWeightTile,
        b_tile: &SharedActTile,
        sfa0_tile: &SharedWeightScales,
        sfa1_tile: &SharedWeightScales,
        sfa2_tile: &SharedWeightScales,
        sfb_tile: &SharedActScales,
        first_burst: bool,
    ) {
        let mma = *tiled_mma;
        let a0 = *a0_tile;
        let a1 = *a1_tile;
        let a2 = *a2_tile;
        let b = *b_tile;
        let sfa0 = *sfa0_tile;
        let sfa1 = *sfa1_tile;
        let sfa2 = *sfa2_tile;
        let sfb = *sfb_tile;
        macro_rules! step_triple {
            ($s:expr, $acc:expr) => {{
                let sa0 = sfa0_tmem_base + ($s as u32) * 4;
                let sa1 = sfa1_tmem_base + ($s as u32) * 4;
                let sa2 = sfa2_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                TiledMma128::copy(sa0, &sfa0, $s);
                TiledMma128::copy(sa1, &sfa1, $s);
                TiledMma128::copy(sa2, &sfa2, $s);
                TiledMma128::copy(sb, &sfb, $s);
                mma.gemm(tmem_d, &a0, &b, sa0, sb, $s, $acc);
                mma.gemm(tmem_d + 128, &a1, &b, sa1, sb, $s, $acc);
                mma.gemm(tmem_d + 256, &a2, &b, sa2, sb, $s, $acc);
            }};
        }
        unsafe {
            step_triple!(0, !first_burst);
            step_triple!(1, true);
            step_triple!(2, true);
            step_triple!(3, true);
            step_triple!(4, true);
            step_triple!(5, true);
            step_triple!(6, true);
            step_triple!(7, true);
        }
    }

    /// Stage 1 (`K1` for `B >= 32`): Single-CTA (`256`-thread) shared-memory route histogram,
    /// parallel 8-warp chunked scan, and group/permutation table builder using `cute-layout` algebra.
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
        let _ = (tile_n, tile_n_log2);
        let tid = thread::threadIdx_x() as usize;
        let (lane32, warp32) = idx2crd_col_major_2d(tid as u32, 32);
        let (warp_id, lane) = (warp32 as usize, lane32 as usize);
        let routes = routed_rows as usize;
        let dyn_smem = DynamicSharedArray::<u8, 16>::get_raw();

        let exp_in = StridedTensor::from_slice(expert_ids);
        let mut group_experts_out = StridedTensorMut::from_disjoint_slice(&mut group_experts);
        let mut group_limits_out = StridedTensorMut::from_disjoint_slice(&mut group_limits);
        let mut group_routes_out = StridedTensorMut::from_disjoint_slice(&mut group_routes);
        let mut tok_to_perm_out = StridedTensorMut::from_disjoint_slice(&mut tok_to_perm);
        let mut group_count_out = StridedTensorMut::from_disjoint_slice(&mut group_count);

        unsafe {
            let mut s_groups = StridedTensorMut::<u32>::from_shared_bytes(dyn_smem, 0, 256);
            let mut s_offsets = StridedTensorMut::<u32>::from_shared_bytes(dyn_smem, 1024, 256);
            let mut s_ranks = StridedTensorMut::<u16>::from_shared_bytes(dyn_smem, 2048, 1024);
            let mut s_exp = StridedTensorMut::<u16>::from_shared_bytes(dyn_smem, 4096, 1024);
            let mut s_sub_counts =
                StridedTensorMut::<u16>::from_shared_bytes(dyn_smem, 6144, 8 * NUM_EXPERTS_192E);
            let mut s_active = StridedTensorMut::<u8>::from_shared_bytes(dyn_smem, 9216, 256);

            s_active.store_at(tid, 0);
            NamedBarrier::<1, 256>::sync();

            let mut r = tid;
            while r < routes {
                let e = exp_in.get_unchecked(r) as usize;
                s_exp.store_at(r, e as u16);
                s_active.store_at(e, 1);
                r += 256;
            }
            NamedBarrier::<1, 256>::sync();

            let s_exp_ro = StridedTensor::<u16>::from_shared_bytes(dyn_smem, 4096, 1024);
            let s_active_ro = StridedTensor::<u8>::from_shared_bytes(dyn_smem, 9216, 256);
            let chunk_len = routes / 8;
            let r_start = warp_id * chunk_len;
            let r_end = if warp_id == 7 {
                routes
            } else {
                r_start + chunk_len
            };

            let mut e_idx = lane;
            while e_idx < NUM_EXPERTS_192E {
                let mut local_c = 0u16;
                if s_active_ro.get_unchecked(e_idx) != 0 {
                    let target_exp = e_idx as u16;
                    let mut i = r_start;
                    while i < r_end {
                        if s_exp_ro.get_unchecked(i) == target_exp {
                            s_ranks.store_at(i, local_c);
                            local_c += 1;
                        }
                        i += 1;
                    }
                }
                s_sub_counts.store_at(crd2idx_2d::<RowMajor<8, 192>>(warp_id, e_idx), local_c);
                e_idx += 32;
            }
            NamedBarrier::<1, 256>::sync();

            let s_sub_ro =
                StridedTensor::<u16>::from_shared_bytes(dyn_smem, 6144, 8 * NUM_EXPERTS_192E);
            let mut count = 0u32;
            if tid < NUM_EXPERTS_192E && s_active_ro.get_unchecked(tid) != 0 {
                let mut w = 0usize;
                while w < 8 {
                    let idx_sub = crd2idx_2d::<RowMajor<8, 192>>(w, tid);
                    let wc = s_sub_ro.get_unchecked(idx_sub) as u32;
                    s_sub_counts.store_at(idx_sub, count as u16);
                    count += wc;
                    w += 1;
                }
            }
            let my_groups = count.div_ceil(128);
            s_groups.store_at(tid, my_groups);
            NamedBarrier::<1, 256>::sync();

            let s_groups_ro = StridedTensor::<u32>::from_shared_bytes(dyn_smem, 0, 256);
            if tid == NUM_EXPERTS_192E - 1 || (tid < NUM_EXPERTS_192E && my_groups > 0) {
                let mut excl_offset = 0u32;
                let mut j = 0usize;
                while j < tid {
                    excl_offset += s_groups_ro.get_unchecked(j);
                    j += 1;
                }
                s_offsets.store_at(tid, excl_offset);
                if tid == NUM_EXPERTS_192E - 1 {
                    group_count_out.store_at(0, (excl_offset + my_groups) as i32);
                }
                if my_groups > 0 {
                    let mut sg = 0u32;
                    while sg < my_groups {
                        let g_idx = (excl_offset + sg) as usize;
                        let rem = count - sg * 128;
                        let lim = rem.min(128);
                        group_experts_out.store_at(g_idx, tid as i32);
                        group_limits_out.store_at(g_idx, lim as i32);
                        sg += 1;
                    }
                }
            }
            NamedBarrier::<1, 256>::sync();

            let s_offsets_ro = StridedTensor::<u32>::from_shared_bytes(dyn_smem, 1024, 256);
            let s_ranks_ro = StridedTensor::<u16>::from_shared_bytes(dyn_smem, 2048, 1024);
            let s_sub_ro2 =
                StridedTensor::<u16>::from_shared_bytes(dyn_smem, 6144, 8 * NUM_EXPERTS_192E);
            let mut r2 = r_start + lane;
            while r2 < r_end {
                let exp = s_exp_ro.get_unchecked(r2) as usize;
                let sub_idx = crd2idx_2d::<RowMajor<8, 192>>(warp_id, exp);
                let rank = (s_ranks_ro.get_unchecked(r2) + s_sub_ro2.get_unchecked(sub_idx)) as u32;
                let (row, g_sub) = idx2crd_col_major_2d(rank, 128);
                let g = s_offsets_ro.get_unchecked(exp) + g_sub;
                tok_to_perm_out.store_at(r2, crd2idx_col_major_2d(row, g, 256) as i32);
                let route_dst = crd2idx_2d::<RowMajor<768, 128>>(g as usize, row as usize);
                group_routes_out.store_at(route_dst, r2 as i32);
                r2 += 32;
            }
        }
    }

    /// Stage 2 (`K2`): Vectorized `FP16 -> NVFP4` input activation quantizer using
    /// `Tensor::zipped_divide::<8>()` + `RegisterTile<f32, 8>` and `cute-layout`
    /// `sm100_noswizzle_burst_u64_crd2idx` / `sm1xx_block_scale_crd2idx` (`H = 6,144`, `12` bursts).
    #[launch_bounds(128)]
    #[kernel]
    pub fn quantize_fp16_192e_cute(
        hidden: &[f16],
        expert_ids: &[i32],
        mut group_experts: DisjointSlice<i32>,
        mut group_limits: DisjointSlice<i32>,
        mut group_routes: DisjointSlice<i32>,
        mut group_count: DisjointSlice<i32>,
        mut tok_to_perm: DisjointSlice<i32>,
        mut b_up_vals: DisjointSlice<u64>,
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
        let (row, grp) = idx2crd_2d::<RowMajor<256, 384>>(idx.min(total_groups));

        if zero_out != 0 && idx < total_groups {
            let zeros = cute_rs::RegisterTile::from_array([BFloat16::from_bits(0); 8]);
            unsafe {
                TensorMut::from_disjoint_slice(&mut output)
                    .zipped_divide::<8>()
                    .slice(idx * 2)
                    .store(zeros);
                TensorMut::from_disjoint_slice(&mut output)
                    .zipped_divide::<8>()
                    .slice(idx * 2 + 1)
                    .store(zeros);
            }
        }

        let dyn_smem = DynamicSharedArray::<u8, 16>::get_raw();
        let perms_4: [i32; 4] = if fuse_route != 0 {
            let routes = (num_tokens as usize) * TOP_K_192E;
            let exp_in = StridedTensor::from_slice(expert_ids);
            let mut group_experts_out = StridedTensorMut::from_disjoint_slice(&mut group_experts);
            let mut group_limits_out = StridedTensorMut::from_disjoint_slice(&mut group_limits);
            let mut group_routes_out = StridedTensorMut::from_disjoint_slice(&mut group_routes);
            let mut tok_to_perm_out = StridedTensorMut::from_disjoint_slice(&mut tok_to_perm);
            let mut group_count_out = StridedTensorMut::from_disjoint_slice(&mut group_count);
            unsafe {
                let mut s_ids = StridedTensorMut::<i32>::from_shared_bytes(dyn_smem, 0, 64);
                let mut s_first = StridedTensorMut::<i32>::from_shared_bytes(dyn_smem, 256, 64);
                let mut s_perm = StridedTensorMut::<i32>::from_shared_bytes(dyn_smem, 512, 64);
                if tid < routes {
                    s_ids.store_at(tid, exp_in.get_unchecked(tid));
                }
                NamedBarrier::<1, 128>::sync();

                let s_ids_ro = StridedTensor::<i32>::from_shared_bytes(dyn_smem, 0, 64);
                let mut my_exp = 0i32;
                let mut my_rank = 0i32;
                let mut my_count = 0i32;
                if tid < routes {
                    my_exp = s_ids_ro.get_unchecked(tid);
                    let mut j = 0usize;
                    while j < routes {
                        if s_ids_ro.get_unchecked(j) == my_exp {
                            my_count += 1;
                            if j < tid {
                                my_rank += 1;
                            }
                        }
                        j += 1;
                    }
                    s_first.store_at(tid, if my_rank == 0 { my_exp } else { i32::MAX });
                }
                NamedBarrier::<1, 128>::sync();

                let s_first_ro = StridedTensor::<i32>::from_shared_bytes(dyn_smem, 256, 64);
                if cta_id == 0 {
                    if tid < routes {
                        let mut my_group = 0i32;
                        let mut total_g = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = s_first_ro.get_unchecked(j);
                            if fj != i32::MAX {
                                total_g += 1;
                                if fj < my_exp {
                                    my_group += 1;
                                }
                            }
                            j += 1;
                        }
                        let perm =
                            crd2idx_col_major_2d(my_rank as u32, my_group as u32, 256) as i32;
                        s_perm.store_at(tid, perm);
                        tok_to_perm_out.store_at(tid, perm);
                        let route_dst =
                            crd2idx_2d::<RowMajor<768, 128>>(my_group as usize, my_rank as usize);
                        group_routes_out.store_at(route_dst, tid as i32);
                        if my_rank == 0 {
                            group_experts_out.store_at(my_group as usize, my_exp);
                            group_limits_out.store_at(my_group as usize, my_count);
                        }
                        if tid == 0 {
                            group_count_out.store_at(0, total_g);
                        }
                    }
                } else {
                    let r_start = row * 4;
                    if tid >= r_start && tid < r_start + 4 {
                        let mut my_group = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = s_first_ro.get_unchecked(j);
                            if fj < my_exp {
                                my_group += 1;
                            }
                            j += 1;
                        }
                        let perm =
                            crd2idx_col_major_2d(my_rank as u32, my_group as u32, 256) as i32;
                        s_perm.store_at(tid, perm);
                    }
                }
                NamedBarrier::<1, 128>::sync();
                let s_perm_ro = StridedTensor::<i32>::from_shared_bytes(dyn_smem, 512, 64);
                if idx < total_groups {
                    let base = row * 4;
                    [
                        s_perm_ro.get_unchecked(base),
                        s_perm_ro.get_unchecked(base + 1),
                        s_perm_ro.get_unchecked(base + 2),
                        s_perm_ro.get_unchecked(base + 3),
                    ]
                } else {
                    [0; 4]
                }
            }
        } else if idx < total_groups {
            let perm_slice = unsafe {
                core::slice::from_raw_parts(
                    tok_to_perm.as_mut_ptr().cast_const(),
                    tok_to_perm.len(),
                )
            };
            let perm_tile = Tensor::from_slice(perm_slice).local_tile::<4>(row);
            unsafe { perm_tile.load().into_array() }
        } else {
            [0; 4]
        };

        if idx >= total_groups {
            return;
        }

        let g_hidden = Tensor::from_slice(hidden).zipped_divide::<8>();
        let h0 = unsafe { g_hidden.slice(idx * 2).load() }.to::<f32>();
        let h1 = unsafe { g_hidden.slice(idx * 2 + 1).load() }.to::<f32>();
        let (q_vec, scale) = quantize_tile16_nvfp4(h0.concat(h1));
        let packed_u64 = q_vec.bitcast::<u64, 1>()[0];
        let scale_code = scale.to_bits();

        let k_bursts = HIDDEN_192E / 512; // 12
        let k_atoms = HIDDEN_192E / 64; // 96
        let mut b_u64_out = StridedTensorMut::from_disjoint_slice(&mut b_up_vals);
        let mut sfb_out = StridedTensorMut::from_disjoint_slice(&mut sfb_up_scales);
        macro_rules! scatter_slot {
            ($slot:expr) => {{
                let (r_u32, g_u32) = idx2crd_col_major_2d(perms_4[$slot] as u32, 256);
                let (g, r) = (g_u32 as usize, r_u32 as usize);
                let u64_idx = sm100_noswizzle_burst_u64_crd2idx::<128>(g, r, grp, k_bursts);
                let s_idx = sm1xx_block_scale_crd2idx(g, r, grp, 1, k_atoms);
                b_u64_out.store_at(u64_idx, packed_u64);
                sfb_out.store_at(s_idx, scale_code);
            }};
        }
        unsafe {
            scatter_slot!(0);
            scatter_slot!(1);
            scatter_slot!(2);
            scatter_slot!(3);
        }
    }

    /// Stage 3 (`K3`): `5`-Warp (`160`-thread) Persistent K-Streaming `tcgen05` `FC1 + ReLU² + NVFP4 Requantize`
    /// (`I = 1,536, H = 6,144`, `12` bursts of `K = 512`) using `PipelineTmaAsync`, `SingleCtaRing`,
    /// `TmemAllocator`, `TmemFragment`, and `Sm100BlockScaledTiledMma::{copy, gemm}`.
    ///
    /// # Safety
    ///
    /// Must be launched with `160` threads per block (`5` warps), at least `148,480` bytes of
    /// 1024-byte-aligned dynamic shared memory, and valid TMA descriptors and slices.
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
        mut inter_vals: DisjointSlice<u64>,
        mut inter_scales: DisjointSlice<u8>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m_tiles = INTERMEDIATE_192E / 128; // 12
        let (worker_id, tile_m128) = idx2crd_2d::<RowMajor<12, 12>>(cta_id);
        let group_count_t = StridedTensor::from_slice(group_count);
        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let w_up_global_t = StridedTensor::from_slice(w_up_global);

        let active_groups = unsafe { group_count_t.get_unchecked(0) } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let (base_g32, rem_g32) = idx2crd_row_major_2d(active_groups as u32, num_workers as u32);
        let (base_g, rem_g) = (base_g32 as usize, rem_g32 as usize);
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let k_bursts = HIDDEN_192E / 512; // 12
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 147_456 / 8 + 4) };
        let mma_done_ring = unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, 147_456 / 8 + 8) };
        let tmem_slot_ptr =
            unsafe { TmemAllocator::<256, 1>::from_raw_bytes(dyn_smem, 147_456 + 80) };

        unsafe {
            let pipeline = Fc1Pipeline::from_shared_bytes(dyn_smem, 147_456);
            if tid == 0 {
                w_up_tma.prefetch_descriptor();
                b_up_tma.prefetch_descriptor();
                sfa_up_tma.prefetch_descriptor();
                sfb_up_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<256, 1>::allocate(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let w_outer = expert * m_tiles + tile_m128;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            b_free_ring.wait(step - 2);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a_tile =
                            SmemTile::<u8, WeightTileSmem>::from_stage_bytes(dyn_smem, 0, slot);
                        let mut b_tile = SmemTile::<u8, ActTile128Smem>::from_stage_bytes(
                            dyn_smem, 65_536, slot,
                        );
                        let mut sfa_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 131_072, slot,
                        );
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 139_264, slot,
                        );
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
                let tmem_addr = TmemAllocator::<256, 1>::retrieve_ptr(tmem_slot_ptr);
                let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                let mut smem_epi =
                    StridedTensorMut::<f32>::from_shared_bytes(dyn_smem, 0, 128 * 128);
                let mut inter_u64_out = StridedTensorMut::from_disjoint_slice(&mut inter_vals);
                let mut scales_out = StridedTensorMut::from_disjoint_slice(&mut inter_scales);

                let (sub_tile16_u32, row_off16_u32) = idx2crd_col_major_2d(tid as u32, 8);
                let (sub_tile16, row_off16) = (sub_tile16_u32 as usize, row_off16_u32 as usize);
                type Fc1ColPermLayout = Layout<(C<4>, (C<4>, C<8>)), (C<1>, (C<32>, C<4>))>;
                let col_perm =
                    <Fc1ColPermLayout as cute_rs::static_layout::StaticLayout>::crd2idx(tid);
                let grp = tile_m128 * 8 + sub_tile16;
                let inter_bursts = INTERMEDIATE_192E / 512; // 3
                let inter_atoms = INTERMEDIATE_192E / 64; // 24

                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut b_free_step = 0u32;
                let mut g_idx = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let limit = group_limits_t.get_unchecked(group) as usize;
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
                            let a_tile = SharedWeightTile::from_stage_bytes(dyn_smem, 0, slot);
                            let b_tile = SharedActTile::from_stage_bytes(dyn_smem, 65_536, slot);
                            let sfa_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                131_072,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfb_tile = SharedActScales::from_stage_bytes(
                                dyn_smem,
                                139_264,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            issue_single_m_burst(
                                &tiled_mma,
                                tmem_addr,
                                sfa_tmem_b,
                                sfb_tmem_b,
                                &a_tile,
                                &b_tile,
                                &sfa_tile,
                                &sfb_tile,
                                burst == 0,
                            );
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            if burst + 2 < k_bursts {
                                b_free_ring.producer_commit(b_free_step + (burst as u32));
                            }
                            if burst + 1 == k_bursts {
                                mma_done_ring.producer_commit(g_idx);
                                mma_done_ring.wait(g_idx);
                            }
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let frag =
                            TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                .to_tensor_ssa();
                        let u = (frag * global_scale).fmax(0.0);
                        let act = u * u;
                        let mut c = 0usize;
                        while c < 16 {
                            let n = n_base + c;
                            if n < limit {
                                smem_epi.store_at(
                                    crd2idx_2d::<RowMajor<128, 128>>(n, col_perm),
                                    act[c],
                                );
                            }
                            c += 1;
                        }
                        n_base += 16;
                    }
                    fence_view_async_tmem_load();
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 128>::sync();

                    let smem_epi_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 0, 128 * 32);
                    let mut n = row_off16;
                    while n < limit {
                        let base_u128 = crd2idx_2d::<RowMajor<128, 32>>(n, sub_tile16);
                        let q0 = smem_epi_u128.load_f32x4(base_u128);
                        let q1 = smem_epi_u128.load_f32x4(base_u128 + 8);
                        let q2 = smem_epi_u128.load_f32x4(base_u128 + 16);
                        let q3 = smem_epi_u128.load_f32x4(base_u128 + 24);
                        let vec = q0.concat(q1).concat(q2.concat(q3));
                        let (q_vec, scale) = quantize_tile16_nvfp4(vec);
                        let u64_idx =
                            sm100_noswizzle_burst_u64_crd2idx::<128>(group, n, grp, inter_bursts);
                        let s_idx = sm1xx_block_scale_crd2idx(group, n, grp, 1, inter_atoms);
                        inter_u64_out.store_at(u64_idx, q_vec.bitcast::<u64, 1>()[0]);
                        scales_out.store_at(s_idx, scale.to_bits());
                        n += 16;
                    }
                    fence_view_async_shared();
                    NamedBarrier::<1, 128>::sync();
                    if group + 1 < group_end && tid == 0 {
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 2);
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 1);
                    }
                    b_free_step += k_bursts as u32;
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<256, 1>::free(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4a (`K4` for `B <= 16`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` with Fused Top-4 `BF16x2` Atomic Combine (`limit <= 16`).
    ///
    /// # Safety
    ///
    /// Must be launched with `160` threads per block (`5` warps), at least `222,208` bytes of
    /// 1024-byte-aligned dynamic shared memory, and valid TMA descriptors and slices.
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
        let (worker_id, tile_m256) = idx2crd_2d::<RowMajor<8, 24>>(cta_id);
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let group_count_t = StridedTensor::from_slice(group_count);
        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let group_routes_t = StridedTensor::from_slice(group_routes);
        let routing_weights_t = StridedTensor::from_slice(routing_weights);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        let active_groups = unsafe { group_count_t.get_unchecked(0) } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let (base_g32, rem_g32) = idx2crd_row_major_2d(active_groups as u32, num_workers as u32);
        let (base_g, rem_g) = (base_g32 as usize, rem_g32 as usize);
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 221_184 / 8 + 4) };
        let mma_done_ring = unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, 221_184 / 8 + 8) };
        let tmem_slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, 221_184 + 80) };

        unsafe {
            let pipeline = Fc2M256Pipeline::from_shared_bytes(dyn_smem, 221_184);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            b_free_ring.wait(step - 2);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile =
                            SmemTile::<u8, WeightTileSmem>::from_stage_bytes(dyn_smem, 0, slot);
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                            dyn_smem, 65_536, slot,
                        );
                        let mut b_tile = SmemTile::<u8, ActTile128Smem>::from_stage_bytes(
                            dyn_smem, 131_072, slot,
                        );
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 196_608, slot,
                        );
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 204_800, slot,
                        );
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 212_992, slot,
                        );
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
                        copy_tma_2d(b_down_tma, (group * k_bursts + burst, 0), &mut b_tile, bar);
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
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot_ptr);
                let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                let tiled_mma = TiledMma128::new_with_lbo(16, 2048);
                let mut smem_epi0 =
                    StridedTensorMut::<f32>::from_shared_bytes(dyn_smem, 0, 16 * 128);
                let mut smem_epi1 =
                    StridedTensorMut::<f32>::from_shared_bytes(dyn_smem, 65_536, 16 * 128);
                let (col16_u32, row8_u32) = idx2crd_col_major_2d(tid as u32, 16);
                let (col16, row8) = (col16_u32 as usize, row8_u32 as usize);
                let col_bf16_0 = tile_m0 * 128 + col16 * 8;
                let col_bf16_1 = tile_m1 * 128 + col16 * 8;
                let output_t = StridedTensorMut::from_disjoint_slice(&mut output);

                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut b_free_step = 0u32;
                let mut g_idx = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let limit = group_limits_t.get_unchecked(group) as usize;
                    let global_scale = w_down_global_t.get_unchecked(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            pipeline.consumer_wait(&cons);
                            let slot = cons.index();
                            let sfa0_tmem_b = tmem_addr + 256 + (slot as u32) * 32;
                            let sfa1_tmem_b = tmem_addr + 320 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 384 + (slot as u32) * 32;
                            let a0_tile = SharedWeightTile::from_stage_bytes(dyn_smem, 0, slot);
                            let a1_tile =
                                SharedWeightTile::from_stage_bytes(dyn_smem, 65_536, slot);
                            let b_tile = SharedActTile::from_stage_bytes(dyn_smem, 131_072, slot);
                            let sfa0_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                196_608,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfa1_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                204_800,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfb_tile = SharedActScales::from_stage_bytes(
                                dyn_smem,
                                212_992,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            issue_dual_m_burst(
                                &tiled_mma,
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
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            if burst + 2 < k_bursts {
                                b_free_ring.producer_commit(b_free_step + (burst as u32));
                            }
                            if burst + 1 == k_bursts {
                                mma_done_ring.producer_commit(g_idx);
                                mma_done_ring.wait(g_idx);
                            }
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let r0 = TmemFragment::<16>::load_deferred(tmem_row_base).to_tensor_ssa();
                    let r1 = TmemFragment::<16>::load_deferred(tmem_row_base + 128).to_tensor_ssa();
                    let mut c = 0usize;
                    while c < 16 {
                        if c < limit {
                            let idx_s = crd2idx_2d::<RowMajor<16, 128>>(c, tid);
                            smem_epi0.store_at(idx_s, r0[c]);
                            smem_epi1.store_at(idx_s, r1[c]);
                        }
                        c += 1;
                    }
                    fence_view_async_tmem_load();
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 128>::sync();

                    let smem_epi0_ro =
                        StridedTensor::<f32>::from_shared_bytes(dyn_smem, 0, 16 * 128);
                    let smem_epi1_ro =
                        StridedTensor::<f32>::from_shared_bytes(dyn_smem, 65_536, 16 * 128);

                    let mut n = row8;
                    while n < limit {
                        let route_idx = crd2idx_2d::<RowMajor<768, 128>>(group, n);
                        let route = group_routes_t.get_unchecked(route_idx) as usize;
                        let token = route / 4;
                        let sw = global_scale * routing_weights_t.get_unchecked(route).to_f32();
                        let base = crd2idx_2d::<RowMajor<16, 128>>(n, col16 * 8);
                        let sc0 = |i: usize| {
                            BFloat16::from_f32(smem_epi0_ro.get_unchecked(base + i) * sw)
                        };
                        let sc1 = |i: usize| {
                            BFloat16::from_f32(smem_epi1_ro.get_unchecked(base + i) * sw)
                        };
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_0,
                            HIDDEN_192E,
                            sc0(0),
                            sc0(1),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_0 + 2,
                            HIDDEN_192E,
                            sc0(2),
                            sc0(3),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_0 + 4,
                            HIDDEN_192E,
                            sc0(4),
                            sc0(5),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_0 + 6,
                            HIDDEN_192E,
                            sc0(6),
                            sc0(7),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_1,
                            HIDDEN_192E,
                            sc1(0),
                            sc1(1),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_1 + 2,
                            HIDDEN_192E,
                            sc1(2),
                            sc1(3),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_1 + 4,
                            HIDDEN_192E,
                            sc1(4),
                            sc1(5),
                        );
                        output_t.atomic_add_bf16x2_at_2d(
                            token,
                            col_bf16_1 + 6,
                            HIDDEN_192E,
                            sc1(6),
                            sc1(7),
                        );
                        n += 8;
                    }
                    fence_view_async_shared();
                    NamedBarrier::<1, 128>::sync();
                    if group + 1 < group_end && tid == 0 {
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 2);
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 1);
                    }
                    b_free_step += k_bursts as u32;
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4b (`K4` for `B = 32, 64, 128`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` writing `grouped_out` (`BFloat16`) via `Tensor` / `TensorMut` (`cute.copy`).
    ///
    /// # Safety
    ///
    /// Must be launched with `160` threads per block (`5` warps), at least `222,208` bytes of
    /// 1024-byte-aligned dynamic shared memory, and valid TMA descriptors and slices.
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
        let (worker_id, tile_m256) = idx2crd_2d::<RowMajor<8, 24>>(cta_id);
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let group_count_t = StridedTensor::from_slice(group_count);
        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        let active_groups = unsafe { group_count_t.get_unchecked(0) } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let (base_g32, rem_g32) = idx2crd_row_major_2d(active_groups as u32, num_workers as u32);
        let (base_g, rem_g) = (base_g32 as usize, rem_g32 as usize);
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let b_free_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 221_184 / 8 + 4) };
        let mma_done_ring = unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, 221_184 / 8 + 8) };
        let tmem_slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, 221_184 + 80) };

        unsafe {
            let pipeline = Fc2M256Pipeline::from_shared_bytes(dyn_smem, 221_184);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            b_free_ring.wait(step - 2);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile =
                            SmemTile::<u8, WeightTileSmem>::from_stage_bytes(dyn_smem, 0, slot);
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                            dyn_smem, 65_536, slot,
                        );
                        let mut b_tile = SmemTile::<u8, ActTile128Smem>::from_stage_bytes(
                            dyn_smem, 131_072, slot,
                        );
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 196_608, slot,
                        );
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 204_800, slot,
                        );
                        let mut sfb_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 212_992, slot,
                        );
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
                        copy_tma_2d(b_down_tma, (group * k_bursts + burst, 0), &mut b_tile, bar);
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
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot_ptr);
                let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                let mut smem_epi0 =
                    StridedTensorMut::<BFloat16>::from_shared_bytes(dyn_smem, 0, 128 * 128);
                let mut smem_epi1 =
                    StridedTensorMut::<BFloat16>::from_shared_bytes(dyn_smem, 65_536, 128 * 128);
                let (col16_u32, row8_u32) = idx2crd_col_major_2d(tid as u32, 16);
                let (col16, row8) = (col16_u32 as usize, row8_u32 as usize);
                let dst_col0_tile8 = tile_m0 * 16 + col16;
                let dst_col1_tile8 = tile_m1 * 16 + col16;
                let h_tile8 = HIDDEN_192E / 8;

                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut b_free_step = 0u32;
                let mut g_idx = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let limit = group_limits_t.get_unchecked(group) as usize;
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
                            let a0_tile = SharedWeightTile::from_stage_bytes(dyn_smem, 0, slot);
                            let a1_tile =
                                SharedWeightTile::from_stage_bytes(dyn_smem, 65_536, slot);
                            let b_tile = SharedActTile::from_stage_bytes(dyn_smem, 131_072, slot);
                            let sfa0_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                196_608,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfa1_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                204_800,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfb_tile = SharedActScales::from_stage_bytes(
                                dyn_smem,
                                212_992,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            issue_dual_m_burst(
                                &tiled_mma,
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
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            if burst + 2 < k_bursts {
                                b_free_ring.producer_commit(b_free_step + (burst as u32));
                            }
                            if burst + 1 == k_bursts {
                                mma_done_ring.producer_commit(g_idx);
                                mma_done_ring.wait(g_idx);
                            }
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let r0 =
                            (TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                .to_tensor_ssa()
                                * global_scale)
                                .to::<BFloat16>();
                        let r1 = (TmemFragment::<16>::load_deferred(
                            tmem_row_base + 128 + (n_base as u32),
                        )
                        .to_tensor_ssa()
                            * global_scale)
                            .to::<BFloat16>();
                        let mut c = 0usize;
                        while c < 16 {
                            let n = n_base + c;
                            if n < limit {
                                let idx_s = crd2idx_2d::<RowMajor<128, 128>>(n, tid);
                                smem_epi0.store_at(idx_s, r0[c]);
                                smem_epi1.store_at(idx_s, r1[c]);
                            }
                            c += 1;
                        }
                        n_base += 16;
                    }
                    fence_view_async_tmem_load();
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 128>::sync();

                    let smem_epi0_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 0, 128 * 16);
                    let smem_epi1_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 65_536, 128 * 16);
                    let mut out_u128 =
                        StridedTensorMut::<u128>::from_disjoint_slice_as(&mut grouped_out);
                    let mut n = row8;
                    while n < limit {
                        let smem_idx = crd2idx_2d::<RowMajor<128, 16>>(n, col16);
                        let q0 = smem_epi0_u128.get_unchecked(smem_idx);
                        let q1 = smem_epi1_u128.get_unchecked(smem_idx);
                        let perm_row = crd2idx_2d::<RowMajor<768, 128>>(group, n);
                        let dst0 = crd2idx_row_major_2d(perm_row, dst_col0_tile8, h_tile8);
                        let dst1 = crd2idx_row_major_2d(perm_row, dst_col1_tile8, h_tile8);
                        out_u128.store_at(dst0, q0);
                        out_u128.store_at(dst1, q1);
                        n += 8;
                    }
                    fence_view_async_shared();
                    NamedBarrier::<1, 128>::sync();
                    if group + 1 < group_end && tid == 0 {
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 2);
                        b_free_ring.commit_threads(b_free_step + (k_bursts as u32) - 1);
                    }
                    b_free_step += k_bursts as u32;
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 4c (`K4` for `B = 256`): `5`-Warp (`160`-thread) Triple-M-Tile (`M = 384` = `3 × 128`)
    /// K-Streaming `tcgen05` `FC2` (`16` M384-tiles × `8` groups = `128 <= 148` CTAs in 1 single wave!).
    ///
    /// # Safety
    ///
    /// Must be launched with `160` threads per block (`5` warps), at least `222,208` bytes of
    /// 1024-byte-aligned dynamic shared memory, and valid TMA descriptors and slices.
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
        let (worker_id, tile_m384) = idx2crd_2d::<RowMajor<9, 16>>(cta_id);
        let tile_m0 = tile_m384 * 3;
        let tile_m1 = tile_m0 + 1;
        let tile_m2 = tile_m0 + 2;
        let group_count_t = StridedTensor::from_slice(group_count);
        let group_experts_t = StridedTensor::from_slice(group_experts);
        let group_limits_t = StridedTensor::from_slice(group_limits);
        let w_down_global_t = StridedTensor::from_slice(w_down_global);

        let active_groups = unsafe { group_count_t.get_unchecked(0) } as usize;
        let num_workers = (group_step as usize).min(active_groups);
        if worker_id >= num_workers {
            return;
        }
        let (base_g32, rem_g32) = idx2crd_row_major_2d(active_groups as u32, num_workers as u32);
        let (base_g, rem_g) = (base_g32 as usize, rem_g32 as usize);
        let group_start = worker_id * base_g + worker_id.min(rem_g);
        let group_end = group_start + base_g + usize::from(worker_id < rem_g);

        let m128_tiles = HIDDEN_192E / 128; // 48
        let k_bursts = INTERMEDIATE_192E / 512; // 3
        let tid = thread::threadIdx_x() as usize;
        let warp_id = unsafe { cute_rs::arch::warp_idx() };

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let a_free_ring = unsafe { SingleCtaRing::<2>::from_raw(dyn_smem, 221_184 / 8 + 4) };
        let b_free_ring = unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, 221_184 / 8 + 8) };
        let mma_done_ring = unsafe { SingleCtaRing::<1>::from_raw(dyn_smem, 221_184 / 8 + 10) };
        let tmem_slot_ptr =
            unsafe { TmemAllocator::<512, 1>::from_raw_bytes(dyn_smem, 221_184 + 96) };

        unsafe {
            let pipeline = Fc2M384Pipeline::from_shared_bytes(dyn_smem, 221_184);
            if tid == 0 {
                w_down_tma.prefetch_descriptor();
                b_down_tma.prefetch_descriptor();
                sfa_down_tma.prefetch_descriptor();
                sfb_down_tma.prefetch_descriptor();
                a_free_ring.init(1, 1);
                b_free_ring.init(1, 1);
                mma_done_ring.init(1, 1);
            }
            if warp_id == 0 {
                TmemAllocator::<512, 1>::allocate(tmem_slot_ptr);
            }
            pipeline.init(0);

            if warp_id == 4 && (tid & 31) == 0 {
                let mut prod = PipelineState::<Producer, 2>::new();
                let mut step = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let w_outer2 = expert * m128_tiles + tile_m2;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = prod.index();
                        if step >= 2 {
                            a_free_ring.wait(step - 2);
                        }
                        pipeline.producer_acquire(&prod);
                        let bar = pipeline.producer_expect_tx(&prod);
                        let mut a0_tile =
                            SmemTile::<u8, WeightTileSmem>::from_stage_bytes(dyn_smem, 0, slot);
                        let mut a1_tile = SmemTile::<u8, WeightTileSmem>::from_stage_bytes(
                            dyn_smem, 65_536, slot,
                        );
                        let mut sfa0_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 196_608, slot,
                        );
                        let mut sfa1_tile = SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(
                            dyn_smem, 204_800, slot,
                        );
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
                            b_free_ring.wait(step - 1);
                        }
                        let mut a2_tile =
                            SmemTile::<u8, WeightTileSmem>::from_stage_bytes(dyn_smem, 131_072, 0);
                        let mut b_tile =
                            SmemTile::<u8, ActTile128Smem>::from_stage_bytes(dyn_smem, 163_840, 0);
                        let mut sfa2_tile =
                            SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(dyn_smem, 212_992, 0);
                        let mut sfb_tile =
                            SmemTile::<u16, ScaleTmaSmem>::from_stage_bytes(dyn_smem, 217_088, 0);
                        copy_tma_2d(
                            w_down_tma,
                            (w_outer2 * k_bursts + burst, 0),
                            &mut a2_tile,
                            bar,
                        );
                        copy_tma_2d(b_down_tma, (group * k_bursts + burst, 0), &mut b_tile, bar);
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
                let tmem_addr = TmemAllocator::<512, 1>::retrieve_ptr(tmem_slot_ptr);
                let tmem_row_base = tmem_row_address(tmem_addr, warp_id, 0);
                let a2_tile = SharedWeightTile::from_stage_bytes(dyn_smem, 131_072, 0);
                let b_tile = SharedActTile::from_stage_bytes(dyn_smem, 163_840, 0);
                let sfa2_tile =
                    SharedWeightScales::from_stage_bytes(dyn_smem, 212_992, SCALE_BURST_BYTES, 0);
                let sfb_tile =
                    SharedActScales::from_stage_bytes(dyn_smem, 217_088, SCALE_BURST_BYTES, 0);
                let mut smem_epi0 =
                    StridedTensorMut::<BFloat16>::from_shared_bytes(dyn_smem, 0, 128 * 128);
                let mut smem_epi1 =
                    StridedTensorMut::<BFloat16>::from_shared_bytes(dyn_smem, 65_536, 128 * 128);
                let mut smem_epi2 =
                    StridedTensorMut::<BFloat16>::from_shared_bytes(dyn_smem, 131_072, 128 * 128);
                let (col16_u32, row8_u32) = idx2crd_col_major_2d(tid as u32, 16);
                let (col16, row8) = (col16_u32 as usize, row8_u32 as usize);
                let dst_col0_tile8 = tile_m0 * 16 + col16;
                let dst_col1_tile8 = tile_m1 * 16 + col16;
                let dst_col2_tile8 = tile_m2 * 16 + col16;
                let h_tile8 = HIDDEN_192E / 8;

                let mut cons = PipelineState::<Consumer, 2>::new();
                let mut step_base = 0u32;
                let mut g_idx = 0u32;
                let mut group = group_start;
                while group < group_end {
                    let expert = group_experts_t.get_unchecked(group) as usize;
                    let limit = group_limits_t.get_unchecked(group) as usize;
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
                            let a0_tile = SharedWeightTile::from_stage_bytes(dyn_smem, 0, slot);
                            let a1_tile =
                                SharedWeightTile::from_stage_bytes(dyn_smem, 65_536, slot);
                            let sfa0_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                196_608,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            let sfa1_tile = SharedWeightScales::from_stage_bytes(
                                dyn_smem,
                                204_800,
                                SCALE_BURST_BYTES,
                                slot,
                            );
                            issue_triple_m_burst(
                                &tiled_mma,
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
                            pipeline.consumer_release(&cons);
                            cons.advance();
                            if burst + 1 < k_bursts {
                                b_free_ring.producer_commit(step_base + (burst as u32));
                            }
                            if burst == 0 {
                                a_free_ring.producer_commit(step_base);
                            }
                            if burst + 1 == k_bursts {
                                mma_done_ring.producer_commit(g_idx);
                                mma_done_ring.wait(g_idx);
                            }
                            burst += 1;
                        }
                    }
                    NamedBarrier::<1, 128>::sync();
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let r0 =
                            (TmemFragment::<16>::load_deferred(tmem_row_base + (n_base as u32))
                                .to_tensor_ssa()
                                * global_scale)
                                .to::<BFloat16>();
                        let r1 = (TmemFragment::<16>::load_deferred(
                            tmem_row_base + 128 + (n_base as u32),
                        )
                        .to_tensor_ssa()
                            * global_scale)
                            .to::<BFloat16>();
                        let r2 = (TmemFragment::<16>::load_deferred(
                            tmem_row_base + 256 + (n_base as u32),
                        )
                        .to_tensor_ssa()
                            * global_scale)
                            .to::<BFloat16>();
                        let mut c = 0usize;
                        while c < 16 {
                            let n = n_base + c;
                            if n < limit {
                                let idx_s = crd2idx_2d::<RowMajor<128, 128>>(n, tid);
                                smem_epi0.store_at(idx_s, r0[c]);
                                smem_epi1.store_at(idx_s, r1[c]);
                                smem_epi2.store_at(idx_s, r2[c]);
                            }
                            c += 1;
                        }
                        n_base += 16;
                    }
                    fence_view_async_tmem_load();
                    tcgen05_fence_before_thread_sync();
                    NamedBarrier::<1, 128>::sync();

                    let smem_epi0_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 0, 128 * 16);
                    let smem_epi1_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 65_536, 128 * 16);
                    let smem_epi2_u128 =
                        StridedTensor::<u128>::from_shared_bytes(dyn_smem, 131_072, 128 * 16);
                    let mut out_u128 =
                        StridedTensorMut::<u128>::from_disjoint_slice_as(&mut grouped_out);
                    let mut n = row8;
                    while n < limit {
                        let smem_idx = crd2idx_2d::<RowMajor<128, 16>>(n, col16);
                        let q0 = smem_epi0_u128.get_unchecked(smem_idx);
                        let q1 = smem_epi1_u128.get_unchecked(smem_idx);
                        let q2 = smem_epi2_u128.get_unchecked(smem_idx);
                        let perm_row = crd2idx_2d::<RowMajor<768, 128>>(group, n);
                        let dst0 = crd2idx_row_major_2d(perm_row, dst_col0_tile8, h_tile8);
                        let dst1 = crd2idx_row_major_2d(perm_row, dst_col1_tile8, h_tile8);
                        let dst2 = crd2idx_row_major_2d(perm_row, dst_col2_tile8, h_tile8);
                        out_u128.store_at(dst0, q0);
                        out_u128.store_at(dst1, q1);
                        out_u128.store_at(dst2, q2);
                        n += 8;
                    }
                    fence_view_async_shared();
                    NamedBarrier::<1, 128>::sync();
                    if group + 1 < group_end && tid == 0 {
                        a_free_ring.commit_threads(step_base + 1);
                        a_free_ring.commit_threads(step_base + 2);
                        b_free_ring.commit_threads(step_base + 2);
                    }
                    step_base += k_bursts as u32;
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    TmemAllocator::<512, 1>::free(tmem_slot_ptr);
                }
            }
        }
    }

    /// Stage 5 (`K5` for `B >= 32`): Top-4 `FP32` weighted combine (`[B, 6,144]`) using
    /// `Tensor::local_tile::<4>` / `zipped_divide::<8>` + `RegisterTile` (`cute.copy`).
    #[kernel]
    pub fn moe_finalize_192e_cute(
        grouped_out: &[BFloat16],
        tok_to_perm: &[i32],
        routing_weights: &[BFloat16],
        mut output: DisjointSlice<BFloat16>,
        num_tokens: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let h_tile8 = HIDDEN_192E / 8; // 768
        let total_tiles = (num_tokens as usize) * h_tile8;
        if idx >= total_tiles {
            return;
        }
        let (token, col_tile8) = idx2crd_2d::<RowMajor<256, 768>>(idx);

        let perms = Tensor::from_slice(tok_to_perm).local_tile::<4>(token);
        let weights = Tensor::from_slice(routing_weights).local_tile::<4>(token);
        let grouped = Tensor::from_slice(grouped_out).zipped_divide::<8>();
        let out_tile = TensorMut::from_disjoint_slice(&mut output).local_tile::<8>(idx);

        let p = unsafe { perms.load() }.into_array();
        let w = unsafe { weights.load() }.to::<f32>();

        let decode_row = |gp: i32| -> usize {
            let (r_u32, g_u32) = idx2crd_col_major_2d(gp as u32, 256);
            let perm_row = crd2idx_2d::<RowMajor<768, 128>>(g_u32 as usize, r_u32 as usize);
            crd2idx_row_major_2d(perm_row, col_tile8, h_tile8)
        };
        let q0 = unsafe { grouped.slice(decode_row(p[0])).load() }.to::<f32>();
        let q1 = unsafe { grouped.slice(decode_row(p[1])).load() }.to::<f32>();
        let q2 = unsafe { grouped.slice(decode_row(p[2])).load() }.to::<f32>();
        let q3 = unsafe { grouped.slice(decode_row(p[3])).load() }.to::<f32>();

        let acc = q0 * w[0] + q1 * w[1] + q2 * w[2] + q3 * w[3];

        unsafe {
            out_tile.store(acc.to::<BFloat16>());
        }
    }
}

/// Device buffers and typed TMA descriptors for the 192-expert `cute-rs` MoE pipeline.
struct CuteMoe192Buffers {
    /// Input `FP16` activations (`[num_tokens, 6144]`).
    hidden: DeviceBuffer<f16>,
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
    /// Routed & quantized `FC1` input tokens (`[max_groups * 12, 128 * 32]` `u64`).
    b_up_vals: DeviceBuffer<u64>,
    /// Routed `FC1` token block scales (`[max_groups * 96, 512]` B).
    sfb_up_scales: DeviceBuffer<u8>,
    /// Quantized `FC1 + ReLU²` intermediate activations (`[max_groups * 3, 128 * 32]` `u64`).
    inter_vals: DeviceBuffer<u64>,
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

    let hidden_host: Vec<f16> = (0..num_tokens)
        .flat_map(|t| std::iter::repeat_n(crate::hidden_value(t) as f16, HIDDEN_192E))
        .collect();
    let expert_ids_host: Vec<i32> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| crate::expert_id(t, s) as i32))
        .collect();
    let routing_host: Vec<BFloat16> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| BFloat16::from_f32(crate::routing_weight(t, s))))
        .collect();

    let w_up_bytes = NUM_EXPERTS_192E * INTERMEDIATE_192E * (HIDDEN_192E / 2);
    let w_up_scale_bytes = NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * (HIDDEN_192E / 64) * 512;
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
    let b_up_vals = DeviceBuffer::from_host(stream, &vec![0u64; max_groups * k_bursts_up * 4096])?;
    let sfb_up_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_up * SCALE_BURST_BYTES],
    )?;
    let inter_vals =
        DeviceBuffer::from_host(stream, &vec![0u64; max_groups * k_bursts_down * 4096])?;
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
            &vec![BFloat16::from_f32(0.0); 768 * GROUP_TILE_ROWS * HIDDEN_192E],
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
        let total_tiles = (bufs.num_tokens * (HIDDEN_192E / 8)) as u32;
        let fin_cfg = LaunchConfig {
            grid_dim: (total_tiles.div_ceil(256), 1, 1),
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
            return Err(
                format!("non-finite element at {idx}: actual={act}, expected={exp}").into(),
            );
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

/// Exact CPU reference for the uniform 192-expert fixture at token `t`.
fn uniform_expected_row_192e(t: usize) -> f32 {
    let q = (crate::hidden_value(t) as f16) as f32;
    let mut total = 0.0f32;
    for slot in 0..TOP_K_192E {
        let expert = crate::expert_id(t, slot);
        let rw = BFloat16::from_f32(crate::routing_weight(t, slot)).to_f32();
        let fc1 = q * UP_FACTORS_192E[expert];
        let u = fc1.max(0.0);
        let iq = u * u;
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
    let w_up_scale_bytes = NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * (HIDDEN_192E / 64) * 512;
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
                let sub_base = (((e * down_m_tiles + mt) * (INTERMEDIATE_192E / 512) + burst) * 8
                    + s_in_b)
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

    const E2M1_POS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let decode_e4m3 = |code: u8| -> f32 {
        let exp = (code >> 3) & 0x0f;
        let mant = code & 0x07;
        if exp == 0 {
            (mant as f32) * (1.0 / 512.0)
        } else {
            (1.0 + (mant as f32) * 0.125) * 2.0f32.powi((exp as i32) - 7)
        }
    };
    let host_quant_nvfp4 = |v: f32| -> f32 {
        let a = v.abs().min(2688.0);
        if a == 0.0 {
            return 0.0;
        }
        let target_s = a / 6.0;
        let mut best_s = 0.0f32;
        let mut best_err = f32::INFINITY;
        for code in 0u8..=0x7E {
            let cand = decode_e4m3(code);
            let err = (target_s - cand).abs();
            if err < best_err || (err == best_err && (code & 1) == 0) {
                best_err = err;
                best_s = cand;
            }
        }
        if best_s == 0.0 {
            return 0.0;
        }
        let scaled = a / best_s;
        let mut best_m = 0.0f32;
        let mut best_m_err = f32::INFINITY;
        for (nib, &cand) in E2M1_POS.iter().enumerate() {
            let err = (scaled - cand).abs();
            if err < best_m_err || (err == best_m_err && (nib & 1) == 0) {
                best_m_err = err;
                best_m = cand;
            }
        }
        v.signum() * best_m * best_s
    };

    let eval_cpu_reference = |hidden_h: &[f16], exp_h: &[i32], rw_h: &[BFloat16]| -> Vec<f32> {
        let mut out = vec![0.0f32; num_tokens * HIDDEN_192E];
        for t in 0..num_tokens {
            let mut q_in = [0.0f32; HIDDEN_192E / 64];
            for kc in 0..(HIDDEN_192E / 64) {
                let x = hidden_h[t * HIDDEN_192E + kc * 64] as f32;
                q_in[kc] = host_quant_nvfp4(x);
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
                        let w_val = E2M1_POS[(NIB_PAIR[(e + mt + kc) & 3] & 7) as usize];
                        let w_sc = decode_e4m3(SCALE_CODES[(e + mt + kc) % 3]);
                        acc += 64.0 * q_in[kc] * w_val * w_sc;
                    }
                    let u = (acc * up_g).max(0.0);
                    q_inter[mt] = host_quant_nvfp4(u * u);
                }
                for mt in 0..(HIDDEN_192E / 128) {
                    let mut acc = 0.0f32;
                    for kc in 0..(INTERMEDIATE_192E / 64) {
                        let w_val = E2M1_POS[(NIB_PAIR[(e + mt * 3 + kc) & 3] & 7) as usize];
                        let w_sc = decode_e4m3(SCALE_CODES[(e + mt + kc * 2) % 3]);
                        acc += 64.0 * q_inter[kc >> 1] * w_val * w_sc;
                    }
                    let fc2_bf16 = BFloat16::from_f32(acc * down_g).to_f32();
                    for c in 0..128 {
                        out[t * HIDDEN_192E + mt * 128 + c] += rw * fc2_bf16;
                    }
                }
            }
            for c in 0..HIDDEN_192E {
                out[t * HIDDEN_192E + c] = BFloat16::from_f32(out[t * HIDDEN_192E + c]).to_f32();
            }
        }
        out
    };

    let mut hidden_nu = vec![0.0f16; num_tokens * HIDDEN_192E];
    for t in 0..num_tokens {
        for kc in 0..(HIDDEN_192E / 64) {
            let v = crate::hidden_value(t) * (0.75 + 0.15 * (((t + kc) % 4) as f32));
            let base = t * HIDDEN_192E + kc * 64;
            hidden_nu[base..base + 64].fill(v as f16);
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
    let mut hidden_cr = vec![0.0f16; num_tokens * HIDDEN_192E];
    for t in 0..num_tokens {
        for kc in 0..(HIDDEN_192E / 64) {
            let v = crate::hidden_value(t + 5) * (0.85 + 0.1 * (((t + kc + 1) % 3) as f32));
            let base = t * HIDDEN_192E + kc * 64;
            hidden_cr[base..base + 64].fill(v as f16);
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

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wide 192-Expert (`H = 6,144, I = 1,536, E = 192, top_k = 4`) NVFP4 `ReLU²` Mixture-of-Experts
//! layer in native `cuda-oxide` SIMT + 5th-Gen Tensor Core (`tcgen05`) on Blackwell (`sm_100a`).
//!
//! # Architecture (`B = 1 .. 256`, On-GPU Route Grouping Inside Captured CUDA Graph)
//!
//! ```text
//! BeginCapture                                                                                    EndCapture
//! ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
//! │ [K1: moe_route_192e] ──► [K2: quantize_fp16_192e] ──► [K3: tcgen05_fc1_192e] ──► [K4: tcgen05_fc2_192e]│
//! │  (1 CTA, SMEM atomic      (Vectorized FP16 load,       (296 CTAs @ 2 CTA/SM,      (288 CTAs @ 2 CTA/SM,│
//! │   histogram + 8-warp       RN-even E4M3 + E2M1,         12x K=512 TMA bursts,      3x K=512 TMA bursts,│
//! │   bit-shift prefix)        4-way 64-row scatter)        register ReLU²+NVFP4)      register BF16 store)│
//! │                                                                                           │            │
//! │                                                                                           ▼            │
//! │                                                                            [K5: moe_finalize_192e]     │
//! │                                                                             (Coalesced u128 FP32 top-4 │
//! │                                                                              combine ──► BF16 output)  │
//! └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::encode_tma_2d;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig, LaunchConfig1D, sys,
};
use cuda_device::atomic::{ AtomicOrdering, BlockAtomicU32, atom_add_bf16x2 };
use cuda_device::barrier::{
    Barrier, barrier_cta_sync, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::convert::{
    bf16_to_f32, cvt_bf16x2_f32, cvt_f32x2_f16x2, cvt_rn_f16x2_e4m3x2, f32_to_bf16_rne,
};
use cuda_device::shared::{ DynamicSharedArray, cvta_generic_to_shared_offset };
use cuda_device::tcgen05::{
    tcgen05_alloc, tcgen05_commit_shared_cluster, tcgen05_cp_32x128b_warpx4, tcgen05_dealloc,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync, tcgen05_ld_32x32b_x16_raw,
    tcgen05_load_wait, tcgen05_mma_shared_mxf4nvf4_block_scale_block16,
};
use cuda_device::tma::{ TmaDescriptor, cp_async_bulk_tensor_2d_g2s_cta };
use cuda_device::{ DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp };
use cuda_host::cuda_module;
use half::{ bf16, f16 };
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Hidden dimension `H` of each input and output token vector (`6,144` elements).
/// Divided into `384` scale groups of `16` elements (`3,072` packed FP4 `E2M1` bytes = `12` bursts of `K = 512`).
pub const HIDDEN_192E: usize = 6144;

/// Intermediate FFN dimension `I` inside each expert (`1,536` elements).
/// Divided into `12` `M = 128` output tiles in `FC1` and `3` `K = 512` reduction bursts in `FC2`.
pub const INTERMEDIATE_192E: usize = 1536;

/// Total number of allocated experts in the wide MoE layer (`E = 192`).
/// Every expert owns an independent `W_up` (`[1536, 6144]`) and `W_down` (`[6144, 1536]`) weight matrix.
pub const NUM_EXPERTS_192E: usize = 192;

/// Number of experts activated per token (`top_k = 4`).
/// Each token's final output is the weighted combination of its `4` routed experts.
pub const TOP_K_192E: usize = 4;

/// Token batch sizes `B` swept from single-token decode (`B = 1`) to `B = 256`.
/// Evaluated with `50` untimed graph warmups and `1,000` timed graph replays per sample.
pub const BATCH_SIZES_192E: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Fixed physical token row capacity per TMA group tile (`128` rows).
/// Covers `tile_n = 16, 32, 64, 128` with a single `512`-byte `(32, 4, 4)` scale atom per `K = 64` step.
const GROUP_TILE_ROWS: usize = 128;

/// Byte size of one `128 × 256` (`K = 512`) packed FP4 weight burst in shared memory (`32,768` bytes).
/// Stage 0 occupies `0..32768` and Stage 1 occupies `32768..65536`.
const WEIGHT_BURST_BYTES: usize = 32_768;

/// Byte size of one `128 × 256` (`K = 512`) packed FP4 activation burst in shared memory (`32,768` bytes).
/// Stage 0 occupies `65536..98304` and Stage 1 occupies `98304..131072`.
const ACT64_BURST_BYTES: usize = 32_768;

/// Byte size of 8 canonical `512`-byte `(32, 4, 4)` scale atoms (`4,096` bytes).
/// Covers one `K = 512` burst for `SFA` (`128` weight rows) or `SFB` (`128` token rows).
const SCALE_BURST_BYTES: usize = 4_096;

/// Byte offset of the 2-stage `SMEM_B` activation buffer (`65,536` bytes after `SMEM_A`) in `FC1`.
/// Holds `2 × 32 KB = 64 KB` of routed token bursts.
const SMEM192_B_OFFSET: usize = 65_536;

/// Byte offset of the 2-stage `SMEM_SFA` weight scale buffer (`131,072` bytes) in `FC1`.
/// Holds `2 × 4 KB = 8 KB` of canonical `(32, 4, 4)` weight scales.
const SMEM192_SFA_OFFSET: usize = 131_072;

/// Byte offset of the 2-stage `SMEM_SFB` activation scale buffer (`139,264` bytes) in `FC1`.
/// Holds `2 × 4 KB = 8 KB` of canonical `(32, 4, 4)` token scales.
const SMEM192_SFB_OFFSET: usize = 139_264;

/// Byte offset of the CTA `mbarrier` array and TMEM column slot (`147,456` bytes) in `FC1`.
/// Placed immediately after `SMEM_SFB`.
const SMEM192_BAR_OFFSET: usize = 147_456;

/// Total dynamic shared memory bytes per `FC1` CTA (`148,480` bytes = `145 KB`).
/// Holds `64 KB` `SMEM_A` + `64 KB` `SMEM_B` + `8 KB` `SMEM_SFA` + `8 KB` `SMEM_SFB` + barriers.
const DYNAMIC_SMEM_192E: u32 = 148_480;

/// Byte offset of `SMEM_A1` (`2 × 32 KB = 64 KB` for the second `M = 128` tile) in `FC2` (`65,536` bytes).
const SMEM_FC2_A1_OFFSET: usize = 65_536;

/// Byte offset of `SMEM_B` (`2 × 32 KB = 64 KB` shared by both `M = 128` tiles) in `FC2` (`131,072` bytes).
const SMEM_FC2_B_OFFSET: usize = 131_072;

/// Byte offset of `SMEM_SFA0` (`2 × 4 KB = 8 KB` for `m0`) in `FC2` (`196,608` bytes).
const SMEM_FC2_SFA0_OFFSET: usize = 196_608;

/// Byte offset of `SMEM_SFA1` (`2 × 4 KB = 8 KB` for `m1`) in `FC2` (`204,800` bytes).
const SMEM_FC2_SFA1_OFFSET: usize = 204_800;

/// Byte offset of `SMEM_SFB` (`2 × 4 KB = 8 KB` shared by `m0` and `m1`) in `FC2` (`212,992` bytes).
const SMEM_FC2_SFB_OFFSET: usize = 212_992;

/// Byte offset of the `mbarrier` array in `FC2` (`221,184` bytes).
const SMEM_FC2_BAR_OFFSET: usize = 221_184;

/// Total dynamic shared memory bytes per `FC2` CTA (`222,208` bytes = `217 KB`).
/// Holds `2` `M = 128` weight tiles (`A0`, `A1`, `SFA0`, `SFA1`) sharing `B` and `SFB`.
const DYNAMIC_SMEM_FC2_192E: u32 = 222_208;

/// Total TMA transaction bytes per `FC2` stage (`32KB A0 + 32KB A1 + 32KB B + 4KB SFA0 + 4KB SFA1 + 4KB SFB = 110,592` B).
const STAGE_TX_BYTES_FC2_192E: u32 = 110_592;

/// Total TMA transaction bytes per `M = 384` (`3 × 128`) `FC2` burst (`3 × 36 KB A/SFA + 36 KB B/SFB = 147,456` B).
const STAGE_TX_BYTES_FC2_M384: u32 = 147_456;

/// Total TMA transaction bytes per `FC1` stage (`32 KB A + 32 KB B + 4 KB SFA + 4 KB SFB = 73,728` bytes).
/// Announced via `mbarrier_arrive_expect_tx` by the TMA producer warp on every `K = 512` burst.
const STAGE_TX_BYTES_192E: u32 = 73_728;

/// Starting TMEM column (`128`) reserved for the 2-stage `SFA` block scales (`2 × 32 = 64` columns).
/// Leaves columns `0..127` for the accumulator and `192..255` for `SFB` inside a `256`-column TMEM allocation.
const TMEM192_SFA_COL: u32 = 128;

/// Starting TMEM column (`192`) reserved for the 2-stage `SFB` block scales (`2 × 32 = 64` columns).
/// Fits within the `256`-column per-CTA TMEM budget so `2` CTAs share each SM's `512` TMEM columns.
const TMEM192_SFB_COL: u32 = 192;

/// Untimed CUDA Graph replay warmups before each timing sample (`50` replays).
/// Matches the evaluation protocol across all compared implementations.
const WARMUPS_192E: usize = 50;

/// Timed CUDA Graph replays per sample (`1,000` replays).
/// Total elapsed time is divided by `1,000` to obtain mean per-replay latency in microseconds.
const REPLAYS_192E: usize = 1000;

/// Number of independent `1,000`-replay samples collected per batch size (`20` samples).
/// Matches the 20-round evaluation protocol.
const SAMPLES_192E: usize = 20;

/// Deterministic top-4 routing weight pattern summing to `1.0` (`[0.125, 0.25, 0.25, 0.375]`).
/// Used by the uniform and nonuniform validation fixtures.
const ROUTING_PATTERN_192E: [f32; TOP_K_192E] = [0.125, 0.25, 0.25, 0.375];

/// Per-expert `FC1` (`W_up`) global scale factors for the first 4 experts (`[0.5, 1.0, 2.0, 1.0]`).
/// Divided by `HIDDEN_192E` (`6,144`) to form `w_up_global`.
const UP_FACTORS_192E: [f32; TOP_K_192E] = [0.5, 1.0, 2.0, 1.0];

/// Per-expert `FC2` (`W_down`) global scale factors for the first 4 experts (`[1.0, 2.0, 0.5, 1.0]`).
/// Divided by `INTERMEDIATE_192E` (`1,536`) to form `w_down_global`.
const DOWN_FACTORS_192E: [f32; TOP_K_192E] = [1.0, 2.0, 0.5, 1.0];

/// Build a `tcgen05` shared-memory descriptor (`SWIZZLE_NONE = 0`) from a 32-bit shared offset,
/// `leading_dim_bytes`, and `stride_bytes`.
#[inline(always)]
const fn build_smem_desc_192e(smem_addr: u64, leading_dim_bytes: u32, stride_bytes: u32) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    addr_enc | (ld_enc << 16) | (stride_enc << 32) | (1u64 << 46)
}

/// Encode the 32-bit `tcgen05.mma.kind::mxf4nvf4.block_scale` instruction descriptor for `M = 128` and `n_cols`.
/// Sets `M = 128` (`bits 28..24 = 8`), `E2M1 × E2M1` (`bits 11..7`), and `N / 8` in `bits 22..17`.
#[inline(always)]
const fn encode_idesc_192e(n_cols: u32) -> u32 {
    0x0800_0480u32 | ((n_cols >> 3) << 17)
}

/// Map a row index `row_in_128` (`0..128`) to its 32-bit word index (`0..128`) in a canonical `(32, 4, 4)` scale atom.
/// Matches the `tcgen05.cp.32x128b.warpx4` per-lane scale layout (`(row % 32) * 4 + (row / 32)`).
#[inline(always)]
const fn scale_atom_word_192e(row_in_128: usize) -> usize {
    ((row_in_128 & 31) * 4) + ((row_in_128 >> 5) & 3)
}

/// Device kernels for the 192-expert (`H = 6,144, I = 1,536, E = 192, top_k = 4`) NVFP4 `ReLU²` MoE pipeline.
/// All five kernels execute inside a single captured CUDA Graph with zero host-side route preparation.
#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub mod kernels_192e {
    use super::*;

    /// Decode one `Float8E4M3FN` scale byte into `f32` via hardware `cvt.rn.f16x2.e4m3x2`.
    #[inline(always)]
    fn dev_e4m3_to_f32_192e(byte: u8) -> f32 {
        let packed = cvt_rn_f16x2_e4m3x2(byte as u16);
        let (lo, _) = cvt_f32x2_f16x2(packed);
        lo
    }

    /// Encode one non-negative `f32` scale factor into `Float8E4M3FN` using round-to-nearest-even.
    /// Matches the `fixture.py` `to(torch.float8_e4m3fn)` scale quantization contract.
    #[inline(always)]
    fn dev_f32_to_e4m3_rne(value: f32) -> u8 {
        let clamped = if value.is_nan() || value <= 0.0 {
            0.0f32
        } else if value > 448.0 {
            448.0f32
        } else {
            value
        };
        let scaled = clamped * f32::from_bits(7 << 23);
        let bits = scaled.to_bits();
        let code = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) as u8;
        if code > 0x7E { 0x7E } else { code }
    }

    /// Compute exact `1.0 / scale_val * 2^-126` for a decoded `Float8E4M3FN` `scale_val` without `rcp.approx` drift.
    /// Uses an 8-entry mantissa reciprocal lookup table packed into a single 64-bit constant.
    #[inline(always)]
    fn exact_inv_scale_e2m1(scale_code: u8) -> f32 {
        if scale_code == 0 {
            return 0.0f32;
        }
        let scale_val = dev_e4m3_to_f32_192e(scale_code);
        let sv_bits = scale_val.to_bits();
        let sv_exp = ((sv_bits >> 23) & 0xFF).max(64);
        let sv_mant = (sv_bits >> 20) & 7;
        let diff = ((0x776E_6255_4633_1C00u64 >> (sv_mant * 8)) & 0xFF) as u32;
        f32::from_bits((0x2000 - diff) << 16) * f32::from_bits((191 - sv_exp) << 23)
    }

    /// Encode one non-negative `f32` pre-multiplied by [`exact_inv_scale_e2m1`] into its 4-bit
    /// `Float4E2M1FN` nibble (`0..7`) using IEEE 754 round-to-nearest-even.
    #[inline(always)]
    fn fast_pos_e2m1_rne(scaled_v: f32) -> u32 {
        let bits = scaled_v.to_bits().min(0x01C0_0000);
        (bits + 0x001F_FFFF + ((bits >> 22) & 1)) >> 22
    }

    /// Stage 1 (`K1`): Single-CTA (`256` threads = `8` warps) shared-memory route histogram,
    /// parallel power-of-two bit-shift prefix scan, and group/permutation table builder.
    ///
    /// Uses `6,176` bytes of dynamic shared memory (zero module-static shared memory) so `FC1` and
    /// `FC2` retain the full `115,200`-byte per-CTA dynamic shared memory budget for `2` CTAs/SM.
    #[launch_bounds(256)]
    #[kernel]
    pub fn moe_route_192e(
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
        let warp_id = tid >> 5;
        let lane = tid & 31;
        let routes = routed_rows as usize;
        let tn = tile_n;
        let tn_mask = tn - 1;
        let tn_shift = tile_n_log2;

        unsafe {
            let dyn_smem = DynamicSharedArray::<u8, 16>::get_raw();
            let counts_ptr = dyn_smem.cast::<u32>();
            let offsets_ptr = dyn_smem.add(1024).cast::<u32>();
            let ranks_ptr = dyn_smem.add(2048).cast::<u16>();
            let exp_ptr = dyn_smem.add(4096).cast::<u16>();
            let warp_sums_ptr = dyn_smem.add(6144).cast::<u32>();

            *counts_ptr.add(tid) = 0;
            thread::sync_threads();

            let mut r = tid;
            while r < routes {
                let exp = *expert_ids.as_ptr().add(r) as u32;
                let atom = BlockAtomicU32::from_ptr(counts_ptr.add(exp as usize));
                let rank = atom.fetch_add(1, AtomicOrdering::Relaxed);
                *exp_ptr.add(r) = exp as u16;
                *ranks_ptr.add(r) = rank as u16;
                r += 256;
            }
            thread::sync_threads();

            let count = if tid < NUM_EXPERTS_192E {
                *counts_ptr.add(tid)
            } else {
                0
            };
            let groups = (count + tn_mask) >> tn_shift;

            let mut prefix = groups;
            #[unroll]
            for s_log2 in 0..5 {
                let step = 1u32 << s_log2;
                let n = warp::shuffle_up(prefix, step);
                if (lane as u32) >= step {
                    prefix += n;
                }
            }
            if lane == 31 {
                *warp_sums_ptr.add(warp_id) = prefix;
            }
            thread::sync_threads();

            if warp_id == 0 {
                let mut w_sum = if lane < 8 {
                    *warp_sums_ptr.add(lane)
                } else {
                    0
                };
                #[unroll]
                for s_log2 in 0..3 {
                    let step = 1u32 << s_log2;
                    let n = warp::shuffle_up(w_sum, step);
                    if (lane as u32) >= step {
                        w_sum += n;
                    }
                }
                if lane < 8 {
                    *warp_sums_ptr.add(lane) = w_sum;
                }
            }
            thread::sync_threads();

            let warp_base = if warp_id > 0 {
                *warp_sums_ptr.add(warp_id - 1)
            } else {
                0
            };
            let group_base = (warp_base + prefix - groups) as usize;
            *offsets_ptr.add(tid) = group_base as u32;

            if tid == 255 {
                *group_count.as_mut_ptr() = (warp_base + prefix) as i32;
            }
            if tid < NUM_EXPERTS_192E && groups > 0 {
                let mut g = 0u32;
                while g < groups {
                    let rem = count - g * tn;
                    let lim = if rem < tn { rem } else { tn };
                    *group_experts.as_mut_ptr().add(group_base + g as usize) = tid as i32;
                    *group_limits.as_mut_ptr().add(group_base + g as usize) = lim as i32;
                    g += 1;
                }
            }
            thread::sync_threads();

            let mut r2 = tid;
            while r2 < routes {
                let exp = *exp_ptr.add(r2) as usize;
                let rank = *ranks_ptr.add(r2) as u32;
                let g = *offsets_ptr.add(exp) + (rank >> tn_shift);
                let row = rank & tn_mask;
                *tok_to_perm.as_mut_ptr().add(r2) = ((g << 8) | row) as i32;
                *group_routes
                    .as_mut_ptr()
                    .add((g as usize) * GROUP_TILE_ROWS + (row as usize)) = r2 as i32;
                r2 += 256;
            }
        }
    }

    /// Stage 2 (`K2`): Vectorized `FP16 -> NVFP4` input activation quantizer and 4-way `128`-row
    /// `K = 512` TMA burst scatter (`H = 6,144`, `12` bursts).
    ///
    /// For `num_tokens <= 16` (`routed_rows <= 64`), fuses route counting and group permutation
    /// directly in shared memory (`fuse_route != 0`), eliminating the separate `moe_route_192e` launch.
    #[launch_bounds(128)]
    #[kernel]
    pub fn quantize_fp16_192e(
        hidden: &[u16],
        expert_ids: &[i32],
        mut group_experts: DisjointSlice<i32>,
        mut group_limits: DisjointSlice<i32>,
        mut group_routes: DisjointSlice<i32>,
        mut group_count: DisjointSlice<i32>,
        mut tok_to_perm: DisjointSlice<i32>,
        mut b_up_vals: DisjointSlice<u8>,
        mut sfb_up_scales: DisjointSlice<u8>,
        mut output: DisjointSlice<u16>,
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

        if zero_out != 0 && idx < total_groups {
            let out_u128_ptr = output.as_mut_ptr().cast::<u128>();
            unsafe {
                *out_u128_ptr.add(idx * 2) = 0;
                *out_u128_ptr.add(idx * 2 + 1) = 0;
            }
        }

        let gp128 = if fuse_route != 0 {
            let routes = (num_tokens as usize) * TOP_K_192E;
            unsafe {
                let smem = DynamicSharedArray::<u8, 16>::get_raw();
                let s_ids = smem.cast::<i32>();
                let s_first = smem.add(256).cast::<i32>();
                let s_perm = smem.add(512).cast::<i32>();

                if tid < routes {
                    *s_ids.add(tid) = *expert_ids.as_ptr().add(tid);
                }
                thread::sync_threads();

                let mut my_exp = 0i32;
                let mut my_rank = 0i32;
                let mut my_count = 0i32;
                if tid < routes {
                    my_exp = *s_ids.add(tid);
                    let mut j = 0usize;
                    while j < routes {
                        if *s_ids.add(j) == my_exp {
                            my_count += 1;
                            if j < tid {
                                my_rank += 1;
                            }
                        }
                        j += 1;
                    }
                    *s_first.add(tid) = if my_rank == 0 { my_exp } else { i32::MAX };
                }
                thread::sync_threads();

                if cta_id == 0 {
                    if tid < routes {
                        let mut my_group = 0i32;
                        let mut total_g = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = *s_first.add(j);
                            if fj != i32::MAX {
                                total_g += 1;
                                if fj < my_exp {
                                    my_group += 1;
                                }
                            }
                            j += 1;
                        }
                        let perm = (my_group << 8) | my_rank;
                        *s_perm.add(tid) = perm;
                        *tok_to_perm.as_mut_ptr().add(tid) = perm;
                        *group_routes
                            .as_mut_ptr()
                            .add((my_group as usize) * GROUP_TILE_ROWS + (my_rank as usize)) =
                            tid as i32;
                        if my_rank == 0 {
                            *group_experts.as_mut_ptr().add(my_group as usize) = my_exp;
                            *group_limits.as_mut_ptr().add(my_group as usize) = my_count;
                        }
                        if tid == 0 {
                            *group_count.as_mut_ptr() = total_g;
                        }
                    }
                } else {
                    let r_start = row * 4;
                    if tid >= r_start && tid < r_start + 4 {
                        let mut my_group = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = *s_first.add(j);
                            if fj < my_exp {
                                my_group += 1;
                            }
                            j += 1;
                        }
                        *s_perm.add(tid) = (my_group << 8) | my_rank;
                    }
                }
                thread::sync_threads();
                if idx < total_groups {
                    *s_perm.cast_const().cast::<u128>().add(row)
                } else {
                    0u128
                }
            }
        } else if idx < total_groups {
            unsafe { *tok_to_perm.as_mut_ptr().cast_const().cast::<u128>().add(row) }
        } else {
            0u128
        };

        if idx >= total_groups {
            return;
        }

        let in_u128_ptr = hidden.as_ptr().cast::<u128>();
        let (q0, q1) = unsafe { (*in_u128_ptr.add(idx * 2), *in_u128_ptr.add(idx * 2 + 1)) };
        let raw_u32 = [
            (q0 & 0xFFFF_FFFF) as u32,
            ((q0 >> 32) & 0xFFFF_FFFF) as u32,
            ((q0 >> 64) & 0xFFFF_FFFF) as u32,
            (q0 >> 96) as u32,
            (q1 & 0xFFFF_FFFF) as u32,
            ((q1 >> 32) & 0xFFFF_FFFF) as u32,
            ((q1 >> 64) & 0xFFFF_FFFF) as u32,
            (q1 >> 96) as u32,
        ];

        let mut vals = [0.0f32; 16];
        let mut amax = 0.0f32;
        #[unroll]
        for i in 0..8 {
            let (lo, hi) = cvt_f32x2_f16x2(raw_u32[i]);
            vals[i * 2] = lo;
            vals[i * 2 + 1] = hi;
            let a0 = f32::from_bits(lo.to_bits() & 0x7FFF_FFFF);
            let a1 = f32::from_bits(hi.to_bits() & 0x7FFF_FFFF);
            if a0 > amax {
                amax = a0;
            }
            if a1 > amax {
                amax = a1;
            }
        }

        let scale_code = dev_f32_to_e4m3_rne(amax * (1.0f32 / 6.0f32));
        let inv_scale_e2m1 = exact_inv_scale_e2m1(scale_code);
        let mut packed_u64 = 0u64;
        #[unroll]
        for i in 0..16 {
            let v = vals[i];
            let sign = ((v.to_bits() >> 28) & 8) as u64;
            let abs_v = f32::from_bits(v.to_bits() & 0x7FFF_FFFF);
            let nib = sign | (fast_pos_e2m1_rne(abs_v * inv_scale_e2m1) as u64);
            packed_u64 |= nib << (i * 4);
        }

        let b_u64_ptr = b_up_vals.as_mut_ptr().cast::<u64>();
        let sfb_ptr = sfb_up_scales.as_mut_ptr();
        let k_bursts = HIDDEN_192E / 512; // 12
        let k_atoms = HIDDEN_192E / 64; // 96
        let burst = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;

        #[unroll]
        for slot in 0..4 {
            let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
            let g = gp >> 8;
            let r = gp & 127;
            let u64_idx = (g * k_bursts + burst) * 4096 + sub_u64_off + r * 2;
            let sfb_word = scale_atom_word_192e(r);
            let s_idx = (g * k_atoms + k_atom) * 512 + sfb_word * 4 + k_in_atom;
            unsafe {
                *b_u64_ptr.add(u64_idx) = packed_u64;
                *sfb_ptr.add(s_idx) = scale_code;
            }
        }
    }

    /// Stage 3 (`K3`): `5`-Warp (`160`-thread) Persistent K-Streaming `tcgen05` `FC1 + ReLU² + NVFP4 Requantize`
    /// (`I = 1,536, H = 6,144`, `12` bursts of `K = 512`).
    ///
    /// Uses `148,480` bytes of dynamic shared memory and `256` TMEM columns, releases shared-memory
    /// stages asynchronously via `tcgen05.commit`, and performs `ReLU²` + NVFP4 requantization
    /// in parallel across all 128 compute threads.
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 148480,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_fc1_relu2_requant_192e(
        #[grid_constant] w_up_tma: &TmaDescriptor,
        #[grid_constant] b_up_tma: &TmaDescriptor,
        #[grid_constant] sfa_up_tma: &TmaDescriptor,
        #[grid_constant] sfb_up_tma: &TmaDescriptor,
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
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM192_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(SMEM192_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM192_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM192_BAR_OFFSET).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { tma_bar_ptr.add(2) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 5 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 256);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 4 && (tid & 31) == 0 {
                let w_desc = core::ptr::from_ref(w_up_tma);
                let b_desc = core::ptr::from_ref(b_up_tma);
                let sfa_desc = core::ptr::from_ref(sfa_up_tma);
                let sfb_desc = core::ptr::from_ref(sfb_up_tma);
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer = expert * m_tiles + tile_m128;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = step & 1;
                        let phase = ((step >> 1) & 1) as u32;
                        if step >= 2 {
                            while !mbarrier_try_wait_parity(b_free_bar_const.add(slot), phase ^ 1) {}
                        }
                        let bar = tma_bar_ptr.add(slot);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, STAGE_TX_BYTES_192E);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_b_ptr.add(slot * ACT64_BURST_BYTES),
                            b_desc,
                            0,
                            ((group * k_bursts + burst) * GROUP_TILE_ROWS) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfb_ptr.add(slot * SCALE_BURST_BYTES),
                            sfb_desc,
                            0,
                            ((group * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a_base0 = cvta_generic_to_shared_offset(smem_a_ptr.cast_const());
                let smem_b_base0 = smem_a_base0 + (SMEM192_B_OFFSET as u64);
                let smem_sfa_base0 = smem_a_base0 + (SMEM192_SFA_OFFSET as u64);
                let smem_sfb_base0 = smem_a_base0 + (SMEM192_SFB_OFFSET as u64);
                let tmem_row_base = tmem_addr + (warp_id << 21);
                let col_perm = ((tid & 12) << 3) | ((tid >> 4) << 2) | (tid & 3);
                let sub_tile16 = tid & 7; // 0..7
                let row_off16 = tid >> 3; // 0..15
                let grp = tile_m128 * 8 + sub_tile16;
                let burst_inter = grp >> 5; // 0..2
                let s_in_burst = (grp >> 2) & 7;
                let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
                let k_atom = grp >> 2;
                let k_in_atom = grp & 3;
                let inter_bursts = INTERMEDIATE_192E / 512; // 3
                let inter_atoms = INTERMEDIATE_192E / 64; // 24
                let inter_u64_ptr = inter_vals.as_mut_ptr().cast::<u64>();
                let inter_scales_ptr = inter_scales.as_mut_ptr();
                let smem_epi_f32 = smem_a_ptr.cast::<f32>();
                let smem_epi_u128 = smem_a_ptr.cast_const().cast::<u128>();

                let mut step = 0usize;
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let idesc = encode_idesc_192e(valid_n as u32);
                    let global_scale = *w_up_global.as_ptr().add(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            let slot = step & 1;
                            let phase = ((step >> 1) & 1) as u32;
                            while !mbarrier_try_wait_parity(tma_bar_const.add(slot), phase) {}
                            let sfa_tmem_b = tmem_addr + TMEM192_SFA_COL + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + TMEM192_SFB_COL + (slot as u32) * 32;
                            let a_desc0 = build_smem_desc_192e(
                                smem_a_base0 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let b_desc0 = build_smem_desc_192e(
                                smem_b_base0 + (slot * ACT64_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa_desc0 = build_smem_desc_192e(
                                smem_sfa_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sb_desc0 = build_smem_desc_192e(
                                smem_sfb_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            #[unroll]
                            for s in 0..8 {
                                let sa = sfa_tmem_b + (s as u32) * 4;
                                let sb = sfb_tmem_b + (s as u32) * 4;
                                tcgen05_cp_32x128b_warpx4(sa, sa_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sb, sb_desc0 + (s as u64) * 32);
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr,
                                    a_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa,
                                    sb,
                                );
                            }
                            if burst + 2 < k_bursts {
                                tcgen05_commit_shared_cluster(
                                    b_free_bar_ptr.add(slot).cast::<u64>(),
                                );
                            }
                            if burst + 1 == k_bursts {
                                tcgen05_commit_shared_cluster(mma_bar_ptr.cast::<u64>());
                                while !mbarrier_try_wait_parity(mma_bar_const, (g_idx & 1) as u32) {}
                            }
                            step += 1;
                            burst += 1;
                        }
                    } else {
                        step += k_bursts;
                    }
                    barrier_cta_sync(1, 128);
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let regs =
                            tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32)).to_array();
                        #[unroll]
                        for c in 0..16 {
                            let n = n_base + c;
                            if n < limit {
                                let a = f32::from_bits(regs[c]) * global_scale;
                                let u = if a > 0.0 { a } else { 0.0 };
                                *smem_epi_f32.add(n * 128 + col_perm) = u * u;
                            }
                        }
                        n_base += 16;
                    }
                    tcgen05_load_wait();
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 128);

                    let mut n = row_off16;
                    while n < limit {
                        let row_u128 = smem_epi_u128.add(n * 32 + sub_tile16);
                        let q0 = *row_u128;
                        let q1 = *row_u128.add(8);
                        let q2 = *row_u128.add(16);
                        let q3 = *row_u128.add(24);
                        let vals = [
                            f32::from_bits((q0 & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q0 >> 32) & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q0 >> 64) & 0xFFFF_FFFF) as u32),
                            f32::from_bits((q0 >> 96) as u32),
                            f32::from_bits((q1 & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q1 >> 32) & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q1 >> 64) & 0xFFFF_FFFF) as u32),
                            f32::from_bits((q1 >> 96) as u32),
                            f32::from_bits((q2 & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q2 >> 32) & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q2 >> 64) & 0xFFFF_FFFF) as u32),
                            f32::from_bits((q2 >> 96) as u32),
                            f32::from_bits((q3 & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q3 >> 32) & 0xFFFF_FFFF) as u32),
                            f32::from_bits(((q3 >> 64) & 0xFFFF_FFFF) as u32),
                            f32::from_bits((q3 >> 96) as u32),
                        ];
                        let mut amax = 0.0f32;
                        #[unroll]
                        for i in 0..16 {
                            if vals[i] > amax {
                                amax = vals[i];
                            }
                        }
                        let scale_code = dev_f32_to_e4m3_rne(amax * (1.0f32 / 6.0f32));
                        let inv_scale = exact_inv_scale_e2m1(scale_code);
                        let mut packed_u64 = 0u64;
                        #[unroll]
                        for i in 0..16 {
                            packed_u64 |=
                                (fast_pos_e2m1_rne(vals[i] * inv_scale) as u64) << (i * 4);
                        }
                        let u64_idx =
                            (group * inter_bursts + burst_inter) * 4096 + sub_u64_off + n * 2;
                        let sfb_word = scale_atom_word_192e(n);
                        let s_idx =
                            (group * inter_atoms + k_atom) * 512 + sfb_word * 4 + k_in_atom;
                        *inter_u64_ptr.add(u64_idx) = packed_u64;
                        *inter_scales_ptr.add(s_idx) = scale_code;
                        n += 16;
                    }
                    fence_proxy_async_shared_cta();
                    barrier_cta_sync(1, 128);
                    if group + 1 < group_end && tid == 0 {
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const);
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const.add(1));
                    }
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 256);
                }
                if tid < 5 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage 4a (`K4` for `B <= 16`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` with Fused Top-4 `BF16x2` Atomic Combine (`limit <= 16`).
    ///
    /// Since `B <= 16` guarantees `limit <= 16` (`valid_n = 16`), unloads `TMEM` in a single
    /// `x16` pass (`n_base = 0`) and fuses `routing_weight * x` + `atom_add_bf16x2` directly,
    /// eliminating the separate `Finalize` kernel launch!
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_fc2_m256_atomic_192e(
        #[grid_constant] w_down_tma: &TmaDescriptor,
        #[grid_constant] b_down_tma: &TmaDescriptor,
        #[grid_constant] sfa_down_tma: &TmaDescriptor,
        #[grid_constant] sfb_down_tma: &TmaDescriptor,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_routes: &[i32],
        group_count: &[i32],
        routing_weights: &[u16],
        mut output: DisjointSlice<u16>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m256_tiles = HIDDEN_192E / 256; // 24
        let tile_m256 = cta_id % m256_tiles;
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let worker_id = cta_id / m256_tiles;
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
        let smem_a1_ptr = unsafe { dyn_smem.add(SMEM_FC2_A1_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_FC2_B_OFFSET) };
        let smem_sfa0_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFA0_OFFSET) };
        let smem_sfa1_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFA1_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_FC2_BAR_OFFSET).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { tma_bar_ptr.add(2) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 5 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 512);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 4 && (tid & 31) == 0 {
                let w_desc = core::ptr::from_ref(w_down_tma);
                let b_desc = core::ptr::from_ref(b_down_tma);
                let sfa_desc = core::ptr::from_ref(sfa_down_tma);
                let sfb_desc = core::ptr::from_ref(sfb_down_tma);
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = step & 1;
                        let phase = ((step >> 1) & 1) as u32;
                        if step >= 2 {
                            while !mbarrier_try_wait_parity(b_free_bar_const.add(slot), phase ^ 1) {}
                        }
                        let bar = tma_bar_ptr.add(slot);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, STAGE_TX_BYTES_FC2_192E);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_b_ptr.add(slot * ACT64_BURST_BYTES),
                            b_desc,
                            0,
                            ((group * k_bursts + burst) * GROUP_TILE_ROWS) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfb_ptr.add(slot * SCALE_BURST_BYTES),
                            sfb_desc,
                            0,
                            ((group * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a0_base0 = cvta_generic_to_shared_offset(smem_a0_ptr.cast_const());
                let smem_a1_base0 = smem_a0_base0 + (SMEM_FC2_A1_OFFSET as u64);
                let smem_b_base0 = smem_a0_base0 + (SMEM_FC2_B_OFFSET as u64);
                let smem_sfa0_base0 = smem_a0_base0 + (SMEM_FC2_SFA0_OFFSET as u64);
                let smem_sfa1_base0 = smem_a0_base0 + (SMEM_FC2_SFA1_OFFSET as u64);
                let smem_sfb_base0 = smem_a0_base0 + (SMEM_FC2_SFB_OFFSET as u64);
                let tmem_row_base = tmem_addr + (warp_id << 21);
                let out_u32_ptr = output.as_mut_ptr().cast::<u32>();
                let smem_epi0_u32 = smem_a0_ptr.cast::<u32>();
                let smem_epi0_u128 = smem_a0_ptr.cast_const().cast::<u128>();
                let smem_epi1_u32 = smem_a1_ptr.cast::<u32>();
                let smem_epi1_u128 = smem_a1_ptr.cast_const().cast::<u128>();
                let col16 = tid & 15;
                let row8 = tid >> 4;
                let col_u32_0 = tile_m0 * 64 + col16 * 4;
                let col_u32_1 = tile_m1 * 64 + col16 * 4;
                let h_u32 = HIDDEN_192E / 2;
                let idesc = encode_idesc_192e(16);

                let mut step = 0usize;
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let global_scale = *w_down_global.as_ptr().add(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            let slot = step & 1;
                            let phase = ((step >> 1) & 1) as u32;
                            while !mbarrier_try_wait_parity(tma_bar_const.add(slot), phase) {}
                            let sfa0_tmem_b = tmem_addr + 256 + (slot as u32) * 32;
                            let sfa1_tmem_b = tmem_addr + 320 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 384 + (slot as u32) * 32;
                            let a0_desc0 = build_smem_desc_192e(
                                smem_a0_base0 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let a1_desc0 = build_smem_desc_192e(
                                smem_a1_base0 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let b_desc0 = build_smem_desc_192e(
                                smem_b_base0 + (slot * ACT64_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa0_desc0 = build_smem_desc_192e(
                                smem_sfa0_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa1_desc0 = build_smem_desc_192e(
                                smem_sfa1_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sb_desc0 = build_smem_desc_192e(
                                smem_sfb_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            #[unroll]
                            for s in 0..8 {
                                let sa0 = sfa0_tmem_b + (s as u32) * 4;
                                let sa1 = sfa1_tmem_b + (s as u32) * 4;
                                let sb = sfb_tmem_b + (s as u32) * 4;
                                tcgen05_cp_32x128b_warpx4(sa0, sa0_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sa1, sa1_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sb, sb_desc0 + (s as u64) * 32);
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr,
                                    a0_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa0,
                                    sb,
                                );
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr + 128,
                                    a1_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa1,
                                    sb,
                                );
                            }
                            if burst + 2 < k_bursts {
                                tcgen05_commit_shared_cluster(
                                    b_free_bar_ptr.add(slot).cast::<u64>(),
                                );
                            }
                            if burst + 1 == k_bursts {
                                tcgen05_commit_shared_cluster(mma_bar_ptr.cast::<u64>());
                                while !mbarrier_try_wait_parity(mma_bar_const, (g_idx & 1) as u32) {}
                            }
                            step += 1;
                            burst += 1;
                        }
                    } else {
                        step += k_bursts;
                    }
                    barrier_cta_sync(1, 128);
                    tcgen05_fence_after_thread_sync();

                    let r0 = tcgen05_ld_32x32b_x16_raw(tmem_row_base).to_array();
                    let r1 = tcgen05_ld_32x32b_x16_raw(tmem_row_base + 128).to_array();
                    #[unroll]
                    for c in 0..16 {
                        if c < limit {
                            *smem_epi0_u32.add(c * 128 + tid) = r0[c];
                            *smem_epi1_u32.add(c * 128 + tid) = r1[c];
                        }
                    }
                    tcgen05_load_wait();
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 128);

                    let mut n = row8;
                    while n < limit {
                        let route = *group_routes
                            .as_ptr()
                            .add(group * GROUP_TILE_ROWS + n)
                            as usize;
                        let token = route >> 2;
                        let sw =
                            global_scale * bf16_to_f32(*routing_weights.as_ptr().add(route));
                        let q0_a = *smem_epi0_u128.add(n * 32 + col16 * 2);
                        let q0_b = *smem_epi0_u128.add(n * 32 + col16 * 2 + 1);
                        let q1_a = *smem_epi1_u128.add(n * 32 + col16 * 2);
                        let q1_b = *smem_epi1_u128.add(n * 32 + col16 * 2 + 1);
                        let d0 = out_u32_ptr.add(token * h_u32 + col_u32_0);
                        let d1 = out_u32_ptr.add(token * h_u32 + col_u32_1);

                        macro_rules! atomic_quad {
                            ($q:ident, $dst:expr) => {{
                                let f0 = f32::from_bits(($q & 0xFFFF_FFFF) as u32) * sw;
                                let f1 = f32::from_bits((($q >> 32) & 0xFFFF_FFFF) as u32) * sw;
                                let f2 = f32::from_bits((($q >> 64) & 0xFFFF_FFFF) as u32) * sw;
                                let f3 = f32::from_bits(($q >> 96) as u32) * sw;
                                let b0 = f32_to_bf16_rne(f0) as u32;
                                let b1 = f32_to_bf16_rne(f1) as u32;
                                let b2 = f32_to_bf16_rne(f2) as u32;
                                let b3 = f32_to_bf16_rne(f3) as u32;
                                let _ = atom_add_bf16x2($dst, b0 | (b1 << 16));
                                let _ = atom_add_bf16x2(($dst).add(1), b2 | (b3 << 16));
                            }};
                        }
                        atomic_quad!(q0_a, d0);
                        atomic_quad!(q0_b, d0.add(2));
                        atomic_quad!(q1_a, d1);
                        atomic_quad!(q1_b, d1.add(2));
                        n += 8;
                    }
                    fence_proxy_async_shared_cta();
                    barrier_cta_sync(1, 128);
                    if group + 1 < group_end && tid == 0 {
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const);
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const.add(1));
                    }
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 512);
                }
                if tid < 5 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage 4b (`K4` for `B = 32, 64, 128`): `5`-Warp (`160`-thread) Persistent Dual-M-Tile (`M = 256` = `2 × 128`)
    /// K-Streaming `tcgen05` `FC2` writing `grouped_out` (`BF16`) for deterministic `FP32` combine.
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_fc2_m256_192e(
        #[grid_constant] w_down_tma: &TmaDescriptor,
        #[grid_constant] b_down_tma: &TmaDescriptor,
        #[grid_constant] sfa_down_tma: &TmaDescriptor,
        #[grid_constant] sfb_down_tma: &TmaDescriptor,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_count: &[i32],
        mut grouped_out: DisjointSlice<u16>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m256_tiles = HIDDEN_192E / 256; // 24
        let tile_m256 = cta_id % m256_tiles;
        let tile_m0 = tile_m256 * 2;
        let tile_m1 = tile_m0 + 1;
        let worker_id = cta_id / m256_tiles;
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
        let smem_a1_ptr = unsafe { dyn_smem.add(SMEM_FC2_A1_OFFSET) };
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_FC2_B_OFFSET) };
        let smem_sfa0_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFA0_OFFSET) };
        let smem_sfa1_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFA1_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_FC2_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_FC2_BAR_OFFSET).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { tma_bar_ptr.add(2) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(2) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 5 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 512);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 4 && (tid & 31) == 0 {
                let w_desc = core::ptr::from_ref(w_down_tma);
                let b_desc = core::ptr::from_ref(b_down_tma);
                let sfa_desc = core::ptr::from_ref(sfa_down_tma);
                let sfb_desc = core::ptr::from_ref(sfb_down_tma);
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = step & 1;
                        let phase = ((step >> 1) & 1) as u32;
                        if step >= 2 {
                            while !mbarrier_try_wait_parity(b_free_bar_const.add(slot), phase ^ 1) {}
                        }
                        let bar = tma_bar_ptr.add(slot);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, STAGE_TX_BYTES_FC2_192E);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_b_ptr.add(slot * ACT64_BURST_BYTES),
                            b_desc,
                            0,
                            ((group * k_bursts + burst) * GROUP_TILE_ROWS) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfb_ptr.add(slot * SCALE_BURST_BYTES),
                            sfb_desc,
                            0,
                            ((group * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a0_base0 = cvta_generic_to_shared_offset(smem_a0_ptr.cast_const());
                let smem_a1_base0 = smem_a0_base0 + (SMEM_FC2_A1_OFFSET as u64);
                let smem_b_base0 = smem_a0_base0 + (SMEM_FC2_B_OFFSET as u64);
                let smem_sfa0_base0 = smem_a0_base0 + (SMEM_FC2_SFA0_OFFSET as u64);
                let smem_sfa1_base0 = smem_a0_base0 + (SMEM_FC2_SFA1_OFFSET as u64);
                let smem_sfb_base0 = smem_a0_base0 + (SMEM_FC2_SFB_OFFSET as u64);
                let tmem_row_base = tmem_addr + (warp_id << 21);
                let out_u128_ptr = grouped_out.as_mut_ptr().cast::<u128>();
                let smem_epi0_u16 = smem_a0_ptr.cast::<u16>();
                let smem_epi0_u128 = smem_a0_ptr.cast_const().cast::<u128>();
                let smem_epi1_u16 = smem_a1_ptr.cast::<u16>();
                let smem_epi1_u128 = smem_a1_ptr.cast_const().cast::<u128>();
                let col16 = tid & 15;
                let row8 = tid >> 4;
                let dst_col0_u128 = tile_m0 * 16 + col16;
                let dst_col1_u128 = tile_m1 * 16 + col16;
                let h_u128 = HIDDEN_192E / 8;

                let mut step = 0usize;
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let idesc = encode_idesc_192e(valid_n as u32);
                    let global_scale = *w_down_global.as_ptr().add(expert);

                    if tid == 0 {
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            let slot = step & 1;
                            let phase = ((step >> 1) & 1) as u32;
                            while !mbarrier_try_wait_parity(tma_bar_const.add(slot), phase) {}
                            let sfa0_tmem_b = tmem_addr + 256 + (slot as u32) * 32;
                            let sfa1_tmem_b = tmem_addr + 320 + (slot as u32) * 32;
                            let sfb_tmem_b = tmem_addr + 384 + (slot as u32) * 32;
                            let a0_desc0 = build_smem_desc_192e(
                                smem_a0_base0 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let a1_desc0 = build_smem_desc_192e(
                                smem_a1_base0 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let b_desc0 = build_smem_desc_192e(
                                smem_b_base0 + (slot * ACT64_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa0_desc0 = build_smem_desc_192e(
                                smem_sfa0_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa1_desc0 = build_smem_desc_192e(
                                smem_sfa1_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sb_desc0 = build_smem_desc_192e(
                                smem_sfb_base0 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            #[unroll]
                            for s in 0..8 {
                                let sa0 = sfa0_tmem_b + (s as u32) * 4;
                                let sa1 = sfa1_tmem_b + (s as u32) * 4;
                                let sb = sfb_tmem_b + (s as u32) * 4;
                                tcgen05_cp_32x128b_warpx4(sa0, sa0_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sa1, sa1_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sb, sb_desc0 + (s as u64) * 32);
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr,
                                    a0_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa0,
                                    sb,
                                );
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr + 128,
                                    a1_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa1,
                                    sb,
                                );
                            }
                            if burst + 2 < k_bursts {
                                tcgen05_commit_shared_cluster(
                                    b_free_bar_ptr.add(slot).cast::<u64>(),
                                );
                            }
                            if burst + 1 == k_bursts {
                                tcgen05_commit_shared_cluster(mma_bar_ptr.cast::<u64>());
                                while !mbarrier_try_wait_parity(mma_bar_const, (g_idx & 1) as u32) {}
                            }
                            step += 1;
                            burst += 1;
                        }
                    } else {
                        step += k_bursts;
                    }
                    barrier_cta_sync(1, 128);
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let r0 = tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32))
                            .to_array();
                        let r1 =
                            tcgen05_ld_32x32b_x16_raw(tmem_row_base + 128 + (n_base as u32))
                                .to_array();
                        #[unroll]
                        for c in 0..16 {
                            let n = n_base + c;
                            if n < limit {
                                let v0 = f32::from_bits(r0[c]) * global_scale;
                                let v1 = f32::from_bits(r1[c]) * global_scale;
                                *smem_epi0_u16.add(n * 128 + tid) = f32_to_bf16_rne(v0);
                                *smem_epi1_u16.add(n * 128 + tid) = f32_to_bf16_rne(v1);
                            }
                        }
                        n_base += 16;
                    }
                    tcgen05_load_wait();
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 128);

                    let group_row_base = group * GROUP_TILE_ROWS;
                    let mut n = row8;
                    while n < limit {
                        let q0 = *smem_epi0_u128.add(n * 16 + col16);
                        let q1 = *smem_epi1_u128.add(n * 16 + col16);
                        let row_off = (group_row_base + n) * h_u128;
                        *out_u128_ptr.add(row_off + dst_col0_u128) = q0;
                        *out_u128_ptr.add(row_off + dst_col1_u128) = q1;
                        n += 8;
                    }
                    fence_proxy_async_shared_cta();
                    barrier_cta_sync(1, 128);
                    if group + 1 < group_end && tid == 0 {
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const);
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const.add(1));
                    }
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 512);
                }
                if tid < 5 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage 4b (`K4` for `B = 256`): `5`-Warp (`160`-thread) Triple-M-Tile (`M = 384` = `3 × 128`)
    /// K-Streaming `tcgen05` `FC2` (`H = 6,144, I = 1,536`, `3` bursts of `K = 512`).
    ///
    /// Double-buffers `A0 + A1 + SFA0 + SFA1` (`72 KB` per stage) while single-buffering `A2 + B + SFA2 + SFB`
    /// (`72 KB`) in `222,208` bytes of shared memory so `2/3` of the cold HBM weights for `burst + 1`
    /// prefetch concurrently with `burst`'s MMA across all `16 × 9 = 144 <= 148` CTAs in 1 single wave!
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        block = (160, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_fc2_192e(
        #[grid_constant] w_down_tma: &TmaDescriptor,
        #[grid_constant] b_down_tma: &TmaDescriptor,
        #[grid_constant] sfa_down_tma: &TmaDescriptor,
        #[grid_constant] sfb_down_tma: &TmaDescriptor,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        group_count: &[i32],
        mut grouped_out: DisjointSlice<u16>,
        group_step: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let m384_tiles = HIDDEN_192E / 384; // 16
        let tile_m384 = cta_id % m384_tiles;
        let tile_m0 = tile_m384 * 3;
        let tile_m1 = tile_m0 + 1;
        let tile_m2 = tile_m0 + 2;
        let worker_id = cta_id / m384_tiles;
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
        let tma_bar_ptr = unsafe { dyn_smem.add(221_184).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let a_free_bar_ptr = unsafe { tma_bar_ptr.add(2) };
        let a_free_bar_const = a_free_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { a_free_bar_ptr.add(2) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(1) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { mma_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 6 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 512);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 4 && (tid & 31) == 0 {
                let w_desc = core::ptr::from_ref(w_down_tma);
                let b_desc = core::ptr::from_ref(b_down_tma);
                let sfa_desc = core::ptr::from_ref(sfa_down_tma);
                let sfb_desc = core::ptr::from_ref(sfb_down_tma);
                let mut step = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let w_outer0 = expert * m128_tiles + tile_m0;
                    let w_outer1 = expert * m128_tiles + tile_m1;
                    let w_outer2 = expert * m128_tiles + tile_m2;
                    let mut burst = 0usize;
                    while burst < k_bursts {
                        let slot = step & 1;
                        let phase = ((step >> 1) & 1) as u32;
                        if step >= 2 {
                            while !mbarrier_try_wait_parity(a_free_bar_const.add(slot), phase ^ 1) {}
                        }
                        let bar = tma_bar_ptr.add(slot);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, STAGE_TX_BYTES_FC2_M384);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a0_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a1_ptr.add(slot * WEIGHT_BURST_BYTES),
                            w_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa0_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer0 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa1_ptr.add(slot * SCALE_BURST_BYTES),
                            sfa_desc,
                            0,
                            ((w_outer1 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        if step >= 1 {
                            while !mbarrier_try_wait_parity(
                                b_free_bar_const,
                                ((step - 1) & 1) as u32,
                            ) {}
                        }
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_a2_ptr,
                            w_desc,
                            0,
                            ((w_outer2 * k_bursts + burst) * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_b_ptr,
                            b_desc,
                            0,
                            ((group * k_bursts + burst) * GROUP_TILE_ROWS) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfa2_ptr,
                            sfa_desc,
                            0,
                            ((w_outer2 * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            smem_sfb_ptr,
                            sfb_desc,
                            0,
                            ((group * k_bursts + burst) * 8) as i32,
                            bar,
                        );
                        step += 1;
                        burst += 1;
                    }
                    group += 1;
                }
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a0_base = cvta_generic_to_shared_offset(smem_a0_ptr.cast_const());
                let a2_desc0 = build_smem_desc_192e(smem_a0_base + 131_072, 2048, 128);
                let b_desc0 = build_smem_desc_192e(smem_a0_base + 163_840, 2048, 128);
                let sa2_desc0 = build_smem_desc_192e(smem_a0_base + 212_992, 2048, 128);
                let sb_desc0 = build_smem_desc_192e(smem_a0_base + 217_088, 2048, 128);

                let tmem_row_base = tmem_addr + (warp_id << 21);
                let out_u128_ptr = grouped_out.as_mut_ptr().cast::<u128>();
                let smem_epi0_u16 = smem_a0_ptr.cast::<u16>();
                let smem_epi0_u128 = smem_a0_ptr.cast_const().cast::<u128>();
                let smem_epi1_u16 = smem_a1_ptr.cast::<u16>();
                let smem_epi1_u128 = smem_a1_ptr.cast_const().cast::<u128>();
                let smem_epi2_u16 = smem_a2_ptr.cast::<u16>();
                let smem_epi2_u128 = smem_a2_ptr.cast_const().cast::<u128>();
                let col16 = tid & 15;
                let row8 = tid >> 4;
                let dst_col0_u128 = tile_m0 * 16 + col16;
                let dst_col1_u128 = tile_m1 * 16 + col16;
                let dst_col2_u128 = tile_m2 * 16 + col16;
                let h_u128 = HIDDEN_192E / 8;

                let mut step = 0usize;
                let mut g_idx = 0usize;
                let mut group = group_start;
                while group < group_end {
                    let expert = *group_experts.as_ptr().add(group) as usize;
                    let limit = *group_limits.as_ptr().add(group) as usize;
                    let valid_n = ((limit + 15) & !15).max(16);
                    let idesc = encode_idesc_192e(valid_n as u32);
                    let global_scale = *w_down_global.as_ptr().add(expert);

                    if tid == 0 {
                        let sfa0_tmem_b = tmem_addr + 384;
                        let sfa1_tmem_b = tmem_addr + 416;
                        let sfa2_tmem_b = tmem_addr + 448;
                        let sfb_tmem_b = tmem_addr + 480;
                        let mut burst = 0usize;
                        while burst < k_bursts {
                            let slot = step & 1;
                            let phase = ((step >> 1) & 1) as u32;
                            while !mbarrier_try_wait_parity(tma_bar_const.add(slot), phase) {}
                            let a0_desc0 = build_smem_desc_192e(
                                smem_a0_base + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let a1_desc0 = build_smem_desc_192e(
                                smem_a0_base + 65_536 + (slot * WEIGHT_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa0_desc0 = build_smem_desc_192e(
                                smem_a0_base + 196_608 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            let sa1_desc0 = build_smem_desc_192e(
                                smem_a0_base + 204_800 + (slot * SCALE_BURST_BYTES) as u64,
                                2048,
                                128,
                            );
                            #[unroll]
                            for s in 0..8 {
                                let sa0 = sfa0_tmem_b + (s as u32) * 4;
                                let sa1 = sfa1_tmem_b + (s as u32) * 4;
                                let sa2 = sfa2_tmem_b + (s as u32) * 4;
                                let sb = sfb_tmem_b + (s as u32) * 4;
                                tcgen05_cp_32x128b_warpx4(sa0, sa0_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sa1, sa1_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sa2, sa2_desc0 + (s as u64) * 32);
                                tcgen05_cp_32x128b_warpx4(sb, sb_desc0 + (s as u64) * 32);
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr,
                                    a0_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa0,
                                    sb,
                                );
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr + 128,
                                    a1_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa1,
                                    sb,
                                );
                                tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                                    tmem_addr + 256,
                                    a2_desc0 + (s as u64) * 256,
                                    b_desc0 + (s as u64) * 256,
                                    idesc,
                                    !(burst == 0 && s == 0),
                                    sa2,
                                    sb,
                                );
                            }
                            if burst + 1 < k_bursts {
                                tcgen05_commit_shared_cluster(b_free_bar_ptr.cast::<u64>());
                            }
                            if burst == 0 {
                                tcgen05_commit_shared_cluster(
                                    a_free_bar_ptr.add(slot).cast::<u64>(),
                                );
                            }
                            if burst + 1 == k_bursts {
                                tcgen05_commit_shared_cluster(mma_bar_ptr.cast::<u64>());
                                while !mbarrier_try_wait_parity(mma_bar_const, (g_idx & 1) as u32) {}
                            }
                            step += 1;
                            burst += 1;
                        }
                    } else {
                        step += k_bursts;
                    }
                    barrier_cta_sync(1, 128);
                    tcgen05_fence_after_thread_sync();

                    let mut n_base = 0usize;
                    while n_base < limit {
                        let r0 =
                            tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32)).to_array();
                        let r1 = tcgen05_ld_32x32b_x16_raw(tmem_row_base + 128 + (n_base as u32))
                            .to_array();
                        let r2 = tcgen05_ld_32x32b_x16_raw(tmem_row_base + 256 + (n_base as u32))
                            .to_array();
                        #[unroll]
                        for c in 0..16 {
                            let n = n_base + c;
                            if n < limit {
                                let v0 = f32::from_bits(r0[c]) * global_scale;
                                let v1 = f32::from_bits(r1[c]) * global_scale;
                                let v2 = f32::from_bits(r2[c]) * global_scale;
                                *smem_epi0_u16.add(n * 128 + tid) = f32_to_bf16_rne(v0);
                                *smem_epi1_u16.add(n * 128 + tid) = f32_to_bf16_rne(v1);
                                *smem_epi2_u16.add(n * 128 + tid) = f32_to_bf16_rne(v2);
                            }
                        }
                        n_base += 16;
                    }
                    tcgen05_load_wait();
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 128);

                    let group_row_base = group * GROUP_TILE_ROWS;
                    let mut n = row8;
                    while n < limit {
                        let q0 = *smem_epi0_u128.add(n * 16 + col16);
                        let q1 = *smem_epi1_u128.add(n * 16 + col16);
                        let q2 = *smem_epi2_u128.add(n * 16 + col16);
                        let row_off = (group_row_base + n) * h_u128;
                        *out_u128_ptr.add(row_off + dst_col0_u128) = q0;
                        *out_u128_ptr.add(row_off + dst_col1_u128) = q1;
                        *out_u128_ptr.add(row_off + dst_col2_u128) = q2;
                        n += 8;
                    }
                    fence_proxy_async_shared_cta();
                    barrier_cta_sync(1, 128);
                    if group + 1 < group_end && tid == 0 {
                        let _ = cuda_device::barrier::mbarrier_arrive(a_free_bar_const);
                        let _ = cuda_device::barrier::mbarrier_arrive(a_free_bar_const.add(1));
                        let _ = cuda_device::barrier::mbarrier_arrive(b_free_bar_const);
                    }
                    g_idx += 1;
                    group += 1;
                }

                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 512);
                }
                if tid < 6 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage 5 (`K5`): Coalesced `u128` (`8 × bf16`) top-4 `FP32` weighted combine (`[B, 6,144]`).
    ///
    /// Accumulates `routing_weight * bf16_to_f32(expert_out)` in `FP32` across the 4 routed experts
    /// and rounds once to `BF16`, matching the reference accumulation order deterministically.
    #[kernel]
    pub fn moe_finalize_192e(
        grouped_out: &[u16],
        tok_to_perm: &[i32],
        routing_weights: &[u16],
        mut output: DisjointSlice<u16>,
        num_tokens: u32,
    ) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let h_u128 = HIDDEN_192E / 8; // 768 u128s per token row
        let total_u128 = (num_tokens as usize) * h_u128;
        if idx >= total_u128 {
            return;
        }
        let token = idx / h_u128;
        let col_u128 = idx % h_u128;

        unsafe {
            let gp128 = *tok_to_perm.as_ptr().cast::<u128>().add(token);
            let w64 = *routing_weights.as_ptr().cast::<u64>().add(token);

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

            let grouped_u128 = grouped_out.as_ptr().cast::<u128>();
            let q0 = *grouped_u128.add(r0);
            let q1 = *grouped_u128.add(r1);
            let q2 = *grouped_u128.add(r2);
            let q3 = *grouped_u128.add(r3);
            let qs = [q0, q1, q2, q3];
            let ws = [s0, s1, s2, s3];

            let mut acc = [0.0f32; 8];
            #[unroll]
            for slot in 0..4 {
                let q = qs[slot];
                let w = ws[slot];
                let u0 = (q & 0xFFFF_FFFF) as u32;
                let u1 = ((q >> 32) & 0xFFFF_FFFF) as u32;
                let u2 = ((q >> 64) & 0xFFFF_FFFF) as u32;
                let u3 = (q >> 96) as u32;
                acc[0] += f32::from_bits(u0 << 16) * w;
                acc[1] += f32::from_bits(u0 & 0xFFFF_0000) * w;
                acc[2] += f32::from_bits(u1 << 16) * w;
                acc[3] += f32::from_bits(u1 & 0xFFFF_0000) * w;
                acc[4] += f32::from_bits(u2 << 16) * w;
                acc[5] += f32::from_bits(u2 & 0xFFFF_0000) * w;
                acc[6] += f32::from_bits(u3 << 16) * w;
                acc[7] += f32::from_bits(u3 & 0xFFFF_0000) * w;
            }
            let p0 = cvt_bf16x2_f32(acc[0], acc[1]);
            let p1 = cvt_bf16x2_f32(acc[2], acc[3]);
            let p2 = cvt_bf16x2_f32(acc[4], acc[5]);
            let p3 = cvt_bf16x2_f32(acc[6], acc[7]);
            *output.as_mut_ptr().cast::<u128>().add(idx) = (p0 as u128)
                | ((p1 as u128) << 32)
                | ((p2 as u128) << 64)
                | ((p3 as u128) << 96);
        }
    }
}

/// Pack logical row-major packed FP4 weights `[rows, k / 2]` into the stage-blocked
/// `[rows / 128, k / 512, 8, 2, 128, 16]` TMA burst layout for one expert.
fn pack_expert_tma_weights(logical: &[u8], rows: usize, k: usize, out: &mut [u8]) {
    let packed_k = k / 2;
    let m_tiles = rows / 128;
    let k_bursts = packed_k / 256;
    for mt in 0..m_tiles {
        for kb in 0..k_bursts {
            for s in 0..8 {
                for half_s in 0..2 {
                    for r in 0..128 {
                        let src_row = mt * 128 + r;
                        let src_col = kb * 256 + s * 32 + half_s * 16;
                        let dst_off =
                            ((((mt * k_bursts + kb) * 8 + s) * 2 + half_s) * 128 + r) * 16;
                        out[dst_off..dst_off + 16]
                            .copy_from_slice(&logical[src_row * packed_k + src_col..src_row * packed_k + src_col + 16]);
                    }
                }
            }
        }
    }
}

/// Pack logical row-major E4M3 scales `[rows, k / 16]` into canonical TensorRT/CUTLASS `R128c4`
/// `(32, 4, 4)` 512-byte scale atoms.
fn pack_expert_r128c4_scales(logical: &[u8], rows: usize, k: usize, out: &mut [u8]) {
    let cols = k / 16;
    let m_tiles = rows / 128;
    let k_atoms = cols / 4;
    for mt in 0..m_tiles {
        for ka in 0..k_atoms {
            for r32 in 0..32 {
                for q4 in 0..4 {
                    for c4 in 0..4 {
                        let src_row = mt * 128 + q4 * 32 + r32;
                        let src_col = ka * 4 + c4;
                        let dst_off = (((mt * k_atoms + ka) * 32 + r32) * 4 + q4) * 4 + c4;
                        out[dst_off] = logical[src_row * cols + src_col];
                    }
                }
            }
        }
    }
}

/// Device buffers and 2D TMA descriptors for the 192-expert (`H = 6,144, I = 1,536`) MoE layer.
struct Moe192Buffers {
    /// Input token matrix `[num_tokens, 6144]` in `FP16` (`u16` bit patterns).
    hidden: DeviceBuffer<u16>,
    /// Supplied expert IDs `[num_tokens * 4]` (`i32` in `0..192`).
    expert_ids: DeviceBuffer<i32>,
    /// Supplied per-token routing weights `[num_tokens * 4]` in `BF16` (`u16` bit patterns).
    routing_weights: DeviceBuffer<u16>,
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
    /// TMA-ready `FC1` activation values (`[max_groups * 12, 64, 256]` bytes).
    b_up_vals: DeviceBuffer<u8>,
    /// TMA-ready `FC1` activation scales (`[max_groups * 12, 8, 256]` `u16`s).
    sfb_up_scales: DeviceBuffer<u8>,
    /// TMA-ready `FC1` intermediate outputs (`[max_groups * 3, 64, 256]` bytes).
    inter_vals: DeviceBuffer<u8>,
    /// TMA-ready `FC1` intermediate scales (`[max_groups * 3, 8, 256]` `u16`s).
    inter_scales: DeviceBuffer<u8>,
    /// Packed `W_up` weights for all `192` experts (`192 × 1536 × 3072` bytes).
    w_up: DeviceBuffer<u8>,
    /// Canonical `R128c4` `W_up` scales for all `192` experts.
    w_up_scales: DeviceBuffer<u8>,
    /// Per-expert `FC1` global dequantization scale (`[192]`).
    w_up_global: DeviceBuffer<f32>,
    /// Packed `W_down` weights for all `192` experts (`192 × 6144 × 768` bytes).
    w_down: DeviceBuffer<u8>,
    /// Canonical `R128c4` `W_down` scales for all `192` experts.
    w_down_scales: DeviceBuffer<u8>,
    /// Per-expert `FC2` global dequantization scale (`[192]`).
    w_down_global: DeviceBuffer<f32>,
    /// 2D TMA descriptor for `W_up`.
    w_up_tma: TmaDescriptor,
    /// 2D TMA descriptor for `b_up_vals`.
    b_up_tma: TmaDescriptor,
    /// 2D TMA descriptor for `w_up_scales`.
    sfa_up_tma: TmaDescriptor,
    /// 2D TMA descriptor for `sfb_up_scales`.
    sfb_up_tma: TmaDescriptor,
    /// 2D TMA descriptor for `W_down`.
    w_down_tma: TmaDescriptor,
    /// 2D TMA descriptor for `inter_vals`.
    b_down_tma: TmaDescriptor,
    /// 2D TMA descriptor for `w_down_scales`.
    sfa_down_tma: TmaDescriptor,
    /// 2D TMA descriptor for `inter_scales`.
    sfb_down_tma: TmaDescriptor,
    /// Grouped `BF16` expert outputs (`[max_groups * 64, 6144]`).
    grouped_out: DeviceBuffer<u16>,
    /// Final `BF16` layer output (`[num_tokens, 6144]`).
    output: DeviceBuffer<u16>,
    /// Token batch size `B`.
    num_tokens: usize,
    /// Tuned logical group capacity (`16, 32, 64`).
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

/// Allocate device buffers and TMA descriptors for the 192-expert MoE layer at `MAX_TOKENS_192E`.
/// Reuses the static 192-expert weight allocations across all batch sizes `B = 1 .. 256`.
fn allocate_192e(
    stream: &Arc<CudaStream>,
    num_tokens: usize,
) -> Result<Moe192Buffers, Box<dyn std::error::Error>> {
    let tile_n = select_tile_n_192e(num_tokens);
    let max_groups = max_groups_192e(num_tokens, tile_n as usize);
    let routed_rows = num_tokens * TOP_K_192E;

    let hidden_host: Vec<u16> = (0..num_tokens)
        .flat_map(|t| {
            std::iter::repeat_n(f16::from_f32(crate::hidden_value(t)).to_bits(), HIDDEN_192E)
        })
        .collect();
    let expert_ids_host: Vec<i32> = (0..num_tokens)
        .flat_map(|t| (0..TOP_K_192E).map(move |s| crate::expert_id(t, s) as i32))
        .collect();
    let routing_host: Vec<u16> = (0..num_tokens)
        .flat_map(|t| {
            (0..TOP_K_192E).map(move |s| bf16::from_f32(crate::routing_weight(t, s)).to_bits())
        })
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
        &vec![0u8; max_groups * k_bursts_up * ACT64_BURST_BYTES],
    )?;
    let sfb_up_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_up * SCALE_BURST_BYTES],
    )?;
    let inter_vals = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_down * ACT64_BURST_BYTES],
    )?;
    let inter_scales = DeviceBuffer::from_host(
        stream,
        &vec![0u8; max_groups * k_bursts_down * SCALE_BURST_BYTES],
    )?;

    let swizzle_none = sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE;
    let w_up_tma = encode_tma_2d::<u8, _>(
        &w_up,
        (NUM_EXPERTS_192E * INTERMEDIATE_192E * k_bursts_up) as u64,
        256,
        128,
        256,
        swizzle_none,
    )?;
    let b_up_tma = encode_tma_2d::<u8, _>(
        &b_up_vals,
        (max_groups * k_bursts_up * GROUP_TILE_ROWS) as u64,
        256,
        GROUP_TILE_ROWS as u32,
        256,
        swizzle_none,
    )?;
    let sfa_up_tma = encode_tma_2d::<u16, _>(
        &w_up_scales,
        (NUM_EXPERTS_192E * (INTERMEDIATE_192E / 128) * k_bursts_up * 8) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;
    let sfb_up_tma = encode_tma_2d::<u16, _>(
        &sfb_up_scales,
        (max_groups * k_bursts_up * 8) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;
    let w_down_tma = encode_tma_2d::<u8, _>(
        &w_down,
        (NUM_EXPERTS_192E * HIDDEN_192E * k_bursts_down) as u64,
        256,
        128,
        256,
        swizzle_none,
    )?;
    let b_down_tma = encode_tma_2d::<u8, _>(
        &inter_vals,
        (max_groups * k_bursts_down * GROUP_TILE_ROWS) as u64,
        256,
        GROUP_TILE_ROWS as u32,
        256,
        swizzle_none,
    )?;
    let sfa_down_tma = encode_tma_2d::<u16, _>(
        &w_down_scales,
        (NUM_EXPERTS_192E * (HIDDEN_192E / 128) * k_bursts_down * 8) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;
    let sfb_down_tma = encode_tma_2d::<u16, _>(
        &inter_scales,
        (max_groups * k_bursts_down * 8) as u64,
        256,
        8,
        256,
        swizzle_none,
    )?;

    Ok(Moe192Buffers {
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
            &vec![0u16; max_groups * GROUP_TILE_ROWS * HIDDEN_192E],
        )?,
        output: DeviceBuffer::from_host(stream, &vec![0u16; num_tokens * HIDDEN_192E])?,
        num_tokens,
        tile_n,
    })
}

/// Enqueue the complete 5-kernel 192-expert MoE pipeline on `stream`.
/// Used both for eager numerical validation and for CUDA Graph capture.
fn launch_pipeline_192e(
    module: &kernels_192e::LoadedModule,
    stream: &Arc<CudaStream>,
    bufs: &mut Moe192Buffers,
) -> Result<(), Box<dyn std::error::Error>> {
    let fuse_route = u32::from(bufs.num_tokens <= 16);
    let fuse_finalize = u32::from(bufs.num_tokens <= 16);
    if fuse_route == 0 {
        let routed_rows = (bufs.num_tokens * TOP_K_192E) as u32;
        let route_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 6176,
        };
        unsafe {
            module.moe_route_192e(
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
        module.quantize_fp16_192e(
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
    let fc1_cfg = LaunchConfig1D::new(fc1_ctas, 160, DYNAMIC_SMEM_192E);
    let fc1_launch = module.prepare_tcgen05_fc1_relu2_requant_192e(fc1_cfg)?;
    unsafe {
        module.tcgen05_fc1_relu2_requant_192e(
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
        let fc2_launch = module.prepare_tcgen05_fc2_192e(fc2_cfg)?;
        unsafe {
            module.tcgen05_fc2_192e(
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
        let fc2_launch = module.prepare_tcgen05_fc2_m256_atomic_192e(fc2_cfg)?;
        unsafe {
            module.tcgen05_fc2_m256_atomic_192e(
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
        let fc2_launch = module.prepare_tcgen05_fc2_m256_192e(fc2_cfg)?;
        unsafe {
            module.tcgen05_fc2_m256_192e(
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
            module.moe_finalize_192e(
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
    bufs: &Moe192Buffers,
    expected: &[f32],
) -> Result<f32, Box<dyn std::error::Error>> {
    stream.synchronize()?;
    let actual_bits = bufs.output.to_host_vec(stream)?;
    let mut max_abs = 0.0f32;
    let mut max_ref = 1.0f32;
    for (idx, (&bits, &exp)) in actual_bits.iter().zip(expected.iter()).enumerate() {
        let act = bf16::from_bits(bits).to_f32();
        if !act.is_finite() || !exp.is_finite() {
            return Err(format!("non-finite element at {idx}: actual={act}, expected={exp}").into());
        }
        max_abs = max_abs.max((act - exp).abs());
        max_ref = max_ref.max(exp.abs());
    }
    let norm_err = max_abs / max_ref;
    if norm_err > 0.01 {
        return Err(format!("output check failed: norm_err={norm_err:.6}, max_abs={max_abs:.6}").into());
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

/// Host reference helper: encode a non-negative `f32` to `Float8E4M3FN` using round-to-nearest-even.
fn host_f32_to_e4m3_rne(value: f32) -> u8 {
    let clamped = value.clamp(0.0, 448.0);
    let scaled = clamped * f32::from_bits(7 << 23);
    let bits = scaled.to_bits();
    let code = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) as u8;
    code.min(0x7E)
}

/// Host reference helper: quantize a slice of `[rows, cols]` `f32` values into packed `E2M1` bytes,
/// `E4M3` scales, and dequantized `f32` values matching `fixture.py`.
fn host_quant_dequant_nvfp4(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    const BOUNDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for g in 0..(cols / 16) {
            let base = r * cols + g * 16;
            let mut amax = 0.0f32;
            for i in 0..16 {
                amax = amax.max(x[base + i].abs());
            }
            let sc_code = host_f32_to_e4m3_rne(amax / 6.0);
            let sc_val = host_e4m3_to_f32(sc_code);
            for i in 0..16 {
                let v = if sc_val > 0.0 { x[base + i] / sc_val } else { 0.0 };
                let abs_v = v.abs();
                let mut code = 0u8;
                for (bi, &b) in BOUNDS.iter().enumerate() {
                    if abs_v > b || (abs_v == b && (bi & 1) == 1) {
                        code = (bi + 1) as u8;
                    }
                }
                if v.is_sign_negative() {
                    code |= 8;
                }
                out[base + i] = LUT[code as usize] * sc_val;
            }
        }
    }
    out
}

/// Generate the `nonuniform` or `changed-routes` inputs and compute the exact quantization-aware
/// `FP32` reference output matching `scripts/fixture.py`.
fn build_nonuniform_case(
    batch: usize,
    changed_routes: bool,
) -> (Vec<u16>, Vec<i32>, Vec<u16>, Vec<f32>) {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let pool = [0i32, 4, (NUM_EXPERTS_192E - 2) as i32, (NUM_EXPERTS_192E - 1) as i32, 2, 5];
    let mut hidden_fp16 = vec![0u16; batch * HIDDEN_192E];
    let mut hidden_f32 = vec![0.0f32; batch * HIDDEN_192E];
    let mut ids = vec![0i32; batch * TOP_K_192E];
    let mut routing_bf16 = vec![0u16; batch * TOP_K_192E];

    for t in 0..batch {
        let base_mag = crate::hidden_value(t);
        for c in 0..HIDDEN_192E {
            let factor = 0.5 + (((c * 7 + t * 3) % 19) as f32) / 16.0;
            let sign = if (c + 3 * t).is_multiple_of(11) { -1.0 } else { 1.0 };
            let h16 = f16::from_f32(base_mag * factor * sign);
            hidden_fp16[t * HIDDEN_192E + c] = h16.to_bits();
            hidden_f32[t * HIDDEN_192E + c] = h16.to_f32();
        }
        if !changed_routes {
            for s in 0..TOP_K_192E {
                let eid = ((t + s) % TOP_K_192E) as i32;
                ids[t * TOP_K_192E + s] = eid;
                routing_bf16[t * TOP_K_192E + s] =
                    bf16::from_f32(ROUTING_PATTERN_192E[eid as usize]).to_bits();
            }
        } else {
            let mut raw_w = [0.0f32; TOP_K_192E];
            let mut sum_w = 0.0f32;
            for s in 0..TOP_K_192E {
                ids[t * TOP_K_192E + s] = pool[(t / 3 + s) % pool.len()];
                raw_w[s] = ROUTING_PATTERN_192E[(t + 2 * s) % TOP_K_192E];
                sum_w += raw_w[s];
            }
            for s in 0..TOP_K_192E {
                routing_bf16[t * TOP_K_192E + s] = bf16::from_f32(raw_w[s] / sum_w).to_bits();
            }
        }
    }

    let qx = host_quant_dequant_nvfp4(&hidden_f32, batch, HIDDEN_192E);
    let mut result_f32 = vec![0.0f32; batch * HIDDEN_192E];

    let mut active_experts: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
    active_experts.sort_unstable();
    active_experts.dedup();

    for &expert in &active_experts {
        let up_g = if expert < TOP_K_192E {
            UP_FACTORS_192E[expert] / HIDDEN_192E as f32
        } else {
            1.0 / HIDDEN_192E as f32
        };
        let down_g = if expert < TOP_K_192E {
            DOWN_FACTORS_192E[expert] / INTERMEDIATE_192E as f32
        } else {
            1.0 / INTERMEDIATE_192E as f32
        };

        let routes_for_exp: Vec<(usize, usize)> = (0..batch)
            .flat_map(|t| (0..TOP_K_192E).map(move |s| (t, s)))
            .filter(|&(t, s)| ids[t * TOP_K_192E + s] as usize == expert)
            .collect();
        let n_tok = routes_for_exp.len();
        let mut fc1_act = vec![0.0f32; n_tok * INTERMEDIATE_192E];

        for r in 0..INTERMEDIATE_192E {
            for col in 0..HIDDEN_192E {
                let mag = 1 + (r * 3 + col * 5 + expert) % 6;
                let sign = if (r + col + expert).is_multiple_of(13) { 8 } else { 0 };
                let sc_code = (0x30 + ((r + (col / 16) * 3 + expert) % 3) * 8) as u8;
                let w_val = LUT[mag | sign] * host_e4m3_to_f32(sc_code);
                for (ti, &(tok, _)) in routes_for_exp.iter().enumerate() {
                    fc1_act[ti * INTERMEDIATE_192E + r] += qx[tok * HIDDEN_192E + col] * w_val;
                }
            }
        }
        for v in &mut fc1_act {
            let u = (*v * up_g).max(0.0);
            *v = u * u;
        }
        let inter_q = host_quant_dequant_nvfp4(&fc1_act, n_tok, INTERMEDIATE_192E);

        let mut fc2_out = vec![0.0f32; n_tok * HIDDEN_192E];
        for r in 0..HIDDEN_192E {
            for col in 0..INTERMEDIATE_192E {
                let mag = 1 + (r * 3 + col * 5 + expert) % 6;
                let sign = if (r + col + expert).is_multiple_of(13) { 8 } else { 0 };
                let sc_code = (0x30 + ((r + (col / 16) * 3 + expert) % 3) * 8) as u8;
                let w_val = LUT[mag | sign] * host_e4m3_to_f32(sc_code);
                for ti in 0..n_tok {
                    fc2_out[ti * HIDDEN_192E + r] += inter_q[ti * INTERMEDIATE_192E + col] * w_val;
                }
            }
        }
        for (ti, &(tok, slot)) in routes_for_exp.iter().enumerate() {
            let rw = bf16::from_bits(routing_bf16[tok * TOP_K_192E + slot]).to_f32();
            for c in 0..HIDDEN_192E {
                let rounded = bf16::from_f32(fc2_out[ti * HIDDEN_192E + c] * down_g).to_f32();
                result_f32[tok * HIDDEN_192E + c] += rounded * rw;
            }
        }
    }

    let expected: Vec<f32> = result_f32
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();
    (hidden_fp16, ids, routing_bf16, expected)
}

/// Populate `bufs.w_up`, `bufs.w_up_scales`, `bufs.w_down`, and `bufs.w_down_scales` with the
/// `nonuniform` weight and scale patterns for active experts `[0, 1, 2, 3, 4, 5, 190, 191]`.
fn upload_nonuniform_weights_192e(
    stream: &Arc<CudaStream>,
    bufs: &mut Moe192Buffers,
) -> Result<(), Box<dyn std::error::Error>> {
    let w1_expert_bytes = INTERMEDIATE_192E * (HIDDEN_192E / 2);
    let s1_expert_bytes = INTERMEDIATE_192E * (HIDDEN_192E / 16);
    let w2_expert_bytes = HIDDEN_192E * (INTERMEDIATE_192E / 2);
    let s2_expert_bytes = HIDDEN_192E * (INTERMEDIATE_192E / 16);

    let mut w1_all = vec![0x22u8; NUM_EXPERTS_192E * w1_expert_bytes];
    let mut s1_all = vec![0x38u8; NUM_EXPERTS_192E * s1_expert_bytes];
    let mut w2_all = vec![0x22u8; NUM_EXPERTS_192E * w2_expert_bytes];
    let mut s2_all = vec![0x38u8; NUM_EXPERTS_192E * s2_expert_bytes];

    for &expert in &[0usize, 1, 2, 3, 4, 5, NUM_EXPERTS_192E - 2, NUM_EXPERTS_192E - 1] {
        let mut w1_log = vec![0u8; w1_expert_bytes];
        let mut s1_log = vec![0u8; s1_expert_bytes];
        for r in 0..INTERMEDIATE_192E {
            for col_pair in 0..(HIDDEN_192E / 2) {
                let c0 = col_pair * 2;
                let c1 = c0 + 1;
                let m0 = (1 + (r * 3 + c0 * 5 + expert) % 6) as u8
                    | if (r + c0 + expert).is_multiple_of(13) { 8 } else { 0 };
                let m1 = (1 + (r * 3 + c1 * 5 + expert) % 6) as u8
                    | if (r + c1 + expert).is_multiple_of(13) { 8 } else { 0 };
                w1_log[r * (HIDDEN_192E / 2) + col_pair] = m0 | (m1 << 4);
            }
            for g in 0..(HIDDEN_192E / 16) {
                s1_log[r * (HIDDEN_192E / 16) + g] = (0x30 + ((r + g * 3 + expert) % 3) * 8) as u8;
            }
        }
        pack_expert_tma_weights(
            &w1_log,
            INTERMEDIATE_192E,
            HIDDEN_192E,
            &mut w1_all[expert * w1_expert_bytes..(expert + 1) * w1_expert_bytes],
        );
        pack_expert_r128c4_scales(
            &s1_log,
            INTERMEDIATE_192E,
            HIDDEN_192E,
            &mut s1_all[expert * s1_expert_bytes..(expert + 1) * s1_expert_bytes],
        );

        let mut w2_log = vec![0u8; w2_expert_bytes];
        let mut s2_log = vec![0u8; s2_expert_bytes];
        for r in 0..HIDDEN_192E {
            for col_pair in 0..(INTERMEDIATE_192E / 2) {
                let c0 = col_pair * 2;
                let c1 = c0 + 1;
                let m0 = (1 + (r * 3 + c0 * 5 + expert) % 6) as u8
                    | if (r + c0 + expert).is_multiple_of(13) { 8 } else { 0 };
                let m1 = (1 + (r * 3 + c1 * 5 + expert) % 6) as u8
                    | if (r + c1 + expert).is_multiple_of(13) { 8 } else { 0 };
                w2_log[r * (INTERMEDIATE_192E / 2) + col_pair] = m0 | (m1 << 4);
            }
            for g in 0..(INTERMEDIATE_192E / 16) {
                s2_log[r * (INTERMEDIATE_192E / 16) + g] =
                    (0x30 + ((r + g * 3 + expert) % 3) * 8) as u8;
            }
        }
        pack_expert_tma_weights(
            &w2_log,
            HIDDEN_192E,
            INTERMEDIATE_192E,
            &mut w2_all[expert * w2_expert_bytes..(expert + 1) * w2_expert_bytes],
        );
        pack_expert_r128c4_scales(
            &s2_log,
            HIDDEN_192E,
            INTERMEDIATE_192E,
            &mut s2_all[expert * s2_expert_bytes..(expert + 1) * s2_expert_bytes],
        );
    }

    bufs.w_up.copy_from_host(stream, &w1_all)?;
    bufs.w_up_scales.copy_from_host(stream, &s1_all)?;
    bufs.w_down.copy_from_host(stream, &w2_all)?;
    bufs.w_down_scales.copy_from_host(stream, &s2_all)?;
    stream.synchronize()?;
    Ok(())
}

/// Run the complete 192-expert (`H = 6,144, I = 1,536, E = 192, B = 1 .. 256`) validation
/// (`uniform`, `nonuniform`, `changed-routes` after capture) and 1,000-replay CUDA Graph benchmark.
pub fn run_benchmark_192e(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n=== Wide 192-Expert NVFP4 ReLU² MoE (H={HIDDEN_192E}, I={INTERMEDIATE_192E}, E={NUM_EXPERTS_192E}, top-k={TOP_K_192E}) ==="
    );
    let module = unsafe { kernels_192e::load(ctx)? };

    // 1. Validate `nonuniform` + `changed-routes-after-capture` at B=1 and B=8
    for &test_b in &[1usize, 8] {
        let mut bufs = allocate_192e(stream, test_b)?;
        upload_nonuniform_weights_192e(stream, &mut bufs)?;
        let (h_nu, ids_nu, rw_nu, exp_nu) = build_nonuniform_case(test_b, false);
        bufs.hidden.copy_from_host(stream, &h_nu)?;
        bufs.expert_ids.copy_from_host(stream, &ids_nu)?;
        bufs.routing_weights.copy_from_host(stream, &rw_nu)?;
        launch_pipeline_192e(&module, stream, &mut bufs)?;
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
        for _ in 0..5 {
            unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
        }
        let nu_err = check_output_192e(stream, &bufs, &exp_nu)?;

        let (h_ch, ids_ch, rw_ch, exp_ch) = build_nonuniform_case(test_b, true);
        bufs.hidden.copy_from_host(stream, &h_ch)?;
        bufs.expert_ids.copy_from_host(stream, &ids_ch)?;
        bufs.routing_weights.copy_from_host(stream, &rw_ch)?;
        for _ in 0..7 {
            unsafe { sys::cuGraphLaunch(exec, stream.cu_stream()).result()? };
        }
        let ch_err = check_output_192e(stream, &bufs, &exp_ch)?;
        unsafe {
            let _ = sys::cuGraphExecDestroy(exec);
            let _ = sys::cuGraphDestroy(graph);
        }
        println!(
            "Validation B={test_b:>3}: nonuniform_err={nu_err:.6}, changed_routes_after_capture_err={ch_err:.6} (PASSED)"
        );
    }

    // 2. Sweep B = 1 .. 256 (1,000x CUDA Graph replay x 20 samples)
    let mut results = Vec::with_capacity(BATCH_SIZES_192E.len());
    for &b in &BATCH_SIZES_192E {
        let mut bufs = allocate_192e(stream, b)?;
        let expected: Vec<f32> = (0..b * HIDDEN_192E)
            .map(|idx| crate::expected_output(idx / HIDDEN_192E).to_f32())
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

    println!(
        "\nBatch | TRTLLM-Gen (us) | FlashInfer CuTeDSL (us) | cuTile NVFP4 (us) | Native SIMT [live] (us) | vs. TRTLLM-Gen"
    );
    println!(
        "------+-----------------+-------------------------+-------------------+-------------------------+---------------"
    );
    for (idx, &(b, native_us)) in results.iter().enumerate() {
        let trt = TRTLLM_GEN_US[idx];
        let fi = FLASHINFER_CUTEDSL_US[idx];
        let ct = CUTILE_US[idx];
        let speedup = trt / native_us;
        println!(
            "{b:>5} | {trt:>15.3} | {fi:>23.3} | {ct:>17.3} | {native_us:>23.3} | {speedup:>9.2}x"
        );
    }
    Ok(())
}

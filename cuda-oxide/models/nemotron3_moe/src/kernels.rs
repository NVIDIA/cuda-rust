// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Byte offset of the `SMEM_B` activation ring buffer (`65,536` bytes after `SMEM_A`) in dynamic shared memory.
/// Follows the `64 KB` (`2 × 32 KB`) `SMEM_A` stationary weight buffer.
const SMEM_B_OFFSET: usize = 65_536;

/// Byte offset of the `8 KB` (`2 × 4 KB`) `SMEM_SFA` weight scale buffer in dynamic shared memory.
/// Aligned to `1,024` bytes for 2D TMA transfers and `tcgen05.cp`.
const SMEM_SFA_OFFSET: usize = 196_608;

/// Byte offset of the `16 KB` `SMEM_SFB` activation scale buffer in dynamic shared memory.
/// Aligned to `1,024` bytes for 2D TMA transfers and `tcgen05.cp`.
const SMEM_SFB_OFFSET: usize = 204_800;

/// Byte offset of the `mbarrier` array (`221,184` bytes) in `FC1`, `FC1_128`, and `FC2`.
/// Placed immediately after `SMEM_SFB`.
const SMEM_BARRIER_OFFSET: usize = 221_184;

/// Byte offset of the `mbarrier` array (`229,376` bytes) in `FC2_128`.
/// Placed after the second `8 KB` S2G TMA Store epilogue buffer (`SMEM_EPI1` at `221,184`).
const SMEM_FC2_128_BARRIER_OFFSET: usize = 229_376;

/// Size in bytes of one `K = 512` `128`-row weight or `128`-token activation burst (`32,768` bytes).
/// Transferred by a single 2D TMA copy (`128 × 256` bytes).
const TILE_128_BURST_BYTES: usize = 32_768;

/// Size in bytes of one `K = 512` `256`-token activation burst (`65,536` bytes).
/// Transferred by a single 2D TMA copy (`256 × 256` bytes) for `B <= 1,024`.
const ACT_256_BURST_BYTES: usize = 65_536;

/// Size in bytes of 8 canonical `512`-byte `(32, 4, 4)` scale atoms (`4,096` bytes).
/// Covers one `K = 512` burst for `128` rows (`SFA` or `128`-token `SFB`).
const SCALE_128_BURST_BYTES: usize = 4_096;

/// Size in bytes of 16 canonical `512`-byte `(32, 4, 4)` scale atoms (`8,192` bytes).
/// Covers one `K = 512` burst for `256` tokens (`SFB` at `B <= 1,024`).
const SCALE_256_BURST_BYTES: usize = 8_192;

/// Starting TMEM column (`256`) reserved for staging `SFA` (`W_up` / `W_down`) block scales.
/// Holds `2` bursts × `32` columns = `64` columns (`TMEM[256..319]`).
const TMEM_SFA_COL: u32 = 256;

/// Starting TMEM column (`320`) reserved for staging `SFB` (routed token) block scales.
/// Holds up to `4` stages × `32` columns = `128` columns (`TMEM[320..447]`).
const TMEM_SFB_COL: u32 = 320;

/// Build a `tcgen05` shared-memory descriptor (`SWIZZLE_NONE = 0`).
/// Encodes `smem_addr >> 4`, `leading_dim_bytes >> 4`, and `stride_bytes >> 4` with bit `46` set.
#[inline(always)]
const fn build_smem_descriptor(smem_addr: u64, leading_dim_bytes: u32, stride_bytes: u32) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit
}

/// Encode the 32-bit `tcgen05.mma.kind::mxf4nvf4.block_scale` instruction descriptor for `M = 128` and `n_cols`.
/// Sets `M = 128` (`bits 28..24 = 8`), `E2M1 × E2M1` (`bits 11..7`), and `N / 8` in `bits 22..17`.
#[inline(always)]
const fn encode_mxf4nvf4_idesc(n_cols: u32) -> u32 {
    0x0800_0480u32 | ((n_cols >> 3) << 17)
}

/// Map a row index `row_in_128` (`0..128`) to its 32-bit word index (`0..128`) in a canonical `(32, 4, 4)` scale atom.
/// Matches the `tcgen05.cp.32x128b.warpx4` per-lane scale layout (`(row % 32) * 4 + (row / 32)`).
#[inline(always)]
const fn sfb_scale_word(row_in_128: usize) -> usize {
    ((row_in_128 & 31) * 4) + ((row_in_128 >> 5) & 3)
}

/// Device kernels for the Nemotron-3 NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` block scales,
/// `SF_VEC = 16`) non-gated `ReLU²` Mixture-of-Experts (MoE) layer on Blackwell (`sm_100a`).
///
/// ```text
/// THE 4 STAGES OF THE MOE LAYER:
/// [K1: Quantize BF16->FP4] ──► [K2: Route] ──► [K3: FC1 + ReLU²] ──► [K4: FC2 + Finalize]
///
/// HARDWARE BLOCK-SCALED `tcgen05` DATA FLOW (`B = 4..2048`)
/// ─────────────────────────────────────────────────────────
/// Global Memory
///   ├── 4-bit Weights/Tokens (`E2M1`) ──► Shared Memory (`SMEM_AB`) ──────────────────┐
///   │                                                                                 ▼
///   └── 8-bit Scales (`E4M3`) ──────────► Shared Memory (`SMEM_SF*`) ──(`tcgen05.cp`)─► TMEM
///                                                                                     │
///                                         5th-Gen Tensor Core (`tcgen05.mma...mxf4nvf4.block_scale`)
///                                                                                     │
///                                         TMEM Unload (`tcgen05.ld.32x32b.x{16,32}`) ◄┘
///                                            ├── K3: `ReLU²` + 16-warp NVFP4 requantization
///                                            └── K4: 16-warp `atom_add_bf16x2` fused finalize
/// ```
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#[cuda_module]
mod kernels {
    use super::*;

    /// Unpack 4 E2M1 nibbles (in the low 16 bits of `packed_u16`) into 4 signed `i8` bytes
    /// representing `2 * value` in `{-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12}`.
    #[inline(always)]
    fn unpack_4_e2m1_to_i8x4(packed_u16: u32) -> u32 {
        let mag4 = prmt(0x0302_0100, 0x0C08_0604, packed_u16 & 0x7777);
        let mask = prmt(0x0000_FF00, 0, (packed_u16 >> 3) & 0x1111);
        (mag4 ^ mask).wrapping_add(mask & 0x0101_0101)
    }

    /// Quantize one signed `f32` into its nearest 4-bit `E2M1` nibble (`bits 3..0`) using
    /// branchless IEEE 754 subnormal bias-shift arithmetic (`0` float comparisons!).
    #[inline(always)]
    fn dev_f32_to_e2m1(value: f32) -> u8 {
        let bits = value.to_bits();
        let sign = ((bits >> 28) & 8) as u8;
        let abs = f32::from_bits(bits & 0x7FFF_FFFF);
        let scaled = (abs * f32::from_bits(1 << 23)).to_bits().min(0x01C0_0000);
        sign | (((scaled + 0x0020_0000) >> 22) as u8)
    }

    /// Quantize a non-negative `f32` (such as `ReLU²` output) into its 3-bit `E2M1` magnitude
    /// using branchless IEEE 754 subnormal bias-shift arithmetic (`0` float comparisons!).
    #[inline(always)]
    fn dev_pos_f32_to_e2m1(abs: f32) -> u32 {
        let scaled = (abs * f32::from_bits(1 << 23)).to_bits().min(0x01C0_0000);
        (scaled + 0x0020_0000) >> 22
    }

    /// Encode one non-negative `f32` already pre-multiplied by `inv_scale * 2^-126` (`f32::from_bits(1 << 23)`)
    /// into its 4-bit `Float4E2M1FN` nibble (`0..7`) in 2 branchless bitwise instructions!
    #[inline(always)]
    fn fast_pos_e2m1(scaled_v: f32) -> u64 {
        let bits = scaled_v.to_bits().min(0x01C0_0000);
        ((bits + 0x0020_0000) >> 22) as u64
    }

    /// Decode one `Float8E4M3FN` scale byte into `f32` via hardware `cvt.rn.f16x2.e4m3x2`.
    #[inline(always)]
    fn dev_e4m3_to_f32(byte: u8) -> f32 {
        let packed = cvt_rn_f16x2_e4m3x2(byte as u16);
        let (lo, _) = cvt_f32x2_f16x2(packed);
        lo
    }

    /// Decode four packed `Float8E4M3FN` scale bytes in a `u32` into four `f32` values.
    #[inline(always)]
    fn dev_e4m3x4_to_f32x4(packed_u32: u32) -> (f32, f32, f32, f32) {
        let p01 = cvt_rn_f16x2_e4m3x2((packed_u32 & 0xFFFF) as u16);
        let p23 = cvt_rn_f16x2_e4m3x2((packed_u32 >> 16) as u16);
        let (s0, s1) = cvt_f32x2_f16x2(p01);
        let (s2, s3) = cvt_f32x2_f16x2(p23);
        (s0, s1, s2, s3)
    }

    /// Encode one non-negative `f32` scale factor into a `Float8E4M3FN` byte via `cvt.rn.satfinite.e4m3x2.f32`.
    #[inline(always)]
    fn dev_f32_to_e4m3(value: f32) -> u8 {
        let clamped = if value.is_nan() || value <= 0.0 {
            0.0f32
        } else if value > 448.0 {
            448.0f32
        } else {
            value
        };
        (cvt_rn_satfinite_e4m3x2_f32(clamped, 0.0) & 0x00FF) as u8
    }

    /// Map logical `(row, group)` coordinates to their byte offset inside a canonical `(32, 4, 4)`
    /// 512-byte Blackwell scale atom.
    #[inline(always)]
    fn canonical_scale_byte_offset(rest_k: usize, row: usize, group: usize) -> usize {
        (row >> 7) * rest_k * 512
            + (group >> 2) * 512
            + ((row & 31) << 4)
            + (((row >> 5) & 3) << 2)
            + (group & 3)
    }

    /// Sum a `f32` value across each aligned 16-lane half-warp (`shuffle_xor` by `8, 4, 2, 1`).
    #[inline(always)]
    fn reduce_16_lanes(mut v: f32) -> f32 {
        v += warp::shuffle_xor_f32(v, 8);
        v += warp::shuffle_xor_f32(v, 4);
        v += warp::shuffle_xor_f32(v, 2);
        v += warp::shuffle_xor_f32(v, 1);
        v
    }

    /// Compute the maximum `f32` across each aligned 16-lane half-warp (`shuffle_xor` by `8, 4, 2, 1`).
    #[inline(always)]
    fn reduce_max_16_lanes(mut amax: f32) -> f32 {
        let o8 = warp::shuffle_xor_f32(amax, 8);
        if o8 > amax {
            amax = o8;
        }
        let o4 = warp::shuffle_xor_f32(amax, 4);
        if o4 > amax {
            amax = o4;
        }
        let o2 = warp::shuffle_xor_f32(amax, 2);
        if o2 > amax {
            amax = o2;
        }
        let o1 = warp::shuffle_xor_f32(amax, 1);
        if o1 > amax {
            amax = o1;
        }
        amax
    }

    // =========================================================================
    // REGIME 1: LOW-LATENCY 2-KERNEL DIRECT PATH (B <= 2)
    // =========================================================================

    /// Low-batch (`B <= 2`) fused `K1 + K3` kernel: quantizes each input token (`FP16 -> NVFP4`),
    /// multiplies against `w_up` via `dp4a_s32`, applies `ReLU²`, and writes quantized NVFP4
    /// intermediate activations (`inter_out` and `inter_scales`).
    #[kernel]
    pub unsafe fn fc1_relu2_requant_direct(
        mut inter_out: DisjointSlice<u8>,
        mut inter_scales: DisjointSlice<u8>,
        hidden: &[u16],
        expert_ids: &[i32],
        w_up: &[u8],
        w_up_scales: &[u8],
        w_up_global: &[f32],
        input_scale: f32,
        hidden_size: u32,
        intermediate_size: u32,
        top_k: u32,
    ) {
        /// Shared-memory buffer staging unpacked `i8x4` (`2 × E2M1`) input token values (`1 KB`).
        /// Broadcast across all 16 warps in the CTA during the `dp4a_s32` dot product.
        static mut S_X_DIRECT: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        /// Shared-memory buffer staging decoded `f32` input scale factors (`64` scale groups).
        /// Multiplied with each expert's `E4M3` weight scale during `FC1` accumulation.
        static mut S_XS_DIRECT: SharedArray<f32, 64, 16> = SharedArray::UNINIT;
        /// Shared-memory buffer staging the 16 `ReLU²` output values produced by the 16 warps.
        /// Read by thread 0 to compute the output `E4M3` scale and pack 8 `E2M1` bytes.
        static mut S_ACT_DIRECT: SharedArray<f32, 16, 16> = SharedArray::UNINIT;

        let output_tile = thread::blockIdx_x() as usize;
        let route = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let m_local = tid >> 4;
        let k_lane = tid & 15;

        let tok = route / (top_k as usize);
        let expert = expert_ids[route] as usize;
        let h = hidden_size as usize;
        let inter = intermediate_size as usize;
        let packed_h = h / 2;
        let rest_k = h / 64;

        let tile128_base = (output_tile / 8) * 128;
        let row_in_128 = (output_tile % 8) * 16 + m_local;
        let m_grp = row_in_128 / 32;
        let rem32 = row_in_128 % 32;
        let phys_in_128 = m_grp * 32 + (rem32 % 4) * 8 + (rem32 / 4);
        let phys_row = expert * inter + tile128_base + phys_in_128;

        let w_u32_ptr = w_up.as_ptr().cast::<u32>();
        let ws_u32_ptr = w_up_scales.as_ptr().cast::<u32>();
        let hidden_u32 = hidden.as_ptr().cast::<u32>();
        let sx_ptr = (&raw mut S_X_DIRECT).cast::<u32>();
        let sxs_ptr = (&raw mut S_XS_DIRECT).cast::<f32>();

        if tid < 64 {
            let in_u32_base = (tok * h + tid * 16) >> 1;
            let mut vals = [0.0f32; 16];
            #[unroll]
            for i in 0..8 {
                let packed = unsafe { *hidden_u32.add(in_u32_base + i) };
                let (lo, hi) = cvt_f32x2_f16x2(packed);
                vals[i * 2] = lo;
                vals[i * 2 + 1] = hi;
            }
            let mut amax = 0.0f32;
            #[unroll]
            for k in 0..16 {
                let a = if vals[k] < 0.0 { -vals[k] } else { vals[k] };
                if a > amax {
                    amax = a;
                }
            }
            let scale_code = dev_f32_to_e4m3(amax * input_scale * (1.0f32 / 6.0f32));
            let scale_val = dev_e4m3_to_f32(scale_code);
            let inv_scale = if scale_val > 0.0 {
                input_scale * rcp_approx_ftz_f32(scale_val)
            } else {
                0.0
            };
            unsafe { *sxs_ptr.add(tid) = scale_val * 0.25f32 };

            #[unroll]
            for w_word in 0..4 {
                let b = w_word * 4;
                let n0 = dev_f32_to_e2m1(vals[b] * inv_scale) as u32;
                let n1 = dev_f32_to_e2m1(vals[b + 1] * inv_scale) as u32;
                let n2 = dev_f32_to_e2m1(vals[b + 2] * inv_scale) as u32;
                let n3 = dev_f32_to_e2m1(vals[b + 3] * inv_scale) as u32;
                let packed16 = n0 | (n1 << 4) | (n2 << 8) | (n3 << 12);
                unsafe {
                    *sx_ptr.add(tid * 4 + w_word) = unpack_4_e2m1_to_i8x4(packed16);
                }
            }
        }
        thread::sync_threads();

        let k_tile = k_lane;
        let x_base = k_tile * 16;
        let w_u32_off = (phys_row * packed_h + k_tile * 32) >> 2;
        let ws_byte_off = canonical_scale_byte_offset(rest_k, phys_row, k_tile * 4);
        let ws_packed = unsafe { *ws_u32_ptr.add(ws_byte_off >> 2) };
        let (ws0, ws1, ws2, ws3) = dev_e4m3x4_to_f32x4(ws_packed);
        let ws = [ws0, ws1, ws2, ws3];

        let mut sum = 0.0f32;
        #[unroll]
        for g in 0..4 {
            let rw_lo = unsafe { *w_u32_ptr.add(w_u32_off + g * 2) };
            let rw_hi = unsafe { *w_u32_ptr.add(w_u32_off + g * 2 + 1) };
            let mut dot = dp4a_s32(
                unpack_4_e2m1_to_i8x4(rw_lo & 0xFFFF),
                unsafe { *sx_ptr.add(x_base + g * 4) },
                0,
            );
            dot = dp4a_s32(
                unpack_4_e2m1_to_i8x4(rw_lo >> 16),
                unsafe { *sx_ptr.add(x_base + g * 4 + 1) },
                dot,
            );
            dot = dp4a_s32(
                unpack_4_e2m1_to_i8x4(rw_hi & 0xFFFF),
                unsafe { *sx_ptr.add(x_base + g * 4 + 2) },
                dot,
            );
            dot = dp4a_s32(
                unpack_4_e2m1_to_i8x4(rw_hi >> 16),
                unsafe { *sx_ptr.add(x_base + g * 4 + 3) },
                dot,
            );
            let xs_g = unsafe { *sxs_ptr.add(k_tile * 4 + g) };
            sum += (dot as f32) * (ws[g] * xs_g);
        }

        let reduced = reduce_16_lanes(sum);
        let s_act_ptr = (&raw mut S_ACT_DIRECT).cast::<f32>();
        if k_lane == 0 {
            let scaled = reduced * w_up_global[expert] * rcp_approx_ftz_f32(input_scale);
            let relu = if scaled > 0.0 { scaled } else { 0.0 };
            unsafe { *s_act_ptr.add(m_local) = relu * relu };
        }
        thread::sync_threads();

        if tid < 16 {
            let act_val = unsafe { *s_act_ptr.add(tid) };
            let amax = reduce_max_16_lanes(act_val);
            let scale_code = dev_f32_to_e4m3(amax * (1.0f32 / 6.0f32));
            let scale_val = dev_e4m3_to_f32(scale_code);
            let inv_scale = if scale_val > 0.0 {
                rcp_approx_ftz_f32(scale_val)
            } else {
                0.0f32
            };
            let nib = dev_pos_f32_to_e2m1(act_val * inv_scale);
            let partner_nib = warp::shuffle_xor(nib, 1);
            if (tid & 1) == 0 {
                let packed_byte = (nib as u8) | ((partner_nib as u8) << 4);
                let packed_inter_cols = inter / 2;
                unsafe {
                    *inter_out.get_unchecked_mut(
                        route * packed_inter_cols + output_tile * 8 + (tid >> 1),
                    ) = packed_byte;
                }
            }
            if tid == 0 {
                let scale_groups_inter = inter / 16;
                unsafe {
                    *inter_scales.get_unchecked_mut(route * scale_groups_inter + output_tile) =
                        scale_code;
                }
            }
        }
    }

    /// Low-batch (`B <= 2`) fused `K4` (`FC2 + Finalize`) kernel: multiplies the `top_k = 4`
    /// routed NVFP4 intermediates against `w_down` via `dp4a_s32` and writes the weighted `bf16`
    /// sum directly to `out` in a single pass.
    #[kernel]
    pub unsafe fn fc2_and_finalize_direct(
        mut out: DisjointSlice<u16>,
        inter: &[u8],
        inter_scales: &[u8],
        expert_ids: &[i32],
        routing_weights: &[u16],
        w_down: &[u8],
        w_down_scales: &[u8],
        w_down_global: &[f32],
        hidden_size: u32,
        intermediate_size: u32,
        top_k: u32,
    ) {
        let output_tile = thread::blockIdx_x() as usize;
        let token = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let m_local = tid >> 4;
        let k_lane = tid & 15;

        let h = hidden_size as usize;
        let inter_dim = intermediate_size as usize;
        let packed_inter = inter_dim / 2;
        let scale_groups_inter = inter_dim / 16;
        let rest_k = inter_dim / 64;
        let tk = top_k as usize;

        let out_col = output_tile * 16 + m_local;
        let tile128_base = (output_tile / 8) * 128;
        let row_in_128 = (output_tile % 8) * 16 + m_local;
        let m_grp = row_in_128 / 32;
        let rem32 = row_in_128 % 32;
        let phys_in_128 = m_grp * 32 + (rem32 % 4) * 8 + (rem32 / 4);

        let w_u32_ptr = w_down.as_ptr().cast::<u32>();
        let ws_u32_ptr = w_down_scales.as_ptr().cast::<u32>();
        let x_u32_ptr = inter.as_ptr().cast::<u32>();
        let xs_u32_ptr = inter_scales.as_ptr().cast::<u32>();

        let mut total_bf16_acc = 0.0f32;
        let mut slot = 0usize;
        while slot < tk {
            let route = token * tk + slot;
            let expert = expert_ids[route] as usize;
            let route_w = bf16_to_f32(routing_weights[route]);
            let phys_row = expert * h + tile128_base + phys_in_128;
            let global_scale = w_down_global[expert];

            let k_tile = k_lane;
            let x_u32_off = (route * packed_inter + k_tile * 32) >> 2;
            let xs_u32_off = (route * scale_groups_inter + k_tile * 4) >> 2;
            let xs_packed = unsafe { *xs_u32_ptr.add(xs_u32_off) };
            let (xs0, xs1, xs2, xs3) = dev_e4m3x4_to_f32x4(xs_packed);
            let xs = [xs0, xs1, xs2, xs3];

            let w_u32_off = (phys_row * packed_inter + k_tile * 32) >> 2;
            let ws_byte_off = canonical_scale_byte_offset(rest_k, phys_row, k_tile * 4);
            let ws_packed = unsafe { *ws_u32_ptr.add(ws_byte_off >> 2) };
            let (ws0, ws1, ws2, ws3) = dev_e4m3x4_to_f32x4(ws_packed);
            let ws = [ws0, ws1, ws2, ws3];

            let mut sum = 0.0f32;
            #[unroll]
            for g in 0..4 {
                let rx_lo = unsafe { *x_u32_ptr.add(x_u32_off + g * 2) };
                let rx_hi = unsafe { *x_u32_ptr.add(x_u32_off + g * 2 + 1) };
                let rw_lo = unsafe { *w_u32_ptr.add(w_u32_off + g * 2) };
                let rw_hi = unsafe { *w_u32_ptr.add(w_u32_off + g * 2 + 1) };
                let mut dot = dp4a_s32(
                    unpack_4_e2m1_to_i8x4(rw_lo & 0xFFFF),
                    unpack_4_e2m1_to_i8x4(rx_lo & 0xFFFF),
                    0,
                );
                dot = dp4a_s32(
                    unpack_4_e2m1_to_i8x4(rw_lo >> 16),
                    unpack_4_e2m1_to_i8x4(rx_lo >> 16),
                    dot,
                );
                dot = dp4a_s32(
                    unpack_4_e2m1_to_i8x4(rw_hi & 0xFFFF),
                    unpack_4_e2m1_to_i8x4(rx_hi & 0xFFFF),
                    dot,
                );
                dot = dp4a_s32(
                    unpack_4_e2m1_to_i8x4(rw_hi >> 16),
                    unpack_4_e2m1_to_i8x4(rx_hi >> 16),
                    dot,
                );
                sum += (dot as f32) * (ws[g] * xs[g]);
            }
            sum *= 0.25f32;

            let reduced = reduce_16_lanes(sum);
            let fc2_bf16 = bf16_to_f32(f32_to_bf16_rne(reduced * global_scale));
            total_bf16_acc += route_w * fc2_bf16;
            slot += 1;
        }

        if k_lane == 0 {
            unsafe {
                *out.get_unchecked_mut(token * h + out_col) = f32_to_bf16_rne(total_bf16_acc);
            }
        }
    }

    // =========================================================================
    // REGIME 2: 5th-GEN BLACKWELL TENSOR CORE (`tcgen05.mma`) PIPELINE (B >= 4)
    // =========================================================================

    /// Fused `K1` quantizer + TMA-tile scatter + output zeroing for the 5th-gen Tensor Core (`tcgen05`) path (`B = 4..2048`).
    /// Quantizes `[num_tokens, hidden_size]` `f16` values into packed `Float4E2M1FN` bytes and canonical `(32, 4)`
    /// `Float8E4M3FN` scales, writing TMA-ready `[256, 256]` burst tiles (`b_up_vals` / `sfb_up_scales`) while zero-initializing `output`.
    #[kernel]
    pub unsafe fn fused_route_and_quantize_tcgen05(
        hidden: &[u16],
        tok_to_perm: &[i32],
        mut b_up_vals: DisjointSlice<u8>,
        mut sfb_up_scales: DisjointSlice<u8>,
        mut output: DisjointSlice<u16>,
        num_tokens: u32,
        hidden_size: u32,
        input_scale: f32,
        zero_out: u32,
    ) {
        let tid = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let h = hidden_size as usize;
        let groups_per_row = h / 16;
        let total_groups = (num_tokens as usize) * groups_per_row;
        if tid >= total_groups {
            return;
        }
        let row = tid / groups_per_row;
        let grp = tid % groups_per_row;
        let u128_base = (row * h + grp * 16) >> 3;

        let hidden_u128 = hidden.as_ptr().cast::<u128>();
        let out_u128 = output.as_mut_ptr().cast::<u128>();
        let (h0, h1, gp128) = unsafe {
            (
                *hidden_u128.add(u128_base),
                *hidden_u128.add(u128_base + 1),
                *tok_to_perm.as_ptr().cast::<u128>().add(row),
            )
        };
        if zero_out != 0 {
            unsafe {
                *out_u128.add(u128_base) = 0u128;
                *out_u128.add(u128_base + 1) = 0u128;
            }
        }
        let raw_u32 = [
            (h0 & 0xFFFF_FFFF) as u32,
            ((h0 >> 32) & 0xFFFF_FFFF) as u32,
            ((h0 >> 64) & 0xFFFF_FFFF) as u32,
            (h0 >> 96) as u32,
            (h1 & 0xFFFF_FFFF) as u32,
            ((h1 >> 32) & 0xFFFF_FFFF) as u32,
            ((h1 >> 64) & 0xFFFF_FFFF) as u32,
            (h1 >> 96) as u32,
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

        let scale_code = dev_f32_to_e4m3(amax * input_scale * (1.0f32 / 6.0f32));
        let scale_val = dev_e4m3_to_f32(scale_code);
        let inv_scale_e2m1 = if scale_val > 0.0 {
            input_scale * rcp_approx_ftz_f32(scale_val) * f32::from_bits(1 << 23)
        } else {
            0.0f32
        };
        let mut packed_u64 = 0u64;
        #[unroll]
        for idx in 0..16 {
            let v = vals[idx];
            let sign = ((v.to_bits() >> 28) & 8) as u64;
            let abs_v = f32::from_bits(v.to_bits() & 0x7FFF_FFFF);
            let nib = sign | fast_pos_e2m1(abs_v * inv_scale_e2m1);
            packed_u64 |= nib << (idx * 4);
        }

        let b_u64_ptr = b_up_vals.as_mut_ptr().cast::<u64>();
        let burst = grp >> 5; // 0..1
        let s_in_burst = (grp >> 2) & 7; // 0..7
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        if zero_out != 0 {
            let sub_u64_off = s_in_burst * 1024 + ((grp >> 1) & 1) * 512 + (grp & 1);
            #[unroll]
            for slot in 0..4 {
                let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
                let g = gp >> 8;
                let r = gp & 255;
                let u64_idx = (g * 2 + burst) * 8192 + sub_u64_off + r * 2;
                let sfb_word = sfb_scale_word(r & 127);
                let s_idx = ((g * 16 + k_atom) * 2 + (r >> 7)) * 512 + sfb_word * 4 + k_in_atom;
                unsafe {
                    *b_u64_ptr.add(u64_idx) = packed_u64;
                    *sfb_up_scales.get_unchecked_mut(s_idx) = scale_code;
                }
            }
        } else {
            let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
            #[unroll]
            for slot in 0..4 {
                let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
                let g = gp >> 7;
                let r = gp & 127;
                let u64_idx = (g * 2 + burst) * 4096 + sub_u64_off + r * 2;
                let sfb_word = sfb_scale_word(r);
                let s_idx = (g * 16 + k_atom) * 512 + sfb_word * 4 + k_in_atom;
                unsafe {
                    *b_u64_ptr.add(u64_idx) = packed_u64;
                    *sfb_up_scales.get_unchecked_mut(s_idx) = scale_code;
                }
            }
        }
    }

    /// Run `Warp 16`'s 2-stage weight-stationary TMA producer loop for the `256`-token (`B <= 1,024`) pipeline.
    /// Loads `W + B + SFA + SFB` (`110,592` B) on `g_idx == 0` and streams `B + SFB` (`73,728` B) on `g_idx > 0`.
    macro_rules! run_tma_producer_256 {
        (
            $w_tma:ident,
            $b_tma:ident,
            $sfa_tma:ident,
            $sfb_tma:ident,
            $smem_a_ptr:ident,
            $smem_b_ptr:ident,
            $smem_sfa_ptr:ident,
            $smem_sfb_ptr:ident,
            $tma_bar_ptr:ident,
            $b_free_bar_const:ident,
            $w_outer:ident,
            $group_start:ident,
            $g_per_cta:ident
        ) => {{
            let w_desc = core::ptr::from_ref($w_tma);
            let b_desc = core::ptr::from_ref($b_tma);
            let sfa_desc = core::ptr::from_ref($sfa_tma);
            let sfb_desc = core::ptr::from_ref($sfb_tma);
            let mut g_idx = 0usize;
            while g_idx < $g_per_cta {
                let group = $group_start + g_idx;
                if g_idx == 0 {
                    let mut burst = 0usize;
                    while burst < 2 {
                        let bar = $tma_bar_ptr.add(burst);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, 110_592);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_a_ptr.add(burst * TILE_128_BURST_BYTES),
                            w_desc,
                            (burst * 256) as i32,
                            ($w_outer * 128) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_b_ptr.add(burst * ACT_256_BURST_BYTES),
                            b_desc,
                            0,
                            ((group * 2 + burst) * 256) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_sfa_ptr.add(burst * SCALE_128_BURST_BYTES),
                            sfa_desc,
                            0,
                            (($w_outer * 2 + burst) * 8) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_sfb_ptr.add(burst * SCALE_256_BURST_BYTES),
                            sfb_desc,
                            0,
                            ((group * 2 + burst) * 16) as i32,
                            bar,
                        );
                        burst += 1;
                    }
                } else {
                    let wait_parity = ((g_idx - 1) & 1) as u32;
                    let mut burst = 0usize;
                    while burst < 2 {
                        while !mbarrier_try_wait_parity($b_free_bar_const.add(burst), wait_parity) {}
                        let bar = $tma_bar_ptr.add(burst);
                        let _ = mbarrier_arrive_expect_tx(bar, 1, 73_728);
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_b_ptr.add(burst * ACT_256_BURST_BYTES),
                            b_desc,
                            0,
                            ((group * 2 + burst) * 256) as i32,
                            bar,
                        );
                        cp_async_bulk_tensor_2d_g2s_cta(
                            $smem_sfb_ptr.add(burst * SCALE_256_BURST_BYTES),
                            sfb_desc,
                            0,
                            ((group * 2 + burst) * 16) as i32,
                            bar,
                        );
                        burst += 1;
                    }
                }
                g_idx += 1;
            }
        }};
    }

    /// Issue both `K = 512` bursts (`16` `mxf4nvf4` MMA steps) for one `256`-token group (`B <= 1,024`)
    /// from thread 0 of `Warp 0` and wait on `mma_bar` completion.
    macro_rules! issue_mxf4nvf4_256_bursts {
        (
            $tmem_addr:ident,
            $smem_a_base0:ident,
            $smem_b_base0:ident,
            $smem_sfa_base0:ident,
            $smem_sfb_base0:ident,
            $idesc:ident,
            $g_idx:ident,
            $parity:ident,
            $tma_bar_const:ident,
            $mma_bar_ptr:ident,
            $mma_bar_const:ident
        ) => {{
            let mut burst = 0usize;
            while burst < 2 {
                while !mbarrier_try_wait_parity($tma_bar_const.add(burst), $parity) {}
                let sfa_tmem_base = $tmem_addr + TMEM_SFA_COL + (burst as u32) * 32;
                let sfb_tmem_base = $tmem_addr + TMEM_SFB_COL + (burst as u32) * 64;
                let a_desc0 = build_smem_descriptor(
                    $smem_a_base0 + (burst as u64) * (TILE_128_BURST_BYTES as u64),
                    2048,
                    128,
                );
                let b_desc0 = build_smem_descriptor(
                    $smem_b_base0 + (burst as u64) * (ACT_256_BURST_BYTES as u64),
                    4096,
                    128,
                );
                let sfa_desc0 = build_smem_descriptor(
                    $smem_sfa_base0 + (burst as u64) * (SCALE_128_BURST_BYTES as u64),
                    2048,
                    128,
                );
                let sfb_desc0 = build_smem_descriptor(
                    $smem_sfb_base0 + (burst as u64) * (SCALE_256_BURST_BYTES as u64),
                    2048,
                    128,
                );
                let mut s = 0usize;
                while s < 8 {
                    let sfa_tmem = sfa_tmem_base + (s as u32) * 4;
                    let sfb_tmem = sfb_tmem_base + (s as u32) * 8;
                    if $g_idx == 0 {
                        tcgen05_cp_32x128b_warpx4(sfa_tmem, sfa_desc0 + (s as u64) * 32);
                    }
                    tcgen05_cp_32x128b_warpx4(sfb_tmem, sfb_desc0 + (s as u64) * 64);
                    tcgen05_cp_32x128b_warpx4(sfb_tmem + 4, sfb_desc0 + (s as u64) * 64 + 32);
                    tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                        $tmem_addr,
                        a_desc0 + (s as u64) * 256,
                        b_desc0 + (s as u64) * 512,
                        $idesc,
                        !(burst == 0 && s == 0),
                        sfa_tmem,
                        sfb_tmem,
                    );
                    s += 1;
                }
                if burst == 1 {
                    tcgen05_commit_shared_cluster($mma_bar_ptr.cast::<u64>());
                    while !mbarrier_try_wait_parity($mma_bar_const, $parity) {}
                }
                burst += 1;
            }
        }};
    }

    /// Run `Warp 5`'s weight-stationary 4-stage TMA producer loop for the `128`-token (`B >= 2,048`) pipeline.
    /// Stages stationary `W` (`64 KB`) + `SFA` (`8 KB`) once, then streams `4` stages of `B` (`32 KB`) + `SFB` (`4 KB`).
    macro_rules! run_tma_producer_128 {
        (
            $w_tma:ident,
            $b_tma:ident,
            $sfa_tma:ident,
            $sfb_tma:ident,
            $smem_a_ptr:ident,
            $smem_b_ptr:ident,
            $smem_sfa_ptr:ident,
            $smem_sfb_ptr:ident,
            $tma_bar_ptr:ident,
            $b_free_bar_const:ident,
            $w_bar_ptr:ident,
            $w_outer:ident,
            $group_start:ident,
            $g_per_cta:ident
        ) => {{
            let w_desc = core::ptr::from_ref($w_tma);
            let b_desc = core::ptr::from_ref($b_tma);
            let sfa_desc = core::ptr::from_ref($sfa_tma);
            let sfb_desc = core::ptr::from_ref($sfb_tma);
            let _ = mbarrier_arrive_expect_tx($w_bar_ptr, 1, 73_728);
            let mut burst = 0usize;
            while burst < 2 {
                cp_async_bulk_tensor_2d_g2s_cta(
                    $smem_a_ptr.add(burst * TILE_128_BURST_BYTES),
                    w_desc,
                    (burst * 256) as i32,
                    ($w_outer * 128) as i32,
                    $w_bar_ptr,
                );
                cp_async_bulk_tensor_2d_g2s_cta(
                    $smem_sfa_ptr.add(burst * SCALE_128_BURST_BYTES),
                    sfa_desc,
                    0,
                    (($w_outer * 2 + burst) * 8) as i32,
                    $w_bar_ptr,
                );
                burst += 1;
            }
            let total_steps = $g_per_cta * 2;
            let mut step = 0usize;
            while step < total_steps {
                let slot = step & 3;
                let step_parity = ((step >> 2) & 1) as u32;
                if step >= 4 {
                    while !mbarrier_try_wait_parity($b_free_bar_const.add(slot), step_parity ^ 1) {}
                }
                let bar = $tma_bar_ptr.add(slot);
                let _ = mbarrier_arrive_expect_tx(bar, 1, 36_864);
                cp_async_bulk_tensor_2d_g2s_cta(
                    $smem_b_ptr.add(slot * TILE_128_BURST_BYTES),
                    b_desc,
                    0,
                    (($group_start * 2 + step) * 128) as i32,
                    bar,
                );
                cp_async_bulk_tensor_2d_g2s_cta(
                    $smem_sfb_ptr.add(slot * SCALE_128_BURST_BYTES),
                    sfb_desc,
                    0,
                    (($group_start * 2 + step) * 8) as i32,
                    bar,
                );
                step += 1;
            }
        }};
    }

    /// Stage one 512-byte `SFA` scale atom at compile-time step `$s` into TMEM.
    macro_rules! cp_sfa_one {
        ($tmem_addr:ident, $sfa_desc0:ident, $s:expr) => {
            tcgen05_cp_32x128b_warpx4(
                $tmem_addr + TMEM_SFA_COL + ($s as u32) * 4,
                $sfa_desc0 + ($s as u64) * 32,
            );
        };
    }

    /// Stage one `SFB` scale atom and issue one `mxf4nvf4` MMA at compile-time step `$s`.
    macro_rules! step_128_one {
        ($tmem_d:ident, $sfa_tmem_b:ident, $sfb_tmem_b:ident, $a_desc0:ident, $b_desc0:ident, $sfb_desc0:ident, $idesc:ident, $s:expr, $acc:expr) => {{
            let sfa_tmem = $sfa_tmem_b + ($s as u32) * 4;
            let sfb_tmem = $sfb_tmem_b + ($s as u32) * 4;
            tcgen05_cp_32x128b_warpx4(sfb_tmem, $sfb_desc0 + ($s as u64) * 32);
            tcgen05_mma_shared_mxf4nvf4_block_scale_block16::<1, 0>(
                $tmem_d,
                $b_desc0 + ($s as u64) * 256,
                $a_desc0 + ($s as u64) * 256,
                $idesc,
                $acc,
                sfb_tmem,
                sfa_tmem,
            );
        }};
    }

    /// Run `Warp 4`'s 2-stage TMEM (`TMEM[0] = cols 0..127`, `TMEM[1] = cols 128..255`) MMA mainloop for `B >= 2,048`.
    /// Stages all 16 `SFA` scale atoms once, then places `Tokens` on `Operand A` (`M = 128`) and `Weights` on `Operand B` (`N = 128`).
    macro_rules! run_mma_consumer_128 {
        (
            $tmem_addr:ident,
            $smem_a_base0:ident,
            $g_per_cta:ident,
            $w_bar_const:ident,
            $tma_bar_const:ident,
            $tmem_free_bar_const:ident,
            $b_free_bar_ptr:ident,
            $mma_bar_ptr:ident
        ) => {{
            let smem_b_base0 = $smem_a_base0 + (SMEM_B_OFFSET as u64);
            let smem_sfa_base0 = $smem_a_base0 + (SMEM_SFA_OFFSET as u64);
            let smem_sfb_base0 = $smem_a_base0 + (SMEM_SFB_OFFSET as u64);
            let idesc = encode_mxf4nvf4_idesc(128);
            while !mbarrier_try_wait_parity($w_bar_const, 0) {}
            let sfa_desc0 = build_smem_descriptor(smem_sfa_base0, 2048, 128);
            cp_sfa_one!($tmem_addr, sfa_desc0, 0);
            cp_sfa_one!($tmem_addr, sfa_desc0, 1);
            cp_sfa_one!($tmem_addr, sfa_desc0, 2);
            cp_sfa_one!($tmem_addr, sfa_desc0, 3);
            cp_sfa_one!($tmem_addr, sfa_desc0, 4);
            cp_sfa_one!($tmem_addr, sfa_desc0, 5);
            cp_sfa_one!($tmem_addr, sfa_desc0, 6);
            cp_sfa_one!($tmem_addr, sfa_desc0, 7);
            cp_sfa_one!($tmem_addr, sfa_desc0, 8);
            cp_sfa_one!($tmem_addr, sfa_desc0, 9);
            cp_sfa_one!($tmem_addr, sfa_desc0, 10);
            cp_sfa_one!($tmem_addr, sfa_desc0, 11);
            cp_sfa_one!($tmem_addr, sfa_desc0, 12);
            cp_sfa_one!($tmem_addr, sfa_desc0, 13);
            cp_sfa_one!($tmem_addr, sfa_desc0, 14);
            cp_sfa_one!($tmem_addr, sfa_desc0, 15);

            let total_steps = $g_per_cta * 2;
            let mut g_idx = 0usize;
            while g_idx < $g_per_cta {
                let tmem_slot = g_idx & 1;
                let tmem_parity = ((g_idx >> 1) & 1) as u32;
                let tmem_d = $tmem_addr + (tmem_slot as u32) * 128;
                if g_idx >= 2 {
                    while !mbarrier_try_wait_parity(
                        $tmem_free_bar_const.add(tmem_slot),
                        tmem_parity ^ 1,
                    ) {}
                }
                let mut burst = 0usize;
                while burst < 2 {
                    let step = g_idx * 2 + burst;
                    let slot = step & 3;
                    let step_parity = ((step >> 2) & 1) as u32;
                    while !mbarrier_try_wait_parity($tma_bar_const.add(slot), step_parity) {}
                    let a_desc0 = build_smem_descriptor(
                        $smem_a_base0 + (burst as u64) * (TILE_128_BURST_BYTES as u64),
                        2048,
                        128,
                    );
                    let b_desc0 = build_smem_descriptor(
                        smem_b_base0 + (slot as u64) * (TILE_128_BURST_BYTES as u64),
                        2048,
                        128,
                    );
                    let sfb_desc0 = build_smem_descriptor(
                        smem_sfb_base0 + (slot as u64) * (SCALE_128_BURST_BYTES as u64),
                        2048,
                        128,
                    );
                    let sfa_tmem_b = $tmem_addr + TMEM_SFA_COL + (burst as u32) * 32;
                    let sfb_tmem_b = $tmem_addr + TMEM_SFB_COL + (slot as u32) * 32;
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 0, burst != 0);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 1, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 2, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 3, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 4, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 5, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 6, true);
                    step_128_one!(tmem_d, sfa_tmem_b, sfb_tmem_b, a_desc0, b_desc0, sfb_desc0, idesc, 7, true);
                    if step + 4 < total_steps {
                        tcgen05_commit_shared_cluster($b_free_bar_ptr.add(slot).cast::<u64>());
                    }
                    if burst == 1 {
                        tcgen05_commit_shared_cluster($mma_bar_ptr.add(tmem_slot).cast::<u64>());
                    }
                    burst += 1;
                }
                g_idx += 1;
            }
        }};
    }

    /// Stage 3 (`tcgen05` Hardware Block-Scaled NVFP4 Tensor Core + Persistent Weight-Stationary 2-Stage Full-TMA):
    /// `128` persistent CTAs (`1` wave on B200) load `W` (`SMEM_A`) and `SFA` (`TMEM`) once on `g_idx == 0`
    /// and stream `g_per_cta` (`1..16` for `B = 1024..16384`) token groups of `B + SFB` via `Warp 16` TMA!
    #[launch_bounds(544)]
    #[launch_contract(
        domain = 1,
        block = (544, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_grouped_fc1_relu2_requant(
        #[grid_constant] w_up_tma: &TmaDescriptor,
        #[grid_constant] b_up_tma: &TmaDescriptor,
        #[grid_constant] sfa_up_tma: &TmaDescriptor,
        #[grid_constant] sfb_up_tma: &TmaDescriptor,
        w_up_global: &[f32],
        group_experts: &[i32],
        group_limits: &[i32],
        mut inter_out: DisjointSlice<u8>,
        mut inter_scales: DisjointSlice<u8>,
        input_scale: f32,
        intermediate_size: u32,
        groups_per_cta: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let tile_m128 = cta_id & 7;
        let g_per_cta = groups_per_cta as usize;
        let group_start = (cta_id >> 3) * g_per_cta;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = group_experts[group_start] as usize;
        let limit = group_limits[group_start] as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let inter = intermediate_size as usize;
        let w_outer = expert * (inter / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = dyn_smem;
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(SMEM_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_BARRIER_OFFSET).cast::<Barrier>() };
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

            if warp_id == 16 && (tid & 31) == 0 {
                run_tma_producer_256!(
                    w_up_tma,
                    b_up_tma,
                    sfa_up_tma,
                    sfb_up_tma,
                    smem_a_ptr,
                    smem_b_ptr,
                    smem_sfa_ptr,
                    smem_sfb_ptr,
                    tma_bar_ptr,
                    b_free_bar_const,
                    w_outer,
                    group_start,
                    g_per_cta
                );
            } else if warp_id < 16 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a_base0 = cvta_generic_to_shared_offset(smem_a_ptr.cast_const());
                let smem_b_base0 = smem_a_base0 + (SMEM_B_OFFSET as u64);
                let smem_sfa_base0 = smem_a_base0 + (SMEM_SFA_OFFSET as u64);
                let smem_sfb_base0 = smem_a_base0 + (SMEM_SFB_OFFSET as u64);
                let idesc = encode_mxf4nvf4_idesc(valid_n as u32);
                let global_scale = w_up_global[expert] * rcp_approx_ftz_f32(input_scale);
                let col_perm = ((tid & 12) << 3) | ((tid >> 4) << 2) | (tid & 3);
                let sub_tile16 = tid & 7;
                let grp = tile_m128 * 8 + sub_tile16;
                let burst_inter = grp >> 5;
                let s_in_burst = (grp >> 2) & 7;
                let sub_u64_off = s_in_burst * 1024 + ((grp >> 1) & 1) * 512 + (grp & 1);
                let k_atom = grp >> 2;
                let k_in_atom = grp & 3;
                let inter_u64_ptr = inter_out.as_mut_ptr().cast::<u64>();
                let smem_ab_f32 = smem_b_ptr.cast::<f32>();
                let smem_ab_u128 = smem_b_ptr.cast_const().cast::<u128>();

                let mut g_idx = 0usize;
                while g_idx < g_per_cta {
                    let group = group_start + g_idx;
                    let parity = (g_idx & 1) as u32;
                    if tid == 0 {
                        issue_mxf4nvf4_256_bursts!(
                            tmem_addr,
                            smem_a_base0,
                            smem_b_base0,
                            smem_sfa_base0,
                            smem_sfb_base0,
                            idesc,
                            g_idx,
                            parity,
                            tma_bar_const,
                            mma_bar_ptr,
                            mma_bar_const
                        );
                    }
                    barrier_cta_sync(1, 512);

                    tcgen05_fence_after_thread_sync();
                    if tid < 128 {
                        let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                        let mut n_base = 0usize;
                        while n_base + 32 <= limit {
                            let r0 =
                                tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32))
                                    .to_array();
                            let r1 = tcgen05_ld_32x32b_x16_raw(
                                tmem_row_base + ((n_base + 16) as u32),
                            )
                            .to_array();
                            #[unroll]
                            for c in 0..16 {
                                let a0 = f32::from_bits(r0[c]) * global_scale;
                                let a1 = f32::from_bits(r1[c]) * global_scale;
                                let u0 = if a0 > 0.0 { a0 } else { 0.0 };
                                let u1 = if a1 > 0.0 { a1 } else { 0.0 };
                                *smem_ab_f32.add((n_base + c) * 128 + col_perm) = u0 * u0;
                                *smem_ab_f32.add((n_base + 16 + c) * 128 + col_perm) = u1 * u1;
                            }
                            n_base += 32;
                        }
                        while n_base < limit {
                            let regs =
                                tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32))
                                    .to_array();
                            #[unroll]
                            for c in 0..16 {
                                if n_base + c < limit {
                                    let acc = f32::from_bits(regs[c]) * global_scale;
                                    let relu = if acc > 0.0 { acc } else { 0.0 };
                                    *smem_ab_f32.add((n_base + c) * 128 + col_perm) = relu * relu;
                                }
                            }
                            n_base += 16;
                        }
                        tcgen05_load_wait();
                    }
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 512);

                    let mut n = tid >> 3;
                    while n < limit {
                        let row_u128 = smem_ab_u128.add(n * 32 + sub_tile16);
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
                        let scale_code = dev_f32_to_e4m3(amax * (1.0f32 / 6.0f32));
                        let scale_val = dev_e4m3_to_f32(scale_code);
                        let inv_scale_e2m1 = if scale_val > 0.0 {
                            rcp_approx_ftz_f32(scale_val) * f32::from_bits(1 << 23)
                        } else {
                            0.0f32
                        };
                        let mut packed_u64 = 0u64;
                        #[unroll]
                        for i in 0..16 {
                            packed_u64 |= fast_pos_e2m1(vals[i] * inv_scale_e2m1) << (i * 4);
                        }
                        let sfb_word = ((n & 31) * 4) + (((n & 127) >> 5) & 3);
                        let scale_idx =
                            ((group * 16 + k_atom) * 2 + (n >> 7)) * 512 + sfb_word * 4 + k_in_atom;
                        *inter_scales.get_unchecked_mut(scale_idx) = scale_code;
                        let u64_idx = (group * 2 + burst_inter) * 8192 + sub_u64_off + n * 2;
                        *inter_u64_ptr.add(u64_idx) = packed_u64;
                        n += 64;
                        if g_idx + 1 < g_per_cta && (n >> 6) == 2 {
                            fence_proxy_async_shared_cta();
                            barrier_cta_sync(1, 512);
                            if tid == 0 {
                                let _ = mbarrier_arrive(b_free_bar_const);
                            }
                        }
                    }

                    if g_idx + 1 < g_per_cta {
                        fence_proxy_async_shared_cta();
                        barrier_cta_sync(1, 512);
                        if tid == 0 {
                            let _ = mbarrier_arrive(b_free_bar_const.add(1));
                        }
                    }
                    g_idx += 1;
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

    /// Stage 3 (`B >= 2048` 6-Warp `192`-Thread Persistent Weight-Stationary 4-Stage TMA + 2-Stage TMEM + Register-Direct Epilogue):
    /// `Warps 0..3` (`128` threads) issue all 8 `tcgen05.ld.32x32b.x16` loads (`128` channels) in a single burst,
    /// release `TMEM[tmem_slot]` immediately, and quantize + store `u128` values and `u32` scales with `0 KB` `SMEM_EPI`!
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_grouped_fc1_relu2_requant_128(
        #[grid_constant] w_up_tma: &TmaDescriptor,
        #[grid_constant] b_up_tma: &TmaDescriptor,
        #[grid_constant] sfa_up_tma: &TmaDescriptor,
        #[grid_constant] sfb_up_tma: &TmaDescriptor,
        w_up_global: &[f32],
        group_experts: &[i32],
        mut inter_out: DisjointSlice<u8>,
        mut inter_scales: DisjointSlice<u8>,
        input_scale: f32,
        intermediate_size: u32,
        groups_per_cta: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let tile_m128 = cta_id & 7;
        let g_per_cta = groups_per_cta as usize;
        let group_start = (cta_id >> 3) * g_per_cta;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = group_experts[group_start] as usize;
        let inter = intermediate_size as usize;
        let w_outer = expert * (inter / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = dyn_smem;
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(SMEM_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_BARRIER_OFFSET).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { tma_bar_ptr.add(4) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(4) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_free_bar_ptr = unsafe { mma_bar_ptr.add(2) };
        let tmem_free_bar_const = tmem_free_bar_ptr.cast_const();
        let w_bar_ptr = unsafe { tmem_free_bar_ptr.add(2) };
        let w_bar_const = w_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { w_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 13 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 512);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 5 && (tid & 31) == 0 {
                run_tma_producer_128!(
                    w_up_tma,
                    b_up_tma,
                    sfa_up_tma,
                    sfb_up_tma,
                    smem_a_ptr,
                    smem_b_ptr,
                    smem_sfa_ptr,
                    smem_sfb_ptr,
                    tma_bar_ptr,
                    b_free_bar_const,
                    w_bar_ptr,
                    w_outer,
                    group_start,
                    g_per_cta
                );
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a_base0 = cvta_generic_to_shared_offset(smem_a_ptr.cast_const());
                run_mma_consumer_128!(
                    tmem_addr,
                    smem_a_base0,
                    g_per_cta,
                    w_bar_const,
                    tma_bar_const,
                    tmem_free_bar_const,
                    b_free_bar_ptr,
                    mma_bar_ptr
                );
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let global_scale = w_up_global[expert] * rcp_approx_ftz_f32(input_scale);
                let n = tid; // 0..127
                let sfb_word = sfb_scale_word(n);
                let burst_inter = tile_m128 >> 2;
                let inter_u128_ptr = inter_out.as_mut_ptr().cast::<u128>();
                let scales_u32_ptr = inter_scales.as_mut_ptr().cast::<u32>();

                let mut g_idx = 0usize;
                while g_idx < g_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        while !mbarrier_try_wait_parity(
                            mma_bar_const.add(tmem_slot),
                            tmem_parity,
                        ) {}
                    }
                    barrier_cta_sync(1, 128);

                    tcgen05_fence_after_thread_sync();
                    let tmem_row_base = tmem_d + ((warp_id * 32) << 16);
                    let scale_sq_div6 = global_scale * global_scale * (1.0f32 / 6.0f32);
                    #[unroll]
                    for half_m in 0..2 {
                        let c_base = (half_m * 64) as u32;
                        let r64 = tcgen05_ld_32x32b_x64_raw(tmem_row_base + c_base).to_array();
                        tcgen05_load_wait();
                        if half_m == 1 {
                            tcgen05_fence_before_thread_sync();
                            barrier_cta_sync(1, 128);
                            if g_idx + 2 < g_per_cta && tid == 0 {
                                let _ = mbarrier_arrive(tmem_free_bar_const.add(tmem_slot));
                            }
                        }
                        let mut p_lo = 0u128;
                        let mut p_hi = 0u128;
                        let mut sc4 = 0u32;
                        #[unroll]
                        for sub in 0..4 {
                            let base16 = sub * 16;
                            let mut vals = [0.0f32; 16];
                            let mut max_raw = 0.0f32;
                            #[unroll]
                            for i in 0..16 {
                                let raw = f32::from_bits(r64[base16 + i]);
                                if raw > max_raw {
                                    max_raw = raw;
                                }
                                let a = raw * global_scale;
                                let u = if a > 0.0 { a } else { 0.0 };
                                vals[i] = u * u;
                            }
                            let scale_code = dev_f32_to_e4m3(max_raw * max_raw * scale_sq_div6);
                            let scale_val = dev_e4m3_to_f32(scale_code);
                            let inv_scale_e2m1 = if scale_val > 0.0 {
                                rcp_approx_ftz_f32(scale_val) * f32::from_bits(1 << 23)
                            } else {
                                0.0f32
                            };
                            let mut packed_u64 = 0u64;
                            #[unroll]
                            for i in 0..16 {
                                packed_u64 |=
                                    fast_pos_e2m1(vals[i] * inv_scale_e2m1) << (i * 4);
                            }
                            if sub < 2 {
                                p_lo |= (packed_u64 as u128) << (sub * 64);
                            } else {
                                p_hi |= (packed_u64 as u128) << ((sub - 2) * 64);
                            }
                            sc4 |= (scale_code as u32) << (sub * 8);
                        }
                        let k_atom = tile_m128 * 2 + half_m;
                        let s_in_burst = k_atom & 7;
                        let scale_u32_idx = (group * 16 + k_atom) * 128 + sfb_word;
                        *scales_u32_ptr.add(scale_u32_idx) = sc4;
                        let u128_base = (group * 2 + burst_inter) * 2048 + s_in_burst * 256 + n;
                        *inter_u128_ptr.add(u128_base) = p_lo;
                        *inter_u128_ptr.add(u128_base + 128) = p_hi;
                    }
                    g_idx += 1;
                }

                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 512);
                }
                if tid < 13 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage `K4` (`tcgen05` Hardware Block-Scaled NVFP4 Tensor Core + Persistent Weight-Stationary 2-Stage Full-TMA):
    /// `128` persistent CTAs (`1` wave on B200) load `W_down` (`SMEM_A`) and `SFA` (`TMEM`) once on `g_idx == 0`
    /// and stream `g_per_cta` (`1..16` for `B = 1024..16384`) token groups of `B + SFB` via `Warp 16` TMA!
    #[launch_bounds(544)]
    #[launch_contract(
        domain = 1,
        block = (544, 1, 1),
        dynamic_shared = 222208,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_grouped_fc2(
        #[grid_constant] w_down_tma: &TmaDescriptor,
        #[grid_constant] b_down_tma: &TmaDescriptor,
        #[grid_constant] sfa_down_tma: &TmaDescriptor,
        #[grid_constant] sfb_down_tma: &TmaDescriptor,
        w_down_global: &[f32],
        group_experts: &[i32],
        group_routes: &[i32],
        group_limits: &[i32],
        routing_weights: &[u16],
        mut out_or_grouped: DisjointSlice<u16>,
        hidden_size: u32,
        groups_per_cta: u32,
    ) {
        let cta_id = thread::blockIdx_x() as usize;
        let tile_m128 = cta_id & 7;
        let g_per_cta = groups_per_cta as usize;
        let group_start = (cta_id >> 3) * g_per_cta;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = group_experts[group_start] as usize;
        let limit = group_limits[group_start] as usize;
        let valid_n = ((limit + 15) & !15).max(16);
        let h = hidden_size as usize;
        let w_outer = expert * (h / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = dyn_smem;
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(SMEM_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_SFB_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_BARRIER_OFFSET).cast::<Barrier>() };
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

            if warp_id == 16 && (tid & 31) == 0 {
                run_tma_producer_256!(
                    w_down_tma,
                    b_down_tma,
                    sfa_down_tma,
                    sfb_down_tma,
                    smem_a_ptr,
                    smem_b_ptr,
                    smem_sfa_ptr,
                    smem_sfb_ptr,
                    tma_bar_ptr,
                    b_free_bar_const,
                    w_outer,
                    group_start,
                    g_per_cta
                );
            } else if warp_id < 16 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a_base0 = cvta_generic_to_shared_offset(smem_a_ptr.cast_const());
                let smem_b_base0 = smem_a_base0 + (SMEM_B_OFFSET as u64);
                let smem_sfa_base0 = smem_a_base0 + (SMEM_SFA_OFFSET as u64);
                let smem_sfb_base0 = smem_a_base0 + (SMEM_SFB_OFFSET as u64);
                let idesc = encode_mxf4nvf4_idesc(valid_n as u32);
                let col4 = tid & 3;
                let row_off = tid >> 2;
                let global_scale = w_down_global[expert];
                let out_u32_ptr = out_or_grouped.as_mut_ptr().cast::<u32>();
                let out_col_u32_base = tile_m128 * 64 + col4 * 2;
                let h_u32 = h >> 1;
                let smem_ab_u32 = smem_b_ptr.cast::<u32>();
                let smem_ab_u128 = smem_b_ptr.cast_const().cast::<u128>();

                let mut g_idx = 0usize;
                while g_idx < g_per_cta {
                    let group = group_start + g_idx;
                    let parity = (g_idx & 1) as u32;
                    if tid == 0 {
                        issue_mxf4nvf4_256_bursts!(
                            tmem_addr,
                            smem_a_base0,
                            smem_b_base0,
                            smem_sfa_base0,
                            smem_sfb_base0,
                            idesc,
                            g_idx,
                            parity,
                            tma_bar_const,
                            mma_bar_ptr,
                            mma_bar_const
                        );
                    }
                    barrier_cta_sync(1, 512);

                    tcgen05_fence_after_thread_sync();
                    if tid < 128 {
                        let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                        let mut n_base = 0usize;
                        while n_base + 32 <= limit {
                            let r0 =
                                tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32))
                                    .to_array();
                            let r1 = tcgen05_ld_32x32b_x16_raw(
                                tmem_row_base + ((n_base + 16) as u32),
                            )
                            .to_array();
                            #[unroll]
                            for c in 0..16 {
                                *smem_ab_u32.add((n_base + c) * 128 + tid) = r0[c];
                                *smem_ab_u32.add((n_base + 16 + c) * 128 + tid) = r1[c];
                            }
                            n_base += 32;
                        }
                        while n_base < limit {
                            let regs =
                                tcgen05_ld_32x32b_x16_raw(tmem_row_base + (n_base as u32))
                                    .to_array();
                            #[unroll]
                            for c in 0..16 {
                                if n_base + c < limit {
                                    *smem_ab_u32.add((n_base + c) * 128 + tid) = regs[c];
                                }
                            }
                            n_base += 16;
                        }
                        tcgen05_load_wait();
                    }
                    tcgen05_fence_before_thread_sync();
                    barrier_cta_sync(1, 512);

                    let half_n = valid_n.min(128);
                    let half_mask = half_n - 1;
                    let n_rot = (expert * (half_n >> 2)) & half_mask;
                    let mut n_step = row_off;
                    while n_step < valid_n {
                        let n = (n_step & 128) | ((n_step + n_rot) & half_mask);
                        if n < limit {
                            let route = group_routes[group * 256 + n] as usize;
                            let token = route >> 2;
                            let scale = global_scale * bf16_to_f32(routing_weights[route]);
                            let row_u128 = smem_ab_u128.add(n * 32 + col4);
                            let qs = [
                                *row_u128,
                                *row_u128.add(4),
                                *row_u128.add(8),
                                *row_u128.add(12),
                                *row_u128.add(16),
                                *row_u128.add(20),
                                *row_u128.add(24),
                                *row_u128.add(28),
                            ];
                            let dst_ptr = out_u32_ptr.add(token * h_u32 + out_col_u32_base);
                            #[unroll]
                            for q_idx in 0..8 {
                                let q = qs[q_idx];
                                let r0 = (q & 0xFFFF_FFFF) as u32;
                                let r1 = ((q >> 32) & 0xFFFF_FFFF) as u32;
                                let r2 = ((q >> 64) & 0xFFFF_FFFF) as u32;
                                let r3 = (q >> 96) as u32;
                                let b0 = f32_to_bf16_rne(f32::from_bits(r0) * scale) as u32;
                                let b1 = f32_to_bf16_rne(f32::from_bits(r1) * scale) as u32;
                                let b2 = f32_to_bf16_rne(f32::from_bits(r2) * scale) as u32;
                                let b3 = f32_to_bf16_rne(f32::from_bits(r3) * scale) as u32;
                                let d = dst_ptr.add(q_idx * 8);
                                let _ = atom_add_bf16x2(d, b0 | (b1 << 16));
                                let _ = atom_add_bf16x2(d.add(1), b2 | (b3 << 16));
                            }
                        }
                        if g_idx + 1 < g_per_cta && n_step < 128 {
                            fence_proxy_async_shared_cta();
                            barrier_cta_sync(1, 512);
                            if tid == 0 {
                                let _ = mbarrier_arrive(b_free_bar_const);
                            }
                        }
                        n_step += 128;
                    }

                    if g_idx + 1 < g_per_cta {
                        fence_proxy_async_shared_cta();
                        barrier_cta_sync(1, 512);
                        if tid == 0 {
                            let _ = mbarrier_arrive(b_free_bar_const.add(1));
                        }
                    }
                    g_idx += 1;
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

    /// Stage `K4` (`B >= 2048` 6-Warp `192`-Thread Persistent Weight-Stationary 4-Stage TMA + 2-Stage TMEM + 2-Stage S2G TMA Store):
    /// `Warps 0..3` (`128` threads) unload `TMEM[tmem_slot]` via `tcgen05.ld.32x32b.x64`, release `TMEM[tmem_slot]` immediately,
    /// and stream `[128, 32]` `bf16` (`8 KB`) tiles to `grouped_out` via hardware 2D TMA Store (`cp.async.bulk.tensor.2d.s2g`)!
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        dynamic_shared = 230400,
        dynamic_shared_alignment = 1024,
        min_compute_capability = (10, 0),
    )]
    #[kernel]
    pub unsafe fn tcgen05_grouped_fc2_128(
        #[grid_constant] w_down_tma: &TmaDescriptor,
        #[grid_constant] b_down_tma: &TmaDescriptor,
        #[grid_constant] sfa_down_tma: &TmaDescriptor,
        #[grid_constant] sfb_down_tma: &TmaDescriptor,
        #[grid_constant] grouped_out_tma: &TmaDescriptor,
        w_down_global: &[f32],
        group_experts: &[i32],
        mut grouped_out: DisjointSlice<u16>,
        hidden_size: u32,
        groups_per_cta: u32,
    ) {
        let _ = grouped_out.as_mut_ptr();
        let cta_id = thread::blockIdx_x() as usize;
        let tile_m128 = cta_id & 7;
        let g_per_cta = groups_per_cta as usize;
        let group_start = (cta_id >> 3) * g_per_cta;
        let tid = thread::threadIdx_x() as usize;
        let warp_id = (tid >> 5) as u32;

        let expert = group_experts[group_start] as usize;
        let h = hidden_size as usize;
        let w_outer = expert * (h / 128) + tile_m128;

        let dyn_smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let smem_a_ptr = dyn_smem;
        let smem_b_ptr = unsafe { dyn_smem.add(SMEM_B_OFFSET) };
        let smem_sfa_ptr = unsafe { dyn_smem.add(SMEM_SFA_OFFSET) };
        let smem_sfb_ptr = unsafe { dyn_smem.add(SMEM_SFB_OFFSET) };
        let smem_epi1_ptr = unsafe { dyn_smem.add(SMEM_BARRIER_OFFSET) };
        let tma_bar_ptr = unsafe { dyn_smem.add(SMEM_FC2_128_BARRIER_OFFSET).cast::<Barrier>() };
        let tma_bar_const = tma_bar_ptr.cast_const();
        let b_free_bar_ptr = unsafe { tma_bar_ptr.add(4) };
        let b_free_bar_const = b_free_bar_ptr.cast_const();
        let mma_bar_ptr = unsafe { b_free_bar_ptr.add(4) };
        let mma_bar_const = mma_bar_ptr.cast_const();
        let tmem_free_bar_ptr = unsafe { mma_bar_ptr.add(2) };
        let tmem_free_bar_const = tmem_free_bar_ptr.cast_const();
        let w_bar_ptr = unsafe { tmem_free_bar_ptr.add(2) };
        let w_bar_const = w_bar_ptr.cast_const();
        let tmem_slot_ptr = unsafe { w_bar_ptr.add(1).cast::<u32>() };

        unsafe {
            if tid < 13 {
                mbarrier_init(tma_bar_ptr.add(tid), 1);
            }
            if warp_id == 0 {
                tcgen05_alloc(tmem_slot_ptr, 512);
            }
            fence_proxy_async_shared_cta();
            thread::sync_threads();

            if warp_id == 5 && (tid & 31) == 0 {
                run_tma_producer_128!(
                    w_down_tma,
                    b_down_tma,
                    sfa_down_tma,
                    sfb_down_tma,
                    smem_a_ptr,
                    smem_b_ptr,
                    smem_sfa_ptr,
                    smem_sfb_ptr,
                    tma_bar_ptr,
                    b_free_bar_const,
                    w_bar_ptr,
                    w_outer,
                    group_start,
                    g_per_cta
                );
            } else if warp_id == 4 && (tid & 31) == 0 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let smem_a_base0 = cvta_generic_to_shared_offset(smem_a_ptr.cast_const());
                run_mma_consumer_128!(
                    tmem_addr,
                    smem_a_base0,
                    g_per_cta,
                    w_bar_const,
                    tma_bar_const,
                    tmem_free_bar_const,
                    b_free_bar_ptr,
                    mma_bar_ptr
                );
            } else if warp_id < 4 {
                let tmem_addr = *tmem_slot_ptr.cast_const();
                let global_scale = w_down_global[expert];
                let out_desc = core::ptr::from_ref(grouped_out_tma);
                let smem_epi0_u128 = smem_sfa_ptr.cast::<u128>();
                let smem_epi1_u128 = smem_epi1_ptr.cast::<u128>();
                let u_swz = (tid >> 1) & 3;

                let mut g_idx = 0usize;
                while g_idx < g_per_cta {
                    let group = group_start + g_idx;
                    let tmem_slot = g_idx & 1;
                    let tmem_parity = ((g_idx >> 1) & 1) as u32;
                    let tmem_d = tmem_addr + (tmem_slot as u32) * 128;
                    if tid == 0 {
                        while !mbarrier_try_wait_parity(
                            mma_bar_const.add(tmem_slot),
                            tmem_parity,
                        ) {}
                    }
                    barrier_cta_sync(1, 128);

                    tcgen05_fence_after_thread_sync();
                    let tmem_row_base = tmem_d + ((warp_id * 32) << 16);
                    #[unroll]
                    for half_m in 0..2 {
                        let c_base = (half_m * 64) as u32;
                        let r64 = tcgen05_ld_32x32b_x64_raw(tmem_row_base + c_base).to_array();
                        tcgen05_load_wait();
                        if half_m == 1 {
                            tcgen05_fence_before_thread_sync();
                            barrier_cta_sync(1, 128);
                            if g_idx + 2 < g_per_cta && tid == 0 {
                                let _ = mbarrier_arrive(tmem_free_bar_const.add(tmem_slot));
                            }
                        }
                        #[unroll]
                        for sub in 0..2 {
                            let base32 = sub * 32;
                            let epi_u128 = if sub == 0 {
                                smem_epi0_u128
                            } else {
                                smem_epi1_u128
                            };
                            if tid == 0 {
                                cp_async_bulk_wait_group_read(1);
                            }
                            barrier_cta_sync(1, 128);
                            #[unroll]
                            for u in 0..4 {
                                let base8 = base32 + u * 8;
                                let p0 = cvt_bf16x2_f32(
                                    f32::from_bits(r64[base8]) * global_scale,
                                    f32::from_bits(r64[base8 + 1]) * global_scale,
                                );
                                let p1 = cvt_bf16x2_f32(
                                    f32::from_bits(r64[base8 + 2]) * global_scale,
                                    f32::from_bits(r64[base8 + 3]) * global_scale,
                                );
                                let p2 = cvt_bf16x2_f32(
                                    f32::from_bits(r64[base8 + 4]) * global_scale,
                                    f32::from_bits(r64[base8 + 5]) * global_scale,
                                );
                                let p3 = cvt_bf16x2_f32(
                                    f32::from_bits(r64[base8 + 6]) * global_scale,
                                    f32::from_bits(r64[base8 + 7]) * global_scale,
                                );
                                let q = (p0 as u128)
                                    | ((p1 as u128) << 32)
                                    | ((p2 as u128) << 64)
                                    | ((p3 as u128) << 96);
                                *epi_u128.add(tid * 4 + (u ^ u_swz)) = q;
                            }
                            fence_proxy_async_shared_cta();
                            barrier_cta_sync(1, 128);
                            if tid == 0 {
                                cp_async_bulk_tensor_2d_s2g(
                                    epi_u128.cast_const().cast::<u8>(),
                                    out_desc,
                                    (tile_m128 * 128 + half_m * 64 + sub * 32) as i32,
                                    (group * 128) as i32,
                                );
                                cp_async_bulk_commit_group();
                            }
                        }
                    }
                    g_idx += 1;
                }

                if tid == 0 {
                    cp_async_bulk_wait_group_read(0);
                }
                if warp_id == 0 {
                    tcgen05_dealloc(tmem_addr, 512);
                }
                if tid < 13 {
                    mbarrier_inval(tma_bar_ptr.add(tid));
                }
            }
        }
    }

    /// Stage `K5` (`B >= 2048` Coalesced 128-Bit `u128` 4-Expert Combine — `1` Warp = `1` Token!):
    /// Each thread `tid_global` (`32` threads = `1` warp per token, `4 × u128` = `32` `bf16`s per thread)
    /// loads `gp128` and `w64` once per token and combines all `4` expert rows with 128-bit vector loads/stores!
    #[kernel]
    pub fn moe_finalize_tcgen05(
        grouped_out: &[u16],
        tok_to_perm: &[i32],
        routing_weights: &[u16],
        mut output: DisjointSlice<u16>,
        num_tokens: u32,
    ) {
        let tid_global =
            (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let token = tid_global >> 5;
        if token >= (num_tokens as usize) {
            return;
        }
        let lane = tid_global & 31;
        unsafe {
            let gp128 = *tok_to_perm.as_ptr().cast::<u128>().add(token);
            let w64 = *routing_weights.as_ptr().cast::<u64>().add(token);
            let gp0 = ((gp128 & 0xFFFF_FFFF) as usize) * 128 + lane;
            let gp1 = (((gp128 >> 32) & 0xFFFF_FFFF) as usize) * 128 + lane;
            let gp2 = (((gp128 >> 64) & 0xFFFF_FFFF) as usize) * 128 + lane;
            let gp3 = ((gp128 >> 96) as usize) * 128 + lane;
            let s0 = bf16_to_f32((w64 & 0xFFFF) as u16);
            let s1 = bf16_to_f32(((w64 >> 16) & 0xFFFF) as u16);
            let s2 = bf16_to_f32(((w64 >> 32) & 0xFFFF) as u16);
            let s3 = bf16_to_f32((w64 >> 48) as u16);
            let grouped_u128 = grouped_out.as_ptr().cast::<u128>();
            let out_u128 = output.as_mut_ptr().cast::<u128>().add(token * 128 + lane);

            #[unroll]
            for k in 0..4 {
                let off = k * 32;
                let q0 = *grouped_u128.add(gp0 + off);
                let q1 = *grouped_u128.add(gp1 + off);
                let q2 = *grouped_u128.add(gp2 + off);
                let q3 = *grouped_u128.add(gp3 + off);
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
                *out_u128.add(off) = (p0 as u128)
                    | ((p1 as u128) << 32)
                    | ((p2 as u128) << 64)
                    | ((p3 as u128) << 96);
            }
        }
    }
}

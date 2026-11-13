// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

//! Fused FP16 QKV projection, FP32 Q/K RMSNorm, FP8 conversion and paged KV stores.
//!
//! The grouped-M traversal and TMA load pattern are adapted from TileGym's
//! MIT-licensed `ops/cutile_rs/matmul_kernel/kernel.rs` (c343dd3).
//! See the `fused_qkv` example for correctness checks and CUDA graph timing.
//! Use the CUDA 13.5 Tile IR assembler for performance evaluation. This starting
//! schedule has not been tuned on B200 and does not guarantee epilogue overlap.

/*
 * MIT License
 *
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// FP16 projection -> FP32 per-head Q/K RMSNorm -> static E4M3FN outputs.
    ///
    /// # Safety
    /// All tensors must be contiguous, on the launch device, non-overlapping,
    /// and live through completion. Outputs are writable despite `&Tensor`.
    /// `x:[M,K]`, `w:[(HQ+2*HKV)*128,K]`, `gamma_q/k:[128]`, `slots:[M]`,
    /// `q:[M,HQ*128]`, `k/v_cache:[CAPACITY,HKV*128]`. The flattened cache is
    /// physically `[pages,page_size,HKV,128]` (NHD), not an HND cache.
    /// Dimensions and grid.x must be positive, with grid.y = grid.z = 1.
    /// K/row strides must satisfy TMA alignment. Use BM/BK positive powers of
    /// two, GROUP_M > 0, and ensure tile counts/scalar dimension arithmetic fit
    /// i32. Slots are int64:
    /// negative means skip KV (still write Q); nonnegative slots must be
    /// unique and < CAPACITY. The bounds mask is not a duplicate-write guard.
    /// Inputs/gamma/scales must be finite; epsilon/scales must be positive.
    /// No concurrent readers/writers may race with these output stores.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn fused_qkv<const BM: i32, const BK: i32, const GROUP_M: i32>(
        x: &Tensor<f16, { [-1, -1] }>,
        w: &Tensor<f16, { [-1, -1] }>,
        gamma_q: &Tensor<f32, { [-1] }>,
        gamma_k: &Tensor<f32, { [-1] }>,
        slots: &Tensor<i64, { [-1] }>,
        q: &Tensor<f8e4m3fn, { [-1, -1] }>,
        k_cache: &Tensor<f8e4m3fn, { [-1, -1] }>,
        v_cache: &Tensor<f8e4m3fn, { [-1, -1] }>,
        m: i32,
        k: i32,
        q_heads: i32,
        kv_heads: i32,
        capacity: i64,
        eps: f32,
        inv_q_scale: f32,
        inv_k_scale: f32,
        inv_v_scale: f32,
    ) {
        // One complete head per N tile: no partial-head normalization or
        // inter-CTA reduction. Wider N tiles are an explicit tuning follow-up.
        let nm = (m + BM - 1i32) / BM;
        let nh = q_heads + 2i32 * kv_heads;
        let tiles = nm * nh;
        let group_tiles = GROUP_M * nh;
        let start = get_tile_block_id().0;
        let stride = get_num_tile_blocks().0;
        let x_part = x.partition(const_shape![BM, BK]);
        let w_part = w.partition(const_shape![128, BK]);
        let gq = gamma_q.partition(const_shape![128]).load([0i32]);
        let gk = gamma_k.partition(const_shape![128]).load([0i32]);

        // grid.x == tiles gives ordinary one-tile-per-program scheduling;
        // fewer programs gives the static-persistent version of the same body.
        for tile in (start..tiles).step_by(stride as usize) {
            let first_m = (tile / group_tiles) * GROUP_M;
            let remaining = nm - first_m;
            let gm = if remaining < GROUP_M {
                remaining
            } else {
                GROUP_M
            };
            let tm = first_m + tile % gm;
            let head = (tile % group_tiles) / gm;
            let mut acc: Tile<f32, { [BM, 128] }> = constant(0.0f32, const_shape![BM, 128]);
            for tk in 0i32..((k + BK - 1i32) / BK) {
                // Partition loads zero-pad tails. W is stored [N,K], so each
                // whole head's K slice is contiguous before the transpose.
                let a: Tile<f16, { [BM, BK] }> = load_view_tko(
                    &x_part,
                    [tm, tk],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let b: Tile<f16, { [128, BK] }> = load_view_tko(
                    &w_part,
                    [head, tk],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                acc = mmaf(a, b.transpose(), acc);
            }

            if head < q_heads + kv_heads {
                let sum: Tile<f32, { [BM] }> = reduce_sum(acc * acc, 1i32);
                let inv_rms: Tile<f32, { [BM] }> = rsqrt(
                    sum * constant(0.0078125f32, const_shape![BM])
                        + broadcast_scalar(eps, const_shape![BM]),
                    ftz::Disabled,
                );
                let gamma = if head < q_heads { gq } else { gk };
                acc = acc
                    * inv_rms
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, 128])
                    * gamma
                        .reshape(const_shape![1, 128])
                        .broadcast(const_shape![BM, 128]);
            }
            let inv_scale = if head < q_heads {
                inv_q_scale
            } else if head < q_heads + kv_heads {
                inv_k_scale
            } else {
                inv_v_scale
            };
            let scaled = acc * broadcast_scalar(inv_scale, const_shape![BM, 128]);
            // Explicit finite saturation before round-to-nearest conversion.
            let lower: Tile<f32, { [BM, 128] }> = constant(-448.0f32, const_shape![BM, 128]);
            let upper: Tile<f32, { [BM, 128] }> = constant(448.0f32, const_shape![BM, 128]);
            let bounded: Tile<f32, { [BM, 128] }> =
                maxf(scaled, lower, nan::Disabled, ftz::Disabled);
            let clamped: Tile<f32, { [BM, 128] }> =
                minf(bounded, upper, nan::Disabled, ftz::Disabled);
            let out: Tile<f8e4m3fn, { [BM, 128] }> = convert_tile(clamped);
            let lanes: Tile<i32, { [BM] }> = iota(const_shape![BM]);
            let rows = lanes + broadcast_scalar(tm * BM, const_shape![BM]);
            let row_valid = cmpi(
                rows,
                broadcast_scalar(m, const_shape![BM]),
                predicate::LessThan,
            );
            let rows64: Tile<i64, { [BM] }> = exti(rows);
            let cols: Tile<i64, { [128] }> = iota(const_shape![128]);

            if head < q_heads {
                let q_stride32: Tile<i32, { [BM] }> =
                    broadcast_scalar(q_heads * 128i32, const_shape![BM]);
                let q_stride64: Tile<i64, { [BM] }> = exti(q_stride32);
                let head_base32: Tile<i32, { [128] }> =
                    broadcast_scalar(head * 128i32, const_shape![128]);
                let head_base64: Tile<i64, { [128] }> = exti(head_base32);
                let offsets = (rows64 * q_stride64)
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, 128])
                    + (cols + head_base64)
                        .reshape(const_shape![1, 128])
                        .broadcast(const_shape![BM, 128]);
                let base: PointerTile<*mut f8e4m3fn, { [] }> =
                    pointer_to_tile(cast_mut(q.as_ptr()));
                let base_2d: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                    base.reshape(const_shape![1, 1]);
                let base_tile: PointerTile<*mut f8e4m3fn, { [BM, 128] }> =
                    base_2d.broadcast(const_shape![BM, 128]);
                let ptrs: PointerTile<*mut f8e4m3fn, { [BM, 128] }> =
                    base_tile.offset_tile(offsets);
                let mask_rows: Tile<bool, { [BM, 1] }> = row_valid.reshape(const_shape![BM, 1]);
                let mask: Tile<bool, { [BM, 128] }> = mask_rows.broadcast(const_shape![BM, 128]);
                unsafe {
                    let _: Token = store_ptr_tko(
                        ptrs,
                        out,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None::<Token>,
                        Latency::<1>,
                    );
                }
            } else {
                let slot_base: PointerTile<*const i64, { [] }> = pointer_to_tile(slots.as_ptr());
                let slot_base_1d: PointerTile<*const i64, { [1] }> =
                    slot_base.reshape(const_shape![1]);
                let slot_base_tile: PointerTile<*const i64, { [BM] }> =
                    slot_base_1d.broadcast(const_shape![BM]);
                let zero_rows: Tile<i64, { [BM] }> = constant(0i64, const_shape![BM]);
                let safe_rows: Tile<i64, { [BM] }> = select(row_valid, rows64, zero_rows);
                let slot_ptrs: PointerTile<*const i64, { [BM] }> =
                    slot_base_tile.offset_tile(safe_rows);
                let (slot, _): (Tile<i64, { [BM] }>, Token) = unsafe {
                    load_ptr_tko(
                        slot_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(row_valid),
                        Some(-1i64),
                        None::<Token>,
                        Latency::<1>,
                    )
                };
                let zero_slots: Tile<i64, { [BM] }> = constant(0i64, const_shape![BM]);
                let slot_limit: Tile<i64, { [BM] }> = broadcast_scalar(capacity, const_shape![BM]);
                let valid: Tile<bool, { [BM] }> = row_valid
                    & cmpi(slot, zero_slots, predicate::GreaterThanOrEqual)
                    & cmpi(slot, slot_limit, predicate::LessThan);
                // Invalid slots get safe addresses as well as a store mask.
                let safe_slot: Tile<i64, { [BM] }> = select(valid, slot, zero_slots);
                let cache_head = if head < q_heads + kv_heads {
                    head - q_heads
                } else {
                    head - q_heads - kv_heads
                };
                let cache_stride32: Tile<i32, { [BM] }> =
                    broadcast_scalar(kv_heads * 128i32, const_shape![BM]);
                let cache_stride64: Tile<i64, { [BM] }> = exti(cache_stride32);
                let head_base32: Tile<i32, { [128] }> =
                    broadcast_scalar(cache_head * 128i32, const_shape![128]);
                let head_base64: Tile<i64, { [128] }> = exti(head_base32);
                let offsets = (safe_slot * cache_stride64)
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, 128])
                    + (cols + head_base64)
                        .reshape(const_shape![1, 128])
                        .broadcast(const_shape![BM, 128]);
                let base: *mut f8e4m3fn = if head < q_heads + kv_heads {
                    cast_mut(k_cache.as_ptr())
                } else {
                    cast_mut(v_cache.as_ptr())
                };
                let base_tile: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(base);
                let base_2d: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                    base_tile.reshape(const_shape![1, 1]);
                let base_broadcast: PointerTile<*mut f8e4m3fn, { [BM, 128] }> =
                    base_2d.broadcast(const_shape![BM, 128]);
                let ptrs: PointerTile<*mut f8e4m3fn, { [BM, 128] }> =
                    base_broadcast.offset_tile(offsets);
                let mask_rows: Tile<bool, { [BM, 1] }> = valid.reshape(const_shape![BM, 1]);
                let mask: Tile<bool, { [BM, 128] }> = mask_rows.broadcast(const_shape![BM, 128]);
                unsafe {
                    let _: Token = store_ptr_tko(
                        ptrs,
                        out,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(mask),
                        None::<Token>,
                        Latency::<1>,
                    );
                }
            }
        }
    }
}

pub use kernels::fused_qkv;

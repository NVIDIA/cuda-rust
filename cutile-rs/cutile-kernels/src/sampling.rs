/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Categorical sampling with temperature, top-k, and top-p filtering.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod sample_module {
    use cutile::core::*;

    /// Samples one token per row using temperature, top-k, and top-p filtering.
    ///
    /// # Safety
    /// `len <= BLOCK`, `u` has one value per row, `out` has one element per
    /// row, and each `u` is in `[0, 1)`.
    #[cutile::entry(print_ir = false, unchecked_accesses = true)]
    unsafe fn sample<E: FloatElement, const BLOCK: i32>(
        logits: &Tensor<E, { [-1, BLOCK] }>, // [batch, vocab] padded to BLOCK
        u: &Tensor<f32, { [-1] }>,           // [batch] uniforms in [0, 1)
        out: &mut Tensor<u32, { [1] }>,      // per-CTA sampled token id
        len: i32,                            // real vocab size (<= BLOCK)
        temperature: f32,
        k: i32, // top-k; k >= len keeps everything
        p: f32, // top-p; p >= 1.0 keeps everything
    ) {
        let row: i32 = get_tile_block_id().0;
        let shape = const_shape![BLOCK];
        let zeros: Tile<f32, { [BLOCK] }> = constant(0.0f32, shape);
        let ones: Tile<f32, { [BLOCK] }> = constant(1.0f32, shape);
        let neg: Tile<f32, { [BLOCK] }> = zeros - constant(1.0e30f32, shape);

        let logits_part: Partition<E, { [1, BLOCK] }> = logits.partition(const_shape![1, BLOCK]);
        let x: Tile<f32, { [BLOCK] }> = convert_tile(logits_part.load([row, 0i32]).reshape(shape));
        let x: Tile<f32, { [BLOCK] }> = x * broadcast_scalar(1.0f32 / temperature, shape);

        let idx: Tile<i32, { [BLOCK] }> = iota(shape);
        let valid: Tile<bool, { [BLOCK] }> = lt_tile(idx, len.broadcast(shape));
        let x: Tile<f32, { [BLOCK] }> = select(valid, x, neg);

        let hi_t: Tile<f32, { [1] }> = reduce_max(x, 0i32);
        let hi0: f32 = tile_to_scalar(hi_t.reshape(const_shape![]));
        let lo_t: Tile<f32, { [1] }> = reduce_min(select(valid, x, hi0.broadcast(shape)), 0i32);
        let lo0: f32 = tile_to_scalar(lo_t.reshape(const_shape![]));
        let mut lo: f32 = lo0;
        let mut hi: f32 = hi0;
        let kf: f32 = convert_scalar(k);
        for _i in 0i32..32i32 {
            let mid: f32 = (lo + hi) * 0.5f32;
            let over: Tile<f32, { [BLOCK] }> =
                select(ge_tile(x, mid.broadcast(shape)), ones, zeros);
            let cnt_t: Tile<f32, { [1] }> = reduce_sum(over, 0i32);
            let cnt: f32 = tile_to_scalar(cnt_t.reshape(const_shape![]));
            let raise: bool = cnt > kf;
            lo = if raise { mid } else { lo };
            hi = if raise { hi } else { mid };
        }
        let x: Tile<f32, { [BLOCK] }> = select(ge_tile(x, hi.broadcast(shape)), x, neg);

        let m_t: Tile<f32, { [1] }> = reduce_max(x, 0i32);
        let m: f32 = tile_to_scalar(m_t.reshape(const_shape![]));
        let e: Tile<f32, { [BLOCK] }> = exp(x - m.broadcast(shape));
        let denom_t: Tile<f32, { [1] }> = reduce_sum(e, 0i32);
        let denom: f32 = tile_to_scalar(denom_t.reshape(const_shape![]));
        let probs: Tile<f32, { [BLOCK] }> = e * broadcast_scalar(1.0f32 / denom, shape);

        let phi_t: Tile<f32, { [1] }> = reduce_max(probs, 0i32);
        let mut plo: f32 = 0.0f32;
        let mut phi: f32 = tile_to_scalar(phi_t.reshape(const_shape![]));
        for _i in 0i32..32i32 {
            let mid: f32 = (plo + phi) * 0.5f32;
            let kept: Tile<f32, { [BLOCK] }> =
                select(ge_tile(probs, mid.broadcast(shape)), probs, zeros);
            let mass_t: Tile<f32, { [1] }> = reduce_sum(kept, 0i32);
            let mass: f32 = tile_to_scalar(mass_t.reshape(const_shape![]));
            let shrink: bool = mass > p;
            plo = if shrink { mid } else { plo };
            phi = if shrink { phi } else { mid };
        }
        let probs: Tile<f32, { [BLOCK] }> =
            select(ge_tile(probs, plo.broadcast(shape)), probs, zeros);
        let z_t: Tile<f32, { [1] }> = reduce_sum(probs, 0i32);
        let z: f32 = tile_to_scalar(z_t.reshape(const_shape![]));
        let probs: Tile<f32, { [BLOCK] }> = probs * broadcast_scalar(1.0f32 / z, shape);

        let u_part: Partition<f32, { [1] }> = u.partition(const_shape![1]);
        let uu: f32 = tile_to_scalar(u_part.load([row]).reshape(const_shape![]));
        let cdf: Tile<f32, { [BLOCK] }> = scan_sum(probs, 0i32, reverse::Forward, 0.0f32);
        let crossed: Tile<bool, { [BLOCK] }> = lt_tile(uu.broadcast(shape), cdf);
        let cand: Tile<i32, { [BLOCK] }> = select(crossed, idx, len.broadcast(shape));
        let tok_t: Tile<i32, { [1] }> = reduce_min(cand, 0i32);
        let tok: i32 = tile_to_scalar(tok_t.reshape(const_shape![]));

        let tok_s: Tile<i32, { [] }> = scalar_to_tile(tok);
        let tok_i32: Tile<i32, { [1] }> = tok_s.reshape(const_shape![1]);
        let tok_u: Tile<u32, { [1] }> = bitcast(tok_i32);
        out.store(tok_u);
    }
}

pub use sample_module::sample;

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Device-side argmax and fused LM-head argmax kernels for sampling or greedy decode.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod argmax_blocks_f16_module {
    use cutile::core::*;

    /// First-stage block argmax over f16 logits. Used for greedy token selection over vocab-sized vectors.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn argmax_blocks_f16<const BLOCK_SIZE: i32>(
        logits: &Tensor<f16, { [-1] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let logits_part = logits.partition(const_shape![BLOCK_SIZE]);
        let logits_f16: Tile<f16, { [BLOCK_SIZE] }> = logits_part.load([block]);
        let logits: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(logits_f16);

        let base: i32 = block * BLOCK_SIZE;
        let base_tile: Tile<i32, { [BLOCK_SIZE] }> = base.broadcast(const_shape![BLOCK_SIZE]);
        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let indices: Tile<i32, { [BLOCK_SIZE] }> = base_tile + offs;

        let len_tile: Tile<i32, { [BLOCK_SIZE] }> = len.broadcast(const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(indices, len_tile);

        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [BLOCK_SIZE] }> = select(valid, logits, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));

        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_logits, max_bcast);
        let invalid_idx: i32 = len + 1i32;
        let invalid_idx: Tile<i32, { [BLOCK_SIZE] }> =
            invalid_idx.broadcast(const_shape![BLOCK_SIZE]);
        let candidate_idx: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }
}

pub use argmax_blocks_f16_module::argmax_blocks_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod lm_head_argmax_blocks_f16_module {
    use cutile::core::*;

    /// Fuses LM-head matvec and block argmax, avoiding materialized logits. Used in decode for [vocab, hidden] x [1, hidden].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn lm_head_argmax_blocks_f16<const K: i32>(
        weights: &Tensor<f16, { [-1, K] }>,
        hidden: &Tensor<f16, { [1, K] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        vocab_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let rows_shape: Shape<{ [64] }> = const_shape![64];
        let weight_shape: Shape<{ [64, 32] }> = const_shape![64, 32];
        let hidden_shape: Shape<{ [1, 32] }> = const_shape![1, 32];
        let weight_part: Partition<f16, { [64, 32] }> = weights.partition(weight_shape);
        let hidden_part: Partition<f16, { [1, 32] }> = hidden.partition(hidden_shape);

        let mut acc: Tile<f32, { [64] }> = constant(0.0f32, rows_shape);
        let num_k_tiles: i32 = K / 32i32;
        for k_block in 0i32..num_k_tiles {
            let w_f16: Tile<f16, { [64, 32] }> = weight_part.load([block, k_block]);
            let h_f16: Tile<f16, { [1, 32] }> = hidden_part.load([0i32, k_block]);
            let w: Tile<f32, { [64, 32] }> = convert_tile(w_f16);
            let h: Tile<f32, { [1, 32] }> = convert_tile(h_f16);
            let h_bc: Tile<f32, { [64, 32] }> = h.broadcast(weight_shape);
            let prod: Tile<f32, { [64, 32] }> = w * h_bc;
            let partial: Tile<f32, { [64] }> = reduce_sum(prod, 1i32);
            acc = acc + partial;
        }

        let base: i32 = block * 64i32;
        let base_tile: Tile<i32, { [64] }> = base.broadcast(rows_shape);
        let offs: Tile<i32, { [64] }> = iota(rows_shape);
        let indices: Tile<i32, { [64] }> = base_tile + offs;

        let vocab_tile: Tile<i32, { [64] }> = vocab_size.broadcast(rows_shape);
        let valid: Tile<bool, { [64] }> = lt_tile(indices, vocab_tile);
        let mask_mag: Tile<f32, { [64] }> = constant(1.0e30f32, rows_shape);
        let zero: Tile<f32, { [64] }> = constant(0.0f32, rows_shape);
        let neg_inf: Tile<f32, { [64] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [64] }> = select(valid, acc, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));
        let max_bcast: Tile<f32, { [64] }> = max_scalar.broadcast(rows_shape);
        let is_max: Tile<bool, { [64] }> = eq_tile(masked_logits, max_bcast);

        let invalid_idx: i32 = vocab_size + 1i32;
        let invalid_idx: Tile<i32, { [64] }> = invalid_idx.broadcast(rows_shape);
        let candidate_idx: Tile<i32, { [64] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }
}

pub use lm_head_argmax_blocks_f16_module::lm_head_argmax_blocks_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod argmax_reduce_blocks_to_u32_module {
    use cutile::core::*;

    /// Reduces per-block argmax results to one token id. Completes device-side greedy decoding after block argmax.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn argmax_reduce_blocks_to_u32<const BLOCK_SIZE: i32>(
        block_max: &Tensor<f32, { [-1] }>,
        block_idx: &Tensor<u32, { [-1] }>,
        out: &mut Tensor<u32, { [1] }>,
        num_blocks: i32,
    ) {
        let bm_part = block_max.partition(const_shape![BLOCK_SIZE]);
        let bi_part = block_idx.partition(const_shape![BLOCK_SIZE]);

        let bm_tile: Tile<f32, { [BLOCK_SIZE] }> = bm_part.load([0i32]);
        let bi_tile_u32: Tile<u32, { [BLOCK_SIZE] }> = bi_part.load([0i32]);
        let bi_tile_i32: Tile<i32, { [BLOCK_SIZE] }> = bitcast(bi_tile_u32);

        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let n_tile: Tile<i32, { [BLOCK_SIZE] }> = num_blocks.broadcast(const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(offs, n_tile);

        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero_f: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero_f - mask_mag;
        let masked_max: Tile<f32, { [BLOCK_SIZE] }> = select(valid, bm_tile, neg_inf);

        let max_t: Tile<f32, { [1] }> = reduce_max(masked_max, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_t.reshape(const_shape![]));
        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_max, max_bcast);

        let big_idx: Tile<i32, { [BLOCK_SIZE] }> =
            constant(2147483647i32, const_shape![BLOCK_SIZE]);
        let candidates: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, bi_tile_i32, big_idx);
        let winner_t: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner_t.reshape(const_shape![]));

        let out_s: Tile<i32, { [] }> = scalar_to_tile(winner);
        let out_i32: Tile<i32, { [1] }> = out_s.reshape(const_shape![1]);
        let out_u32: Tile<u32, { [1] }> = bitcast(out_i32);
        out.store(out_u32);
    }
}

pub use argmax_reduce_blocks_to_u32_module::argmax_reduce_blocks_to_u32;

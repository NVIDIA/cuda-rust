/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Embedding lookup kernels for token-id batches.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod embedding_batch_f16_module {
    use cutile::core::*;

    /// Batched embedding lookup from token ids. Used at the input of transformer models with output shape [tokens, hidden_size].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn embedding_batch_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let d_block = pid.1;

        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, d_block]);
        out.store(emb);
    }
}

pub use embedding_batch_f16_module::embedding_batch_f16;

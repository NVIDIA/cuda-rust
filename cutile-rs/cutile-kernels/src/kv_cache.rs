/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! KV-cache update kernels for prefill and decode graph execution.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod kv_cache_update_seq_f16_module {
    use cutile::core::*;

    /// Writes a sequence of K/V rows into the KV cache at a host-known offset. Used during prefill for [seq_len, kv_heads, head_dim].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    unsafe fn kv_cache_update_seq_f16<const D: i32, const BLOCK_SIZE: i32, const BM_S: i32>(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, BM_S, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, BM_S, BLOCK_SIZE] }>,
        _position_start: i32, // asserted == 0 at call site; kept for ABI parity
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let s_tile_idx = pid.1;
        let d_block = pid.2;

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]);
        let mut v_cache_part = v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]);

        let s_start: i32 = s_tile_idx * BM_S;
        // Skip trailing CTAs that are entirely beyond seq_len. The
        // per-CTA tile view naturally covers absolute cache positions
        // [s_start, s_start + BM_S) so indexing is local [0, BM_S).
        if s_start < seq_len {
            for s_local in 0i32..BM_S {
                let s_global: i32 = s_start + s_local;
                if s_global < seq_len {
                    let k_tile = new_k_part
                        .load([s_global, head, d_block])
                        .reshape(const_shape![1, 1, BLOCK_SIZE]);
                    let v_tile = new_v_part
                        .load([s_global, head, d_block])
                        .reshape(const_shape![1, 1, BLOCK_SIZE]);
                    // Local index within per-CTA tile; position_start
                    // is assumed 0 (see function docstring).
                    k_cache_part.store(k_tile, [0i32, s_local, 0i32]);
                    v_cache_part.store(v_tile, [0i32, s_local, 0i32]);
                }
            }
        }
    }
}

pub use kv_cache_update_seq_f16_module::kv_cache_update_seq_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod kv_cache_update_seq_dynpos_f16_module {
    use cutile::core::*;

    /// Writes K/V rows into the KV cache using a device position tensor. Used by decode CUDA graphs for single-token updates.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn kv_cache_update_seq_dynpos_f16<
        const D: i32,
        const BLOCK_SIZE: i32,
        const MAX_SEQ: i32,
    >(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position_start: &Tensor<u32, { [1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let pos_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]);
        let mut v_cache_part = v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]);

        for s in 0i32..seq_len {
            let k_tile = new_k_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let v_tile = new_v_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let cache_pos = pos_start + s;
            k_cache_part.store(k_tile, [0i32, cache_pos, 0i32]);
            v_cache_part.store(v_tile, [0i32, cache_pos, 0i32]);
        }
    }
}

pub use kv_cache_update_seq_dynpos_f16_module::kv_cache_update_seq_dynpos_f16;

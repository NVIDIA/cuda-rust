/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Fused attention kernels for causal, grouped-query, prefill, and split-K decode transformer attention.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod flash_attn_causal_seq_f16_module {
    use cutile::core::*;

    /// Causal attention fallback over sequence Q/K/V tensors. Used for transformer attention with [q_len, q_heads, head_dim] and KV cache tensors.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_causal_seq_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }
}

pub use flash_attn_causal_seq_f16_module::flash_attn_causal_seq_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod flash_attn_causal_seq_dynpos_f16_module {
    use cutile::core::*;

    /// Dynamic-position causal attention fallback. Used in decode graph paths where the KV length is read from a device position tensor.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=4, max_divisibility=16,),
                         sm_120 = (occupancy=4, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_causal_seq_dynpos_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let query_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        // Decode graph uses q_len=1, BM=1: kv_len is position+1.
        let kv_len: i32 = query_start + 1i32;

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }
}

pub use flash_attn_causal_seq_dynpos_f16_module::flash_attn_causal_seq_dynpos_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod fmha_prefill_causal_module {
    use cutile::core::*;

    /// Fused causal prefill attention using tensor views and TMA loads. Targets prefill shapes [q_len, q_heads, head_dim] with q_len > 1.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_causal<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const CAUSAL: i32,
        const EVEN_K: i32,
        const LATENCY: i32, // pipeline depth for K/V load_from_view; tune per arch
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, D]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, D]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, D]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // per-CTA [BM, 1, D]
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        // Scale to log2 base: exp2(x * s / log2) = exp(x * s). Scalar since
        // we fuse the multiply into the m_ij subtract inside the loop.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2;
        let qk_scale_tile: Tile<f32, { [BM, BN] }> = qk_scale_log2.broadcast(const_shape![BM, BN]);
        let qk_scale_col: Tile<f32, { [BM, 1] }> = qk_scale_log2.broadcast(const_shape![BM, 1]);

        // Query position offsets for causal mask.
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m_1d: Tile<i32, { [BM] }> =
            offs_m_base.broadcast(const_shape![BM]) + iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m_1d
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        // KV-tile offsets (within a tile).
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let kv_len_tile: Tile<i32, { [BM, BN] }> = kv_len.broadcast(const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> =
            constant(0.0f32, const_shape![BM, BN]) - constant(1.0e30f32, const_shape![BM, BN]);

        // Accumulators (rank 2 to avoid the cutile const-generic unification
        // issues we hit in the split-K kernel).
        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        // Load Q tile (one CTA = BM queries for one head).
        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq_raw: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f16, { [BM, D] }> = tq_raw.reshape(const_shape![BM, D]);

        // Tile iteration bounds (match flash_attn_causal_seq_f16's loop, just
        // hoisted out of the inner body).
        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = k_seqlen_tiles;
        let mut tc: i32 = ceil_div(kv_len, BN);
        if CAUSAL == 1i32 {
            mask_start = (query_start + q_m_idx * BM) / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_len), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..tc {
            // QK^T via a [D, BN]-shape K transpose; accumulator stays f32.
            // Both K and V go through load_from_view with Some(LATENCY):
            // swept on sm_120 and the two-loads-pipelined config is flat
            // across LAT ∈ {0..4} at OCC=2 (~116 ms at pp=2048), while
            // K-plain introduces a cliff at LAT<3 (128 ms regression).
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            qk = mma(tq, k_trans, qk);

            // Causal + OOB mask only on tiles where it can be violated.
            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, const_shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, const_shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                qk = qk + select(mask, mask_true, mask_false);
            }

            // Online softmax in log2 space. Reduce BEFORE scaling; apply scale
            // inside the `qk * scale - m_ij` fused op (TileGym perf note).
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max_col: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let qk_max_scaled: Tile<f32, { [BM, 1] }> = qk_max_col * qk_scale_col;
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max_scaled);
            let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![BM, BN]);
            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);

            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BM, D]);

            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        // Normalize and cast back to f16.
        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_safe: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [BM, D] }> = true_div(acc, l_safe.broadcast(const_shape![BM, D]));
        let out_tile: Tile<f16, { [BM, 1, D] }> =
            convert_tile(acc_norm.reshape(const_shape![BM, 1, D]));
        out.store(out_tile);
    }
}

pub use fmha_prefill_causal_module::fmha_prefill_causal;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod fmha_prefill_gqa_module {
    use cutile::core::*;

    /// Grouped-query prefill attention that packs multiple query heads per CTA. Used by GQA GQA transformer models where query_group_size > 1.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_gqa<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const GROUP: i32,
        const M_EFF: i32, // caller MUST pass BM * GROUP
        const CAUSAL: i32,
        const EVEN_K: i32,
        const LATENCY: i32, // pipeline depth for Q/K/V load_from_view (gemma_attention-style)
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,          // [q_len, q_heads, D]
        k: &Tensor<f16, { [-1, -1, D] }>,          // [kv_heads, kv_len, D]
        v: &Tensor<f16, { [-1, -1, D] }>,          // [kv_heads, kv_len, D]
        out: &mut Tensor<f16, { [BM, GROUP, D] }>, // per-CTA [BM, GROUP, D]
        qk_scale: f32,
        // query_group_size = q_heads / kv_heads (transformer: 32/8=4). GROUP is
        // the packing factor and must divide query_group_size. When GROUP
        // == query_group_size (default), each grid-dim-1 index maps 1:1
        // to a kv_head. For smaller GROUP, multiple grid-1 indices share
        // the same kv_head: kv_head_idx = pid.1 * GROUP / query_group_size.
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let kv_head_idx = pid.1 * GROUP / query_group_size;

        // Scale to log2 base: exp2(x * s / log2) = exp(x * s). Fused into
        // the `qk * scale - m_ij` op below.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2;
        let qk_scale_tile: Tile<f32, { [M_EFF, BN] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, BN]);
        let qk_scale_col: Tile<f32, { [M_EFF, 1] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, 1]);

        // Build offs_m so row r has value query_start + q_m_idx*BM + r/GROUP:
        // iota(BM) reshaped to [BM, 1] and broadcast to [BM, GROUP], then
        // reshaped to [M_EFF, 1] yields [0,…,0, 1,…,1, …, BM-1,…] in
        // row-major order — exactly r/GROUP.
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let iota_bm: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let iota_bm_col: Tile<i32, { [BM, 1] }> = iota_bm.reshape(const_shape![BM, 1]);
        let iota_bm_grp: Tile<i32, { [BM, GROUP] }> =
            iota_bm_col.broadcast(const_shape![BM, GROUP]);
        let base_bg: Tile<i32, { [BM, GROUP] }> = offs_m_base.broadcast(const_shape![BM, GROUP]);
        let offs_m_bg: Tile<i32, { [BM, GROUP] }> = base_bg + iota_bm_grp;
        let offs_m_col: Tile<i32, { [M_EFF, 1] }> = offs_m_bg.reshape(const_shape![M_EFF, 1]);
        let offs_m: Tile<i32, { [M_EFF, BN] }> = offs_m_col.broadcast(const_shape![M_EFF, BN]);

        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [M_EFF, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![M_EFF, BN]);
        let kv_len_tile: Tile<i32, { [M_EFF, BN] }> = kv_len.broadcast(const_shape![M_EFF, BN]);
        let mask_false: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN])
            - constant(1.0e30f32, const_shape![M_EFF, BN]);

        // Rank-2 accumulators (match fmha_prefill_causal / decode-split
        // convention to dodge cutile const-generic unification issues).
        let max_mag: Tile<f32, { [M_EFF, 1] }> = constant(1.0e30f32, const_shape![M_EFF, 1]);
        let mut m_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]) - max_mag;
        let mut l_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]);
        let mut acc: Tile<f32, { [M_EFF, D] }> = constant(0.0f32, const_shape![M_EFF, D]);

        // Load Q tile once: [BM, GROUP, D] → [M_EFF, D]. Pipelined via
        // load_from_view with Some(LATENCY) — mirrors gemma_attention's
        // reference pattern which hints Q, K, V uniformly.
        let q_part: Partition<f16, { [BM, GROUP, D] }> = q.partition(const_shape![BM, GROUP, D]);
        let tq_raw: Tile<f16, { [BM, GROUP, D] }> = load_view_tko(
            &q_part,
            [q_m_idx, kv_head_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let tq: Tile<f16, { [M_EFF, D] }> = tq_raw.reshape(const_shape![M_EFF, D]);

        // Tile iteration bounds. All GROUP queries at a given m share the
        // same q_pos, so the max q_pos in this CTA is still
        // (query_start + (q_m_idx+1)*BM - 1) — same as the non-grouped
        // kernel; group index doesn't affect the KV upper bound.
        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = k_seqlen_tiles;
        let mut tc: i32 = ceil_div(kv_len, BN);
        if CAUSAL == 1i32 {
            mask_start = (query_start + q_m_idx * BM) / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_len), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..tc {
            // ONE K load per iteration, reused across all GROUP queries.
            // Pipelined via load_from_view with Some(LATENCY) (reference
            // gemma_attention hints all of Q/K/V uniformly).
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
            qk = mma(tq, k_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [M_EFF, BN] }> =
                    broadcast_scalar(j * BN, const_shape![M_EFF, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [M_EFF, BN] }> = constant(true, const_shape![M_EFF, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [M_EFF, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [M_EFF, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                let mask_true: Tile<f32, { [M_EFF, BN] }> =
                    constant(0.0f32, const_shape![M_EFF, BN]);
                qk = qk + select(mask, mask_true, mask_false);
            }

            // Online softmax in log2 space (rowwise over M_EFF).
            let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
            let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
            let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
            let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
            let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
            let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

            let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
            let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

            // ONE V load per iteration, reused across all GROUP queries.
            // Pipelined via load_from_view with Some(LATENCY).
            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        // Normalize and reshape acc [M_EFF, D] → [BM, GROUP, D] for store.
        let eps: Tile<f32, { [M_EFF, 1] }> = constant(1.0e-8f32, const_shape![M_EFF, 1]);
        let l_safe: Tile<f32, { [M_EFF, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [M_EFF, D] }> =
            true_div(acc, l_safe.broadcast(const_shape![M_EFF, D]));
        let out_tile: Tile<f16, { [BM, GROUP, D] }> =
            convert_tile(acc_norm.reshape(const_shape![BM, GROUP, D]));
        out.store(out_tile);
    }
}

pub use fmha_prefill_gqa_module::fmha_prefill_gqa;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod fmha_causal_module {
    use cutile::core::*;

    /// Dynamic-position fused causal attention for decode. Used with [1, q_heads, head_dim] query tensors and KV caches.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn fmha_causal<
        const BM: i32, // Query sequence tile size.
        const BN: i32, // KV sequence tile size.
        const D: i32,  // Head dimension.
        const CAUSAL: i32,
        const EVEN_K: i32,
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,      // (m, h, d)
        k: &Tensor<f16, { [-1, -1, D] }>,      // (hkv, n, d)
        v: &Tensor<f16, { [-1, -1, D] }>,      // (hkv, n, d)
        out: &mut Tensor<f16, { [BM, 1, D] }>, // (m, b*h, d)
        qk_scale: f16,
        query_group_size: i32,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let input_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_f32: f32 = convert_scalar(qk_scale);
        let qk_scale: Tile<f32, { [BM, BN] }> =
            broadcast_scalar(qk_scale_f32 / log2, const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let k_seqlen: i32 = get_shape_dim(k.shape(), 1i32);
        let m_end: i32 = input_pos + (q_m_idx + 1i32) * BM;
        let mut mask_start: i32 = k_seqlen / BN;
        let mut tc: i32 = ceil_div(k_seqlen, BN);
        if CAUSAL == 1i32 {
            mask_start = (input_pos + q_m_idx * BM) / BN;
            let k_seqlen_tiles = k_seqlen / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, k_seqlen), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);

        let offs_m_iota: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m_iota = offs_m_iota.reshape(const_shape![BM, 1]);
        let offs_m: Tile<i32, { [BM, 1] }> =
            broadcast_scalar(q_m_idx * BM + input_pos, const_shape![BM, 1]) + offs_m_iota;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m.broadcast(const_shape![BM, BN]);
        let k_seqlen_tile: Tile<i32, { [BM, BN] }> = k_seqlen.broadcast(const_shape![BM, BN]);
        let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> =
            constant(0.0f32, const_shape![BM, BN]) - constant(1.0e30f32, const_shape![BM, BN]);

        for j in 0i32..tc {
            let k_tile: Tile<f16, { [BN, D] }> = k_part
                .load([kv_head_idx, j, 0i32])
                .reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            qk = mma(tq, k_tile_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, const_shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, const_shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, k_seqlen_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                qk = qk + select(mask, mask_true, mask_false);
            }

            qk = qk * qk_scale;
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BM, D]);

            let v_tile: Tile<f16, { [BN, D] }> = v_part
                .load([kv_head_idx, j, 0i32])
                .reshape(const_shape![BN, D]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }
}

pub use fmha_causal_module::fmha_causal;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod fmha_decode_gqa_split_module {
    use cutile::core::*;

    /// Split-K grouped-query decode attention partial kernel. Produces per-split attention/LSE scratch for GQA decode.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn fmha_decode_gqa_split<
        const GROUP: i32,
        const BN: i32,
        const D: i32,
        const NUM_KV_SPLITS: i32,
        const LATENCY: i32, // pipeline depth for K/V load_from_view; tune per arch
    >(
        // Whole-tensor views (K/V partitioned internally by tile-index j).
        q: &Tensor<f16, { [-1, GROUP, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        // Per-CTA output tiles. The scratch tensors are
        //   att: [kv_heads, NUM_KV_SPLITS * GROUP, D]  — each CTA gets [1, GROUP, D]
        //   lse: [kv_heads, NUM_KV_SPLITS * GROUP]     — each CTA gets [1, GROUP]
        att_out: &mut Tensor<f16, { [1, GROUP, D] }>,
        lse_out: &mut Tensor<f32, { [1, GROUP] }>,
        qk_scale: f16,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let kv_head_id = pid.0;
        let split_id = pid.1;

        // s_kv = position_start + 1 (number of valid KV tokens at this step).
        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let input_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));
        let s_kv: i32 = input_pos + 1i32;

        // qk_scale is passed in natural-log scale (1/sqrt(d)); we convert to
        // log2 scale once so the inner loop can use exp2 directly.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let ln2: f32 = tile_to_scalar(log(two));
        let qk_scale_f32: f32 = convert_scalar(qk_scale);
        let qk_scale_log2: Tile<f32, { [BN, GROUP] }> =
            broadcast_scalar(qk_scale_f32 / ln2, const_shape![BN, GROUP]);

        // Split range over KV tiles (in units of BN tokens).
        let k_seqlen_tiles: i32 = ceil_div(s_kv, BN);
        let tiles_per_split: i32 = ceil_div(k_seqlen_tiles, NUM_KV_SPLITS);
        let start_tile: i32 = split_id * tiles_per_split;
        let mut end_tile: i32 = start_tile + tiles_per_split;
        end_tile = min(end_tile, k_seqlen_tiles);

        // Accumulators. m_i is kept rank-2 [GROUP, 1] (to match the rank-2
        // shape cutile's reduce_max produces after reshape, which avoids
        // constant-fold mismatches between `Tile<…{[GROUP]}>` and
        // `Tile<…{[4]}>` in max_tile).
        let neg_inf: Tile<f32, { [GROUP, 1] }> =
            constant(0.0f32, const_shape![GROUP, 1]) - constant(1.0e30f32, const_shape![GROUP, 1]);
        let mut m_i: Tile<f32, { [GROUP, 1] }> = neg_inf;
        let mut l_i: Tile<f32, { [BN, GROUP] }> = constant(1.0f32, const_shape![BN, GROUP]);
        let mut acc: Tile<f32, { [D, GROUP] }> = constant(0.0f32, const_shape![D, GROUP]);

        // Load Q once: [1, GROUP, D] → [GROUP, D] → [D, GROUP] (transposed).
        let q_part: Partition<f16, { [1, GROUP, D] }> = q.partition(const_shape![1, GROUP, D]);
        let q_tile: Tile<f16, { [1, GROUP, D] }> = q_part.load([kv_head_id, 0i32, 0i32]);
        let q_tile: Tile<f16, { [GROUP, D] }> = q_tile.reshape(const_shape![GROUP, D]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let q_trans: Tile<f16, { [D, GROUP] }> = permute(q_tile, transpose_2d);

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_col: Tile<i32, { [BN, 1] }> = offs_n_tile.reshape(const_shape![BN, 1]);
        let offs_n_2d: Tile<i32, { [BN, GROUP] }> = offs_n_col.broadcast(const_shape![BN, GROUP]);

        let s_kv_tile: Tile<i32, { [BN, GROUP] }> = s_kv.broadcast(const_shape![BN, GROUP]);
        let mask_true: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP]);
        let mask_false: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP])
            - constant(1.0e30f32, const_shape![BN, GROUP]);

        for j in start_tile..end_tile {
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_id, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);

            // qk = k @ q_T → [BN, GROUP]
            let mut qk: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP]);
            qk = mma(k_tile, q_trans, qk);

            // Mask out-of-range KV positions (only matters at the last tile).
            if j == k_seqlen_tiles - 1i32 {
                let j_base: Tile<i32, { [BN, GROUP] }> =
                    broadcast_scalar(j * BN, const_shape![BN, GROUP]);
                let kv_pos: Tile<i32, { [BN, GROUP] }> = j_base + offs_n_2d;
                let valid: Tile<bool, { [BN, GROUP] }> = lt_tile(kv_pos, s_kv_tile);
                qk = qk + select(valid, mask_true, mask_false);
            }

            // Convert to log2 scale. Transpose qk to [GROUP, BN] so we can
            // reduce along the last axis (cutile's reduce_max only cleanly
            // supports axis=last in the existing grout/cutile examples).
            qk = qk * qk_scale_log2;
            let qk_t: Tile<f32, { [GROUP, BN] }> = permute(qk, transpose_2d);
            let qk_max_raw: Tile<f32, { [GROUP] }> = reduce_max(qk_t, 1i32);
            let qk_max_col: Tile<f32, { [GROUP, 1] }> = qk_max_raw.reshape(const_shape![GROUP, 1]);
            let m_ij: Tile<f32, { [GROUP, 1] }> = max_tile(m_i, qk_max_col);
            let qk_shifted: Tile<f32, { [GROUP, BN] }> =
                qk_t - m_ij.broadcast(const_shape![GROUP, BN]);
            let p_t: Tile<f32, { [GROUP, BN] }> = exp2(qk_shifted, ftz::Disabled);
            // Transpose p back to [BN, GROUP] for the V-mma below.
            let p: Tile<f32, { [BN, GROUP] }> = permute(p_t, transpose_2d);

            let alpha: Tile<f32, { [GROUP, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            let alpha_row: Tile<f32, { [1, GROUP] }> = alpha.reshape(const_shape![1, GROUP]);
            // Rescale l_i by alpha, accumulate p.
            l_i = l_i * alpha_row.broadcast(const_shape![BN, GROUP]) + p;
            // Rescale acc by alpha.
            acc = acc * alpha_row.broadcast(const_shape![D, GROUP]);

            // V tile: load [BN, D], transpose to [D, BN] for MMA.
            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_id, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            let v_trans: Tile<f16, { [D, BN] }> = permute(v_tile, transpose_2d);

            // acc[D, GROUP] += v_T[D, BN] @ p[BN, GROUP]
            // p is f32; cast to f16 to match v_trans dtype for mma.
            let p_f16: Tile<f16, { [BN, GROUP] }> = convert_tile(p);
            acc = mma(v_trans, p_f16, acc);
            m_i = m_ij;
        }

        // Finalize this split: normalize acc by sum(l_i across BN) and emit
        // LSE = m_i + log2(l_sum) for the merge. Transpose first so we
        // reduce along the last axis (cutile pattern). Keep shapes rank-2
        // so subsequent max_tile etc. see matching symbolic shapes.
        let l_i_t: Tile<f32, { [GROUP, BN] }> = permute(l_i, transpose_2d);
        let l_sum_raw: Tile<f32, { [GROUP] }> = reduce_sum(l_i_t, 1i32);
        let l_sum: Tile<f32, { [GROUP, 1] }> = l_sum_raw.reshape(const_shape![GROUP, 1]);
        let eps_g: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let l_sum_safe: Tile<f32, { [GROUP, 1] }> = max_tile(l_sum, eps_g);
        let l_row: Tile<f32, { [1, GROUP] }> = l_sum_safe.reshape(const_shape![1, GROUP]);
        let acc_norm: Tile<f32, { [D, GROUP] }> =
            true_div(acc, l_row.broadcast(const_shape![D, GROUP]));

        // Transpose acc back to [GROUP, D] and store this CTA's per-split tile.
        let acc_out_t: Tile<f32, { [GROUP, D] }> = permute(acc_norm, transpose_2d);
        let acc_out_f16: Tile<f16, { [GROUP, D] }> = convert_tile(acc_out_t);
        let acc_out_3d: Tile<f16, { [1, GROUP, D] }> =
            acc_out_f16.reshape(const_shape![1, GROUP, D]);
        att_out.store(acc_out_3d);

        // LSE in log2 base: m_i + log2(l_sum). Both rank-2 [GROUP, 1].
        let lse_col: Tile<f32, { [GROUP, 1] }> = m_i + log2(l_sum_safe);
        let lse_out_tile: Tile<f32, { [1, GROUP] }> = lse_col.reshape(const_shape![1, GROUP]);
        lse_out.store(lse_out_tile);
    }
}

pub use fmha_decode_gqa_split_module::fmha_decode_gqa_split;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod splitk_reduce_merge_module {
    use cutile::core::*;

    /// Merges split-K decode attention partials. Combines per-split attention and LSE scratch into [kv_heads, group, head_dim] output chunks.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=4, max_divisibility=16,),
                         sm_120 = (occupancy=4, max_divisibility=16,),
                       ))]
    unsafe fn splitk_reduce_merge<
        const GROUP: i32,
        const D: i32,
        const CHUNK_D: i32, // per-CTA D chunk; grid dim 2 = D / CHUNK_D
        const NUM_KV_SPLITS: i32,
        const NS_GROUP: i32, // NUM_KV_SPLITS * GROUP, passed explicitly
        const LATENCY: i32,  // pipeline depth for input load_from_view
    >(
        // Scratch tensors from the split pass, with split and group flattened
        // into a single dim:
        //   att_partial: [kv_heads, NS_GROUP, D]   — per-CTA: [1, NS_GROUP, CHUNK_D]
        //   lse_partial: [kv_heads, NS_GROUP]      — per-CTA: [1, NS_GROUP]
        //   out:         [kv_heads, GROUP, D]      — per-CTA: [1, GROUP, CHUNK_D]
        //
        // Grid = (kv_heads, 1, D/CHUNK_D). Each CTA produces one [GROUP,
        // CHUNK_D] output slice. Splitting D across CTAs expands the grid
        // from (kv_heads,) = 8 CTAs to (kv_heads × D/CHUNK_D) = up to 64+
        // CTAs, closing the SM-undersub gap on 64-SM Blackwell. LSE and
        // weights are recomputed per-CTA (trivially cheap: GROUP ×
        // NUM_KV_SPLITS = ~32 ops) vs. sharing across CTAs.
        att_partial: &Tensor<f16, { [-1, NS_GROUP, D] }>,
        lse_partial: &Tensor<f32, { [-1, NS_GROUP] }>,
        out: &mut Tensor<f16, { [1, GROUP, CHUNK_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let kv_head_id = pid.0;
        let d_chunk_id = pid.2;

        // Load this CTA's [1, NS_GROUP] LSE tile and reshape to [NUM_KV_SPLITS, GROUP].
        let lse_part: Partition<f32, { [1, NS_GROUP] }> =
            lse_partial.partition(const_shape![1, NS_GROUP]);
        let lse_tile: Tile<f32, { [1, NS_GROUP] }> = load_view_tko(
            &lse_part,
            [kv_head_id, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        // Layout: split-major within NS_GROUP (split * GROUP + g), so reshape
        // to [NUM_KV_SPLITS, GROUP] then transpose → [GROUP, NUM_KV_SPLITS]
        // to match downstream accumulation.
        let lse_ns_g: Tile<f32, { [NUM_KV_SPLITS, GROUP] }> =
            lse_tile.reshape(const_shape![NUM_KV_SPLITS, GROUP]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let lse_tile: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = permute(lse_ns_g, transpose_2d);

        // Compute per-split weight w_s normalized across splits.
        let lse_max: Tile<f32, { [GROUP] }> = reduce_max(lse_tile, 1i32);
        let lse_max_col: Tile<f32, { [GROUP, 1] }> = lse_max.reshape(const_shape![GROUP, 1]);
        let lse_shifted: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> =
            lse_tile - lse_max_col.broadcast(const_shape![GROUP, NUM_KV_SPLITS]);
        let scale_raw: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = exp2(lse_shifted, ftz::Disabled);
        let scale_sum: Tile<f32, { [GROUP] }> = reduce_sum(scale_raw, 1i32);
        let scale_sum_col: Tile<f32, { [GROUP, 1] }> = scale_sum.reshape(const_shape![GROUP, 1]);
        let eps: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let scale_sum_safe: Tile<f32, { [GROUP, 1] }> = max_tile(scale_sum_col, eps);
        let weights: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = true_div(
            scale_raw,
            scale_sum_safe.broadcast(const_shape![GROUP, NUM_KV_SPLITS]),
        );

        // Load this CTA's CHUNK_D slice of [1, NS_GROUP, CHUNK_D] and
        // reshape to [NUM_KV_SPLITS, GROUP, CHUNK_D], then transpose
        // first two dims to get [GROUP, NUM_KV_SPLITS, CHUNK_D].
        let att_part: Partition<f16, { [1, NS_GROUP, CHUNK_D] }> =
            att_partial.partition(const_shape![1, NS_GROUP, CHUNK_D]);
        let att_tile: Tile<f16, { [1, NS_GROUP, CHUNK_D] }> = load_view_tko(
            &att_part,
            [kv_head_id, 0i32, d_chunk_id],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let att_ns_g_d: Tile<f16, { [NUM_KV_SPLITS, GROUP, CHUNK_D] }> =
            att_tile.reshape(const_shape![NUM_KV_SPLITS, GROUP, CHUNK_D]);
        let transpose_3d_01: Array<{ [1, 0, 2] }> = Array::<{ [1, 0, 2] }> {
            dims: &[1i32, 0i32, 2i32],
        };
        let att_g_ns_d: Tile<f16, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> =
            permute(att_ns_g_d, transpose_3d_01);
        let att_tile: Tile<f32, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> = convert_tile(att_g_ns_d);

        // Broadcast weights to match att_tile dims.
        let w_3d: Tile<f32, { [GROUP, NUM_KV_SPLITS, 1] }> =
            weights.reshape(const_shape![GROUP, NUM_KV_SPLITS, 1]);
        let weighted: Tile<f32, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> =
            att_tile * w_3d.broadcast(const_shape![GROUP, NUM_KV_SPLITS, CHUNK_D]);
        let out_tile: Tile<f32, { [GROUP, CHUNK_D] }> = reduce_sum(weighted, 1i32);

        let out_f16: Tile<f16, { [GROUP, CHUNK_D] }> = convert_tile(out_tile);
        let out_3d: Tile<f16, { [1, GROUP, CHUNK_D] }> =
            out_f16.reshape(const_shape![1, GROUP, CHUNK_D]);
        out.store(out_3d);
    }
}

pub use splitk_reduce_merge_module::splitk_reduce_merge;

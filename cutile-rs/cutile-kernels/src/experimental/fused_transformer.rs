/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Experimental raw-pointer fusions that combine normalization, RoPE, and KV-cache writes.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod qk_norm_rope_kv_prefill_raw_f16_module {
    use cutile::core::*;

    /// Raw-pointer fusion of Q/K norm, RoPE, and KV-cache update for prefill. Targets transformer attention projections shaped [seq_len, heads, head_dim].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_norm_rope_kv_prefill_raw_f16<
        const D: i32,
        const HALF_D: i32,
        const MAX_SEQ: i32,
    >(
        q_ptr: *mut f16,
        k_ptr: *mut f16,
        v_ptr: *mut f16,
        q_weight_ptr: *mut f16,
        k_weight_ptr: *mut f16,
        inv_freq_ptr: *mut f32,
        q_out_ptr: *mut f16,
        k_cache_ptr: *mut f16,
        v_cache_ptr: *mut f16,
        eps: f32,
        position_start: i32,
        seq_len: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
    ) {
        let half_shape: Shape<{ [1, 1, HALF_D] }> = const_shape![1, 1, HALF_D];
        let half_shape_2d: Shape<{ [1, HALF_D] }> = const_shape![1, HALF_D];

        let q_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_ptr) };
        let k_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_ptr) };
        let v_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_ptr) };
        let q_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_weight_ptr) };
        let k_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_weight_ptr) };
        let q_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_out_ptr) };
        let k_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_cache_ptr) };
        let v_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_cache_ptr) };
        let seq_len: i32 = unsafe { assume_bounds_lower::<_, 0>(seq_len) };
        let num_q_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_q_heads) };
        let num_kv_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_kv_heads) };

        let tok: Token = new_token_unordered();

        let q_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[seq_len, num_q_heads],
        };
        let q_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[num_q_heads * D, D],
        };
        let q_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_ptr), q_shape, q_strides, tok) };
        let q_out_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_out_ptr), q_shape, q_strides, tok) };

        let kv_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[seq_len, num_kv_heads],
        };
        let kv_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[num_kv_heads * D, D],
        };
        let k_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_ptr), kv_shape, kv_strides, tok) };
        let v_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(v_ptr), kv_shape, kv_strides, tok) };

        let cache_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[num_kv_heads, MAX_SEQ],
        };
        let cache_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[MAX_SEQ * D, D],
        };
        let k_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(k_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let v_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(v_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };

        let w_shape: Shape<{ [D] }> = const_shape![D];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let q_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_weight_ptr), w_shape, w_strides, tok) };
        let k_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_weight_ptr), w_shape, w_strides, tok) };
        let inv_shape: Shape<{ [HALF_D] }> = const_shape![HALF_D];
        let inv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let inv_freq_tv: Tensor<f32, { [HALF_D] }> =
            unsafe { make_tensor_view(pointer_to_tile(inv_freq_ptr), inv_shape, inv_strides, tok) };

        let q_part: Partition<f16, { [1, 1, HALF_D] }> =
            q_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let k_part: Partition<f16, { [1, 1, HALF_D] }> =
            k_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let v_part: Partition<f16, { [1, 1, HALF_D] }> =
            v_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let q_weight_part: Partition<f16, { [HALF_D] }> =
            q_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let k_weight_part: Partition<f16, { [HALF_D] }> =
            k_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let inv_part: Partition<f32, { [HALF_D] }> =
            inv_freq_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let mut q_out_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { q_out_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut k_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { k_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut v_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { v_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };

        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };

        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16.reshape(half_shape_2d));
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16.reshape(half_shape_2d));

        let rms_vec: Tile<f32, { [1, HALF_D] }> = x_lo * x_lo + x_hi * x_hi;
        let rms: Tile<f32, { [1] }> = reduce_sum(rms_vec, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(D);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, HALF_D] }> = inv_rms.broadcast(half_shape_2d);

        let w_lo_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_hi_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(w_lo_f16.reshape(half_shape_2d));
        let w_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(w_hi_f16.reshape(half_shape_2d));

        let norm_lo: Tile<f32, { [1, HALF_D] }> = x_lo * inv_rms * w_lo;
        let norm_hi: Tile<f32, { [1, HALF_D] }> = x_hi * inv_rms * w_hi;

        let freq: Tile<f32, { [HALF_D] }> = load_view_tko(
            &inv_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let pos_i: i32 = position_start + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [1, HALF_D] }> = (pos * freq).reshape(half_shape_2d);
        let cos_t: Tile<f32, { [1, HALF_D] }> = cos(theta);
        let sin_t: Tile<f32, { [1, HALF_D] }> = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = norm_lo * cos_t - norm_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = norm_hi * cos_t + norm_lo * sin_t;
        let y_lo_f16_2d: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16_2d: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = y_lo_f16_2d.reshape(half_shape);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = y_hi_f16_2d.reshape(half_shape);

        if is_q {
            unsafe {
                store_view_tko_mut(
                    &mut q_out_part,
                    y_lo_f16,
                    [seq_idx, local_head, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut q_out_part,
                    y_hi_f16,
                    [seq_idx, local_head, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        } else {
            let v_lo: Tile<f16, { [1, 1, HALF_D] }> = load_view_tko(
                &v_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let v_hi: Tile<f16, { [1, 1, HALF_D] }> = load_view_tko(
                &v_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let cache_pos: i32 = position_start + seq_idx;
            unsafe {
                store_view_tko_mut(
                    &mut k_cache_part,
                    y_lo_f16,
                    [local_head, cache_pos, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut k_cache_part,
                    y_hi_f16,
                    [local_head, cache_pos, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_lo,
                    [local_head, cache_pos, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_hi,
                    [local_head, cache_pos, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }
}

pub use qk_norm_rope_kv_prefill_raw_f16_module::qk_norm_rope_kv_prefill_raw_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod qk_norm_rope_kv_decode_raw_f16_module {
    use cutile::core::*;

    /// Raw-pointer fusion of decode Q/K norm, RoPE, and KV-cache update. Targets single-token QKV projections and graph replay.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_norm_rope_kv_decode_raw_f16<
        const D: i32,
        const HALF_D: i32,
        const MAX_SEQ: i32,
    >(
        qkv_ptr: *mut f16,
        q_weight_ptr: *mut f16,
        k_weight_ptr: *mut f16,
        inv_freq_ptr: *mut f32,
        q_out_ptr: *mut f16,
        k_cache_ptr: *mut f16,
        v_cache_ptr: *mut f16,
        position_start: &Tensor<u32, { [1] }>,
        eps: f32,
        num_q_heads: i32,
        num_kv_heads: i32,
    ) {
        let half_shape_2d: Shape<{ [1, HALF_D] }> = const_shape![1, HALF_D];

        let qkv_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(qkv_ptr) };
        let q_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_weight_ptr) };
        let k_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_weight_ptr) };
        let q_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_out_ptr) };
        let k_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_cache_ptr) };
        let v_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_cache_ptr) };
        let num_q_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_q_heads) };
        let num_kv_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_kv_heads) };
        let total_heads: i32 = num_q_heads + num_kv_heads;
        let qkv_elems: i32 = (num_q_heads + 2i32 * num_kv_heads) * D;

        let tok: Token = new_token_unordered();

        let qkv_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[qkv_elems] };
        let qkv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let qkv_tv: Tensor<f16, { [-1] }> =
            unsafe { make_tensor_view(pointer_to_tile(qkv_ptr), qkv_shape, qkv_strides, tok) };
        let qkv_part: Partition<f16, { [HALF_D] }> =
            qkv_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let q_out_shape: Shape<{ [-1, D] }> = Shape::<{ [-1, D] }> {
            dims: &[total_heads],
        };
        let q_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[D] };
        let q_out_tv: Tensor<f16, { [-1, D] }> = unsafe {
            make_tensor_view(pointer_to_tile(q_out_ptr), q_out_shape, q_out_strides, tok)
        };
        let mut q_out_part: PartitionMut<f16, { [1, HALF_D] }> =
            unsafe { q_out_tv.partition_full_mut(const_shape![1, HALF_D]) };

        let cache_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[num_kv_heads, MAX_SEQ],
        };
        let cache_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[MAX_SEQ * D, D],
        };
        let k_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(k_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let v_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(v_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let mut k_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { k_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut v_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { v_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };

        let w_shape: Shape<{ [D] }> = const_shape![D];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let q_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_weight_ptr), w_shape, w_strides, tok) };
        let k_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_weight_ptr), w_shape, w_strides, tok) };
        let q_weight_part: Partition<f16, { [HALF_D] }> =
            q_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let k_weight_part: Partition<f16, { [HALF_D] }> =
            k_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let inv_shape: Shape<{ [HALF_D] }> = const_shape![HALF_D];
        let inv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let inv_freq_tv: Tensor<f32, { [HALF_D] }> =
            unsafe { make_tensor_view(pointer_to_tile(inv_freq_ptr), inv_shape, inv_strides, tok) };
        let inv_part: Partition<f32, { [HALF_D] }> =
            inv_freq_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let pid: (i32, i32, i32) = get_tile_block_id();
        let head_idx = pid.0;
        let half_idx = pid.1;
        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };
        let q_base_block: i32 = local_head * 2i32;
        let k_base_block: i32 = num_q_heads * 2i32 + local_head * 2i32;
        let v_base_block: i32 = num_q_heads * 2i32 + num_kv_heads * 2i32 + local_head * 2i32;
        let x_base_block: i32 = if is_q { q_base_block } else { k_base_block };

        let x_lo_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
            &qkv_part,
            [x_base_block],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let x_hi_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
            &qkv_part,
            [x_base_block + 1i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16.reshape(half_shape_2d));
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16.reshape(half_shape_2d));

        let rms_vec: Tile<f32, { [1, HALF_D] }> = x_lo * x_lo + x_hi * x_hi;
        let rms: Tile<f32, { [1] }> = reduce_sum(rms_vec, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(D);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, HALF_D] }> = inv_rms.broadcast(half_shape_2d);

        let w_lo_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_hi_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(w_lo_f16.reshape(half_shape_2d));
        let w_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(w_hi_f16.reshape(half_shape_2d));
        let norm_lo: Tile<f32, { [1, HALF_D] }> = x_lo * inv_rms * w_lo;
        let norm_hi: Tile<f32, { [1, HALF_D] }> = x_hi * inv_rms * w_hi;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let cache_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let freq: Tile<f32, { [HALF_D] }> = load_view_tko(
            &inv_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let pos: f32 = convert_scalar(cache_pos);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [1, HALF_D] }> = (pos * freq).reshape(half_shape_2d);
        let cos_t: Tile<f32, { [1, HALF_D] }> = cos(theta);
        let sin_t: Tile<f32, { [1, HALF_D] }> = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = norm_lo * cos_t - norm_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = norm_hi * cos_t + norm_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);

        if is_q {
            if half_idx == 0i32 {
                unsafe {
                    store_view_tko_mut(
                        &mut q_out_part,
                        y_lo_f16,
                        [local_head, 0i32],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(1i32),
                        tma::Disabled,
                    );
                }
            } else {
                unsafe {
                    store_view_tko_mut(
                        &mut q_out_part,
                        y_hi_f16,
                        [local_head, 1i32],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(1i32),
                        tma::Disabled,
                    );
                }
            }
        } else {
            let v_half_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
                &qkv_part,
                [v_base_block + half_idx],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let v_half: Tile<f16, { [1, 1, HALF_D] }> =
                v_half_f16.reshape(const_shape![1, 1, HALF_D]);
            let k_half: Tile<f16, { [1, 1, HALF_D] }> = if half_idx == 0i32 {
                y_lo_f16.reshape(const_shape![1, 1, HALF_D])
            } else {
                y_hi_f16.reshape(const_shape![1, 1, HALF_D])
            };
            unsafe {
                store_view_tko_mut(
                    &mut k_cache_part,
                    k_half,
                    [local_head, cache_pos, half_idx],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_half,
                    [local_head, cache_pos, half_idx],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }
}

pub use qk_norm_rope_kv_decode_raw_f16_module::qk_norm_rope_kv_decode_raw_f16;

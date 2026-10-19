/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Rotary position embedding kernels for sequence and decode graph paths.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rope_seq_f16_module {
    use cutile::core::*;

    /// Applies rotary position embedding with a host-provided position start. Used for Q/K tensors shaped [seq_len, heads, head_dim].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // transformer text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = position_start + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }
}

pub use rope_seq_f16_module::rope_seq_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rope_seq_dynpos_f16_module {
    use cutile::core::*;

    /// Applies rotary position embedding with device-provided position. Used in decode CUDA graphs for [1, heads, head_dim] Q/K tensors.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_dynpos_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position_start: &Tensor<u32, { [1] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // transformer text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let pos_part = position_start.partition(const_shape![1]);
        let base_pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let base_pos_t: Tile<i32, { [1] }> = bitcast(base_pos_t_u32);
        let base_pos: i32 = tile_to_scalar(base_pos_t.reshape(const_shape![]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = base_pos + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }
}

pub use rope_seq_dynpos_f16_module::rope_seq_dynpos_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod qk_rope_dynpos_f16_module {
    use cutile::core::*;

    /// Applies RoPE to Q/K tensors using a device position tensor. Used in decode graph paths over [seq_len, heads, head_dim].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_rope_dynpos_f16<const D: i32, const HALF_D: i32, const LATENCY: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position_start: &Tensor<u32, { [1] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        num_q_heads: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };

        // Load input from Q or K based on head index. Both halves go through
        // load_from_view with Some(LATENCY) so the compiler can pipeline the
        // two cp_async issues with the constant-table loads + cos/sin compute.
        let q_part: Partition<f16, { [1, 1, HALF_D] }> = q.partition(const_shape![1, 1, HALF_D]);
        let k_part: Partition<f16, { [1, 1, HALF_D] }> = k.partition(const_shape![1, 1, HALF_D]);

        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        };
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        };
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        // Position and frequency
        let pos_part = position_start.partition(const_shape![1]);
        let base_pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let base_pos_t: Tile<i32, { [1] }> = bitcast(base_pos_t_u32);
        let base_pos: i32 = tile_to_scalar(base_pos_t.reshape(const_shape![]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = base_pos + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        // Apply rotation
        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }
}

pub use qk_rope_dynpos_f16_module::qk_rope_dynpos_f16;

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Pointwise and row-wise utility kernels used around transformer blocks.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod add_2d_f16_module {
    use cutile::core::*;

    /// Adds two rank-2 f16 tensors tile-by-tile. Used for residual/pointwise adds over [tokens, hidden] or [tokens, projection] activations.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn add_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        lhs: &Tensor<f16, { [-1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let lhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            lhs.partition(const_shape![1, BLOCK_SIZE]);
        let rhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            rhs.partition(const_shape![1, BLOCK_SIZE]);
        let lhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = lhs_part.load([row, col]);
        let rhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = rhs_part.load([row, col]);
        out.store(lhs_tile + rhs_tile);
    }
}

pub use add_2d_f16_module::add_2d_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod silu_mul_2d_f16_module {
    use cutile::core::*;

    /// Fuses SiLU(gate) * up for SwiGLU-style MLPs. Typically used in SwiGLU feed-forward blocks with shape [tokens, intermediate_size].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn silu_mul_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        gate: &Tensor<f16, { [-1, -1] }>,
        up: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let gate_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            gate.partition(const_shape![1, BLOCK_SIZE]);
        let up_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            up.partition(const_shape![1, BLOCK_SIZE]);
        let gate_f16: Tile<f16, { [1, BLOCK_SIZE] }> = gate_part.load([row, col]);
        let up_f16: Tile<f16, { [1, BLOCK_SIZE] }> = up_part.load([row, col]);
        let gate_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(gate_f16);
        let up_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(up_f16);
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, const_shape![1, BLOCK_SIZE]);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, const_shape![1, BLOCK_SIZE]);
        let neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = exp(neg_gate);
        let denom: Tile<f32, { [1, BLOCK_SIZE] }> = one + exp_neg_gate;
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, denom);
        let y: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(y);
        out.store(y_f16);
    }
}

pub use silu_mul_2d_f16_module::silu_mul_2d_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod gather_row_f16_module {
    use cutile::core::*;

    /// Copies one row from a rank-2 f16 tensor. Used for row slicing/gathering activations with shape [rows, width].
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn gather_row_f16<const BLOCK_SIZE: i32>(
        src: &Tensor<f16, { [-1, -1] }>,
        out: &mut Tensor<f16, { [BLOCK_SIZE] }>,
        row_idx: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;
        let src_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            src.partition(const_shape![1, BLOCK_SIZE]);
        let tile: Tile<f16, { [1, BLOCK_SIZE] }> = src_part.load([row_idx, block]);
        out.store(tile.reshape(const_shape![BLOCK_SIZE]));
    }
}

pub use gather_row_f16_module::gather_row_f16;

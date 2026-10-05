/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! RMSNorm-family kernels, including fused residual-add plus RMSNorm transformer block kernels.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod rms_norm_f16_module {
    use cutile::core::*;

    /// Computes RMSNorm over the last dimension. Used for transformer hidden states and per-head Q/K normalization, commonly [tokens, hidden_size] or [heads, head_dim].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            out_part.store(tout_f16, [0i32, j]);
        }
    }
}

pub use rms_norm_f16_module::rms_norm_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod add_rms_norm_f16_module {
    use cutile::core::*;

    /// Fuses residual add and RMSNorm while also writing the residual output. Used between attention/MLP blocks over [tokens, hidden_size].
    #[cutile::entry(print_ir=false,
                        unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn add_rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        residual: &Tensor<f16, { [-1, N] }>,
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        residual_out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        // Ceiling division so BLOCK_SIZE does not have to divide N — lets us
        // ablate BLOCK_SIZE over pow-2 values (512 is the tuned default per
        // cutile-benchmarks/benches/rmsnorm.rs). Overhang lanes mask to zero
        // on load and are dropped on store via tile IR.
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let residual_part: Partition<f16, { [1, BLOCK_SIZE] }> = residual.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);

        // First pass: add residual + x, accumulate sum of squares for RMS.
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            rms = rms + combined * combined;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        // Second pass: write normalized output and updated residual.
        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        let mut res_out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            residual_out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            let normed: Tile<f32, { [1, BLOCK_SIZE] }> = combined * inv_rms * tw;
            let normed_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(normed);
            let combined_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(combined);
            out_part.store(normed_f16, [0i32, j]);
            res_out_part.store(combined_f16, [0i32, j]);
        }
    }
}

pub use add_rms_norm_f16_module::add_rms_norm_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod qk_norm_f16_module {
    use cutile::core::*;

    /// Applies RMSNorm to Q or K head rows. Used by transformer variants with QK norm over [heads, head_dim].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn qk_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        q: &Tensor<f16, { [-1, N] }>,
        k: &Tensor<f16, { [-1, N] }>,
        q_weight: &Tensor<f16, { [N] }>,
        k_weight: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
        num_q_rows: i32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let is_q: bool = row < num_q_rows;
        let local_row: i32 = if is_q { row } else { row - num_q_rows };

        let q_part: Partition<f16, { [1, BLOCK_SIZE] }> = q.partition(tile_shape);
        let k_part: Partition<f16, { [1, BLOCK_SIZE] }> = k.partition(tile_shape);

        // Pass 1: compute RMS
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                q_part.load([local_row, j])
            } else {
                k_part.load([local_row, j])
            };
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        // Pass 2: normalize with the appropriate weight vector
        let qw_part: Partition<f16, { [BLOCK_SIZE] }> =
            q_weight.partition(const_shape![BLOCK_SIZE]);
        let kw_part: Partition<f16, { [BLOCK_SIZE] }> =
            k_weight.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                q_part.load([local_row, j])
            } else {
                k_part.load([local_row, j])
            };
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                qw_part.load([j]).reshape(tile_shape)
            } else {
                kw_part.load([j]).reshape(tile_shape)
            };
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            out_part.store(tout_f16, [0i32, j]);
        }
    }
}

pub use qk_norm_f16_module::qk_norm_f16;

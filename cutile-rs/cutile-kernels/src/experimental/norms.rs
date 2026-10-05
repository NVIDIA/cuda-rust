/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Experimental raw-pointer normalization kernels for graph-oriented decode paths.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod add_rms_norm_decode_raw_f16_module {
    use cutile::core::*;

    /// Raw-pointer fused residual add and RMSNorm for decode CUDA graphs. Used to avoid tensor wrapper overhead for [1, hidden_size] decode activations.
    #[cutile::entry(print_ir=false,
                        unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn add_rms_norm_decode_raw_f16<const N: i32, const BLOCK_SIZE: i32>(
        residual_ptr: *mut f16,
        x_ptr: *mut f16,
        w_ptr: *mut f16,
        out_ptr: *mut f16,
        residual_out_ptr: *mut f16,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let residual_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(residual_ptr) };
        let x_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(x_ptr) };
        let w_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(w_ptr) };
        let out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let residual_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(residual_out_ptr) };

        let tok: Token = new_token_unordered();
        let shape_2d: Shape<{ [-1, N] }> = Shape::<{ [-1, N] }> { dims: &[1i32] };
        let strides_2d: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[N] };

        let residual_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(residual_ptr), shape_2d, strides_2d, tok) };
        let x_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(x_ptr), shape_2d, strides_2d, tok) };
        let w_shape: Shape<{ [N] }> = const_shape![N];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let w_tv: Tensor<f16, { [N] }> =
            unsafe { make_tensor_view(pointer_to_tile(w_ptr), w_shape, w_strides, tok) };
        let out_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(out_ptr), shape_2d, strides_2d, tok) };
        let residual_out_tv: Tensor<f16, { [-1, N] }> = unsafe {
            make_tensor_view(pointer_to_tile(residual_out_ptr), shape_2d, strides_2d, tok)
        };

        let residual_part: Partition<f16, { [1, BLOCK_SIZE] }> = residual_tv.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x_tv.partition(tile_shape);

        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &residual_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &x_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            rms = rms + combined * combined;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(N);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Enabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w_tv.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out_tv.partition_full_mut(tile_shape) };
        let mut res_out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { residual_out_tv.partition_full_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &residual_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &x_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tw_1d: Tile<f16, { [BLOCK_SIZE] }> = load_view_tko(
                &w_part,
                [j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = tw_1d.reshape(tile_shape);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            let normed: Tile<f32, { [1, BLOCK_SIZE] }> = combined * inv_rms * tw;
            let normed_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(normed);
            let combined_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(combined);
            unsafe {
                store_view_tko_mut(
                    &mut out_part,
                    normed_f16,
                    [0i32, j],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut res_out_part,
                    combined_f16,
                    [0i32, j],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }
}

pub use add_rms_norm_decode_raw_f16_module::add_rms_norm_decode_raw_f16;

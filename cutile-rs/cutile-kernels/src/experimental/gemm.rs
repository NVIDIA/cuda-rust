/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Experimental mixed-precision GEMM kernels.

#[cutile::module]
pub mod gemm_f16_module {
    use cutile::core::*;

    /// GEMM with f16 operands, f32 accumulation, and f16 output.
    ///
    /// Keep the operands in f16 for MMA; widening them before multiplication
    /// changes the operation selected by the compiler. Only the accumulator
    /// needs f32 precision. The output is rounded to f16 once after the K loop.
    ///
    /// # Safety
    /// X[M,K], Y[K,N], and Z[M,N] must have matching positive dimensions and
    /// TMA-compatible strides. K must be divisible by BK. Use supported MMA
    /// tile sizes, output partitions [BM,BN], and grid (ceil(M/BM),ceil(N/BN),1).
    /// Partial M/N tiles are masked. Output must not alias inputs or other
    /// writers, and padded indices must fit i32.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    unsafe fn gemm_f16<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f16, { [BM, BN] }>,
        x: &Tensor<f16, { [-1, K] }>,
        y: &Tensor<f16, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x: Tile<f16, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f16, { [BK, BN] }> = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        let tile_z: Tile<f16, { [BM, BN] }> = convert_tile(tile_z);
        z.store(tile_z);
    }
}

pub use gemm_f16_module::gemm_f16;

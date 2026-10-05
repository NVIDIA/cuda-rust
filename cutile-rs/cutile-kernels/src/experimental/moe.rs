/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Experimental MoE kernels, including grouped GEMM over expert descriptor tables.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod group_gemm_f16_nt_desc_module {
    use cutile::core::*;

    unsafe fn load_f16_ptr(
        ptrs: &Tensor<i64, { [-1] }>,
        group_id: i32,
    ) -> PointerTile<*mut f16, { [] }> {
        let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
        let ptr_part: Partition<i64, { [1] }> = ptrs.partition(one_shape);
        let ptr_int: Tile<i64, { [1] }> = ptr_part.load([group_id]);
        let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
        let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
        unsafe { assume_div_by::<_, 16>(ptr) }
    }

    unsafe fn load_f16_desc_2d(
        ptrs: &Tensor<i64, { [-1] }>,
        metas: &Tensor<i32, { [-1, 8] }>,
        group_id: i32,
    ) -> (Tensor<f16, { [-1, -1] }>, i32, i32) {
        let meta_part: Partition<i32, { [1, 8] }> = metas.partition(const_shape![1, 8]);
        let row: Tile<i32, { [1, 8] }> = meta_part.load([group_id, 0i32]);
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let idx2: Tile<i32, { [] }> = scalar_to_tile(2i32);
        let rows_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx0]);
        let cols_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx1]);
        let stride0_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx2]);
        let rows_tile: Tile<i32, { [] }> = rows_tile.reshape(const_shape![]);
        let cols_tile: Tile<i32, { [] }> = cols_tile.reshape(const_shape![]);
        let stride0_tile: Tile<i32, { [] }> = stride0_tile.reshape(const_shape![]);
        let ptr: PointerTile<*mut f16, { [] }> = unsafe { load_f16_ptr(ptrs, group_id) };
        let rows: i32 = unsafe {
            assume_div_by::<_, 16>(assume_bounds_lower::<_, 0>(tile_to_scalar(rows_tile)))
        };
        let cols: i32 = unsafe {
            assume_div_by::<_, 16>(assume_bounds_lower::<_, 0>(tile_to_scalar(cols_tile)))
        };
        let stride0: i32 = unsafe {
            assume_div_by::<_, 8>(assume_bounds_lower::<_, 0>(tile_to_scalar(stride0_tile)))
        };
        let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[rows, cols],
        };
        let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[stride0] };
        let tensor: Tensor<f16, { [-1, -1] }> =
            unsafe { make_tensor_view(ptr, shape, strides, new_token_unordered()) };
        (tensor, rows, cols)
    }

    // TileGym-style persistent group GEMM over vectors of tensor pointers.
    // Tensor extents vary per group through compact metadata tables; BM/BN/BK
    // remain compile-time tile shapes, so the host buckets by tile shape.

    /// Persistent grouped GEMM over descriptor tables and tensor pointer arrays. Intended for MoE expert batches where each group has its own M/N/K.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, num_cta_in_cga=2, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn group_gemm_f16_nt_desc<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const NUM_SM: i32,
    >(
        a_ptrs: &Tensor<i64, { [-1] }>,
        b_ptrs: &Tensor<i64, { [-1] }>,
        c_ptrs: &Tensor<i64, { [-1] }>,
        a_metas: &Tensor<i32, { [-1, 8] }>,
        b_metas: &Tensor<i32, { [-1, 8] }>,
        c_metas: &Tensor<i32, { [-1, 8] }>,
        num_groups: i32,
    ) {
        let mut tile_idx: i32 = get_tile_block_id().0;
        let mut last_problem_end: i32 = 0;

        for group_id in 0i32..num_groups {
            let (a, m, k): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(a_ptrs, a_metas, group_id) };
            let (b, _bk_rows, n): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(b_ptrs, b_metas, group_id) };
            let (c, _cm, _cn): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(c_ptrs, c_metas, group_id) };

            let num_m_tiles: i32 = ceil_div(m, BM);
            let num_n_tiles: i32 = ceil_div(n, BN);
            let num_k_tiles: i32 = ceil_div(k, BK);
            let num_tiles: i32 = num_m_tiles * num_n_tiles;

            let a_part: Partition<f16, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
            let b_part: Partition<f16, { [BK, BN] }> = b.partition(const_shape![BK, BN]);
            let mut c_part: PartitionMut<f16, { [BM, BN] }> =
                unsafe { c.partition_full_mut(const_shape![BM, BN]) };

            while tile_idx >= last_problem_end && tile_idx < last_problem_end + num_tiles {
                let tile_idx_in_group: i32 = tile_idx - last_problem_end;
                let tile_m_idx: i32 = tile_idx_in_group / num_n_tiles;
                let tile_n_idx: i32 = tile_idx_in_group - tile_m_idx * num_n_tiles;

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                for kk in 0i32..num_k_tiles {
                    let ta: Tile<f16, { [BM, BK] }> = a_part.load([tile_m_idx, kk]);
                    let tb: Tile<f16, { [BK, BN] }> = b_part.load([kk, tile_n_idx]);
                    acc = mma(ta, tb, acc);
                }

                let out: Tile<f16, { [BM, BN] }> = convert_tile(acc);
                c_part.store(out, [tile_m_idx, tile_n_idx]);

                tile_idx = tile_idx + NUM_SM;
            }

            last_problem_end = last_problem_end + num_tiles;
        }
    }
}

pub use group_gemm_f16_nt_desc_module::group_gemm_f16_nt_desc;

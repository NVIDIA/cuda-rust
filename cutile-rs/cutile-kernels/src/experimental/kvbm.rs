/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Experimental KVBM block-layout conversion kernels.
//!
//! These kernels move attention KV cache blocks between stacked per-layer block
//! tensors and contiguous storage layouts. They are intended for Dynamo-style
//! KVBM paths where storage blocks may be represented as NHD/HND stacked block
//! chunks, operational flat buffers, or universal `[heads, layers, offsets,
//! tokens, head_dim]` buffers.

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod copy_stacked_to_contiguous_f16_module {
    use cutile::core::*;

    fn load_ptr_tensor_tile<const TILE_SHAPE: [i32; 5], const TILE_VIEW_SHAPE: [i32; 3]>(
        tensors: &Tensor<i64, { [-1] }>,
        shape_dims: &[i32],
        tensor_idx: [i32; 1],
        tile_idx: [i32; 5],
    ) -> Tile<f16, TILE_SHAPE> {
        let flat_shape: Shape<{ [-1, -1, -1] }> = Shape::<{ [-1, -1, -1] }> {
            dims: &[
                shape_dims[0] * shape_dims[1] * shape_dims[2],
                shape_dims[3],
                shape_dims[4],
            ],
        };
        let flat_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[shape_dims[3] * shape_dims[4], shape_dims[4]],
        };
        let flat_tile_idx = [
            (tile_idx[0] * shape_dims[1] + tile_idx[1]) * shape_dims[2] + tile_idx[2],
            tile_idx[3],
            tile_idx[4],
        ];
        unsafe {
            let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
            let ptr_part: Partition<i64, { [1] }> = tensors.partition(one_shape);
            let ptr_int: Tile<i64, { [1] }> = ptr_part.load(tensor_idx);
            let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
            let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
            let ptr: PointerTile<*mut f16, { [] }> = assume_div_by::<_, 16>(ptr);
            let tensor: Tensor<f16, { [-1, -1, -1] }> =
                make_tensor_view(ptr, flat_shape, flat_strides, new_token_unordered());
            let tensor_part: Partition<f16, TILE_VIEW_SHAPE> =
                tensor.partition(const_shape!(TILE_VIEW_SHAPE));
            let tile: Tile<f16, TILE_VIEW_SHAPE> = tensor_part.load(flat_tile_idx);
            tile.reshape(const_shape!(TILE_SHAPE))
        }
    }

    fn store_ptr_tensor_tile<const TILE_SHAPE: [i32; 5], const TILE_VIEW_SHAPE: [i32; 3]>(
        tensors: &Tensor<i64, { [-1] }>,
        shape_dims: &[i32],
        tile: Tile<f16, TILE_SHAPE>,
        tensor_idx: [i32; 1],
        tile_idx: [i32; 5],
    ) {
        let flat_shape: Shape<{ [-1, -1, -1] }> = Shape::<{ [-1, -1, -1] }> {
            dims: &[
                shape_dims[0] * shape_dims[1] * shape_dims[2],
                shape_dims[3],
                shape_dims[4],
            ],
        };
        let flat_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[shape_dims[3] * shape_dims[4], shape_dims[4]],
        };
        let flat_tile_idx = [
            (tile_idx[0] * shape_dims[1] + tile_idx[1]) * shape_dims[2] + tile_idx[2],
            tile_idx[3],
            tile_idx[4],
        ];
        unsafe {
            let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
            let ptr_part: Partition<i64, { [1] }> = tensors.partition(one_shape);
            let ptr_int: Tile<i64, { [1] }> = ptr_part.load(tensor_idx);
            let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
            let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
            let ptr: PointerTile<*mut f16, { [] }> = assume_div_by::<_, 16>(ptr);
            let mut tensor: Tensor<f16, { [-1, -1, -1] }> =
                make_tensor_view(ptr, flat_shape, flat_strides, new_token_unordered());
            let mut tensor_part: PartitionMut<f16, TILE_VIEW_SHAPE> =
                tensor.partition_mut(const_shape!(TILE_VIEW_SHAPE));
            let tile: Tile<f16, TILE_VIEW_SHAPE> = tile.reshape(const_shape!(TILE_VIEW_SHAPE));
            tensor_part.store(tile, flat_tile_idx);
        }
    }

    fn fold_iteration_indexes(tile_idx: i32, shape: [i32; 4]) -> (i32, i32, i32, i32) {
        let o_idx = tile_idx % shape[3];
        let rest_idx = tile_idx / shape[3];
        let l_idx = rest_idx % shape[2];
        let rest_idx = rest_idx / shape[2];
        let h_idx = rest_idx % shape[1];
        let block_idx = rest_idx / shape[1];
        (block_idx, h_idx, l_idx, o_idx)
    }

    /// Copies f16 KVBM stacked NHD/HND block chunks into a contiguous operational or universal block. This is used when cache blocks leave per-layer chunk storage for compact transfer or backend-neutral storage.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    unsafe fn copy_stacked_to_contiguous_f16<
        const STACKED_TILE: [i32; 5],
        const CONTIGUOUS_TILE: [i32; 5],
        const STACKED_TILE_VIEW: [i32; 3],
        const CONTIGUOUS_TILE_VIEW: [i32; 3],
        const UNIVERSAL_TO_STACKED_MAP: [i32; 5],
        const STACKED_TO_CONTIGUOUS_MAP: [i32; 5],
    >(
        num_blocks: i32,
        stacked_tensors: &Tensor<i64, { [-1] }>,
        contiguous_tensors: &Tensor<i64, { [-1] }>,
        nh: i32,
        nl: i32,
        no: i32,
        nt: i32,
        hd: i32,
    ) {
        let universal_shape: [i32; 5] = [nh, nl, no, nt, hd];
        let stacked_shape: [i32; 5] =
            permute_array(universal_shape, const_array!(UNIVERSAL_TO_STACKED_MAP));
        let contiguous_shape: [i32; 5] =
            permute_array(stacked_shape, const_array!(STACKED_TO_CONTIGUOUS_MAP));

        let pid: (i32, i32, i32) = get_tile_block_id();
        let t_idx = pid.1;
        let d_idx = pid.2;

        for tile_idx in 0i32..(num_blocks * nh * nl * no) {
            let (block_idx, h_idx, l_idx, o_idx) =
                fold_iteration_indexes(tile_idx, [num_blocks, nh, nl, no]);
            let universal_tile_idx: [i32; 5] = [h_idx, l_idx, o_idx, t_idx, d_idx];
            let stacked_tile_idx: [i32; 5] =
                permute_array(universal_tile_idx, const_array!(UNIVERSAL_TO_STACKED_MAP));
            let contiguous_tile_idx: [i32; 5] =
                permute_array(stacked_tile_idx, const_array!(STACKED_TO_CONTIGUOUS_MAP));

            let stacked_tensor_shape_dims: &[i32] = &[
                1i32,
                1i32,
                stacked_shape[2],
                stacked_shape[3],
                stacked_shape[4],
            ];
            let stacked_tensor_idx = [stacked_shape[0] * stacked_shape[1] * block_idx
                + stacked_shape[1] * stacked_tile_idx[0]
                + stacked_tile_idx[1]];
            let stacked_tile_idx = [
                0i32,
                0i32,
                stacked_tile_idx[2],
                stacked_tile_idx[3],
                stacked_tile_idx[4],
            ];
            let stacked_tile: Tile<f16, STACKED_TILE> =
                load_ptr_tensor_tile::<STACKED_TILE, STACKED_TILE_VIEW>(
                    stacked_tensors,
                    stacked_tensor_shape_dims,
                    stacked_tensor_idx,
                    stacked_tile_idx,
                );
            let contiguous_tile: Tile<f16, CONTIGUOUS_TILE> =
                permute(stacked_tile, const_array!(STACKED_TO_CONTIGUOUS_MAP));
            store_ptr_tensor_tile::<CONTIGUOUS_TILE, CONTIGUOUS_TILE_VIEW>(
                contiguous_tensors,
                &contiguous_shape,
                contiguous_tile,
                [block_idx],
                contiguous_tile_idx,
            );
        }
    }
}

pub use copy_stacked_to_contiguous_f16_module::copy_stacked_to_contiguous_f16;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod copy_contiguous_to_stacked_f16_module {
    use cutile::core::*;

    fn load_ptr_tensor_tile<const TILE_SHAPE: [i32; 5], const TILE_VIEW_SHAPE: [i32; 3]>(
        tensors: &Tensor<i64, { [-1] }>,
        shape_dims: &[i32],
        tensor_idx: [i32; 1],
        tile_idx: [i32; 5],
    ) -> Tile<f16, TILE_SHAPE> {
        let flat_shape: Shape<{ [-1, -1, -1] }> = Shape::<{ [-1, -1, -1] }> {
            dims: &[
                shape_dims[0] * shape_dims[1] * shape_dims[2],
                shape_dims[3],
                shape_dims[4],
            ],
        };
        let flat_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[shape_dims[3] * shape_dims[4], shape_dims[4]],
        };
        let flat_tile_idx = [
            (tile_idx[0] * shape_dims[1] + tile_idx[1]) * shape_dims[2] + tile_idx[2],
            tile_idx[3],
            tile_idx[4],
        ];
        unsafe {
            let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
            let ptr_part: Partition<i64, { [1] }> = tensors.partition(one_shape);
            let ptr_int: Tile<i64, { [1] }> = ptr_part.load(tensor_idx);
            let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
            let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
            let ptr: PointerTile<*mut f16, { [] }> = assume_div_by::<_, 16>(ptr);
            let tensor: Tensor<f16, { [-1, -1, -1] }> =
                make_tensor_view(ptr, flat_shape, flat_strides, new_token_unordered());
            let tensor_part: Partition<f16, TILE_VIEW_SHAPE> =
                tensor.partition(const_shape!(TILE_VIEW_SHAPE));
            let tile: Tile<f16, TILE_VIEW_SHAPE> = tensor_part.load(flat_tile_idx);
            tile.reshape(const_shape!(TILE_SHAPE))
        }
    }

    fn store_ptr_tensor_tile<const TILE_SHAPE: [i32; 5], const TILE_VIEW_SHAPE: [i32; 3]>(
        tensors: &Tensor<i64, { [-1] }>,
        shape_dims: &[i32],
        tile: Tile<f16, TILE_SHAPE>,
        tensor_idx: [i32; 1],
        tile_idx: [i32; 5],
    ) {
        let flat_shape: Shape<{ [-1, -1, -1] }> = Shape::<{ [-1, -1, -1] }> {
            dims: &[
                shape_dims[0] * shape_dims[1] * shape_dims[2],
                shape_dims[3],
                shape_dims[4],
            ],
        };
        let flat_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[shape_dims[3] * shape_dims[4], shape_dims[4]],
        };
        let flat_tile_idx = [
            (tile_idx[0] * shape_dims[1] + tile_idx[1]) * shape_dims[2] + tile_idx[2],
            tile_idx[3],
            tile_idx[4],
        ];
        unsafe {
            let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
            let ptr_part: Partition<i64, { [1] }> = tensors.partition(one_shape);
            let ptr_int: Tile<i64, { [1] }> = ptr_part.load(tensor_idx);
            let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
            let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
            let ptr: PointerTile<*mut f16, { [] }> = assume_div_by::<_, 16>(ptr);
            let mut tensor: Tensor<f16, { [-1, -1, -1] }> =
                make_tensor_view(ptr, flat_shape, flat_strides, new_token_unordered());
            let mut tensor_part: PartitionMut<f16, TILE_VIEW_SHAPE> =
                tensor.partition_mut(const_shape!(TILE_VIEW_SHAPE));
            let tile: Tile<f16, TILE_VIEW_SHAPE> = tile.reshape(const_shape!(TILE_VIEW_SHAPE));
            tensor_part.store(tile, flat_tile_idx);
        }
    }

    fn fold_iteration_indexes(tile_idx: i32, shape: [i32; 4]) -> (i32, i32, i32, i32) {
        let o_idx = tile_idx % shape[3];
        let rest_idx = tile_idx / shape[3];
        let l_idx = rest_idx % shape[2];
        let rest_idx = rest_idx / shape[2];
        let h_idx = rest_idx % shape[1];
        let block_idx = rest_idx / shape[1];
        (block_idx, h_idx, l_idx, o_idx)
    }

    /// Copies f16 KVBM contiguous operational or universal blocks back into stacked NHD/HND block chunks. This restores per-layer cache block tensors after compact storage or transfer.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    unsafe fn copy_contiguous_to_stacked_f16<
        const STACKED_TILE: [i32; 5],
        const CONTIGUOUS_TILE: [i32; 5],
        const STACKED_TILE_VIEW: [i32; 3],
        const CONTIGUOUS_TILE_VIEW: [i32; 3],
        const UNIVERSAL_TO_STACKED_MAP: [i32; 5],
        const STACKED_TO_CONTIGUOUS_MAP: [i32; 5],
        const CONTIGUOUS_TO_STACKED_MAP: [i32; 5],
    >(
        num_blocks: i32,
        stacked_tensors: &Tensor<i64, { [-1] }>,
        contiguous_tensors: &Tensor<i64, { [-1] }>,
        nh: i32,
        nl: i32,
        no: i32,
        nt: i32,
        hd: i32,
    ) {
        let universal_shape: [i32; 5] = [nh, nl, no, nt, hd];
        let stacked_shape: [i32; 5] =
            permute_array(universal_shape, const_array!(UNIVERSAL_TO_STACKED_MAP));
        let contiguous_shape: [i32; 5] =
            permute_array(stacked_shape, const_array!(STACKED_TO_CONTIGUOUS_MAP));

        let pid: (i32, i32, i32) = get_tile_block_id();
        let t_idx = pid.1;
        let d_idx = pid.2;

        for tile_idx in 0i32..(num_blocks * nh * nl * no) {
            let (block_idx, h_idx, l_idx, o_idx) =
                fold_iteration_indexes(tile_idx, [num_blocks, nh, nl, no]);
            let universal_tile_idx: [i32; 5] = [h_idx, l_idx, o_idx, t_idx, d_idx];
            let stacked_tile_idx: [i32; 5] =
                permute_array(universal_tile_idx, const_array!(UNIVERSAL_TO_STACKED_MAP));
            let contiguous_tile_idx: [i32; 5] =
                permute_array(stacked_tile_idx, const_array!(STACKED_TO_CONTIGUOUS_MAP));

            let stacked_tensor_shape_dims: &[i32] = &[
                1i32,
                1i32,
                stacked_shape[2],
                stacked_shape[3],
                stacked_shape[4],
            ];
            let stacked_tensor_idx = [stacked_shape[0] * stacked_shape[1] * block_idx
                + stacked_shape[1] * stacked_tile_idx[0]
                + stacked_tile_idx[1]];
            let stacked_tile_idx = [
                0i32,
                0i32,
                stacked_tile_idx[2],
                stacked_tile_idx[3],
                stacked_tile_idx[4],
            ];
            let contiguous_tile: Tile<f16, CONTIGUOUS_TILE> =
                load_ptr_tensor_tile::<CONTIGUOUS_TILE, CONTIGUOUS_TILE_VIEW>(
                    contiguous_tensors,
                    &contiguous_shape,
                    [block_idx],
                    contiguous_tile_idx,
                );
            let stacked_tile: Tile<f16, STACKED_TILE> =
                permute(contiguous_tile, const_array!(CONTIGUOUS_TO_STACKED_MAP));
            store_ptr_tensor_tile::<STACKED_TILE, STACKED_TILE_VIEW>(
                stacked_tensors,
                stacked_tensor_shape_dims,
                stacked_tile,
                stacked_tensor_idx,
                stacked_tile_idx,
            );
        }
    }
}

pub use copy_contiguous_to_stacked_f16_module::copy_contiguous_to_stacked_f16;

/// Stacked KVBM block chunk layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StackedLayout {
    /// `[num_layers][num_offsets][tokens, heads, head_dim]`.
    Nhd,
    /// `[num_layers][num_offsets][heads, tokens, head_dim]`.
    Hnd,
}

/// Contiguous KVBM storage layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContiguousLayout {
    /// `[num_layers, num_offsets, tokens * heads * head_dim]` with NHD inner order.
    OperationalNhd,
    /// `[num_layers, num_offsets, heads * tokens * head_dim]` with HND inner order.
    OperationalHnd,
    /// `[heads, num_layers, num_offsets, tokens, head_dim]`.
    Universal,
}

impl StackedLayout {
    pub fn map(self, dst_layout: ContiguousLayout) -> [usize; 5] {
        match (self, dst_layout) {
            (StackedLayout::Nhd, ContiguousLayout::Universal) => [3, 0, 1, 2, 4],
            (StackedLayout::Hnd, ContiguousLayout::Universal) => [2, 0, 1, 3, 4],
            (StackedLayout::Nhd, ContiguousLayout::OperationalNhd) => [0, 1, 2, 3, 4],
            (StackedLayout::Hnd, ContiguousLayout::OperationalHnd) => [0, 1, 2, 3, 4],
            _ => panic!("unsupported KVBM stacked-to-contiguous layout map"),
        }
    }
}

impl ContiguousLayout {
    pub fn map(self, dst_layout: StackedLayout) -> [usize; 5] {
        match (self, dst_layout) {
            (ContiguousLayout::Universal, StackedLayout::Nhd) => [1, 2, 3, 0, 4],
            (ContiguousLayout::Universal, StackedLayout::Hnd) => [1, 2, 0, 3, 4],
            (ContiguousLayout::OperationalNhd, StackedLayout::Nhd) => [0, 1, 2, 3, 4],
            (ContiguousLayout::OperationalHnd, StackedLayout::Hnd) => [0, 1, 2, 3, 4],
            _ => panic!("unsupported KVBM contiguous-to-stacked layout map"),
        }
    }
}

pub fn remap<const RANK: usize>(src: &[usize], map: [usize; RANK]) -> [usize; RANK] {
    let mut result: [usize; RANK] = [0; RANK];
    for i in 0..RANK {
        result[i] = src[map[i]];
    }
    result
}

fn push_usize_array<const RANK: usize>(generics: &mut Vec<String>, values: [usize; RANK]) {
    generics.extend(values.into_iter().map(|value| value.to_string()));
}

fn flat_tile_shape(tile: [usize; 5]) -> [usize; 3] {
    [tile[0] * tile[1] * tile[2], tile[3], tile[4]]
}

/// Builds const-generic arguments for a stacked-to-contiguous f16 KVBM specialization.
pub fn stacked_to_contiguous_f16_generics(
    stacked_layout: StackedLayout,
    contiguous_layout: ContiguousLayout,
    tile_tokens: usize,
    tile_head_dim: usize,
) -> Vec<String> {
    let universal_tile = [1usize, 1usize, 1usize, tile_tokens, tile_head_dim];
    let universal_to_stacked = ContiguousLayout::Universal.map(stacked_layout);
    let stacked_to_contiguous = stacked_layout.map(contiguous_layout);
    let stacked_tile = remap(&universal_tile, universal_to_stacked);
    let contiguous_tile = remap(&stacked_tile, stacked_to_contiguous);
    let stacked_tile_view = flat_tile_shape(stacked_tile);
    let contiguous_tile_view = flat_tile_shape(contiguous_tile);

    let mut generics = Vec::new();
    push_usize_array(&mut generics, stacked_tile);
    push_usize_array(&mut generics, contiguous_tile);
    push_usize_array(&mut generics, stacked_tile_view);
    push_usize_array(&mut generics, contiguous_tile_view);
    push_usize_array(&mut generics, universal_to_stacked);
    push_usize_array(&mut generics, stacked_to_contiguous);
    generics
}

/// Builds const-generic arguments for a contiguous-to-stacked f16 KVBM specialization.
pub fn contiguous_to_stacked_f16_generics(
    stacked_layout: StackedLayout,
    contiguous_layout: ContiguousLayout,
    tile_tokens: usize,
    tile_head_dim: usize,
) -> Vec<String> {
    let universal_tile = [1usize, 1usize, 1usize, tile_tokens, tile_head_dim];
    let universal_to_stacked = ContiguousLayout::Universal.map(stacked_layout);
    let stacked_to_contiguous = stacked_layout.map(contiguous_layout);
    let contiguous_to_stacked = contiguous_layout.map(stacked_layout);
    let stacked_tile = remap(&universal_tile, universal_to_stacked);
    let contiguous_tile = remap(&stacked_tile, stacked_to_contiguous);
    let stacked_tile_view = flat_tile_shape(stacked_tile);
    let contiguous_tile_view = flat_tile_shape(contiguous_tile);

    let mut generics = Vec::new();
    push_usize_array(&mut generics, stacked_tile);
    push_usize_array(&mut generics, contiguous_tile);
    push_usize_array(&mut generics, stacked_tile_view);
    push_usize_array(&mut generics, contiguous_tile_view);
    push_usize_array(&mut generics, universal_to_stacked);
    push_usize_array(&mut generics, stacked_to_contiguous);
    push_usize_array(&mut generics, contiguous_to_stacked);
    generics
}

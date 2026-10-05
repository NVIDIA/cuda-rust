/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/// Tensor creation and initialization kernels.
///
/// This module contains GPU kernels for creating and initializing tensors with
/// specific patterns. These kernels are used internally by the `api` module functions
/// like `zeros`, `ones`, `full`, and `arange`.

#[crate::module(tile_rust_crate = true)]
pub mod creation {
    use crate::core::*;

    /// Fills a tensor with a constant value.
    ///
    /// This kernel broadcasts a scalar value to fill an entire tensor partition.
    /// Each thread block processes one partition.
    ///
    /// ## Parameters
    ///
    /// - `value`: The constant value to fill the tensor with
    /// - `tensor`: Mutable tensor to store the result
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Element type (must implement `ElementType`)
    /// - `S`: Partition shape (compile-time constant)
    ///
    /// ## Usage
    ///
    /// This kernel is used by `api::full()`, `api::zeros()`, and `api::ones()`.
    ///
    /// ```rust,ignore
    /// use cutile::kernels::creation::full;
    ///
    /// let val = 42.0f32;
    /// let tensor = api::zeros(&[1024]).partition([128]);
    /// let result = value((val, tensor))
    ///     .then(full)
    ///     .unzip();
    /// ```
    #[crate::entry()]
    pub fn full<T: ElementType, const S: [i32; 1]>(value: T, tensor: &mut Tensor<T, S>) {
        let value_tile: Tile<T, S> = value.broadcast(tensor.shape());
        tensor.store(value_tile);
    }

    /// Creates a sequence of consecutive integers starting from the partition offset.
    ///
    /// This kernel generates values like [0, 1, 2, ...] across all partitions,
    /// with each partition computing its portion based on the thread block ID.
    ///
    /// ## Parameters
    ///
    /// - `tensor`: Mutable tensor to store the sequence
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Element type (must implement `ElementType`)
    /// - `S`: Partition shape (compile-time constant)
    ///
    /// ## Usage
    ///
    /// This kernel is used by `api::arange()`.
    ///
    /// ```rust,ignore
    /// use cutile::kernels::creation::arange;
    ///
    /// let tensor = Tensor::<i32>::uninitialized(256)
    ///     .await
    ///     .partition([64]);
    /// let result = value((tensor,))
    ///     .then(arange)
    ///     .unzip();
    /// // Result contains [0, 1, 2, ..., 255]
    /// ```
    #[crate::entry()]
    pub fn arange<T: ElementType, const S: [i32; 1]>(tensor: &mut Tensor<T, S>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offset = pid.0 * S[0];
        let offset_tile = broadcast_scalar(offset, tensor.shape());
        let range = iota::<i32, S>(tensor.shape());
        let offset_range = offset_tile + range;
        let offset_range: Tile<T, S> = convert_tile(offset_range);
        tensor.store(offset_range);
    }

    /// Creates evenly spaced values: start + i * step.
    ///
    /// `start` and `step` are passed as scalars. Each thread block
    /// computes its partition using a global index offset.
    #[crate::entry()]
    pub fn linspace<const S: [i32; 1]>(tensor: &mut Tensor<f32, S>, start: f32, step: f32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offset: i32 = pid.0 * S[0];
        let indices: Tile<i32, S> = iota(tensor.shape());
        let offset_indices: Tile<i32, S> = indices + broadcast_scalar(offset, tensor.shape());
        let float_indices: Tile<f32, S> = convert_tile(offset_indices);
        let start_tile: Tile<f32, S> = broadcast_scalar(start, tensor.shape());
        let step_tile: Tile<f32, S> = broadcast_scalar(step, tensor.shape());
        tensor.store(start_tile + float_indices * step_tile);
    }

    /// Creates an identity-like matrix: 1 where row == col, 0 elsewhere.
    ///
    /// Each thread block handles one tile. Row and column offsets are
    /// derived from the 2D block ID and the partition shape.
    #[crate::entry()]
    pub fn eye<const BR: i32, const BC: i32>(tensor: &mut Tensor<f32, { [BR, BC] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_offset: i32 = pid.0 * BR;
        let col_offset: i32 = pid.1 * BC;

        // Row indices: [row_offset, row_offset+1, ..., row_offset+BR-1]
        // broadcast to [BR, BC]
        let row_iota: Tile<i32, { [BR] }> = iota(shape![BR]);
        let row_base: Tile<i32, { [BR] }> = broadcast_scalar(row_offset, shape![BR]);
        let rows: Tile<i32, { [BR] }> = row_iota + row_base;
        let rows_2d: Tile<i32, { [BR, 1] }> = rows.reshape(shape![BR, 1]);
        let rows_bc: Tile<i32, { [BR, BC] }> = rows_2d.broadcast(shape![BR, BC]);

        // Col indices: [col_offset, col_offset+1, ..., col_offset+BC-1]
        // broadcast to [BR, BC]
        let col_iota: Tile<i32, { [BC] }> = iota(shape![BC]);
        let col_base: Tile<i32, { [BC] }> = broadcast_scalar(col_offset, shape![BC]);
        let cols: Tile<i32, { [BC] }> = col_iota + col_base;
        let cols_2d: Tile<i32, { [1, BC] }> = cols.reshape(shape![1, BC]);
        let cols_bc: Tile<i32, { [BR, BC] }> = cols_2d.broadcast(shape![BR, BC]);

        let is_diag: Tile<bool, { [BR, BC] }> = eq_tile(rows_bc, cols_bc);
        let ones: Tile<f32, { [BR, BC] }> = constant(1.0f32, shape![BR, BC]);
        let zeros: Tile<f32, { [BR, BC] }> = constant(0.0f32, shape![BR, BC]);
        tensor.store(select(is_diag, ones, zeros));
    }
}

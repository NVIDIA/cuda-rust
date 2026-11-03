/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/// Type conversion GPU kernels.
///
/// This module provides kernels for converting tensors between different element types.
/// The conversion is performed efficiently on the GPU by loading source tiles,
/// converting element types, and storing the result.
///
/// ## Available Kernels
///
/// - [`convert`] - Converts a tensor from one element type to another
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
/// use cutile::kernels::conversion::convert;
///
/// // Convert f32 tensor to f16
/// let src: Arc<Tensor<f32>> = api::randn(0.0, 1.0, [1024], None).await?.into();
/// let dst = api::zeros::<f16>(&[1024]).partition([128]);
///
/// let (src, dst) = convert(src, dst).await?;
/// ```

#[crate::module(tile_rust_crate = true)]
pub mod conversion {
    use crate::core::*;

    /// Converts a tensor from one element type to another.
    ///
    /// This kernel performs element-wise type conversion on GPU tiles. Each thread block
    /// processes one partition, loading source elements, converting them to the destination
    /// type, and storing the result.
    ///
    /// ## Type Parameters
    ///
    /// - `SrcType`: Source element type (e.g., `f32`, `f16`, `i32`)
    /// - `DstType`: Destination element type
    /// - `S`: Partition shape (1D)
    ///
    /// ## Parameters
    ///
    /// - `src`: Source tensor with dynamic shape (read-only)
    /// - `dst`: Destination tensor (mutable, partitioned)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// use cutile::kernels::conversion::convert;
    ///
    /// // Convert f32 to f16 — unified launcher, no zip! needed.
    /// let src_f32: Arc<Tensor<f32>> = api::arange(1024).await?.into();
    /// let dst_f16 = api::zeros::<f16>(&[1024]).partition([128]);
    /// let (src, dst) = convert(src_f32, dst_f16).await?;
    ///
    /// // Convert i32 to f32
    /// let src_i32: Arc<Tensor<i32>> = api::arange(1024).await?.into();
    /// let dst_f32 = api::zeros::<f32>(&[1024]).partition([128]);
    /// let (src, dst) = convert(src_i32, dst_f32).await?;
    /// ```
    ///
    /// ## Supported Conversions
    ///
    /// All conversions between the following types are supported:
    /// - `f16`, `f32`, `f64`
    /// - `i8`, `i16`, `i32`, `i64`
    /// - `u8`, `u16`, `u32`, `u64`
    ///
    /// ## Performance Notes
    ///
    /// - Type conversion is performed in registers (very fast)
    /// - Memory bandwidth is the primary bottleneck
    /// - Larger partition sizes improve memory access efficiency
    #[crate::entry()]
    pub fn convert<SrcType: ElementType, DstType: ElementType, const S: [i32; 1]>(
        src: &Tensor<SrcType, { [-1] }>,
        dst: &mut Tensor<DstType, S>,
    ) {
        let src_tile: Tile<SrcType, S> = load_tile_like(src, dst);
        let dst_tile: Tile<DstType, S> = convert_tile(src_tile);
        dst.store(dst_tile);
    }
}

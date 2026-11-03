/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tiling utilities for tcgen05.mma shared memory layouts.
//!
//! tcgen05 (Tensor Core Gen 5, Blackwell) requires specific tiled layouts in shared memory:
//! - **K-major**: within 8×8 tile = row-major, tiles arranged column-major (down then right)
//! - **MN-major**: within 8×8 tile = column-major, tiles arranged column-major (down then right)
//!
//! ## Key Insight
//!
//! Testing with Mojo revealed that Mojo's `tile_layout_mn_major` does NOT match NVIDIA's
//! tcgen05 MN-major format - Mojo arranges tiles in row-major order while NVIDIA uses
//! column-major tile ordering for both K-major and MN-major.
//!
//! These functions transform standard row-major matrices to tcgen05-compatible tiled formats.
//!
//! ## Layout Visualizations
//!
//! For a 16×16 bf16/f16 matrix with 8×8 tiles:
//!
//! ### K-major (NVIDIA Figure 188)
//! ```text
//! Layout: (((8, 2), (8, 2)):((8, 64), (1, 128)))
//!
//! Position (0,0) = 0      Position (0,8) = 128
//! Position (1,0) = 8      Position (8,0) = 64
//! Position (0,1) = 1
//!
//! Tiles arranged column-major:
//!   [Tile 0: 0-63  ] [Tile 2: 128-191]
//!   [Tile 1: 64-127] [Tile 3: 192-255]
//!
//! Within each tile: row-major (K-dimension contiguous)
//! ```
//!
//! ### MN-major (NVIDIA Figure 190)
//! ```text
//! Layout: (((8, 2), (8, 2)):((1, 64), (8, 128)))
//!
//! Position (0,0) = 0      Position (0,8) = 128
//! Position (1,0) = 1      Position (8,0) = 64
//! Position (0,1) = 8
//!
//! Tiles arranged column-major (same as K-major):
//!   [Tile 0: 0-63  ] [Tile 2: 128-191]
//!   [Tile 1: 64-127] [Tile 3: 192-255]
//!
//! Within each tile: column-major (M/N-dimension contiguous)
//! ```

use half::f16;

/// Tile size for tcgen05 with 16-bit types (f16/bf16).
///
/// This is fixed at 8×8 elements for tcgen05 tensor core operations.
pub const TILE_SIZE: usize = 8;

/// Transform row-major matrix to tcgen05 K-major layout (no swizzle).
///
/// K-major layout is used for **matrix A** in the computation `C = A × B`.
///
/// # Arguments
/// * `src` - Source matrix in row-major layout
/// * `dst` - Destination buffer for K-major layout
/// * `rows` - Number of rows (must be multiple of 8)
/// * `cols` - Number of columns (must be multiple of 8)
///
/// # Panics
/// Panics if dimensions are not multiples of [`TILE_SIZE`] or buffer sizes don't match.
///
/// # Example
/// ```
/// use cuda_host::tiling::{to_k_major_f16, TILE_SIZE};
/// use half::f16;
///
/// let rows = 16;
/// let cols = 16;
/// let src: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
/// let mut dst = vec![f16::ZERO; 256];
///
/// to_k_major_f16(&src, &mut dst, rows, cols);
///
/// // Element at logical position (0, 8) is now at linear index 128
/// assert_eq!(dst[128].to_f32(), 8.0);
/// ```
pub fn to_k_major_f16(src: &[f16], dst: &mut [f16], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols, "Source size mismatch");
    assert_eq!(dst.len(), rows * cols, "Destination size mismatch");
    assert_eq!(rows % TILE_SIZE, 0, "Rows must be multiple of {TILE_SIZE}");
    assert_eq!(cols % TILE_SIZE, 0, "Cols must be multiple of {TILE_SIZE}");

    let tiles_per_row = cols / TILE_SIZE;
    let tiles_per_col = rows / TILE_SIZE;

    for tile_row in 0..tiles_per_col {
        for tile_col in 0..tiles_per_row {
            // Tile index in COLUMN-MAJOR order (down then right)
            let tile_idx = tile_col * tiles_per_col + tile_row;
            let tile_base = tile_idx * TILE_SIZE * TILE_SIZE;

            for row_in_tile in 0..TILE_SIZE {
                for col_in_tile in 0..TILE_SIZE {
                    // Source: row-major
                    let src_row = tile_row * TILE_SIZE + row_in_tile;
                    let src_col = tile_col * TILE_SIZE + col_in_tile;
                    let src_idx = src_row * cols + src_col;

                    // Dest: K-major (row-major within tile)
                    let dst_idx = tile_base + row_in_tile * TILE_SIZE + col_in_tile;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

/// Transform row-major matrix to tcgen05 MN-major layout (no swizzle).
///
/// MN-major layout is used for **matrix B** in `C = A × B` when B is NOT pre-transposed.
///
/// # Arguments
/// * `src` - Source matrix in row-major layout
/// * `dst` - Destination buffer for MN-major layout
/// * `rows` - Number of rows (must be multiple of 8)
/// * `cols` - Number of columns (must be multiple of 8)
///
/// # Panics
/// Panics if dimensions are not multiples of [`TILE_SIZE`] or buffer sizes don't match.
///
/// # Example
/// ```
/// use cuda_host::tiling::{to_mn_major_f16, TILE_SIZE};
/// use half::f16;
///
/// let rows = 16;
/// let cols = 16;
/// let src: Vec<f16> = (0..256).map(|i| f16::from_f32(i as f32)).collect();
/// let mut dst = vec![f16::ZERO; 256];
///
/// to_mn_major_f16(&src, &mut dst, rows, cols);
///
/// // Element at logical position (0, 8) is now at linear index 128
/// assert_eq!(dst[128].to_f32(), 8.0);
/// ```
pub fn to_mn_major_f16(src: &[f16], dst: &mut [f16], rows: usize, cols: usize) {
    assert_eq!(src.len(), rows * cols, "Source size mismatch");
    assert_eq!(dst.len(), rows * cols, "Destination size mismatch");
    assert_eq!(rows % TILE_SIZE, 0, "Rows must be multiple of {TILE_SIZE}");
    assert_eq!(cols % TILE_SIZE, 0, "Cols must be multiple of {TILE_SIZE}");

    let tiles_per_row = cols / TILE_SIZE;
    let tiles_per_col = rows / TILE_SIZE;

    for tile_row in 0..tiles_per_col {
        for tile_col in 0..tiles_per_row {
            // Tile index in COLUMN-MAJOR order (down then right)
            let tile_idx = tile_col * tiles_per_col + tile_row;
            let tile_base = tile_idx * TILE_SIZE * TILE_SIZE;

            for row_in_tile in 0..TILE_SIZE {
                for col_in_tile in 0..TILE_SIZE {
                    // Source: row-major
                    let src_row = tile_row * TILE_SIZE + row_in_tile;
                    let src_col = tile_col * TILE_SIZE + col_in_tile;
                    let src_idx = src_row * cols + src_col;

                    // Dest: MN-major (column-major within tile)
                    let dst_idx = tile_base + col_in_tile * TILE_SIZE + row_in_tile;

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

/// Compute the linear index in K-major layout for a given (row, col).
///
/// This is useful for verification and debugging.
///
/// # Example
/// ```
/// use cuda_host::tiling::k_major_index;
///
/// // For a 16×16 matrix:
/// assert_eq!(k_major_index(0, 0, 16, 16), 0);
/// assert_eq!(k_major_index(0, 8, 16, 16), 128);  // Horizontal tile boundary
/// assert_eq!(k_major_index(8, 0, 16, 16), 64);   // Vertical tile boundary
/// ```
pub fn k_major_index(row: usize, col: usize, total_rows: usize, _total_cols: usize) -> usize {
    let tiles_per_col = total_rows / TILE_SIZE;

    let tile_row = row / TILE_SIZE;
    let tile_col = col / TILE_SIZE;
    let row_in_tile = row % TILE_SIZE;
    let col_in_tile = col % TILE_SIZE;

    // Column-major tile ordering
    let tile_idx = tile_col * tiles_per_col + tile_row;
    let tile_base = tile_idx * TILE_SIZE * TILE_SIZE;

    // Row-major within tile
    tile_base + row_in_tile * TILE_SIZE + col_in_tile
}

/// Compute the linear index in MN-major layout for a given (row, col).
///
/// This is useful for verification and debugging.
///
/// # Example
/// ```
/// use cuda_host::tiling::mn_major_index;
///
/// // For a 16×16 matrix:
/// assert_eq!(mn_major_index(0, 0, 16, 16), 0);
/// assert_eq!(mn_major_index(0, 8, 16, 16), 128);  // Horizontal tile boundary
/// assert_eq!(mn_major_index(8, 0, 16, 16), 64);   // Vertical tile boundary
/// assert_eq!(mn_major_index(0, 1, 16, 16), 8);    // Next column within tile
/// ```
pub fn mn_major_index(row: usize, col: usize, total_rows: usize, _total_cols: usize) -> usize {
    let tiles_per_col = total_rows / TILE_SIZE;

    let tile_row = row / TILE_SIZE;
    let tile_col = col / TILE_SIZE;
    let row_in_tile = row % TILE_SIZE;
    let col_in_tile = col % TILE_SIZE;

    // Column-major tile ordering
    let tile_idx = tile_col * tiles_per_col + tile_row;
    let tile_base = tile_idx * TILE_SIZE * TILE_SIZE;

    // Column-major within tile
    tile_base + col_in_tile * TILE_SIZE + row_in_tile
}

/// Print a visual representation of the layout indices for a matrix.
///
/// Useful for debugging and understanding layout transformations.
///
/// # Example
/// ```
/// use cuda_host::tiling::{print_layout_indices, k_major_index};
///
/// // Print K-major layout for 8×8 matrix
/// print_layout_indices(8, 8, k_major_index);
/// ```
pub fn print_layout_indices<F>(rows: usize, cols: usize, index_fn: F)
where
    F: Fn(usize, usize, usize, usize) -> usize,
{
    // Header row with column indices
    print!("      ");
    for c in 0..cols {
        print!("{:>5} ", c);
    }
    println!();
    println!("    +{}", "-----+".repeat(cols));

    for r in 0..rows {
        print!("{:>2}  |", r);
        for c in 0..cols {
            let idx = index_fn(r, c, rows, cols);
            print!("{:>4} |", idx);
        }
        println!();
        println!("    +{}", "-----+".repeat(cols));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_major_16x16() {
        println!("\n=== K-major 16x16 (NVIDIA tcgen05 format) ===");
        print_layout_indices(16, 16, k_major_index);

        // Verify key positions match NVIDIA docs
        assert_eq!(k_major_index(0, 0, 16, 16), 0);
        assert_eq!(k_major_index(0, 8, 16, 16), 128); // Horizontal tile = +128
        assert_eq!(k_major_index(8, 0, 16, 16), 64); // Vertical tile = +64
        assert_eq!(k_major_index(1, 0, 16, 16), 8); // Next row in tile = +8
        assert_eq!(k_major_index(0, 1, 16, 16), 1); // Next col in row = +1
    }

    #[test]
    fn test_mn_major_16x16() {
        println!("\n=== MN-major 16x16 (NVIDIA tcgen05 format) ===");
        print_layout_indices(16, 16, mn_major_index);

        // Verify key positions match NVIDIA docs
        assert_eq!(mn_major_index(0, 0, 16, 16), 0);
        assert_eq!(mn_major_index(0, 8, 16, 16), 128); // Horizontal tile = +128
        assert_eq!(mn_major_index(8, 0, 16, 16), 64); // Vertical tile = +64
        assert_eq!(mn_major_index(1, 0, 16, 16), 1); // Next row in col = +1
        assert_eq!(mn_major_index(0, 1, 16, 16), 8); // Next col in tile = +8
    }

    #[test]
    fn test_to_k_major_transform() {
        let rows = 16;
        let cols = 16;
        let src: Vec<f16> = (0..rows * cols).map(|i| f16::from_f32(i as f32)).collect();
        let mut dst = vec![f16::ZERO; rows * cols];

        to_k_major_f16(&src, &mut dst, rows, cols);

        // Element at (0, 8) in row-major is at index 8
        // In K-major it should be at index 128
        assert_eq!(dst[128].to_f32(), 8.0);

        // Element at (8, 0) in row-major is at index 128
        // In K-major it should be at index 64
        assert_eq!(dst[64].to_f32(), 128.0);
    }

    #[test]
    fn test_to_mn_major_transform() {
        let rows = 16;
        let cols = 16;
        let src: Vec<f16> = (0..rows * cols).map(|i| f16::from_f32(i as f32)).collect();
        let mut dst = vec![f16::ZERO; rows * cols];

        to_mn_major_f16(&src, &mut dst, rows, cols);

        // Element at (0, 8) in row-major is at index 8
        // In MN-major it should be at index 128
        assert_eq!(dst[128].to_f32(), 8.0);

        // Element at (8, 0) in row-major is at index 128
        // In MN-major it should be at index 64
        assert_eq!(dst[64].to_f32(), 128.0);
    }

    #[test]
    fn test_128x16_matrix() {
        // Test with actual matrix A dimensions
        let rows = 128;
        let cols = 16;
        let src: Vec<f16> = (0..rows * cols).map(|i| f16::from_f32(i as f32)).collect();
        let mut dst = vec![f16::ZERO; rows * cols];

        to_k_major_f16(&src, &mut dst, rows, cols);

        // Verify a few key positions
        assert_eq!(k_major_index(0, 0, rows, cols), 0);
        assert_eq!(k_major_index(0, 8, rows, cols), 1024); // 16 tiles × 64 elements
        assert_eq!(k_major_index(8, 0, rows, cols), 64);
    }
}

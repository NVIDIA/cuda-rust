/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `block_tile_ptr` on tensors of more than `i32::MAX` elements. Each tile
//! block writes a marker byte through its pointer to the first element of
//! its tile; the host checks that every tile, on both sides of the
//! 2^31-element boundary, got its own marker. An `i32` offset wraps at the
//! boundary and the pointer lands before the allocation.
//!
//! Each test allocates a little over 2 GiB of device memory.

use cutile::api;
use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::error::Error;
use cutile::prelude::*;
use std::sync::Arc;

/// Elements in one tile: a full row of the innermost axis.
const ROW: usize = 65536;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// `(row mod 251) + 1`: never 0, and different for adjacent rows.
    fn marker(row: i32) -> Tile<u8, { [] }> {
        let value: i32 = row % 251i32 + 1i32;
        let value: Tile<i32, { [] }> = scalar_to_tile(value);
        trunci(value, overflow::NoUnsignedWrap)
    }

    /// Writes `marker` to the first element of this block's tile.
    unsafe fn write_marker(ptr: *mut u8, marker: Tile<u8, { [] }>) {
        let dst: PointerTile<*mut u8, { [] }> = pointer_to_tile(ptr);
        let _token: Token = store_ptr_tko(
            dst,
            marker,
            ordering::Relaxed,
            Some(scope::Device),
            None,
            None,
            Latency::<0>,
        );
    }

    /// Block `i` owns row `i`: offset `i × 65536`.
    #[cutile::entry()]
    fn mark_rank_2_block_tiles(y: &mut Tensor<u8, { [1, 65536] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ptr: *mut u8 = y.block_tile_ptr();
        // SAFETY: the tile is this block's own row of `y`, and the row holds
        // at least one element.
        unsafe { write_marker(ptr, marker(pid.0)) };
    }

    /// Block `(i, j)` owns row `i × rows + j`: offset
    /// `i × rows × 65536 + j × 65536`.
    #[cutile::entry()]
    fn mark_rank_3_block_tiles(y: &mut Tensor<u8, { [1, 1, 65536] }>, rows: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ptr: *mut u8 = y.block_tile_ptr();
        // SAFETY: as for the rank-2 kernel.
        unsafe { write_marker(ptr, marker(pid.0 * rows + pid.1)) };
    }
}

fn marker(row: usize) -> u8 {
    (row % 251 + 1) as u8
}

/// A zeroed `u32` tensor, the storage of a byte tensor of more than
/// `i32::MAX` elements: no tensor may have that many elements along one
/// axis, and `api::zeros` stops at `i32::MAX` elements in total.
struct Words(Arc<Tensor<u32>>);

// SAFETY: the pointer, size and context are those of the tensor `Words`
// owns, which lives as long as `Words`.
unsafe impl DeviceAllocation for Words {
    fn context(&self) -> Arc<cuda_core::Context> {
        self.0.context().clone()
    }
    fn device_ptr(&self) -> u64 {
        self.0.device_pointer().cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.0.num_bytes()
    }
    fn device_id(&self) -> usize {
        self.0.device_id()
    }
}

/// A zeroed contiguous byte tensor of `shape`, and its storage.
fn zeroed_bytes(shape: &[usize]) -> Result<(Tensor<u8>, Arc<Words>), Error> {
    let len: usize = shape.iter().product();
    let words = Arc::new(Words(Arc::new(
        api::zeros::<u32>(&[len.div_ceil(4)]).sync()?,
    )));
    let mut strides = vec![1i32; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1] as i32;
    }
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    // SAFETY: only this tensor accesses the bytes until the kernel finished;
    // the host reads them back through `words` afterwards. Every byte is a
    // valid `u8`.
    let bytes = unsafe { Tensor::<u8>::from_foreign(words.clone(), shape, strides) };
    Ok((bytes, words))
}

/// Reads `words` back after the kernel and checks each row's marker, and
/// that the rows from two before `boundary_row` to the last hold nothing
/// else. Rows start at a multiple of 4 bytes, so a row's marker is the low
/// byte of a word (little-endian).
fn check_rows(words: &Words, boundary_row: usize) -> Result<(), Error> {
    const ROW_WORDS: usize = ROW / 4;
    let words: Vec<u32> = (&words.0).to_host_vec().sync()?;
    assert_eq!(words.len() % ROW_WORDS, 0);
    let rows = words.len() / ROW_WORDS;
    assert!(
        rows > boundary_row,
        "{rows} rows do not cross row {boundary_row}"
    );
    for row in 0..rows {
        assert_eq!(words[row * ROW_WORDS], u32::from(marker(row)), "row {row}");
    }
    for row in boundary_row - 2..rows {
        let row_words = &words[row * ROW_WORDS + 1..(row + 1) * ROW_WORDS];
        assert!(row_words.iter().all(|&w| w == 0), "row {row}");
    }
    Ok(())
}

/// The reviewer's case: a `[32769, 65536]` byte tensor in `[1, 65536]`
/// tiles. Block 32768 starts at element 2^31, one past `i32::MAX`; the
/// single-axis product `32768 × 1 × 65536` overflows `i32`.
#[test]
fn rank_2_block_tile_pointers_cross_the_i32_boundary() -> Result<(), Error> {
    let rows = (1usize << 31) / ROW + 1;
    let (y, words) = zeroed_bytes(&[rows, ROW])?;
    let (y,) = kernels::mark_rank_2_block_tiles(y.partition([1, ROW])).sync()?;
    drop(y);
    check_rows(&words, (1usize << 31) / ROW)
}

/// A `[2, 16385, 65536]` tensor in `[1, 1, 65536]` tiles. Every per-axis
/// term fits in `i32`, but block `(1, 16383)` starts at element
/// `(16385 + 16383) × 65536 = 2^31`: the sum overflows `i32`.
#[test]
fn rank_3_block_tile_pointers_cross_the_i32_boundary() -> Result<(), Error> {
    let rows = (1usize << 30) / ROW + 1;
    let (y, words) = zeroed_bytes(&[2, rows, ROW])?;
    let (y, _) = kernels::mark_rank_3_block_tiles(y.partition([1, 1, ROW]), rows as i32).sync()?;
    drop(y);
    check_rows(&words, (1usize << 31) / ROW)
}

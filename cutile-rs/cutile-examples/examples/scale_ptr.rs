/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates PointerTile reshape/broadcast and for-loop step_by with grid stride.
//!
//! This example implements a pointer-based vector scale kernel that:
//! 1. Constructs pointer tiles using the reshape + broadcast pattern
//!    (matching the cuTile-python/nv-triton pointer scatter/gather idiom).
//! 2. Uses a persistent-kernel style grid-stride loop via step_by(grid.0 as usize).
//!
//! The reshape/broadcast pattern avoids the ptr-to-int/int-to-ptr workaround
//! that was previously required for PointerTile shape manipulation.

use cutile::api::{arange, zeros};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::future::IntoFuture;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn scale_ptr(out_ptr: *mut f32, in_ptr: *mut f32, scale: f32, len: i32) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        // Build pointer tiles using reshape + broadcast.
        // This is the standard cuTile-python/nv-triton pattern:
        //   base_ptr -> reshape [1] -> broadcast [TILE_SIZE]
        let in_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_1d: PointerTile<*mut f32, { [1] }> = in_base.reshape(shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_1d.broadcast(shape![128]);

        let out_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_1d: PointerTile<*mut f32, { [1] }> = out_base.reshape(shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_1d.broadcast(shape![128]);

        let scale_tile: Tile<f32, { [128] }> = broadcast_scalar(scale, shape![128]);
        let len_tile: Tile<i32, { [128] }> = broadcast_scalar(len, shape![128]);

        // Grid-stride loop: each block starts at pid.0 * 128 and steps by grid.0 * 128.
        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..len).step_by(step as usize) {
            let offsets: Tile<i32, { [128] }> =
                iota(shape![128]) + broadcast_scalar(offset, shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(offsets, len_tile);

            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(offsets);
            let result: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let tile: Tile<f32, { [128] }> = result.0;

            let scaled: Tile<f32, { [128] }> = tile * scale_tile;

            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(offsets);
            store_ptr_tko(
                dst,
                scaled,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
}

use my_module::scale_ptr;

#[tokio::main()]
async fn main() -> Result<(), cutile::error::Error> {
    if !cutile_examples::requirements::BASELINE.check("scale_ptr", 0)? {
        return Ok(());
    }

    let len = 1024usize;
    let scale = 3.0f32;

    let input: Tensor<f32> = arange(len).await?;
    let output: Tensor<f32> = zeros(&[len]).await?;

    let out_ptr = output.device_pointer();
    let in_ptr = input.device_pointer();

    // Multiple blocks process elements via the grid-stride loop.
    let num_blocks = 4u32;
    let op = unsafe { scale_ptr(out_ptr, in_ptr, scale, len as i32) }.grid((num_blocks, 1, 1));
    tokio::spawn(op.into_future())
        .await
        .expect("Failed to execute tokio task.")?;

    let input_host = input.to_host_vec().await?;
    let output_host = output.to_host_vec().await?;
    for i in 0..len {
        let expected = input_host[i] * scale;
        assert_eq!(
            output_host[i], expected,
            "Mismatch at index {}: {} != {}",
            i, output_host[i], expected
        );
    }
    println!(
        "Scaled {} elements by {}. First 8: {:?} -> {:?}",
        len,
        scale,
        &input_host[..8],
        &output_host[..8]
    );
    Ok(())
}

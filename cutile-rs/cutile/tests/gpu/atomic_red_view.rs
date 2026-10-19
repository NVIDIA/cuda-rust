/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value test: `atomic_red_view_tko` accumulates concurrently into one tile.
//!
//! Every block atomically adds a ones-tile into tile [0, 0] of the output;
//! that tile must equal the block count in every element (a non-atomic
//! store would lose updates), and every other tile must stay zero.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod atomic_red_kernels {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    unsafe fn atomic_red_accumulate(out: &Tensor<f32, { [-1, -1] }>) {
        let mut out_part: PartitionMut<f32, { [16, 16] }> =
            unsafe { out.partition_full_mut(shape![16, 16]) };
        let tile: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let _tok: Token = unsafe {
            atomic_red_view_tko(
                &mut out_part,
                tile,
                [0i32, 0i32],
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
            )
        };
    }
}

use atomic_red_kernels::atomic_red_accumulate;

#[test]
fn atomic_red_view_accumulates_across_blocks() {
    common::with_test_stack(|| {
        if !common::supports_tile_ir(&[cutile_ir::requirements::Feature::AtomicRedView]) {
            return;
        }
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        const BLOCKS: usize = 32;
        let out = api::zeros::<f32>(&[512, 16])
            .sync_on(&stream)
            .expect("zeros");
        unsafe { atomic_red_accumulate(&out) }
            .grid((BLOCKS as u32, 1, 1))
            .sync_on(&stream)
            .expect("launch");
        let host: Vec<f32> = out.dup().to_host_vec().sync_on(&stream).expect("to_host");
        assert_eq!(host.len(), 512 * 16);
        let (first_tile, rest) = host.split_at(16 * 16);
        assert!(
            first_tile.iter().all(|&v| v == BLOCKS as f32),
            "expected tile [0, 0] to accumulate to {BLOCKS}, got {:?}",
            &first_tile[..8]
        );
        assert!(
            rest.iter().all(|&v| v == 0.0),
            "expected all other tiles untouched"
        );
    });
}

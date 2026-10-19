/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: borrow-based kernel API. Pass `&Tensor` for inputs and
//! `Partition<&mut Tensor>` for outputs; the kernel writes in place.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

use my_module::add;

#[test]
fn smoke_add_refs() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let x = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let y = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut z = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();

        // Borrow-based launch: &x, &y for inputs, &mut z for output.
        let _ = add((&mut z).partition([4]), &x, &y)
            .sync_on(&stream)
            .unwrap();

        let z_host: Vec<f32> = z.dup().to_host_vec().sync_on(&stream).unwrap();
        assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));

        // Run again — reuse the same buffers, no allocation.
        let _ = add((&mut z).partition([4]), &x, &y)
            .sync_on(&stream)
            .unwrap();

        let z_host: Vec<f32> = z.to_host_vec().sync_on(&stream).unwrap();
        assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    });
}

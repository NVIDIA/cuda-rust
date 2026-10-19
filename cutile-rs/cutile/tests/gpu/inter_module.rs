/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: inter-module device function calls. Device functions defined
//! in one `#[cutile::module]` are imported and called from another; the JIT
//! inlines them into the entry point.

use cutile::prelude::*;

use crate::common;

/// Module A: reusable activation device functions. No entry points.
#[cutile::module]
mod activations {
    use cutile::core::*;

    pub fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        max_tile(x, zero)
    }

    pub fn square<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        x * x
    }
}

/// Module B: kernels that use device functions from Module A.
#[cutile::module]
mod my_kernels {
    use super::activations::{relu, square};
    use cutile::core::*;

    #[cutile::entry()]
    fn apply_relu_square<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(shape!(S), [pid.0]);
        let activated: Tile<f32, S> = relu(tile);
        output.store(square(activated));
    }

    #[cutile::entry()]
    fn apply_relu<const S: [i32; 1]>(output: &mut Tensor<f32, S>, input: &Tensor<f32, { [-1] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(shape!(S), [pid.0]);
        output.store(relu(tile));
    }
}

use my_kernels::{apply_relu, apply_relu_square};

#[test]
fn smoke_inter_module() {
    common::with_test_stack(|| {
        let device: std::sync::Arc<cuda_core::Device> = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");
        let block: usize = 64;
        let n: usize = 128;

        let input: Tensor<f32> = api::arange::<f32>(n).sync_on(&stream).expect("arange");

        // -- relu_square: calls relu() and square() from Module A --
        let mut rs_out: Tensor<f32> = api::zeros::<f32>(&[n]).sync_on(&stream).expect("zeros");
        apply_relu_square((&mut rs_out).partition([block]), &input)
            .sync_on(&stream)
            .expect("apply_relu_square");
        let rs_host: Vec<f32> = rs_out.dup().to_host_vec().sync_on(&stream).expect("host");
        for (i, &v) in rs_host.iter().enumerate() {
            let expected: f32 = (i as f32) * (i as f32);
            assert!(
                (v - expected).abs() < 1.0,
                "relu_square[{i}]: got {v}, expected {expected}"
            );
        }

        // -- relu only: calls relu() from Module A --
        let mut relu_out: Tensor<f32> = api::zeros::<f32>(&[n]).sync_on(&stream).expect("zeros");
        apply_relu((&mut relu_out).partition([block]), &input)
            .sync_on(&stream)
            .expect("apply_relu");
        let relu_host: Vec<f32> = relu_out.dup().to_host_vec().sync_on(&stream).expect("host");
        for (i, &v) in relu_host.iter().enumerate() {
            let expected: f32 = i as f32;
            assert!(
                (v - expected).abs() < 1e-3,
                "relu[{i}]: got {v}, expected {expected}"
            );
        }
    });
}

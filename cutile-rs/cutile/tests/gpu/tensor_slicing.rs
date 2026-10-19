/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: tensor views and slicing. Borrows subregions of a tensor
//! and passes them to GPU kernels without copying data.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn scale<const B: i32>(out: &mut Tensor<f32, { [B] }>, a: &Tensor<f32, { [-1] }>, scalar: f32) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let s: Tile<f32, { [B] }> = scalar.broadcast(out.shape());
        out.store(tile_a * s);
    }
}

use my_module::{add, scale};

#[test]
fn smoke_tensor_slicing() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");
        let block = 128;

        let data = api::arange::<f32>(1024).sync_on(&stream).expect("arange");

        // -- Single slice [256..512] + add ones --
        let slice_a = data.slice(&[256..512]).expect("slice");
        let ones = api::ones::<f32>(&[256]).sync_on(&stream).expect("ones");
        let mut out = api::zeros::<f32>(&[256]).sync_on(&stream).expect("zeros");
        add((&mut out).partition([block]), &slice_a, &ones)
            .sync_on(&stream)
            .expect("add");
        let host: Vec<f32> = out.dup().to_host_vec().sync_on(&stream).expect("host");
        assert!((host[0] - 257.0).abs() < 1e-3);
        assert!((host[255] - 512.0).abs() < 1e-3);

        // -- Chained slices: data[128..896][128..384] == data[256..512] --
        let outer = data.slice(&[128..896]).expect("outer slice");
        let inner = outer.slice(&[128..384]).expect("inner slice");
        let mut out2 = api::zeros::<f32>(&[256]).sync_on(&stream).expect("zeros");
        scale((&mut out2).partition([block]), &inner, 2.0)
            .sync_on(&stream)
            .expect("scale");
        let host2: Vec<f32> = out2.dup().to_host_vec().sync_on(&stream).expect("host");
        assert!((host2[0] - 512.0).abs() < 1e-3);
        assert!((host2[255] - 1022.0).abs() < 1e-3);

        // -- View + slice (2D) --
        let matrix = data.view(&[32, 32]).expect("view");
        let row_slice = matrix.slice(&[8..16]).expect("row slice");
        assert_eq!(row_slice.shape(), &[8, 32]);
    });
}

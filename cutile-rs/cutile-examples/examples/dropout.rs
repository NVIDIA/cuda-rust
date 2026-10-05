/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{rand, randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn dropout<const S: [i32; 1]>(
        p: f32,
        _x: &Tensor<f32, { [-1] }>,
        x_keep: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, S>,
    ) {
        let zeros: Tile<f32, S> = constant(0.0, out.shape());
        let ones: Tile<f32, S> = constant(1.0, out.shape());
        let p_tile = p.broadcast(out.shape());
        let x_keep_tile = x_keep.load_like(out);
        // x_keep_tile is the probability of keeping (higher is more likely).
        // p_tile is the probability of dropout (lower keeps more values).
        let out_tile = select(gt_tile(x_keep_tile, p_tile), x_keep_tile, zeros);
        // Rescale outputs based on dropout probability.
        let out_tile = out_tile / (ones - p_tile);
        out.store(out_tile);
    }
}

use my_module::dropout;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("dropout", 0)? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let (m,) = (16,);
    let bm = 4;
    let p: f32 = 0.4;
    let seed = 123;
    let x: Arc<Tensor<f32>> = randn(0.0, 1.0, [m], Some(seed)).sync_on(&stream)?.into();
    let x_keep: Arc<Tensor<f32>> = rand([m], Some(seed)).sync_on(&stream)?.into();
    let out: Partition<Tensor<f32>> = zeros(&[m]).sync_on(&stream)?.partition([bm]);
    let (_, _x, _x_keep, out) = dropout(p, x, x_keep, out).sync_on(&stream)?;
    let out_host: Vec<f32> = out.unpartition().to_host_vec().sync_on(&stream)?;
    println!("{out_host:?}");
    Ok(())
}

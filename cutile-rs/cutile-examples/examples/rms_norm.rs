/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        // The launch grid is (M, 1, 1).
        // row is a pid in [0, M).
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        // TODO (hme): Parse 0.0f32 syntax properly.
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            rms = rms + tx * tx;
        }
        // TODO (hme): Try to make this something like:
        //  let rms = (1.0 / (rms.sum(/*axis=*/1, /*keepdims=*/true) / N + eps).sqrt()).broadcast(tile_shape);
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let rms: f32 = 1.0f32 / (rms / n + eps);
        let rms: Tile<f32, { [] }> =
            sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
        let rms: f32 = tile_to_scalar(rms);
        let rms: Tile<f32, { [1, BLOCK_SIZE] }> = rms.broadcast(tile_shape);

        let w_part: Partition<f32, { [BLOCK_SIZE] }> = w.partition(shape![BLOCK_SIZE]);
        // TODO (hme): This is a safety leak. If this partition goes out of scope, we can partition out again,
        //  and any memory ops will not succeed tokens corresponding to write operations (since those will also be dropped).
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * rms * tw;
            out_part.store(tout, [0i32, j]);
        }
    }
}

use my_module::rms_norm;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("rms_norm", 0)? {
        return Ok(());
    }

    // Create a context. Device 0 is associated with the context.
    let device = Device::new(0)?;
    // Create a new stream on which we run CUDA operations.
    let stream = device.new_stream()?;
    let (m, n) = (4, 8);
    let block_size = 2;
    let generics = vec![n.to_string(), block_size.to_string()];
    let eps: f32 = 1e-8; // A sufficiently small number.
    let x: Arc<Tensor<f32>> = randn(0.0, 1.0, [m, n], None).sync_on(&stream)?.into();
    let w: Arc<Tensor<f32>> = randn(0.0, 1.0, [n], None).sync_on(&stream)?.into();
    let out: Partition<Tensor<f32>> = zeros(&[m, n]).sync_on(&stream)?.partition([1, n]);
    let (_x, _w, out, _eps) = rms_norm(x, w, out, eps)
        .generics(generics)
        .sync_on(&stream)?;
    let out_host: Vec<f32> = out.unpartition().to_host_vec().sync_on(&stream)?;
    for i in (0..out_host.len()).step_by(8) {
        let x = out_host[i..i + 8].to_vec();
        println!("{x:?}");
        // let sum: f32 = x.iter().sum();
        // println!("layer_norm(x).sum(axis=1) = {}", sum);
        // assert!(sum.epsilon_close(1.0));
    }
    Ok(())
}

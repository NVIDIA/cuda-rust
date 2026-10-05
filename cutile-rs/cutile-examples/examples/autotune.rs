/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Autotune a kernel's block size with `cutile::tune` (experimental).
//!
//! Run with:
//! ```sh
//! cargo run -p cutile-examples --example autotune --features experimental-tune
//! ```

use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor};
use cutile::tile_kernel::TileKernel;
use cutile::tune::{Autotuner, Config, ParamValue};
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    /// Row-wise RMS normalization; BLOCK_SIZE is the tuned axis.
    #[cutile::entry()]
    fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            rms = rms + tx * tx;
        }
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
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * rms * tw;
            unsafe { out_part.store(tout, [0i32, j]) };
        }
    }
}

use my_module::rms_norm;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("autotune", 0)? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let (m, n) = (512usize, 1024usize);
    let eps: f32 = 1e-8;

    // Shared inputs; each candidate gets its own output buffer.
    let x: Arc<Tensor<f32>> = randn(0.0, 1.0, [m, n], None).sync_on(&stream)?.into();
    let w: Arc<Tensor<f32>> = randn(0.0, 1.0, [n], None).sync_on(&stream)?.into();

    // The search space: one block size per candidate. 4096 exceeds N and is
    // pruned before it is ever visited.
    let configs: Vec<Config> = [32i64, 64, 128, 256, 4096]
        .into_iter()
        .map(|bs| Config::new([("BLOCK_SIZE", ParamValue::Int(bs))]))
        .collect();

    let output = Autotuner::new("rms_norm")
        .configs(configs)
        .prune(move |c| c.int("BLOCK_SIZE").unwrap_or(0) <= n as i64)
        .run(&stream, |stream, config| {
            let block_size = config.int("BLOCK_SIZE").unwrap();
            let generics = vec![n.to_string(), block_size.to_string()];
            let out = zeros(&[m, n]).sync_on(stream)?.partition([1, n]);
            let (x, w) = (x.clone(), w.clone());
            let mut out_slot = Some(out);
            let mut launch = move |s: &Arc<cuda_core::Stream>| {
                let out = out_slot.take().expect("output buffer");
                let (_x, _w, out, _eps) = rms_norm(x.clone(), w.clone(), out, eps)
                    .generics(generics.clone())
                    .sync_on(s)?;
                out_slot = Some(out);
                Ok(())
            };
            // Gate: one launch catches compile and launch errors for this
            // candidate before any timing happens.
            launch(stream)?;
            Ok(launch)
        })?;

    for trial in &output.trials {
        println!("{}: {:?}", trial.config_id, trial.state);
    }
    let best = output.best.expect("a winner");
    println!("best: {}", best.id);
    Ok(())
}

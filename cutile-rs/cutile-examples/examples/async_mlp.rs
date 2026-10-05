/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_async::device_context::global_policy;
use cuda_async::device_operation::*;
use cutile::api::dup;
use cutile::tensor::{Partition, Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{PartitionOp, TileKernel};
use cutile::{api, error::Error};
use std::sync::Arc;
use tokio::task::JoinHandle;

#[cutile::module]
pub mod my_kernels {

    use cutile::core::*;

    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            // TODO (hme): Inject continue.
            continue;
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load().reshape(shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i]).reshape(shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

// Simulate loading input data.
fn load_data<const RANK: usize>(batch_size: [usize; RANK]) -> impl DeviceOp<Output = Tensor<f32>> {
    api::randn(0.0, 1.0, batch_size, None)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("async_mlp", 0)? {
        return Ok(());
    }

    // Get device scheduling policies.
    let num_devices = 4;
    let devices = {
        let mut r = vec![];
        for _ in 0..num_devices {
            // Pretend we have multiple devices...
            r.push(global_policy(0)?);
        }
        r
    };

    let dim = 16;
    let block_dim = 4;
    let fully_connected_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let output_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let w0 = api::randn(0.0f32, 1.0, [dim, dim], None); // impl DeviceOp
    let w1 = api::randn(0.0f32, 1.0, [dim], None); // impl DeviceOp
    let w: (Arc<Tensor<f32>>, Arc<Tensor<f32>>) = zip!(w0.map(Into::into), w1.map(Into::into))
        .schedule(devices.first().unwrap())?
        .await?;
    let mut joins = vec![];
    for device in devices.iter().skip(1) {
        let w_copy = tokio::spawn(
            zip!(dup(&w.0).map(Into::into), dup(&w.1).map(Into::into)).schedule(device)?,
        );
        joins.push(w_copy);
    }
    let mut model_weights = vec![w];
    for join in joins {
        model_weights.push(join.await.unwrap()?);
    }

    // Asynchronously compute forward pass for each batch of data on each device.
    let mut futures: Vec<
        JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>,
    > = vec![];
    for i in 0..num_devices {
        let w = &model_weights[i];
        let (w0, w1) = (w.0.clone(), w.1.clone());
        let data = load_data([dim, dim]);
        let out0 = api::zeros::<f32>(&[dim, dim]).partition([block_dim, block_dim]);
        let fully_connected_layer = fully_connected_layer.to_vec();
        let (out0, _, _) = my_kernels::gemm(out0, data, w0)
            .generics(fully_connected_layer)
            .unzip();
        let out1 = api::zeros::<f32>(&[dim]).partition([block_dim]);
        let output_layer = output_layer.to_vec();
        let (out1, _, _) = my_kernels::matvec(out1, out0.unpartition(), w1)
            .generics(output_layer)
            .unzip();
        let (out1,) = my_kernels::relu(out1).unzip();
        futures.push(tokio::spawn(out1.schedule(&devices[i])?));
    }

    // Wait on results.
    let mut outputs: Vec<Tensor<f32>> = vec![];
    for future in futures.into_iter() {
        let tensor = future.await.unwrap()?.unpartition();
        outputs.push(tensor);
    }
    for output in outputs {
        println!("{:?}", output.to_host_vec().await?);
    }
    Ok(())
}

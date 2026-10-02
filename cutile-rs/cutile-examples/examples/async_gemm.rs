/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::*;
use cutile::api;
use cutile::half::f16;
use cutile::tensor::{Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{PartitionOp, TileKernel};
use cutile::DType;
// Kernel function accessed via my_module::gemm below (not imported to avoid shadowing local fn gemm)
use std::sync::Arc;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<
        E1: ElementType,
        E2: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
    >(
        z: &mut Tensor<E1, { [BM, BN] }>,
        x: &Tensor<E2, { [-1, K] }>,
        y: &Tensor<E2, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z);
    }
}

fn gemm<T1: DType, T2: DType>(
    x: Arc<Tensor<T2>>,
    y: Arc<Tensor<T2>>,
) -> impl DeviceOp<Output = Tensor<T1>> {
    let (m, n, k) = (
        x.shape()[0] as usize,
        y.shape()[1] as usize,
        x.shape()[1] as usize,
    );
    let (bm, bn, bk) = (16, 16, 8);
    let generics = [
        T1::DTYPE.as_str().to_string(),
        T2::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];
    let z = api::zeros(&[m, n]).partition([bm, bn]); // impl DeviceOp
    let (z, _x, _y) = my_module::gemm(z, x, y).generics(generics.to_vec()).unzip();
    z.unpartition()
}

use cutile_examples::to_candle_tensor;

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), cuda_async::error::DeviceError> {
    if !cutile_examples::requirements::BASELINE.check("async_gemm", 0)? {
        return Ok(());
    }

    type In = f16;
    type Out = f32;
    let (m, n, k) = (64, 64, 16);
    let x: Arc<Tensor<In>> = api::randn_f16(In::zero(), In::one(), [m, k], None)
        .await?
        .into();
    let y: Arc<Tensor<In>> = api::randn_f16(In::zero(), In::one(), [k, n], None)
        .await?
        .into();
    let z = gemm::<Out, In>(x.clone(), y.clone()).await?;
    let z_host: Vec<Out> = z.to_host_vec().await?;
    let x_host: Vec<In> = x.to_host_vec().await?;
    let y_host: Vec<In> = y.to_host_vec().await?;
    let x_candle = to_candle_tensor(&x_host, &[m, k]);
    let y_candle = to_candle_tensor(&y_host, &[k, n]);
    let answer_host: Vec<f16> = x_candle
        .matmul(&y_candle)
        .unwrap()
        .reshape(((),))
        .unwrap()
        .to_vec1()
        .unwrap();
    for i in 0..(m * n) {
        println!(
            "z_host[{i}] == answer_host[{i}]? {} == {}",
            z_host[i], answer_host[i]
        );
    }
    Ok(())
}

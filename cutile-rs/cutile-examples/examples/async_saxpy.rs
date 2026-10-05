/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::value;
use cutile::api::{arange, DeviceOpReshape};
use cutile::error::Error;
use cutile::prelude::*;
use cutile::tensor::ToHostVec;
use cutile::tile_kernel::ToHostVecOp;
use cutile::DType;
use std::fmt::Display;
use std::ops::{Add, Mul};

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2], T: ElementType>(
        y: &mut Tensor<T, S>,
        a: T,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

use my_module::saxpy;

async fn execute<T: DType + Display + PartialEq + Mul<Output = T> + Add<Output = T>>(
    size: usize,
) -> Result<(), Error> {
    let a = <T as DType>::one() + <T as DType>::one();
    let x = arange(size).reshape(&[4, 8]);
    let y = arange(size).reshape(&[4, 8]);
    let y_host: Vec<T> = saxpy(y.partition([2, 4]), value(a), x)
        .first()
        .unpartition()
        .to_host_vec()
        .await?;
    let input_host: Vec<T> = arange(size).await?.to_host_vec().await?;
    for i in 0..input_host.len() {
        let x_i: T = input_host[i];
        let y_i: T = input_host[i];
        let answer = a * x_i + y_i;
        println!("{} * {} + {} = {}", a, x_i, y_i, y_host[i]);
        assert_eq!(answer, y_host[i], "{} != {} ?", answer, y_host[i]);
    }
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("async_saxpy", 0)? {
        return Ok(());
    }

    execute::<f32>(2usize.pow(5)).await?;
    execute::<f64>(2usize.pow(5)).await?;
    Ok(())
}

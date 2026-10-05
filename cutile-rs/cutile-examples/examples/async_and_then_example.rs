/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_context::global_policy;
use cuda_async::device_operation::{value, with_context, DeviceOp};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;
use cutile::api::arange;
use cutile::api::DeviceOpReshape;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::{compile_from_context, CompileOptions};
use my_module::__module_ast_self;
use std::sync::Arc;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(y: &mut Tensor<f32, S>, a: f32, x: &Tensor<f32, { [-1, -1] }>) {
        let tile_a: Tile<f32, S> = a.broadcast(y.shape());
        let tile_x: Tile<f32, S> = x.load_like(y);
        let tile_y: Tile<f32, S> = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), DeviceError> {
    if !cutile_examples::requirements::BASELINE.check("async_and_then_example", 0)? {
        return Ok(());
    }

    let policy = global_policy(0)?;
    let num_elements: usize = 2usize.pow(5);
    let strides = &[8, 1];
    let y_partition_shape = [2, 4];
    let y_partition_strides = strides;
    let function_generics: Vec<String> = y_partition_shape.iter().map(|x| x.to_string()).collect();
    let stride_args: Vec<(String, Vec<i32>)> = vec![
        ("y".to_string(), y_partition_strides.to_vec()),
        ("x".to_string(), strides.to_vec()),
    ];

    // We can start compiling the function while we fetch input.
    let compilation_task = tokio::spawn(async move {
        with_context(|ctx| {
            let func = compile_from_context(
                ctx,
                __module_ast_self,
                "my_module",
                "saxpy",
                "saxpy_entry",
                function_generics,
                stride_args,
                vec![],
                vec![],
                None,
                CompileOptions::default(),
                my_module::_SOURCE_HASH,
            );
            value(func)
        })
        .schedule(&policy)?
        .await
    });

    // Spawn allocation of tensor 1 and 2.
    let a: f32 = 2.0;
    let x: Arc<_> = arange::<f32>(num_elements).reshape(&[4, 8]).await?.into();
    let y = arange::<f32>(num_elements).reshape(&[4, 8]).await?;

    // We need the function to build the launcher, so we wait on compilation.
    let (function, _) = compilation_task
        .await
        .expect("Failed to compile module.")??;
    // Chain a bunch of kernel launches.
    let cuda_async_ops = tokio::spawn(async move {
        let y_part = y.partition(y_partition_shape);
        let mut launcher = AsyncKernelLaunch::new(function.clone());
        launcher
            .push_arg(&y_part)
            .push_arg(a)
            .push_arg_arc(&x)
            .set_launch_config(LaunchConfig {
                grid_dim: y_part.grid().expect("Invalid grid."),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            });
        launcher
            .then(|_| {
                let mut launcher = AsyncKernelLaunch::new(function.clone());
                launcher
                    .push_arg(&y_part)
                    .push_arg(a)
                    .push_arg_arc(&x)
                    .set_launch_config(LaunchConfig {
                        grid_dim: y_part.grid().expect("Invalid grid."),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    });
                launcher
            })
            .then(|_| {
                let mut launcher = AsyncKernelLaunch::new(function.clone());
                launcher
                    .push_arg(&y_part)
                    .push_arg(a)
                    .push_arg_arc(&x)
                    .set_launch_config(LaunchConfig {
                        grid_dim: y_part.grid().expect("Invalid grid."),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    });
                launcher
            })
            .await
            .expect("Kernel launch failed.");
        y_part.unpartition()
    });
    let res = cuda_async_ops.await;
    assert!(res.is_ok());
    let y = res.unwrap();

    // Check output.
    let y_host = y.to_host_vec().await?;
    let input_host: Vec<f32> = arange(num_elements).await?.to_host_vec().await?;
    for (&input_host, &y_host) in input_host.iter().zip(y_host.iter()) {
        let x_i = input_host;
        let y_i = input_host;
        // We're applying the saxpy function 3 times.
        let mut answer = y_i;
        answer += a * x_i;
        answer += a * x_i;
        answer += a * x_i;
        println!("{} * {} + {} = {}", a, x_i, y_i, y_host);
        assert_eq!(answer, y_host, "{} != {} ?", answer, y_host);
    }
    Ok(())
}

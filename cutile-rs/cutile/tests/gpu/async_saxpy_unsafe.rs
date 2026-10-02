/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: raw async compile-and-launch path via `compile_from_context`
//! and `AsyncKernelLaunch`. Exercises the low-level async API that higher-
//! level kernel launchers wrap.

use cuda_async::device_operation::{value, with_context, DeviceOp};
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;
use cutile::api::{arange, DeviceOpReshape};
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::{compile_from_context, global_policy, CompileOptions};
use std::sync::Arc;

use my_module::__module_ast_self;

use crate::common;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(y: &mut Tensor<f32, S>, a: f32, x: &Tensor<f32, { [-1, -1] }>) {
        let tile_a = broadcast_scalar(a, y.shape());
        let tile_x = load_tile_like(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

#[test]
fn smoke_async_saxpy_unsafe() {
    common::with_test_stack(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async {
            let policy = global_policy(0).expect("policy");
            let num_elements: usize = 2usize.pow(5);
            let strides = &[8, 1];
            let y_partition_shape = [2, 4];
            let y_partition_strides = strides;
            let function_generics: Vec<String> =
                y_partition_shape.iter().map(|x| x.to_string()).collect();
            let stride_args: Vec<(String, Vec<i32>)> = vec![
                ("y".to_string(), y_partition_strides.to_vec()),
                ("x".to_string(), strides.to_vec()),
            ];

            // Compile and fetch inputs concurrently.
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
                .schedule(&global_policy(0).expect("policy"))
                .expect("schedule")
                .await
            });
            let a: f32 = 2.0;
            let x = arange::<f32>(num_elements)
                .reshape(&[4, 8])
                .schedule(&policy)
                .expect("schedule x")
                .await
                .expect("x");
            let y = arange::<f32>(num_elements)
                .reshape(&[4, 8])
                .schedule(&policy)
                .expect("schedule y")
                .await
                .expect("y");
            let (function, _) = compilation_task
                .await
                .expect("compile join")
                .expect("compile schedule")
                .expect("compile result");

            let cuda_async_op = async move {
                let y_part = y.partition(y_partition_shape);
                let mut launcher = AsyncKernelLaunch::new(function.clone());
                let x_arc: Arc<_> = x.into();
                launcher
                    .push_arg(&y_part)
                    .push_arg(a)
                    .push_arg_arc(&x_arc)
                    .set_launch_config(LaunchConfig {
                        grid_dim: y_part.grid().expect("Invalid grid."),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    });
                launcher.await.expect("Kernel launch failed.");
                y_part.unpartition()
            };
            let y = tokio::spawn(cuda_async_op).await.expect("task");

            let y_host = y.to_host_vec().await.expect("y to_host");
            let input_host: Vec<f32> = arange(num_elements)
                .await
                .expect("input arange")
                .to_host_vec()
                .await
                .expect("input to_host");
            for (&input_host, &y_host) in input_host.iter().zip(y_host.iter()) {
                let answer = a * input_host + input_host;
                assert_eq!(answer, y_host);
            }
        });
    });
}

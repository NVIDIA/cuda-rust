/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: `DeviceOp` async combinators (`with_context`, `zip!`,
//! `and_then_with_context`) over raw CUDA malloc/memcpy/free.

use cuda_async::device_operation;
use cuda_async::device_operation::*;
use cuda_core::*;
use cutile::tile_kernel::global_policy;

use crate::common;

#[test]
fn smoke_async_device_op() {
    common::with_test_stack(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async {
            let _policy = global_policy(0);

            let num_elements: usize = 2usize.pow(12);
            let num_bytes = 4 * num_elements;
            let x_host_1 = vec![1u32; num_elements];
            let y_host_1 = vec![1u32; num_elements];

            let x_op = device_operation::with_context(|ctx: &ExecutionContext| unsafe {
                let dptr = malloc_async(num_bytes, ctx.get_cuda_stream()).expect("malloc x");
                memcpy_htod_async(dptr, x_host_1.as_ptr(), num_elements, ctx.get_cuda_stream())
                    .expect("h2d x");
                Value::new(dptr)
            });
            let y_op = device_operation::with_context(|ctx| unsafe {
                let dptr = malloc_async(num_bytes, ctx.get_cuda_stream()).expect("malloc y");
                memcpy_htod_async(dptr, y_host_1.as_ptr(), num_elements, ctx.get_cuda_stream())
                    .expect("h2d y");
                Value::new(dptr)
            });

            let op = zip!(x_op, y_op).and_then_with_context(|ctx, (x_dptr, y_dptr)| {
                let mut x_host = vec![0u32; num_elements];
                let mut y_host = vec![0u32; num_elements];
                unsafe {
                    memcpy_dtoh_async(
                        x_host.as_mut_ptr(),
                        x_dptr,
                        num_elements,
                        ctx.get_cuda_stream(),
                    )
                    .expect("d2h x");
                    memcpy_dtoh_async(
                        y_host.as_mut_ptr(),
                        y_dptr,
                        num_elements,
                        ctx.get_cuda_stream(),
                    )
                    .expect("d2h y");
                }
                Value::new(((x_host, y_host), (x_dptr, y_dptr)))
            });

            let ((x_host_2, y_host_2), dptrs) = op.await.expect("zip await");

            assert!(x_host_1.iter().all(|&v| v == 1));
            assert!(x_host_2.iter().all(|&v| v == 1));
            assert!(y_host_2.iter().all(|&v| v == 1));

            // Free.
            Value::new(dptrs)
                .and_then_with_context(|ctx, (x_dptr, y_dptr)| {
                    unsafe {
                        free_async(x_dptr, ctx.get_cuda_stream()).expect("free x");
                        free_async(y_dptr, ctx.get_cuda_stream()).expect("free y");
                    }
                    Value::new(())
                })
                .await
                .expect("free await");
        });
    });
}

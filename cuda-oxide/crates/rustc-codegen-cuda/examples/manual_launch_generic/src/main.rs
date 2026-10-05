/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual low-level launch API regression test.
//!
//! This example intentionally uses `load_kernel_module` plus `cuda_launch!`
//! instead of `#[cuda_module]`. It keeps the explicit API covered while the
//! typed embedded-module API is the preferred ergonomic path.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::ops::{Add, Mul};

#[kernel]
pub fn affine<T: Copy + Add<Output = T> + Mul<Output = T>>(
    alpha: T,
    x: &[T],
    beta: T,
    y: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = alpha * x[idx_raw] + beta * y[idx_raw];
    }
}

#[kernel]
pub fn add_const<const VALUE: u32>(input: &[u32], mut output: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(out_elem) = output.get_mut(idx) {
        *out_elem = input[idx_raw] + VALUE;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Manual Generic Launch API Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "manual_launch_generic")?;

    const N: usize = 1024;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    {
        let x_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let y_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
        let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
        let y_dev = DeviceBuffer::from_host(&stream, &y_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let alpha = 2.0f32;
        let beta = 3.0f32;

        // SAFETY: args mirror `affine::<f32>`'s signature (scalar, slice, scalar,
        // slice, slice_mut); all buffers are live DeviceBuffers on this stream.
        unsafe {
            cuda_launch! {
                kernel: affine::<f32>,
                stream: stream,
                module: module,
                config: cfg,
                args: [alpha, slice(x_dev), beta, slice(y_dev), slice_mut(out_dev)]
            }
        }?;

        let out = out_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (out[i] - (alpha * x_host[i] + beta * y_host[i])).abs() > 1e-5)
            .count();
        assert_eq!(errors, 0, "affine::<f32> produced {errors} errors");
        println!("affine::<f32>: PASS");
    }

    {
        let x_host: Vec<i32> = (0..N as i32).collect();
        let y_host: Vec<i32> = (0..N as i32).map(|i| i * 2).collect();
        let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
        let y_dev = DeviceBuffer::from_host(&stream, &y_host)?;
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        let alpha = 2i32;
        let beta = 3i32;

        // SAFETY: args mirror `affine::<i32>`'s signature (scalar, slice, scalar,
        // slice, slice_mut); all buffers are live DeviceBuffers on this stream.
        unsafe {
            cuda_launch! {
                kernel: affine::<i32>,
                stream: stream,
                module: module,
                config: cfg,
                args: [alpha, slice(x_dev), beta, slice(y_dev), slice_mut(out_dev)]
            }
        }?;

        let out = out_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| out[i] != alpha * x_host[i] + beta * y_host[i])
            .count();
        assert_eq!(errors, 0, "affine::<i32> produced {errors} errors");
        println!("affine::<i32>: PASS");
    }

    {
        let input_host: Vec<u32> = (0..N as u32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
        let mut output_4 = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        let mut output_8 = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        // SAFETY: each argument packet matches `add_const`'s slice and
        // mutable-slice parameters, and all buffers remain live through the
        // stream synchronization in `to_host_vec`.
        unsafe {
            cuda_launch! {
                kernel: add_const::<4>,
                stream: stream,
                module: module,
                config: cfg,
                args: [slice(input_dev), slice_mut(output_4)]
            }
        }?;
        unsafe {
            cuda_launch! {
                kernel: add_const::<8>,
                stream: stream,
                module: module,
                config: cfg,
                args: [slice(input_dev), slice_mut(output_8)]
            }
        }?;

        let result_4 = output_4.to_host_vec(&stream)?;
        let result_8 = output_8.to_host_vec(&stream)?;
        assert!((0..N).all(|i| result_4[i] == input_host[i] + 4));
        assert!((0..N).all(|i| result_8[i] == input_host[i] + 8));
        println!("add_const::<4/8>: PASS");
    }

    println!("\nSUCCESS: manual type- and const-generic launches passed");
    Ok(())
}

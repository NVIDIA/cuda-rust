/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async Vector Addition Example
//!
//! Demonstrates the cuda-async execution model:
//! - `#[cuda_module]` generates typed async launch methods
//! - `vecadd_async` returns a lazy `DeviceOperation`, no GPU work yet
//! - `.await` schedules it on a round-robin stream pool and waits
//! - `.sync()` does the same but blocks the calling thread
//! - `and_then` chains operations on the same stream
//! - `zip!` runs independent operations on the same stream
//!
//! Build and run with:
//!   cargo oxide run async_vecadd

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNEL -- compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

// =============================================================================
// HOST CODE -- compiled to native x86_64 by LLVM
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_box::DeviceBox;
    use cuda_async::simt::device_context::init_device_contexts;
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::simt::LaunchConfig;
    use cuda_core::simt::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    println!("=== Async Vector Addition (cuda-async) ===\n");

    // 1. Initialize the async device context (round-robin pool of 4 streams).
    init_device_contexts(0, 1)?;

    // 2. Load the embedded CUDA module from the async device context.
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input vectors (first 5 elements):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // 3. Allocate device memory and copy host data.
    //    We use cuda_core memory APIs with the context's default stream.
    let (a_dev, b_dev, mut c_dev) =
        cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
            let stream = ctx.default_stream();
            let num_bytes = N * mem::size_of::<f32>();

            unsafe {
                let a_dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let b_dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let c_dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();

                memcpy_htod_async(a_dptr, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
                memcpy_htod_async(b_dptr, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();

                stream.synchronize().unwrap();

                let a_dev = DeviceBox::<[f32]>::from_raw_parts(a_dptr, N, 0);
                let b_dev = DeviceBox::<[f32]>::from_raw_parts(b_dptr, N, 0);
                let c_dev = DeviceBox::<[f32]>::from_raw_parts(c_dptr, N, 0);
                (a_dev, b_dev, c_dev)
            }
        })?;

    // 4. Launch the kernel asynchronously.
    //    `vecadd_async` returns an AsyncKernelLaunch (a DeviceOperation).
    //    No GPU work happens until we `.sync()` or `.await`.
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;

    // 5. Copy results back.
    let mut c_host = vec![0.0f32; N];
    cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    println!("\nOutput vector (first 5 elements):");
    println!("  c = {:?}", &c_host[0..5]);

    // 6. Verify.
    let mut errors = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  Error at [{}]: expected {}, got {}",
                    i, expected, c_host[i]
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\nSUCCESS: All {} elements correct!", N);
    } else {
        println!("\nFAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}

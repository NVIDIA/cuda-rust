/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked Slice Indexing Bounds Check Test (issue #396)
//!
//! A `#[kernel]` that indexes a device slice with a data-dependent index
//! using safe checked indexing (`src[j]`) must not read out of bounds when `j >= src.len()`
//!
//! Test 1 gathers through in-range indices and verifies the values
//! Test 2 plants one out-of-range index.
//! The corresponding output element must keep its sentinel value (thread exited at the check)
//! or the launch must fail (thread trapped).
//! If the sentinel is overwritten, the kernel performed an out-of-bounds global read.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const N: usize = 1024;
const SENTINEL: f32 = -12345.5;

#[cuda_module]
mod kernels {
    use super::*;

    /// Gather `src[indices[i]]` into `out[i]` using checked indexing
    #[kernel]
    pub fn gather(src: &[f32], indices: &[u32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let j = indices[i] as usize;
            *out_elem = src[j];
        }
    }
}

fn launch_gather(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    indices_host: &[u32],
) -> Result<Vec<f32>, cuda_core::DriverError> {
    let src_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let src = DeviceBuffer::from_host(stream, &src_host)?;
    let indices = DeviceBuffer::from_host(stream, indices_host)?;
    let mut out = DeviceBuffer::from_host(stream, &vec![SENTINEL; N])?;

    let config = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: one thread per output element
    unsafe { module.gather(stream.as_ref(), config, &src, &indices, &mut out) }?;
    out.to_host_vec(stream)
}

fn run_in_bounds_gather(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> bool {
    let indices_host: Vec<u32> = (0..N).map(|i| (N - 1 - i) as u32).collect();
    let out = match launch_gather(module, stream, &indices_host) {
        Ok(out) => out,
        Err(e) => {
            println!("in_bounds_gather: FAIL (launch error: {})", e);
            return false;
        }
    };

    let ok = out
        .iter()
        .enumerate()
        .all(|(i, &v)| v == (N - 1 - i) as f32);
    if ok {
        println!("in_bounds_gather: PASS");
    } else {
        println!("in_bounds_gather: FAIL (wrong gather results)");
    }
    ok
}

fn run_out_of_bounds_gather(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> bool {
    let mut indices_host: Vec<u32> = (0..N).map(|i| (N - 1 - i) as u32).collect();
    indices_host[0] = 1_000_000; // thread 0 tries to read ~4 MB outside the allocation

    match launch_gather(module, stream, &indices_host) {
        Ok(out) => {
            if out[0] == SENTINEL {
                println!("out_of_bounds_gather: PASS (thread exited at the bounds check)");
                true
            } else {
                println!(
                    "out_of_bounds_gather: FAIL (out-of-bounds read happened, out[0] = {})",
                    out[0]
                );
                false
            }
        }
        // CUDA_ERROR_ILLEGAL_ADDRESS means the out-of-bounds load reached memory
        // and the hardware caught it, which means the bounds check is gone
        Err(e) if e.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_ILLEGAL_ADDRESS => {
            println!(
                "out_of_bounds_gather: FAIL (out-of-bounds read reached memory: {})",
                e
            );
            false
        }
        // Any other launch/copy-back failure is the bounds check trapping before the load
        // PTX `trap` reports as a generic launch failure
        Err(e) => {
            println!("out_of_bounds_gather: PASS (kernel trapped: {})", e);
            true
        }
    }
}

fn main() {
    println!("=== Checked Slice Indexing Bounds Check Test (issue #396) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    // The out-of-bounds test runs last
    // If the kernel traps, the context is poisoned for every launch after it
    let in_bounds_ok = run_in_bounds_gather(&module, &stream);
    let oob_ok = run_out_of_bounds_gather(&module, &stream);

    if in_bounds_ok && oob_ok {
        println!("\nSUCCESS");
    } else {
        println!("\nFAILURE");
        std::process::exit(1);
    }
}

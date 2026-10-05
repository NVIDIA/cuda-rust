/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Vector Addition Example
//!
//! THIS IS THE GOAL: Single file, single compilation, no cfg splits.
//!
//! Build and run with:
//!   cargo oxide run vecadd
//!
//! What happens:
//! 1. rustc parses this file, generates MIR for everything
//! 2. rustc-codegen-cuda intercepts codegen:
//!    - Finds `cuda_oxide_kernel_<hash>_vecadd` (from #[kernel])
//!    - Routes it to mir-importer → PTX
//!    - Routes `main` and other host code to standard LLVM
//! 3. Final binary has both host code and embedded PTX

// No #![cfg_attr(cuda_device, no_std)] - this compiles as ONE unit!

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// KERNEL - This gets compiled to PTX by rustc-codegen-cuda
// =============================================================================

/// Vector addition kernel: c[i] = a[i] + b[i]
///
/// This function exists in BOTH host MIR and device PTX:
/// - Host: The function body is never called, but types are checked
/// - Device: Compiled to PTX via mir-importer pipeline
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
// HOST CODE - This gets compiled to native x86_64 by LLVM
// =============================================================================

fn main() {
    println!("=== Unified Compilation Vector Addition ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test data
    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input vectors (first 5 elements):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Allocate device memory
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // Load the embedded PTX bundle and launch through the typed module API.
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }
    .expect("Kernel launch failed");

    // Get results
    let c_host = c_dev.to_host_vec(&stream).unwrap();

    println!("\nOutput vector (first 5 elements):");
    println!("  c = {:?}", &c_host[0..5]);

    // Verify
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
        println!("\n✓ SUCCESS: All {} elements correct!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Hello Constant Example
//!
//! Demonstrates a minimal kernel that writes a constant value to memory.
//! This tests raw pointer support in the unified compilation pipeline.
//!
//! Build and run with:
//!   cargo oxide run hello_constant

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::kernel;
use cuda_host::cuda_module;

// =============================================================================
// KERNEL - Compiled to PTX by rustc-codegen-cuda
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Minimal kernel that writes 42 to a memory location.
    /// Tests raw pointer passing to kernels.
    #[kernel]
    pub unsafe fn hello_constant(out: *mut i32) {
        unsafe { *out = 42 };
    }
}

// =============================================================================
// HOST CODE - Compiled to native x86_64 by LLVM
// =============================================================================

fn main() {
    println!("=== Unified Hello Constant Example ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let out_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("Failed to allocate");

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    println!("Launching kernel...");
    unsafe {
        module.hello_constant(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(1),
            out_dev.cu_deviceptr() as *mut i32,
        )
    }
    .expect("Kernel launch failed");

    let result = out_dev.to_host_vec(&stream).expect("Failed to copy result");
    println!("Output: {}", result[0]);

    // Verify
    if result[0] == 42 {
        println!("\n✓ SUCCESS: Kernel wrote 42 correctly!");
    } else {
        println!("\n✗ FAILED: Expected 42, got {}", result[0]);
        std::process::exit(1);
    }
}

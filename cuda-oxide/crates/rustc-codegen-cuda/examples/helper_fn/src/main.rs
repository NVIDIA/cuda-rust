/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Helper Function Example
//!
//! Demonstrates the #[device] attribute for helper functions that are called
//! from kernels. This tests that the pipeline correctly inlines or links
//! device functions into kernels.
//!
//! Build and run with:
//!   cargo oxide run helper_fn

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// DEVICE FUNCTION - Helper that gets inlined/linked into the kernel
// =============================================================================

/// Device helper function for vector addition.
/// Demonstrates the #[device] attribute for GPU-only functions.
///
/// `mut c` is required because `DisjointSlice::get_mut()` takes `&mut self`.
/// The `#[device]` macro generates a thin wrapper with the original name,
/// so callers just write `vecadd_device(...)` (not the prefixed internal name).
#[device]
pub fn vecadd_device(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();

    if let Some(c_elem) = c.get_mut(idx) {
        let i = idx_raw;
        *c_elem = a[i] + b[i] + __rust_alloc(0.0);
    }
}

/// Name-collision probe for the heap-allocation guard (issue #108).
///
/// `__rust_alloc` is NOT a reserved name: a user may define their own
/// function with it, and it must compile for the device like any other
/// helper. The real allocator entry points are recognized by sysroot
/// origin, never by name alone. This helper adds 0.0, so the example's
/// expected results are unchanged.
// The #[device] macro prefixes the internal symbol with
// `cuda_oxide_device_<hash>_`, and the double underscore in the probe's
// name makes that generated identifier trip the snake-case lint. The
// odd name is the entire point of the probe, so allow it here.
#[allow(non_snake_case)]
#[device]
pub fn __rust_alloc(x: f32) -> f32 {
    x
}

// =============================================================================
// KERNEL - Entry point that calls the device function
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Kernel that delegates to the device helper function.
    /// This tests the full pipeline: kernel -> device function -> PTX
    ///
    /// Note: The kernel takes `c: DisjointSlice<f32>` (no `mut`) because it just
    /// forwards to the device function. In Rust, `mut` on a by-value parameter is
    /// purely local binding mutability — the caller doesn't need `mut` to pass
    /// an owned value.
    #[kernel]
    pub fn vecadd_with_helper(a: &[f32], b: &[f32], c: DisjointSlice<f32>) {
        // A named generic helper nested inside a kernel carries the kernel's
        // generated symbol in an earlier def-path segment. It must remain a
        // normal device callee under its canonical mangled symbol, not become
        // a second `_TID_` kernel entry.
        #[inline(never)]
        fn nested_identity<T: Copy>(value: T) -> T {
            value
        }

        // Keep two concrete monomorphizations reachable without changing the
        // example's result. `inline(never)` makes them visible to the device
        // collector and this guard is always false at runtime.
        if nested_identity::<f32>(0.0) != 0.0 || nested_identity::<u32>(0) != 0 {
            return;
        }

        // Call the device function by its original name
        vecadd_device(a, b, c);
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Helper Function Example ===\n");

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

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // Launch kernel
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.vecadd_with_helper(
            (stream).as_ref(),
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
        println!("  (Kernel called device helper function successfully)");
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}

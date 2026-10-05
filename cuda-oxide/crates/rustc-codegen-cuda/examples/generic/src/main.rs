/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Generic Kernel Example
//!
//! Tests whether the collector correctly handles monomorphized generic kernels.
//!
//! Build and run with:
//!   cargo oxide pipeline generic
//!   cargo oxide run generic
//!
//! ## What This Tests
//!
//! 1. Generic kernel definition: `fn scale<T>(factor: T, ...)`
//! 2. Monomorphization: `scale::<f32>` and `scale::<i32>` create separate versions
//! 3. Collection: Does our collector find the monomorphized instance?
//! 4. PTX generation: Does the backend generate valid PTX?
//!
//! ## Expected PTX
//!
//! We should see a PTX entry point for `scale` (or `scale_f32` if we add type info).

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::ops::{Add, Mul};

// =============================================================================
// GENERIC KERNELS
// =============================================================================

/// Generic scale kernel - multiplies each element by a factor.
///
/// This kernel is generic over T. Each typed module method call
/// monomorphizes it to a concrete device entry point.
#[allow(dead_code)]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[idx_raw] * factor;
        }
    }

    /// Generic add kernel - adds two arrays element-wise.
    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    #[inline(never)]
    fn apply_unary<T, F>(f: F, x: T) -> T
    where
        F: FnOnce(T) -> T,
    {
        f(x)
    }

    /// Generic kernel whose body defines a captured closure.
    ///
    /// rustc prepends the parent generic args before a closure's fixed
    /// `[kind, sig, tupled_upvars]` suffix. The importer must read the upvars
    /// tuple from the suffix, not from a hard-coded substitution index.
    #[kernel]
    pub fn closure_capture<T>(bias: T, scale: T, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let transform = move |x: T| (x + bias) * scale;
            *out_elem = apply_unary(transform, input[idx_raw]);
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================
//
// Calling the typed module method with type parameters triggers monomorphization.

fn main() {
    println!("=== Unified Generic Kernel Test ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // =========================================================================
    // THE KEY: typed method call with type parameter forces monomorphization!
    // =========================================================================
    //
    // module.scale::<f32>(...) expands to code that references
    // cuda_oxide_kernel_scale::<f32>, forcing rustc to monomorphize it and
    // making it visible to the backend's collector.

    println!("\nLaunching scale::<f32> kernel...");
    {
        let factor: f32 = 2.5;
        let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev =
            DeviceBuffer::from_host(&stream, &input_data).expect("Failed to copy f32 input");
        let mut output_dev =
            DeviceBuffer::<f32>::zeroed(&stream, N).expect("Failed to alloc f32 output");

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale::<f32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("scale::<f32> launch failed");

        let output_host = output_dev
            .to_host_vec(&stream)
            .expect("Failed to copy f32 output back");
        let errors = (0..N)
            .filter(|&i| (output_host[i] - input_data[i] * factor).abs() > 1e-5)
            .count();
        assert_eq!(errors, 0, "scale::<f32> produced {errors} errors");
    }

    println!("Launching scale::<i32> kernel...");
    {
        let factor: i32 = 3;
        let input_data: Vec<i32> = (0..N as i32).collect();
        let input_dev =
            DeviceBuffer::from_host(&stream, &input_data).expect("Failed to copy i32 input");
        let mut output_dev =
            DeviceBuffer::<i32>::zeroed(&stream, N).expect("Failed to alloc i32 output");

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale::<i32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("scale::<i32> launch failed");

        let output_host = output_dev
            .to_host_vec(&stream)
            .expect("Failed to copy i32 output back");
        let errors = (0..N)
            .filter(|&i| output_host[i] != input_data[i] * factor)
            .count();
        assert_eq!(errors, 0, "scale::<i32> produced {errors} errors");
    }

    println!("Launching closure_capture::<f32> kernel...");
    {
        let bias: f32 = 1.25;
        let scale: f32 = 0.5;
        let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev =
            DeviceBuffer::from_host(&stream, &input_data).expect("Failed to copy f32 input");
        let mut output_dev =
            DeviceBuffer::<f32>::zeroed(&stream, N).expect("Failed to alloc f32 output");

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.closure_capture::<f32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                bias,
                scale,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("closure_capture::<f32> launch failed");

        let output_host = output_dev
            .to_host_vec(&stream)
            .expect("Failed to copy f32 output back");
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (input_data[i] + bias) * scale).abs() > 1e-5)
            .count();
        assert_eq!(errors, 0, "closure_capture::<f32> produced {errors} errors");
    }

    println!("Launching closure_capture::<i32> kernel...");
    {
        let bias: i32 = 7;
        let scale: i32 = 2;
        let input_data: Vec<i32> = (0..N as i32).collect();
        let input_dev =
            DeviceBuffer::from_host(&stream, &input_data).expect("Failed to copy i32 input");
        let mut output_dev =
            DeviceBuffer::<i32>::zeroed(&stream, N).expect("Failed to alloc i32 output");

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.closure_capture::<i32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                bias,
                scale,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("closure_capture::<i32> launch failed");

        let output_host = output_dev
            .to_host_vec(&stream)
            .expect("Failed to copy i32 output back");
        let errors = (0..N)
            .filter(|&i| output_host[i] != (input_data[i] + bias) * scale)
            .count();
        assert_eq!(errors, 0, "closure_capture::<i32> produced {errors} errors");
    }

    println!("\n✓ SUCCESS: typed generic launches and captured closures worked for f32 and i32");
}

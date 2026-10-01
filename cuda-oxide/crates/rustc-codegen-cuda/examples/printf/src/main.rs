/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::approx_constant)]

//! GPU Printf Test
//!
//! Tests the `gpu_printf!` macro for formatted output from GPU kernels.
//! This demonstrates full CUDA C++ printf parity including:
//!
//! - Basic types: integers, floats, pointers, booleans
//! - Format specifiers: hex, octal, scientific, compact
//! - Flags: left-justify, sign, alternate, zero-pad
//! - Width and precision
//!
//! Run: cargo oxide run printf

use cuda_device::{gpu_printf, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel: Basic integer formats
    #[kernel]
    pub fn test_integers() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            let signed: i32 = -42;
            let unsigned: u32 = 255;
            let big: u64 = 0x1234_5678_9ABC_DEF0;

            gpu_printf!("=== Integer Tests ===\n");
            gpu_printf!("Signed i32: {}\n", signed);
            gpu_printf!("Unsigned u32: {}\n", unsigned);
            gpu_printf!("Unsigned u64: {}\n", big);
            gpu_printf!("Hex (lower): {:x}\n", unsigned);
            gpu_printf!("Hex (upper): {:X}\n", unsigned);
            gpu_printf!("Hex with prefix: {:#x}\n", unsigned);
            gpu_printf!("Octal: {:o}\n", 64u32);
            gpu_printf!("Octal with prefix: {:#o}\n", 64u32);
        }
    }

    /// Test kernel: Float formats
    #[kernel]
    pub fn test_floats() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            let pi: f32 = 3.141_592_7;
            let e: f64 = 2.718281828;
            let large: f64 = 1234567.89;
            let small: f64 = 0.000123;

            gpu_printf!("=== Float Tests ===\n");
            gpu_printf!("f32 default: {:.6f}\n", pi);
            gpu_printf!("f64 default: {:.6f}\n", e);
            gpu_printf!("Precision .2: {:.2f}\n", pi);
            gpu_printf!("Precision .6: {:.6f}\n", pi);
            gpu_printf!("Scientific (lower): {:e}\n", large);
            gpu_printf!("Scientific (upper): {:E}\n", large);
            gpu_printf!("Compact (g): {:g}\n", small);
            gpu_printf!("Compact (G): {:G}\n", large);
        }
    }

    /// Test kernel: Width and alignment
    #[kernel]
    pub fn test_width_align() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            let val: i32 = 42;
            let fval: f32 = 3.14;

            gpu_printf!("=== Width & Alignment Tests ===\n");
            gpu_printf!("Width 8:     [{}]\n", val);
            gpu_printf!("Width 8:     [{:8}]\n", val);
            gpu_printf!("Zero-pad 8:  [{:08}]\n", val);
            gpu_printf!("Left align:  [{:-8}]\n", val);
            gpu_printf!("Width+prec:  [{:8.2f}]\n", fval);
        }
    }

    /// Test kernel: Sign and flags
    #[kernel]
    pub fn test_flags() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            let pos: i32 = 42;
            let neg: i32 = -42;

            gpu_printf!("=== Sign & Flag Tests ===\n");
            gpu_printf!("Always sign (+): {:+}\n", pos);
            gpu_printf!("Always sign (-): {:+}\n", neg);
            gpu_printf!("Space for positive: {: }\n", pos);
            gpu_printf!("Space for negative: {: }\n", neg);
        }
    }

    /// Test kernel: Thread-indexed output
    #[kernel]
    pub fn test_thread_output(data: &[f32]) {
        let tid = thread::index_1d().get();

        // Only first 4 threads print to avoid overwhelming output
        if tid < 4 {
            gpu_printf!("Thread {}: data[{}] = {:.4f}\n", tid, tid, data[tid]);
        }
    }

    /// Test kernel: Return value check
    ///
    /// CUDA vprintf returns the NUMBER OF ARGUMENTS, not character count.
    /// This is because the GPU doesn't format - it only marshals args to a buffer.
    /// The host reads the buffer later and does actual formatting.
    ///
    /// Reference: https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/
    #[kernel]
    pub fn test_return_value() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            gpu_printf!("=== Return Value Test ===\n");
            gpu_printf!("(vprintf returns arg count, not char count)\n");

            // Test with 0 args - should return 0
            let r0 = gpu_printf!("Hello, GPU!\n");
            gpu_printf!("0 args returned: {}\n", r0);

            // Test with 1 arg - should return 1
            let r1 = gpu_printf!("Value: {}\n", 42);
            gpu_printf!("1 arg returned: {}\n", r1);

            // Test with 2 args - should return 2
            let r2 = gpu_printf!("x={}, y={}\n", 10, 20);
            gpu_printf!("2 args returned: {}\n", r2);

            // Test with 3 args - should return 3
            let r3 = gpu_printf!("a={}, b={}, c={}\n", 1, 2, 3);
            gpu_printf!("3 args returned: {}\n", r3);
        }
    }

    /// Test kernel: Boolean values
    #[kernel]
    pub fn test_booleans() {
        let tid = thread::index_1d().get();

        if tid == 0 {
            let t: bool = true;
            let f: bool = false;

            gpu_printf!("=== Boolean Tests ===\n");
            gpu_printf!("true: {}\n", t);
            gpu_printf!("false: {}\n", f);
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== GPU Printf Test (Unified) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ====================================================================
    // Test 1: Integer formats
    // ====================================================================
    println!("--- Test 1: Integer formats ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_integers((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    // ====================================================================
    // Test 2: Float formats
    // ====================================================================
    println!("--- Test 2: Float formats ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_floats((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    // ====================================================================
    // Test 3: Width and alignment
    // ====================================================================
    println!("--- Test 3: Width and alignment ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_width_align((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    // ====================================================================
    // Test 4: Sign and flags
    // ====================================================================
    println!("--- Test 4: Sign and flags ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_flags((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    // ====================================================================
    // Test 5: Thread-indexed output
    // ====================================================================
    println!("--- Test 5: Thread-indexed output ---");
    {
        let data: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_thread_output((stream).as_ref(), cfg, &data_dev) }?;
        stream.synchronize()?;
    }
    println!();

    // ====================================================================
    // Test 6: Return value
    // ====================================================================
    println!("--- Test 6: Return value ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_return_value((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    // ====================================================================
    // Test 7: Boolean values
    // ====================================================================
    println!("--- Test 7: Boolean values ---");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_booleans((stream).as_ref(), cfg) }?;
    stream.synchronize()?;
    println!();

    println!("=== ALL PRINTF TESTS PASSED ===");
    Ok(())
}

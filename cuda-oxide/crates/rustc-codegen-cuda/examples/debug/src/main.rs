/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::erasing_op)]

//! Debug and Utility Intrinsics Test
//!
//! Tests GPU debug/utility features:
//! - `clock64()` - Read GPU clock cycles
//! - `globaltimer()` - Read the GPU global timer
//! - `trap()` - Abort kernel execution
//! - `gpu_assert!()` - Runtime assertion
//! - `breakpoint()` - cuda-gdb breakpoint
//! - `prof_trigger()` - NVIDIA profiler signals
//! - `#[launch_bounds]` - Hint thread/block counts to compiler
//!
//! Run: cargo oxide run debug

use cuda_device::{DisjointSlice, debug, gpu_assert, kernel, launch_bounds, shared, thread, warp};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel: measures clock and global timer ticks for a simple operation.
    #[kernel] // CUDA_OXIDE_DEBUG_KERNEL_ATTRIBUTE_LINE
    #[launch_bounds(256, 2)] // CUDA_OXIDE_DEBUG_LAUNCH_BOUNDS_ATTRIBUTE_LINE
    pub fn clock_test(mut output: DisjointSlice<u64>) {
        let idx = thread::index_1d(); // CUDA_OXIDE_DEBUG_KERNEL_ENTRY_LINE
        if let Some(output_elem) = output.get_mut(idx) {
            let start_cycles = debug::clock64();
            let start_timer = debug::globaltimer();

            // Some work to measure
            let mut sum: u64 = 0;
            for i in 0..100u64 {
                sum = sum.wrapping_add(i);
            }

            let end_timer = debug::globaltimer();
            let end_cycles = debug::clock64();

            // Write elapsed ticks (use sum to prevent optimization)
            *output_elem = end_cycles
                .wrapping_sub(start_cycles)
                .wrapping_add(end_timer.wrapping_sub(start_timer))
                .wrapping_add(sum & 0);
        }
    }

    /// Compile-time coverage for location, launch, and shared-memory special
    /// registers. Both location registers are sampled twice so optimized PTX
    /// must retain two `%warpid` and two `%smid` moves.
    #[kernel]
    pub fn special_register_test(mut output: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let Some(output_elem) = output.get_mut(idx) else {
            return;
        };

        let warp_before = warp::warpid();
        let sm_before = thread::smid();
        let warp_bound = warp::nwarpid();
        let sm_bound = thread::nsmid();
        let launch = thread::gridid();
        let dynamic_bytes = shared::dynamic_smem_size();
        let total_bytes = shared::total_smem_size();
        let warp_after = warp::warpid();
        let sm_after = thread::smid();

        *output_elem = launch
            .wrapping_add(warp_before as u64)
            .wrapping_add((sm_before as u64).rotate_left(7))
            .wrapping_add((warp_bound as u64).rotate_left(13))
            .wrapping_add((sm_bound as u64).rotate_left(19))
            .wrapping_add((dynamic_bytes as u64).rotate_left(29))
            .wrapping_add((total_bytes as u64).rotate_left(37))
            .wrapping_add((warp_after as u64).rotate_left(43))
            .wrapping_add((sm_after as u64).rotate_left(53));
    }

    /// Test kernel: demonstrates trap() for error handling
    ///
    /// Traps if any thread sees a negative value.
    #[kernel]
    pub fn trap_test(input: &[i32], mut output: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(output_elem) = output.get_mut(idx) {
            let val = input[idx_raw];

            if val < 0 {
                debug::trap(); // Kernel dies here if any value is negative
            }

            *output_elem = val * 2;
        }
    }

    /// Test kernel: demonstrates gpu_assert!() macro
    ///
    /// Asserts that all values are non-negative.
    #[kernel]
    pub fn assert_test(input: &[i32], mut output: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(output_elem) = output.get_mut(idx) {
            let val = input[idx_raw];

            // Assert that values are non-negative and within bounds
            gpu_assert!(val >= 0, "Expected non-negative value");
            gpu_assert!(val < 1000); // Simple assertion

            *output_elem = val + 1;
        }
    }

    /// Test kernel: demonstrates breakpoint() for cuda-gdb debugging
    #[kernel]
    pub fn breakpoint_test(mut output: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(output_elem) = output.get_mut(idx) {
            if idx_raw == 0 {
                debug::breakpoint(); // CUDA_OXIDE_DEBUG_INTRINSIC_BREAKPOINT
            }

            *output_elem = idx_raw as i32;
        }
    }

    /// Test kernel: demonstrates prof_trigger() for profiler signals
    #[kernel]
    pub fn profiler_test(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(output_elem) = output.get_mut(idx) {
            debug::prof_trigger::<0>(); // Signal "region 0 start"

            let val = input[idx_raw];
            let result = val * val; // Some computation

            debug::prof_trigger::<1>(); // Signal "region 0 end"

            *output_elem = result;
        }
    }

    /// Test kernel: demonstrates #[launch_bounds] attribute
    #[kernel]
    #[launch_bounds(128, 4)] // Max 128 threads/block, min 4 blocks/SM
    pub fn launch_bounds_test(input: &[i32], mut output: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(output_elem) = output.get_mut(idx) {
            let val = input[idx_raw];
            *output_elem = val * 3 + 1;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== GPU Debug & Utility Intrinsics Test (Unified) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    // cuda-gdb-only path:
    // brkpt -> debugger stop -> continue -> output [0, 1, ..., 7]
    if std::env::args().any(|a| a == "--breakpoint") {
        println!("--- cuda-gdb mode: breakpoint() ---");
        const N: usize = 8;
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;

        // SAFETY: launch shape/resources match the kernel; the output covers
        // every access by the one-block, eight-thread launch.
        unsafe {
            module.breakpoint_test(
                (stream).as_ref(),
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (N as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut output_dev,
            )
        }?;
        stream.synchronize()?;

        let output = output_dev.to_host_vec(&stream)?;
        let expected: Vec<_> = (0..N as i32).collect();
        assert_eq!(output, expected, "breakpoint_test output mismatch");
        println!("PASS breakpoint_test: runtime values = {output:?}");
        return Ok(());
    }

    // ====================================================================
    // Manually invoked failing mode: `debug --fail-assert`
    //
    // A failing device-side assert poisons the CUDA context, so this mode
    // runs INSTEAD of the normal test suite, never alongside it. The
    // driver prints the assertion message (file:line: Assertion `...`
    // failed) to stderr and the sync returns CUDA_ERROR_ASSERT (710).
    // ====================================================================
    if std::env::args().any(|a| a == "--fail-assert") {
        println!("--- Failing mode: gpu_assert!() with negative input ---");
        let input: Vec<i32> = vec![0, 1, -3, 3]; // lane 2 fails `val >= 0`
        let n = input.len();
        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, n)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let launch = unsafe {
            module.assert_test(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &input_dev,
                &mut output_dev,
            )
        };
        let result = launch.and_then(|_| stream.synchronize());
        match result {
            Err(e) => {
                println!("✓ assertion failure surfaced as expected: {e:?}");
                println!("  (assertion message printed to stderr by the CUDA driver)");
                return Ok(());
            }
            Ok(()) => {
                println!("✗ FAILED: kernel with a failing gpu_assert! completed normally");
                std::process::exit(1);
            }
        }
    }

    // ====================================================================
    // Test 1: Clock cycles measurement
    // ====================================================================
    println!("--- Test 1: clock64() / globaltimer() measurement ---");
    {
        const N: usize = 256;
        let mut output_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.clock_test((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<u64> = output_dev.to_host_vec(&stream)?;
        let avg_cycles: f64 = output.iter().map(|&x| x as f64).sum::<f64>() / N as f64;
        println!("  Average cycles for 100 iterations: {:.0}", avg_cycles);
        println!("✓ clock_test completed\n");
    }

    // ====================================================================
    // Test 2: Trap test (with valid input - should NOT trap)
    // ====================================================================
    println!("--- Test 2: trap() with valid input ---");
    {
        let input: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let n = input.len();

        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, n)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.trap_test(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &input_dev,
                &mut output_dev,
            )
        }?;
        stream.synchronize()?;

        let output: Vec<i32> = output_dev.to_host_vec(&stream)?;
        let expected: Vec<i32> = input.iter().map(|&x| x * 2).collect();

        if output == expected {
            println!("  Input:  {:?}", input);
            println!("  Output: {:?}", output);
            println!("✓ trap_test PASSED (no trap triggered)\n");
        } else {
            println!("✗ trap_test FAILED\n");
        }
    }

    // ====================================================================
    // Test 3: Assert test (with valid input - should NOT assert)
    // ====================================================================
    println!("--- Test 3: gpu_assert!() with valid input ---");
    {
        let input: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let n = input.len();

        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, n)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.assert_test(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &input_dev,
                &mut output_dev,
            )
        }?;
        stream.synchronize()?;

        let output: Vec<i32> = output_dev.to_host_vec(&stream)?;
        let expected: Vec<i32> = input.iter().map(|&x| x + 1).collect();

        if output == expected {
            println!("  Input:  {:?}", input);
            println!("  Output: {:?}", output);
            println!("✓ assert_test PASSED (no assertion failed)\n");
        } else {
            println!("✗ assert_test FAILED\n");
        }
    }

    // ====================================================================
    // Test 4: Breakpoint test (skipped - requires cuda-gdb)
    // ====================================================================
    println!("--- Test 4: breakpoint() ---");
    {
        // Note: brkpt instruction causes launch failure when not running
        // under cuda-gdb. This is expected behavior.
        println!("  ⚠ Skipping breakpoint_test (requires cuda-gdb)");
        println!("  To test: cuda-gdb --args ./target/release/debug --breakpoint");
        println!("  Then: run and hit breakpoint at thread 0\n");
    }

    // ====================================================================
    // Test 5: Profiler trigger test
    // ====================================================================
    println!("--- Test 5: prof_trigger() ---");
    {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = input.len();

        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.profiler_test(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &input_dev,
                &mut output_dev,
            )
        }?;
        stream.synchronize()?;

        let output: Vec<f32> = output_dev.to_host_vec(&stream)?;
        let expected: Vec<f32> = input.iter().map(|&x| x * x).collect();

        if output == expected {
            println!("  Input:  {:?}", input);
            println!("  Output: {:?}", output);
            println!("✓ profiler_test PASSED");
            println!("  (Use nsys/ncu to see profiler trigger events)\n");
        } else {
            println!("✗ profiler_test FAILED\n");
        }
    }

    // ====================================================================
    // Test 6: Launch bounds test
    // ====================================================================
    println!("--- Test 6: #[launch_bounds(128, 4)] ---");
    {
        let input: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let n = input.len();

        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, n)?;

        // launch_bounds_test has #[launch_bounds(128, 4)]
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.launch_bounds_test((stream).as_ref(), cfg, &input_dev, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<i32> = output_dev.to_host_vec(&stream)?;
        let expected: Vec<i32> = input.iter().map(|&x| x * 3 + 1).collect();

        if output == expected {
            println!("  Input:  {:?}", input);
            println!("  Output: {:?}", output);
            println!("✓ launch_bounds_test PASSED");
            println!("  (Check PTX for .maxntid 128 .minnctapersm 4)\n");
        } else {
            println!("✗ launch_bounds_test FAILED\n");
        }
    }

    println!("=== ALL DEBUG TESTS COMPLETED ===");
    Ok(())
}

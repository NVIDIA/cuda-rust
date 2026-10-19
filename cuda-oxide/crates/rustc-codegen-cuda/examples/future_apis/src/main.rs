/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Future APIs Test - Testing CuSimd<T, N> and ManagedBarrier typestate
//!
//! This example tests the new type-safe abstractions:
//! - `CuSimd<T, N>`: Generic SIMD type for multi-register values
//! - `ManagedBarrier<State, Kind>`: Typestate-based barrier management
//!
//! Run: cargo oxide run future_apis

use cuda_device::cusimd::CuSimd;
use cuda_device::{Barrier, GeneralBarrier, ManagedBarrier, TmaBarrier, Uninit};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// CUSIMD KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel demonstrating CuSimd<T, N> usage patterns.
    #[kernel]
    pub fn test_cusimd(mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let tid = idx.get();

        // Test 1: Construct CuSimd<f32, 4> from array
        let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let simd4 = CuSimd::<f32, 4>::new(values);

        // Test 2: Runtime index access via Index trait
        let val_at_tid = simd4[tid % 4];

        // Test 3: Shorthand accessors
        let x_val = simd4.x();
        let y_val = simd4.y();
        let z_val = simd4.z();
        let w_val = simd4.w();

        // Test 4: Compile-time indexed access
        let first = simd4.get::<0>();
        let last = simd4.get::<3>();

        // Test 5: Runtime at() method
        let dynamic = simd4.at(tid % 4);

        // Test 6: Convert to array
        let arr = simd4.to_array();

        // Test 7: CuSimd<f32, 2> with lo/hi accessors
        let simd2 = CuSimd::<f32, 2>::new([10.0, 20.0]);
        let (lo, hi) = simd2.xy();

        // Compute result based on thread ID
        let result = match tid % 8 {
            0 => val_at_tid,      // simd4[0] = 1.0
            1 => x_val,           // 1.0
            2 => y_val + z_val,   // 2.0 + 3.0 = 5.0
            3 => w_val,           // 4.0
            4 => first + last,    // 1.0 + 4.0 = 5.0
            5 => dynamic,         // simd4[5%4] = 2.0
            6 => arr[0] + arr[3], // 1.0 + 4.0 = 5.0
            7 => lo + hi,         // 10.0 + 20.0 = 30.0
            _ => 0.0,
        };

        if let Some(output_elem) = output.get_mut(idx) {
            *output_elem = result;
        }
    }

    /// Test kernel for CuSimd<u32, 4>
    #[kernel]
    pub fn test_cusimd_u32(mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let tid = idx.get();

        let simd = CuSimd::<u32, 4>::new([10, 20, 30, 40]);

        // Use runtime indexing
        let val = simd[tid % 4];

        if let Some(output_elem) = output.get_mut(idx) {
            *output_elem = val;
        }
    }

    // =============================================================================
    // BARRIER TYPESTATE KERNELS
    // =============================================================================

    /// Test kernel demonstrating ManagedBarrier<State, Kind> typestate pattern.
    ///
    /// This kernel tests the basic typestate flow with a single barrier:
    /// 1. from_static() - Create handle from static
    /// 2. init() - State transition from Uninit -> Ready
    /// 3. arrive() - All threads arrive at barrier
    /// 4. wait() - All threads wait for completion
    /// 5. inval() - State transition from Ready -> Invalidated
    #[kernel]
    pub fn test_managed_barrier(mut output: DisjointSlice<u32>) {
        // Declare barrier explicitly (like SharedArray)
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let is_thread0 = tid == 0;

        // ALL threads create Uninit handle (wraps same shared static)
        let bar = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR);

        // ALL threads call init - only thread 0 actually initializes
        // init() includes: thread 0 mbarrier_init + fence + sync_threads
        // All threads get Ready handle
        let bar = unsafe { bar.init(32) }; // 32 threads will participate

        // All threads arrive and wait
        let token = bar.arrive();
        bar.wait(token);

        // Thread 0 invalidates the barrier
        thread::sync_threads();
        if is_thread0 {
            // inval() transforms Ready -> Invalidated (consumes bar)
            let _dead = unsafe { bar.inval() };
        }

        // Write results - simple success indicator
        let idx = thread::index_1d();
        if let Some(output_elem) = output.get_mut(idx) {
            // If we reach here, barrier sync worked - write 42
            *output_elem = 42;
        }
    }

    /// Test kernel for multiple barriers using explicit static declarations.
    #[kernel]
    pub fn test_multi_barrier(mut output: DisjointSlice<u32>) {
        // Declare barriers explicitly - each gets a unique AllocId
        static mut BAR_TMA: Barrier = Barrier::UNINIT;
        static mut BAR_GEN: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let is_thread0 = tid == 0;

        // ALL threads create Uninit handles
        let bar_tma = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR_TMA);
        let bar_gen = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR_GEN);

        // ALL threads call init - each includes sync_threads internally
        let bar_tma = unsafe { bar_tma.init(32) };
        let bar_gen = unsafe { bar_gen.init(32) };

        // Use TMA barrier
        let token_tma = bar_tma.arrive();
        bar_tma.wait(token_tma);

        // Use General barrier
        let token_gen = bar_gen.arrive();
        bar_gen.wait(token_gen);

        // Thread 0 invalidates
        thread::sync_threads();
        if is_thread0 {
            unsafe {
                bar_tma.inval();
                bar_gen.inval();
            }
        }

        // Write results
        let idx = thread::index_1d();
        if let Some(output_elem) = output.get_mut(idx) {
            // If we reach here, both barrier syncs worked - write 99
            *output_elem = 99;
        }
    }

    /// Test kernel for double-buffered barriers (ping-pong pattern).
    #[kernel]
    pub fn test_double_buffered_barriers(mut output: DisjointSlice<u32>) {
        // Two barriers for double-buffering (explicit statics)
        static mut BUF0_BAR: Barrier = Barrier::UNINIT;
        static mut BUF1_BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let is_thread0 = tid == 0;

        // ALL threads create Uninit handles
        let b0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BUF0_BAR);
        let b1 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BUF1_BAR);

        // ALL threads call init - each includes sync_threads internally
        let buf0_bar = unsafe { b0.init(32) };
        let buf1_bar = unsafe { b1.init(32) };

        // Simulate double-buffering: alternate between barriers
        let mut sum: u32 = 0;

        // Iteration 0: use buf0
        let t0 = buf0_bar.arrive();
        buf0_bar.wait(t0);
        sum += 1;

        // Iteration 1: use buf1
        let t1 = buf1_bar.arrive();
        buf1_bar.wait(t1);
        sum += 1;

        // Iteration 2: use buf0 again (barriers are reusable within phase)
        let t2 = buf0_bar.arrive();
        buf0_bar.wait(t2);
        sum += 1;

        // Cleanup
        thread::sync_threads();
        if is_thread0 {
            unsafe {
                buf0_bar.inval();
                buf1_bar.inval();
            }
        }

        // Write result
        let idx = thread::index_1d();
        if let Some(output_elem) = output.get_mut(idx) {
            *output_elem = sum; // Should be 3
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== Future APIs Test (Unified) ===\n");

    let ctx = CudaContext::new(0)?;

    // sm_90, not sm_80. `ManagedBarrier::init_by` emits `fence.proxy.async`
    // for every barrier kind (the TmaBarrier typestate parameter is inert
    // PhantomData and detects nothing), and per the PTX ISA that fence needs
    // sm_90 or newer: ptxas rejects it below that with "Modifier '.async'
    // requires .target sm_90 or higher". The backend's feature scan classifies
    // the fence as Tma; with a device hint the target resolves to the device
    // (a Hopper box builds and runs this example at sm_90), while a hint-less
    // cross-compile defaults the module to sm_100 and a pre-Hopper device then
    // fails to load it with DriverError(218). Skip below sm_90 so the example
    // exercises real hardware everywhere it can actually run.
    let (major, minor) = ctx.compute_capability()?;
    if major < 9 {
        println!(
            "skipping: fence.proxy.async (emitted by ManagedBarrier::init_by) requires sm_90+ (device is sm_{major}{minor})"
        );
        return Ok(());
    }

    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    // ====================================================================
    // Test 1: CuSimd<f32, 4>
    // ====================================================================
    println!("--- Test 1: CuSimd<f32, 4> operations ---");
    {
        const N: usize = 32;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cusimd((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<f32> = output_dev.to_host_vec(&stream)?;

        let expected = [1.0f32, 1.0, 5.0, 4.0, 5.0, 2.0, 5.0, 30.0];

        let mut pass = true;
        for i in 0..N.min(16) {
            let exp = expected[i % 8];
            if (output[i] - exp).abs() > 0.001 {
                eprintln!("  Thread {}: expected {}, got {}", i, exp, output[i]);
                pass = false;
            }
        }

        if pass {
            println!("  Thread 0: {:.1} (val_at_tid)", output[0]);
            println!("  Thread 7: {:.1} (lo+hi)", output[7]);
            println!("✓ CuSimd<f32, 4> PASSED\n");
        } else {
            println!("✗ CuSimd<f32, 4> FAILED\n");
        }
    }

    // ====================================================================
    // Test 2: CuSimd<u32, 4>
    // ====================================================================
    println!("--- Test 2: CuSimd<u32, 4> indexing ---");
    {
        const N: usize = 8;
        let mut output_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cusimd_u32((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<u32> = output_dev.to_host_vec(&stream)?;

        let expected = [10u32, 20, 30, 40, 10, 20, 30, 40];
        let pass = output.iter().zip(expected.iter()).all(|(a, b)| a == b);

        if pass {
            println!("  Results: {:?}", output);
            println!("✓ CuSimd<u32, 4> PASSED\n");
        } else {
            println!("✗ CuSimd<u32, 4> FAILED\n");
        }
    }

    // ====================================================================
    // Test 3: Single ManagedBarrier
    // ====================================================================
    println!("--- Test 3: ManagedBarrier<Uninit/Ready> typestate ---");
    {
        const N: usize = 32;
        let mut output_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_managed_barrier((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<u32> = output_dev.to_host_vec(&stream)?;

        // All threads should write 42 if barrier worked
        let pass = output.iter().all(|&v| v == 42);

        if pass {
            println!("  All {} threads wrote 42 after barrier sync", N);
            println!("✓ ManagedBarrier PASSED\n");
        } else {
            println!("  Results: {:?}", &output[..8.min(output.len())]);
            println!("✗ ManagedBarrier FAILED\n");
        }
    }

    // ====================================================================
    // Test 4: Multiple Barriers (TMA + General)
    // ====================================================================
    println!("--- Test 4: Multiple barriers (TmaBarrier + GeneralBarrier) ---");
    {
        const N: usize = 32;
        let mut output_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_multi_barrier((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<u32> = output_dev.to_host_vec(&stream)?;

        // All threads should write 99 if both barriers worked
        let pass = output.iter().all(|&v| v == 99);

        if pass {
            println!("  All {} threads wrote 99 after both barrier syncs", N);
            println!("✓ Multi-barrier PASSED\n");
        } else {
            println!("  Results: {:?}", &output[..8.min(output.len())]);
            println!("✗ Multi-barrier FAILED\n");
        }
    }

    // ====================================================================
    // Test 5: Double-Buffered Barriers
    // ====================================================================
    println!("--- Test 5: Double-buffered barriers (ping-pong) ---");
    {
        const N: usize = 32;
        let mut output_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_double_buffered_barriers((stream).as_ref(), cfg, &mut output_dev) }?;
        stream.synchronize()?;

        let output: Vec<u32> = output_dev.to_host_vec(&stream)?;

        // All threads should write 3 (3 iterations)
        let pass = output.iter().all(|&v| v == 3);

        if pass {
            println!("  All {} threads completed 3 iterations", N);
            println!("✓ Double-buffered PASSED\n");
        } else {
            println!("  Results: {:?}", &output[..8.min(output.len())]);
            println!("✗ Double-buffered FAILED\n");
        }
    }

    println!("=== ALL FUTURE APIS TESTS COMPLETED ===");
    Ok(())
}

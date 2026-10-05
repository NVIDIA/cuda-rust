/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared memory is accessed by thread-derived index, not an iterator.
#![allow(clippy::needless_range_loop)]

//! Unified Dynamic Shared Memory Example
//!
//! Demonstrates `DynamicSharedArray<T, ALIGN>` for runtime-sized shared memory.
//!
//! ## Test Scenarios
//!
//! 1. **dynamic_smem_basic** - Default alignment (16 bytes)
//! 2. **dynamic_smem_partition** - Partitioning multiple arrays
//! 3. **dynamic_smem_explicit_align** - Explicit 128-byte alignment (TMA-compatible)
//! 4. **dynamic_smem_mixed_align** - Multiple calls with different alignments
//!
//! ## Key Features Tested
//!
//! - Per-kernel symbols: Each kernel gets `__dynamic_smem_{crate_id}_{kernel_name}`
//! - Max alignment: Multiple calls with different ALIGN values use the maximum
//! - User-specified alignment: Can specify any power-of-2 alignment
//!
//! Build and run with:
//!   cargo oxide run dynamic_smem

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, DynamicSharedArray, gpu_printf, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test 1: Basic dynamic shared memory (default 16-byte alignment).
    ///
    /// Each thread writes its value to dynamic shared memory, syncs,
    /// then reads from its neighbor.
    ///
    /// PTX output: `.extern .shared .align 16 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_basic[];`
    #[kernel]
    pub fn dynamic_smem_basic(data: &[f32], mut out: DisjointSlice<f32>) {
        // Default alignment (16 bytes, matches nvcc)
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Print address from thread 0 (sanity check)
        if tid == 0 {
            gpu_printf!("[basic] smem addr: {:x} (align 16)\n", smem as u64);
        }

        unsafe {
            *smem.add(tid) = data[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem.add(neighbor_idx);
            }
        }
    }

    /// Test 2: Dynamic shared memory with partitioning.
    ///
    /// Uses `DynamicSharedArray::offset()` to partition the shared memory into
    /// two arrays (A and B).
    ///
    /// Memory layout:
    /// ```text
    /// |<-- array_a (N f32s) -->|<-- array_b (N f32s) -->|
    /// |        offset 0        |    offset N*4 bytes    |
    /// ```
    ///
    /// PTX output: `.extern .shared .align 16 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_partition[];`
    #[kernel]
    pub fn dynamic_smem_partition(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // First array at offset 0 (default alignment)
        let smem_a: *mut f32 = DynamicSharedArray::<f32>::get();

        // Second array at offset = N * sizeof(f32)
        // For 256 elements: 256 * 4 = 1024 bytes
        let smem_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Print addresses from thread 0 (sanity check)
        if tid == 0 {
            gpu_printf!(
                "[partition] smem_a: {:x}, smem_b: {:x} (diff={})\n",
                smem_a as u64,
                smem_b as u64,
                (smem_b as u64) - (smem_a as u64)
            );
        }

        unsafe {
            *smem_a.add(tid) = a[gid];
            *smem_b.add(tid) = b[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem_a.add(neighbor_idx) + *smem_b.add(neighbor_idx);
            }
        }
    }

    /// Test 3: Explicit 128-byte alignment (TMA-compatible).
    ///
    /// Demonstrates explicit alignment specification for TMA operations.
    ///
    /// PTX output: `.extern .shared .align 128 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_explicit_align[];`
    #[kernel]
    pub fn dynamic_smem_explicit_align(data: &[f32], mut out: DisjointSlice<f32>) {
        // Explicit 128-byte alignment (required for TMA)
        let smem: *mut f32 = DynamicSharedArray::<f32, 128>::get();

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Print address from thread 0 (sanity check - should be 128-byte aligned)
        if tid == 0 {
            let addr = smem as u64;
            gpu_printf!(
                "[explicit] smem addr: {:x} (align 128, mod128={})\n",
                addr,
                addr % 128
            );
        }

        unsafe {
            *smem.add(tid) = data[gid] * 2.0;
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem.add(neighbor_idx);
            }
        }
    }

    /// Test 4: Mixed alignments within same kernel (uses maximum).
    ///
    /// This kernel has multiple DynamicSharedArray calls with different alignments.
    /// The compiler pre-pass computes max(16, 128, 256) = 256 and uses that.
    ///
    /// PTX output: `.extern .shared .align 256 .b8 __dynamic_smem_00f223822842cf67_dynamic_smem_mixed_align[];`
    #[kernel]
    pub fn dynamic_smem_mixed_align(a: &[f32], b: &[f32], c: &[f32], mut out: DisjointSlice<f32>) {
        // Three partitions with different alignment requirements
        // The compiler uses max(16, 128, 256) = 256 for the global
        let smem_a: *mut f32 = DynamicSharedArray::<f32>::get(); // ALIGN = 16 (default)
        let smem_b: *mut f32 = DynamicSharedArray::<f32, 128>::offset(1024); // ALIGN = 128
        let smem_c: *mut f32 = DynamicSharedArray::<f32, 256>::offset(2048); // ALIGN = 256

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Print addresses from thread 0 (sanity check - base should be 256-byte aligned)
        if tid == 0 {
            let addr_a = smem_a as u64;
            let addr_b = smem_b as u64;
            let addr_c = smem_c as u64;
            gpu_printf!("[mixed] smem_a: {:x} (mod256={})\n", addr_a, addr_a % 256);
            gpu_printf!(
                "[mixed] smem_b: {:x} (offset={})\n",
                addr_b,
                addr_b - addr_a
            );
            gpu_printf!(
                "[mixed] smem_c: {:x} (offset={})\n",
                addr_c,
                addr_c - addr_a
            );
        }

        unsafe {
            *smem_a.add(tid) = a[gid];
            *smem_b.add(tid) = b[gid];
            *smem_c.add(tid) = c[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            // Sum all three neighbors
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem_a.add(neighbor_idx)
                    + *smem_b.add(neighbor_idx)
                    + *smem_c.add(neighbor_idx);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Dynamic Shared Memory Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test size
    const N: usize = 256;
    const BLOCK_SIZE: u32 = 256;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // ===== Test 1: Basic Dynamic Shared Memory (default 16-byte alignment) =====
    println!("=== Test 1: Basic DynamicSharedArray (default alignment) ===");
    {
        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

        println!("Input data[0..5] = {:?}", &data_host[0..5]);

        let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (N * core::mem::size_of::<f32>()) as u32,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.dynamic_smem_basic((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = data[(i + 1) % N]
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (data[{}]), got {}",
                    i, expected, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Basic DynamicSharedArray (align 16): correct neighbor read\n");
    }

    // ===== Test 2: Partitioned Dynamic Shared Memory =====
    println!("=== Test 2: Partitioned DynamicSharedArray ===");
    {
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i + 100) as f32).collect();

        println!("Input a[0..5] = {:?}", &a_host[0..5]);
        println!("Input b[0..5] = {:?}", &b_host[0..5]);

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // 2 arrays * N elements * 4 bytes = 2048 bytes
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (2 * N * core::mem::size_of::<f32>()) as u32,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.dynamic_smem_partition((stream).as_ref(), cfg, &a_dev, &b_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = a[(i+1)%N] + b[(i+1)%N]
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (a[{}]+b[{}]), got {}",
                    i, expected, neighbor_idx, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Partitioned DynamicSharedArray (align 16): correct neighbor sum\n");
    }

    // ===== Test 3: Explicit 128-byte alignment (TMA-compatible) =====
    println!("=== Test 3: Explicit 128-byte Alignment ===");
    {
        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

        println!("Input data[0..5] = {:?}", &data_host[0..5]);

        let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (N * core::mem::size_of::<f32>()) as u32,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.dynamic_smem_explicit_align((stream).as_ref(), cfg, &data_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = data[(i+1)%N] * 2.0
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx] * 2.0;
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (data[{}]*2), got {}",
                    i, expected, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Explicit alignment (align 128): correct neighbor read\n");
    }

    // ===== Test 4: Mixed alignments (uses maximum = 256) =====
    println!("=== Test 4: Mixed Alignments (max = 256) ===");
    {
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let c_host: Vec<f32> = (0..N).map(|i| (i * 3) as f32).collect();

        println!("Input a[0..5] = {:?}", &a_host[0..5]);
        println!("Input b[0..5] = {:?}", &b_host[0..5]);
        println!("Input c[0..5] = {:?}", &c_host[0..5]);

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let c_dev = DeviceBuffer::from_host(&stream, &c_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // 3 arrays * N elements * 4 bytes = 3072 bytes
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (3 * N * core::mem::size_of::<f32>()) as u32,
        };

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.dynamic_smem_mixed_align(
                (stream).as_ref(),
                cfg,
                &a_dev,
                &b_dev,
                &c_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = a[(i+1)%N] + b[(i+1)%N] + c[(i+1)%N]
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx] + c_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (a+b+c at {}), got {}",
                    i, expected, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Mixed alignments (max align 256): correct sum of all three\n");
    }

    println!("✓ SUCCESS: All dynamic shared memory tests passed!");
    println!("\nPTX symbols to verify (<crate-id> is this crate's stable id):");
    println!("  - __dynamic_smem_<crate-id>_dynamic_smem_basic        (align 16)");
    println!("  - __dynamic_smem_<crate-id>_dynamic_smem_partition    (align 16)");
    println!("  - __dynamic_smem_<crate-id>_dynamic_smem_explicit_align (align 128)");
    println!("  - __dynamic_smem_<crate-id>_dynamic_smem_mixed_align  (align 256)");
}

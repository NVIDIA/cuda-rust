/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared memory is accessed by thread-derived index, not an iterator.
#![allow(clippy::needless_range_loop)]

//! Unified Shared Memory Example
//!
//! Demonstrates SharedArray for block-level cooperation:
//! 1. shared_test - single SharedArray: loads data, syncs, reads neighbor
//! 2. shared_dual - two SharedArrays: loads from a and b, syncs, adds neighbors
//!
//! Build and run with:
//!   cargo oxide run sharedmem

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel for shared memory with single array
    #[kernel]
    pub fn shared_test(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Write to shared memory
        unsafe {
            TILE[tid] = data[gid];
        }

        thread::sync_threads();

        // Read from shared memory (neighbor)
        unsafe {
            let neighbor_idx = (tid + 1) % 256;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE[neighbor_idx];
            }
        }
    }

    /// Test kernel with TWO shared arrays - tests multiple shared allocations
    #[kernel]
    pub fn shared_dual(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // Two separate shared memory allocations
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Load both arrays into shared memory
        unsafe {
            TILE_A[tid] = a[gid];
            TILE_B[tid] = b[gid];
        }

        thread::sync_threads();

        // Read neighbor's value from both tiles and add them
        unsafe {
            let neighbor_idx = (tid + 1) % 256;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE_A[neighbor_idx] + TILE_B[neighbor_idx];
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Shared Memory Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test size - must match TILE size (256 elements)
    const N: usize = 256;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // Launch config for shared memory kernels
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: Single SharedArray =====
    println!("=== Test 1: Single SharedArray ===");
    {
        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

        println!("Input data[0..5] = {:?}", &data_host[0..5]);

        let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.shared_test((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = data[(i + 1) % 256]
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
        println!("✓ Single SharedArray: correct neighbor read\n");
    }

    // ===== Test 2: Dual SharedArray =====
    println!("=== Test 2: Dual SharedArray ===");
    {
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i + 100) as f32).collect();

        println!("Input a[0..5] = {:?}", &a_host[0..5]);
        println!("Input b[0..5] = {:?}", &b_host[0..5]);

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.shared_dual((stream).as_ref(), cfg, &a_dev, &b_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = a[(i+1)%256] + b[(i+1)%256]
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
        println!("✓ Dual SharedArray: correct neighbor sum from both tiles\n");
    }

    println!("✓ SUCCESS: All shared memory tests passed!");
}

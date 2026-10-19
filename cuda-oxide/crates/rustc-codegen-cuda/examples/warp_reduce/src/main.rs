/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::needless_range_loop)]

//! Unified Warp Reduction Example
//!
//! Demonstrates warp-level primitives: shuffle_xor, shuffle_down, shuffle (broadcast),
//! and the packaged reduce utilities (reduce_sum/max/min for f32 and f64).
//! Uses butterfly reduction pattern for efficient parallel sum.
//!
//! Build and run with:
//!   cargo oxide run warp_reduce

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::cooperative_groups::{
    ThreadGroup, block_reduce, block_scan, ops::Sum, this_thread_block,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Warp reduction using shuffle_xor (butterfly pattern).
    /// All lanes end up with the complete sum.
    #[kernel]
    pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        // Load value (or 0 if out of bounds)
        let mut val = if gid.in_bounds(out.len() * 32) {
            data[gid.get()]
        } else {
            0.0
        };

        // Butterfly reduction using shuffle_xor
        val = val + warp::shuffle_xor_f32(val, 16);
        val = val + warp::shuffle_xor_f32(val, 8);
        val = val + warp::shuffle_xor_f32(val, 4);
        val = val + warp::shuffle_xor_f32(val, 2);
        val = val + warp::shuffle_xor_f32(val, 1);

        // Lane 0 writes the result
        if lane == 0 {
            let warp_idx = gid.get() / 32;
            unsafe {
                *out.get_unchecked_mut(warp_idx) = val;
            }
        }
    }

    /// Warp reduction using shuffle_down (sequential pattern).
    /// Only lane 0 has the complete sum.
    #[kernel]
    pub fn warp_reduce_sum_down(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        let mut val = if gid.in_bounds(out.len() * 32) {
            data[gid.get()]
        } else {
            0.0
        };

        // Sequential reduction using shuffle_down
        val = val + warp::shuffle_down_f32(val, 16);
        val = val + warp::shuffle_down_f32(val, 8);
        val = val + warp::shuffle_down_f32(val, 4);
        val = val + warp::shuffle_down_f32(val, 2);
        val = val + warp::shuffle_down_f32(val, 1);

        // Only lane 0 has the complete sum
        if lane == 0 {
            let warp_idx = gid.get() / 32;
            unsafe {
                *out.get_unchecked_mut(warp_idx) = val;
            }
        }
    }

    /// Broadcast kernel - shuffle lane 0's value to all lanes.
    #[kernel]
    pub fn warp_broadcast(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();

        let my_val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        // Broadcast lane 0's value to all lanes
        let broadcast_val = warp::shuffle_f32(my_val, 0);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = broadcast_val;
        }
    }

    /// Test lane_id() intrinsic - each thread writes its lane ID.
    #[kernel]
    pub fn test_lane_id(mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        if gid.in_bounds(out.len()) {
            // Each thread writes its lane ID
            unsafe {
                *out.get_unchecked_mut(gid.get()) = lane;
            }
        }
    }

    /// Reduce-sum utility - every lane writes the warp sum it received.
    #[kernel]
    pub fn warp_reduce_sum_util(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();

        let val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        let sum = warp::reduce_sum_f32(val);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = sum;
        }
    }

    /// Reduce-max utility - every lane writes the warp maximum it received.
    #[kernel]
    pub fn warp_reduce_max_util(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();

        let val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        let max = warp::reduce_max_f32(val);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = max;
        }
    }

    /// Reduce-min utility - every lane writes the warp minimum it received.
    #[kernel]
    pub fn warp_reduce_min_util(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();

        let val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        let min = warp::reduce_min_f32(val);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = min;
        }
    }

    /// Reduce-sum utility (f64) - every lane writes the warp sum it received.
    #[kernel]
    pub fn warp_reduce_sum_f64_util(data: &[f64], mut out: DisjointSlice<f64>) {
        let gid = thread::index_1d();

        let val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        let sum = warp::reduce_sum_f64(val);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = sum;
        }
    }

    /// Exercise all live warp prefixes, extra scratch, and reuse in 1D/2D/3D.
    #[kernel]
    pub fn block_reduce_partial_warp(mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 32> = SharedArray::UNINIT;
        let block = this_thread_block();
        let rank = block.thread_rank();
        let index = thread::blockIdx_x() as usize * block.size() as usize + rank as usize;
        // SAFETY: every block thread uses this allocation, with no other accesses.
        let total = unsafe { block_reduce::<u32, Sum, 32>(&block, rank + 1, &raw mut SMEM) };
        block.sync();
        // Reuse catches a missing publication/read barrier between scratch phases.
        let second = unsafe { block_reduce::<u32, Sum, 32>(&block, 2, &raw mut SMEM) };
        cuda_device::gpu_assert!(second == 2 * block.size());
        if index < out.len() {
            // SAFETY: the linear rank is distinct for every thread in this launch.
            unsafe { *out.get_unchecked_mut(index) = total };
        }
    }

    #[kernel]
    pub fn block_scan_partial_warp_distinct(mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 32> = SharedArray::UNINIT;
        let block = this_thread_block();
        let rank = block.thread_rank();
        let index = thread::blockIdx_x() as usize * block.size() as usize + rank as usize;
        // SAFETY: every block thread uses this allocation, with no other accesses.
        let prefix = unsafe { block_scan::<u32, Sum, 32>(&block, rank + 1, &raw mut SMEM) };
        block.sync();
        let second = unsafe { block_scan::<u32, Sum, 32>(&block, 2, &raw mut SMEM) };
        cuda_device::gpu_assert!(second == 2 * (rank + 1));
        if index < out.len() {
            // SAFETY: the linear rank is distinct for every thread in this launch.
            unsafe { *out.get_unchecked_mut(index) = prefix };
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Warp Reduction Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 256;
    const WARPS: usize = N / 32;

    // Initialize input: each warp has values 0-31, so sum = 496
    let data_host: Vec<f32> = (0..N).map(|i| (i % 32) as f32).collect();

    println!("Input data: {} elements, {} warps", N, WARPS);
    println!("  First warp values: {:?}", &data_host[0..8]);
    println!("  Expected warp sum: {}", (0..32).sum::<i32>());

    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: Butterfly reduction =====
    println!("\n--- Test 1: Butterfly Reduction (shuffle_xor) ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_sum((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    let expected_sum = (0..32).sum::<i32>() as f32;
    let all_correct = out_result.iter().all(|&x| (x - expected_sum).abs() < 1e-5);

    if all_correct {
        println!(
            "✓ All {} warp sums correct (each = {})",
            WARPS, expected_sum
        );
    } else {
        println!("✗ Some warp sums incorrect!");
        std::process::exit(1);
    }

    // ===== Test 2: Sequential reduction =====
    println!("\n--- Test 2: Sequential Reduction (shuffle_down) ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_sum_down((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    let all_correct = out_result.iter().all(|&x| (x - expected_sum).abs() < 1e-5);
    if all_correct {
        println!(
            "✓ All {} warp sums correct (each = {})",
            WARPS, expected_sum
        );
    } else {
        println!("✗ Some warp sums incorrect!");
        std::process::exit(1);
    }

    // ===== Test 3: Broadcast =====
    println!("\n--- Test 3: Broadcast (shuffle to lane 0) ---");
    let broadcast_input: Vec<f32> = (0..N).map(|i| (i * 10) as f32).collect();
    let broadcast_dev = DeviceBuffer::from_host(&stream, &broadcast_input).unwrap();
    let mut broadcast_out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.warp_broadcast(
            (stream).as_ref(),
            cfg,
            &broadcast_dev,
            &mut broadcast_out_dev,
        )
    }
    .expect("Kernel launch failed");

    let broadcast_result = broadcast_out_dev.to_host_vec(&stream).unwrap();

    println!("Broadcast results (first 8 of each warp):");
    for warp in 0..WARPS.min(4) {
        let start = warp * 32;
        println!("  Warp {}: {:?}", warp, &broadcast_result[start..start + 8]);
    }

    // Verify each warp has all same values
    let mut broadcast_correct = true;
    for warp in 0..WARPS {
        let start = warp * 32;
        let expected = broadcast_input[start]; // Lane 0's value
        for lane in 0..32 {
            if (broadcast_result[start + lane] - expected).abs() > 1e-5 {
                println!(
                    "Mismatch: warp {} lane {} expected {} got {}",
                    warp,
                    lane,
                    expected,
                    broadcast_result[start + lane]
                );
                broadcast_correct = false;
            }
        }
    }

    if broadcast_correct {
        println!("✓ Broadcast correct: all lanes have lane 0's value");
    } else {
        println!("✗ Broadcast failed!");
        std::process::exit(1);
    }

    // ===== Test 4: Lane ID =====
    println!("\n--- Test 4: Lane ID ---");
    let mut lane_out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_lane_id((stream).as_ref(), cfg, &mut lane_out_dev) }
        .expect("Kernel launch failed");

    let lane_result = lane_out_dev.to_host_vec(&stream).unwrap();

    println!("Lane IDs (first 8 of each of first 2 warps):");
    for warp in 0..2.min(WARPS) {
        let start = warp * 32;
        println!("  Warp {}: {:?}", warp, &lane_result[start..start + 8]);
    }

    // Verify lane IDs are 0-31 repeating for each warp
    let mut lane_correct = true;
    for i in 0..N {
        let expected_lane = (i % 32) as u32;
        if lane_result[i] != expected_lane {
            println!(
                "Mismatch at {}: expected lane {} got {}",
                i, expected_lane, lane_result[i]
            );
            lane_correct = false;
        }
    }

    if lane_correct {
        println!("✓ Lane IDs correct: 0-31 pattern for each warp");
    } else {
        println!("✗ Lane ID test failed!");
        std::process::exit(1);
    }

    // ===== Test 5: Reduce utilities (f32) =====
    // Each warp holds lane indices 0-31: sum = 496, max = 31, min = 0.
    // All values are exact in f32, so exact comparison is safe, and every
    // lane must observe the reduced value (not just lane 0).
    println!("\n--- Test 5: Reduce Utilities (f32) ---");

    let mut sum_out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_sum_util((stream).as_ref(), cfg, &data_dev, &mut sum_out_dev) }
        .expect("Kernel launch failed");
    let sum_result = sum_out_dev.to_host_vec(&stream).unwrap();
    if sum_result.iter().all(|&x| x == 496.0) {
        println!("✓ reduce_sum_f32 correct: all {} lanes hold 496", N);
    } else {
        println!("✗ reduce_sum_f32 failed: {:?}", &sum_result[0..8]);
        std::process::exit(1);
    }

    let mut max_out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_max_util((stream).as_ref(), cfg, &data_dev, &mut max_out_dev) }
        .expect("Kernel launch failed");
    let max_result = max_out_dev.to_host_vec(&stream).unwrap();
    if max_result.iter().all(|&x| x == 31.0) {
        println!("✓ reduce_max_f32 correct: all {} lanes hold 31", N);
    } else {
        println!("✗ reduce_max_f32 failed: {:?}", &max_result[0..8]);
        std::process::exit(1);
    }

    // Seed with a sentinel so a kernel that never wrote would be caught
    // (the expected minimum is 0.0, which a zeroed buffer already holds).
    let min_sentinel = vec![-1.0f32; N];
    let mut min_out_dev = DeviceBuffer::from_host(&stream, &min_sentinel).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_min_util((stream).as_ref(), cfg, &data_dev, &mut min_out_dev) }
        .expect("Kernel launch failed");
    let min_result = min_out_dev.to_host_vec(&stream).unwrap();
    if min_result.iter().all(|&x| x == 0.0) {
        println!("✓ reduce_min_f32 correct: all {} lanes hold 0", N);
    } else {
        println!("✗ reduce_min_f32 failed: {:?}", &min_result[0..8]);
        std::process::exit(1);
    }

    // ===== Test 6: Reduce utility (f64) =====
    println!("\n--- Test 6: Reduce Utility (f64) ---");
    let data_f64_host: Vec<f64> = (0..N).map(|i| (i % 32) as f64).collect();
    let data_f64_dev = DeviceBuffer::from_host(&stream, &data_f64_host).unwrap();
    let mut f64_out_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.warp_reduce_sum_f64_util((stream).as_ref(), cfg, &data_f64_dev, &mut f64_out_dev)
    }
    .expect("Kernel launch failed");

    let f64_result = f64_out_dev.to_host_vec(&stream).unwrap();
    let expected_sum_f64 = (0..32).sum::<i32>() as f64;
    let all_lanes_correct = f64_result.iter().all(|&x| x == expected_sum_f64);
    if all_lanes_correct {
        println!(
            "✓ reduce_sum_f64 correct: all {} lanes hold {}",
            N, expected_sum_f64
        );
    } else {
        println!("✗ reduce_sum_f64 failed: {:?}", &f64_result[0..8]);
        std::process::exit(1);
    }

    // All prefix lengths, full-warp controls, maximum capacity, and 2D/3D ranks.
    let shapes = (1..=65).map(|x| (x, 1, 1)).chain([
        (95, 1, 1),
        (96, 1, 1),
        (97, 1, 1),
        (255, 1, 1),
        (256, 1, 1),
        (257, 1, 1),
        (1023, 1, 1),
        (1024, 1, 1),
        (3, 5, 3),
        (8, 3, 2),
        (7, 3, 3),
        (8, 8, 8),
    ]);
    let mut shape_count = 0;
    for shape in shapes {
        let threads = shape.0 * shape.1 * shape.2;
        let cfg = LaunchConfig {
            block_dim: shape,
            grid_dim: (2, 1, 1),
            shared_mem_bytes: 0,
        };
        let sentinel = vec![u32::MAX; 2 * threads as usize];
        let mut reduced = DeviceBuffer::from_host(&stream, &sentinel).unwrap();
        let mut scanned = DeviceBuffer::from_host(&stream, &sentinel).unwrap();
        // SAFETY: both CTAs run every thread through the collectives and outputs
        // contain one slot per linear thread rank.
        unsafe {
            module
                .block_reduce_partial_warp(stream.as_ref(), cfg, &mut reduced)
                .expect("block_reduce launch failed");
            module
                .block_scan_partial_warp_distinct(stream.as_ref(), cfg, &mut scanned)
                .expect("block_scan launch failed");
        }
        let reduced = reduced.to_host_vec(&stream).unwrap();
        let scanned = scanned.to_host_vec(&stream).unwrap();
        let total = threads * (threads + 1) / 2;
        for (index, (&sum, &prefix)) in reduced.iter().zip(&scanned).enumerate() {
            let rank = index as u32 % threads + 1;
            assert_eq!(sum, total, "reduce {shape:?}, index {index}");
            assert_eq!(
                prefix,
                rank * (rank + 1) / 2,
                "scan {shape:?}, index {index}"
            );
        }
        shape_count += 1;
    }
    println!(
        "✓ block reduction and scan passed {shape_count} launch shapes, two CTAs, and scratch reuse"
    );
    println!("\n✓ SUCCESS: All warp tests passed!");
}

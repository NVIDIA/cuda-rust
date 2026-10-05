/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread Block Cluster Test - Testing Hopper cluster intrinsics (sm_90+)
//!
//! This example demonstrates:
//! - Cluster special registers (`cluster_ctaidX`, `cluster_nctaidX`, etc.)
//! - Cluster synchronization (`cluster_sync`)
//! - Distributed shared memory (`map_shared_rank`)
//!
//! **Hardware Requirements:** Hopper (H100, H200) or newer GPUs with sm_90+
//!
//! Uses unified compilation: single `cargo oxide run cluster`

use core::ptr::{addr_of, addr_of_mut};
use cuda_device::{DisjointSlice, SharedArray, cluster, cluster_launch, kernel, thread};
use cuda_host::cuda_module;

// ============================================================================
// Test 0: Compile-Time Cluster Configuration with #[cluster(x,y,z)]
// ============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Keeps the generated cluster-grid helper path in the compiled module.
    #[kernel]
    pub fn compile_cluster_grid_helpers(mut output: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0 && output.len() >= 2 {
            // SAFETY: the length check covers both output slots.
            unsafe {
                *output.get_unchecked_mut(0) = cluster::cluster_idx();
                *output.get_unchecked_mut(1) = cluster::num_clusters();
            }
        }
    }

    /// Test kernel with compile-time cluster configuration.
    ///
    /// The `#[cluster_launch(4, 1, 1)]` attribute tells the compiler to emit:
    /// ```ptx
    /// .entry test_cluster_compile_time .reqnctapercluster 4, 1, 1 { ... }
    /// ```
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn test_cluster_compile_time(mut output: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let my_rank = cluster::block_rank();
        let cluster_size = cluster::cluster_size();

        // Write cluster info to output
        if tid == 0 {
            let idx = my_rank as usize;
            if idx < output.len() {
                // Encode: high 16 bits = rank, low 16 bits = cluster_size
                let value = ((my_rank as u32) << 16) | (cluster_size as u32);
                unsafe { *output.get_unchecked_mut(idx) = value };
            }
        }
    }

    // ============================================================================
    // Test 1: Basic Cluster Intrinsics
    // ============================================================================

    /// Test kernel for cluster special registers.
    #[kernel]
    pub fn test_cluster_intrinsics(mut output: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();

        // Offset for this block's output
        let base = (bid as usize) * 8;

        let value = match tid {
            0 => cluster::cluster_ctaidX(),
            1 => cluster::cluster_ctaidY(),
            2 => cluster::cluster_ctaidZ(),
            3 => cluster::cluster_nctaidX(),
            4 => cluster::cluster_nctaidY(),
            5 => cluster::cluster_nctaidZ(),
            6 => cluster::block_rank(),
            7 => cluster::cluster_size(),
            _ => 0xDEADBEEF,
        };

        if tid < 8 {
            let idx = base + tid as usize;
            if idx < output.len() {
                unsafe { *output.get_unchecked_mut(idx) = value };
            }
        }
    }

    // ============================================================================
    // Test 2: Cluster Synchronization
    // ============================================================================

    /// Test kernel for cluster_sync().
    ///
    /// `cluster_sync` only has meaning when the kernel actually runs as a
    /// thread-block cluster, so this must carry `#[cluster_launch(...)]` like
    /// the DSMEM tests below. Without it the grid launches as ordinary blocks,
    /// every block is its own size-1 cluster, `block_rank()` is always 0, and
    /// only `output[0]` is written.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn test_cluster_sync(mut output: DisjointSlice<u32>) {
        static mut SHMEM: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let my_rank = cluster::block_rank();

        // Step 1: Thread 0 writes block's rank to shared memory
        if tid == 0 {
            unsafe { (addr_of_mut!(SHMEM) as *mut u32).write(my_rank * 100 + 42) };
        }
        thread::sync_threads();

        // Step 2: Synchronize entire cluster
        cluster::cluster_sync();

        // Step 3: Write result (proves we passed the barrier)
        if tid == 0 {
            let idx = my_rank as usize;
            if idx < output.len() {
                let local_value = unsafe { *(addr_of!(SHMEM) as *const u32) };
                unsafe { *output.get_unchecked_mut(idx) = local_value };
            }
        }
    }

    // ============================================================================
    // Test 3: Distributed Shared Memory (Ring Exchange)
    // ============================================================================

    /// Test kernel for distributed shared memory via map_shared_rank() + dereference.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn test_dsmem_ring_exchange(mut output: DisjointSlice<u32>) {
        static mut SHMEM: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let my_rank = cluster::block_rank();
        let cluster_size = cluster::cluster_size();

        // Step 1: Each block writes its unique value to shared memory
        if tid == 0 {
            unsafe { (addr_of_mut!(SHMEM) as *mut u32).write(1000 + my_rank) };
        }
        thread::sync_threads();

        // Step 2: Cluster-wide sync to ensure all blocks have written
        cluster::cluster_sync();

        // Step 3: Read neighbor's shared memory (ring pattern)
        if tid == 0 {
            let neighbor_rank = (my_rank + 1) % cluster_size;
            let neighbor_ptr =
                unsafe { cluster::map_shared_rank(addr_of!(SHMEM) as *const u32, neighbor_rank) };
            let neighbor_value = unsafe { *neighbor_ptr };

            let idx = my_rank as usize;
            if idx < output.len() {
                unsafe { *output.get_unchecked_mut(idx) = neighbor_value };
            }
        }
    }

    // ============================================================================
    // Test 4: Distributed Reduction (All-to-One)
    // ============================================================================

    /// Test kernel for distributed reduction using DSMEM.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn test_dsmem_reduction(mut output: DisjointSlice<u32>) {
        static mut LOCAL_VAL: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let my_rank = cluster::block_rank();
        let cluster_size = cluster::cluster_size();

        // Step 1: Each block writes its contribution
        if tid == 0 {
            unsafe { (addr_of_mut!(LOCAL_VAL) as *mut u32).write((my_rank + 1) * 10) };
        }
        thread::sync_threads();

        // Step 2: Cluster-wide sync
        cluster::cluster_sync();

        // Step 3: Block 0 reads all blocks' values and sums them
        if tid == 0 && my_rank == 0 {
            let mut total = unsafe { *(addr_of!(LOCAL_VAL) as *const u32) };

            let mut rank = 1u32;
            while rank < cluster_size {
                total +=
                    unsafe { cluster::dsmem_read_u32(addr_of!(LOCAL_VAL) as *const u32, rank) };
                rank += 1;
            }

            if !output.is_empty() {
                unsafe { *output.get_unchecked_mut(0) = total };
            }
        }

        // All blocks must stay alive while block 0 reads DSMEM
        cluster::cluster_sync();
    }

    // ============================================================================
    // Test 5: Distributed Shared Memory (Mapped Remote Store)
    // ============================================================================

    /// Test map_shared_rank_mut() followed by an ordinary Rust pointer store.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn test_dsmem_mapped_store(mut output: DisjointSlice<u32>) {
        static mut SHMEM: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let my_rank = cluster::block_rank();
        let cluster_size = cluster::cluster_size();

        if tid == 0 {
            unsafe { (addr_of_mut!(SHMEM) as *mut u32).write(2000 + my_rank) };
        }
        thread::sync_threads();
        cluster::cluster_sync();

        if tid == 0 {
            let neighbor_rank = (my_rank + 1) % cluster_size;
            let neighbor_ptr = unsafe {
                cluster::map_shared_rank_mut(addr_of_mut!(SHMEM) as *mut u32, neighbor_rank)
            };
            unsafe { *neighbor_ptr = 3000 + my_rank };
        }

        cluster::cluster_sync();

        if tid == 0 {
            let idx = my_rank as usize;
            if idx < output.len() {
                let local_value = unsafe { *(addr_of!(SHMEM) as *const u32) };
                unsafe { *output.get_unchecked_mut(idx) = local_value };
            }
        }

        cluster::cluster_sync();
    }
}

// ============================================================================
// HOST CODE
// ============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== Thread Block Cluster Tests (sm_90+) ===\n");

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    if major < 9 {
        println!("\nskipping: Thread Block Clusters require sm_90+ (Hopper)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cluster_size = 4u32;

    // ====================================================================
    // Test 0: Compile-Time Cluster Config (uses cuLaunchKernelEx)
    // ====================================================================

    println!("=== Test 0: Compile-Time Cluster Configuration ===\n");

    let mut ct_output = DeviceBuffer::<u32>::zeroed(&stream, cluster_size as usize).unwrap();

    println!("Launching test_cluster_compile_time via cuLaunchKernelEx");
    println!("  Grid: 4x1x1, Block: 32, Cluster: 4x1x1\n");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_cluster_compile_time(
            (stream).as_ref(),
            LaunchConfig {
                grid_dim: (cluster_size, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut ct_output,
        )
    }
    .expect("Launch compile-time cluster kernel");
    stream.synchronize().expect("Synchronize");

    let ct_results: Vec<u32> = ct_output.to_host_vec(&stream).unwrap();
    println!("Results (each block writes: (rank << 16) | cluster_size):");
    for (i, &val) in ct_results.iter().enumerate() {
        let rank = val >> 16;
        let size = val & 0xFFFF;
        println!(
            "  Block {}: raw=0x{:08X}, rank={}, cluster_size={}",
            i, val, rank, size
        );
    }

    let mut ct_pass = true;
    for (i, &val) in ct_results.iter().enumerate() {
        let rank = val >> 16;
        let size = val & 0xFFFF;
        if size != cluster_size || rank != i as u32 {
            println!(
                "  ❌ Block {} mismatch: expected rank={}, size={}",
                i, i, cluster_size
            );
            ct_pass = false;
        }
    }
    if ct_pass && ct_results.iter().any(|&v| (v & 0xFFFF) == cluster_size) {
        println!("✓ Compile-time cluster config test PASSED\n");
    } else {
        println!(
            "⚠ Cluster intrinsics returned defaults (driver may not support .reqnctapercluster)\n"
        );
    }

    // ====================================================================
    // Test 1: Cluster Intrinsics
    // ====================================================================

    println!("=== Test 1: Cluster Intrinsics (no cluster config) ===\n");

    let n_blocks = 4u32;
    let output_size = (n_blocks * 8) as usize;
    let mut output_dev = DeviceBuffer::<u32>::zeroed(&stream, output_size).unwrap();

    println!("Launching test_cluster_intrinsics...");
    println!("  Grid: 4x1x1 blocks, Block: 32 threads\n");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_cluster_intrinsics(
            (stream).as_ref(),
            LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut output_dev,
        )
    }
    .expect("Launch kernel");
    stream.synchronize().expect("Synchronize");

    let output: Vec<u32> = output_dev.to_host_vec(&stream).unwrap();

    println!("Results per block:");
    for block in 0..n_blocks as usize {
        let base = block * 8;
        println!(
            "  Block {}: ctaid=({},{},{}), nctaid=({},{},{}), rank={}, size={}",
            block,
            output[base],
            output[base + 1],
            output[base + 2],
            output[base + 3],
            output[base + 4],
            output[base + 5],
            output[base + 6],
            output[base + 7]
        );
    }
    println!("✓ Cluster intrinsics test completed\n");

    // ====================================================================
    // Test 2: Cluster Sync
    // ====================================================================

    println!("=== Test 2: Cluster Synchronization ===\n");

    let mut sync_output = DeviceBuffer::<u32>::zeroed(&stream, cluster_size as usize).unwrap();

    println!("Launching test_cluster_sync via cuLaunchKernelEx");
    println!("  Grid: 4x1x1, Block: 32, Cluster: 4x1x1\n");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_cluster_sync(
            (stream).as_ref(),
            LaunchConfig {
                grid_dim: (cluster_size, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut sync_output,
        )
    }
    .expect("Launch sync kernel");
    stream.synchronize().expect("Synchronize");

    let sync_results: Vec<u32> = sync_output.to_host_vec(&stream).unwrap();
    println!("Results: {:?}", sync_results);

    let sync_pass = sync_results
        .iter()
        .enumerate()
        .all(|(i, &val)| val == (i as u32) * 100 + 42);
    if sync_pass {
        println!("✓ Cluster sync test PASSED\n");
    } else {
        println!("⚠ Cluster sync returned unexpected values\n");
    }

    // ====================================================================
    // Test 3: DSMEM Ring Exchange
    // ====================================================================

    println!("=== Test 3: DSMEM Ring Exchange (cluster launch) ===\n");

    let mut ring_output = DeviceBuffer::<u32>::zeroed(&stream, cluster_size as usize).unwrap();

    println!("Launching test_dsmem_ring_exchange via cuLaunchKernelEx");
    println!("  Grid: 4x1x1, Block: 32, Cluster: 4x1x1");
    println!("  Expected: Block 0 reads 1001, Block 1 reads 1002, ...\n");

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    let ring_result = unsafe {
        module.test_dsmem_ring_exchange(
            (stream).as_ref(),
            LaunchConfig {
                grid_dim: (cluster_size, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut ring_output,
        )
    };

    let (ring_pass, dsmem_context_error) = match ring_result {
        Ok(_) => match stream.synchronize() {
            Ok(()) => {
                let results: Vec<u32> = ring_output.to_host_vec(&stream).unwrap();
                println!("Results (each block reads neighbor's value):");
                for (i, &val) in results.iter().enumerate() {
                    let expected = 1000 + ((i as u32 + 1) % cluster_size);
                    let status = if val == expected { "✓" } else { "?" };
                    println!(
                        "  Block {}: got {}, expected {} {}",
                        i, val, expected, status
                    );
                }
                let pass = results
                    .iter()
                    .enumerate()
                    .all(|(i, &val)| val == 1000 + ((i as u32 + 1) % cluster_size));
                (pass, false)
            }
            Err(e) => {
                println!("⚠ DSMEM synchronize failed: {:?}", e);
                (false, true)
            }
        },
        Err(e) => {
            println!("⚠ Cluster launch failed: {:?}", e);
            (false, true)
        }
    };

    if ring_pass {
        println!("✓ DSMEM ring exchange PASSED\n");
    } else if !dsmem_context_error {
        println!("⚠ DSMEM ring returned incorrect values\n");
    } else {
        println!("⚠ DSMEM ring exchange failed (see error above)\n");
    }

    // ====================================================================
    // Test 4: DSMEM Reduction
    // ====================================================================

    println!("=== Test 4: DSMEM Reduction (cluster launch) ===\n");

    let reduce_pass = if dsmem_context_error {
        println!("⚠ Skipped - CUDA context corrupted by previous DSMEM error\n");
        false
    } else {
        let mut reduce_output = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

        println!("Launching test_dsmem_reduction via cuLaunchKernelEx");
        println!("  Grid: 4x1x1, Block: 32, Cluster: 4x1x1");
        println!("  Expected: Block 0 sums all blocks' values: 10+20+30+40 = 100\n");

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let reduce_result = unsafe {
            module.test_dsmem_reduction(
                (stream).as_ref(),
                LaunchConfig {
                    grid_dim: (cluster_size, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut reduce_output,
            )
        };

        match reduce_result {
            Ok(_) => match stream.synchronize() {
                Ok(()) => {
                    let results: Vec<u32> = reduce_output.to_host_vec(&stream).unwrap();
                    let expected_sum = (1..=cluster_size).map(|i| i * 10).sum::<u32>();
                    println!("Result: {}, expected: {}", results[0], expected_sum);
                    results[0] == expected_sum
                }
                Err(e) => {
                    println!("⚠ DSMEM synchronize failed: {:?}", e);
                    false
                }
            },
            Err(e) => {
                println!("⚠ Cluster launch failed: {:?}", e);
                false
            }
        }
    };

    if reduce_pass {
        println!("✓ DSMEM reduction PASSED\n");
    } else if !dsmem_context_error {
        println!("⚠ DSMEM reduction returned incorrect value\n");
    }

    // ====================================================================
    // Test 5: DSMEM mapped remote store
    // ====================================================================

    println!("=== Test 5: DSMEM Mapped Remote Store (cluster launch) ===\n");

    let store_pass = if dsmem_context_error {
        println!("⚠ Skipped - CUDA context corrupted by previous DSMEM error\n");
        false
    } else {
        let mut store_output = DeviceBuffer::<u32>::zeroed(&stream, cluster_size as usize).unwrap();

        println!("Launching test_dsmem_mapped_store via cuLaunchKernelEx");
        println!("  Grid: 4x1x1, Block: 32, Cluster: 4x1x1");
        println!("  Each block writes through a mapped pointer into its next rank\n");

        let store_result = unsafe {
            module.test_dsmem_mapped_store(
                (stream).as_ref(),
                LaunchConfig {
                    grid_dim: (cluster_size, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut store_output,
            )
        };

        match store_result {
            Ok(_) => match stream.synchronize() {
                Ok(()) => {
                    let results: Vec<u32> = store_output.to_host_vec(&stream).unwrap();
                    println!("Results (each block observes the write from its previous rank):");
                    for (i, &val) in results.iter().enumerate() {
                        let previous_rank = (i as u32 + cluster_size - 1) % cluster_size;
                        let expected = 3000 + previous_rank;
                        let status = if val == expected { "✓" } else { "?" };
                        println!(
                            "  Block {}: got {}, expected {} {}",
                            i, val, expected, status
                        );
                    }
                    results.iter().enumerate().all(|(i, &val)| {
                        let previous_rank = (i as u32 + cluster_size - 1) % cluster_size;
                        val == 3000 + previous_rank
                    })
                }
                Err(e) => {
                    println!("⚠ DSMEM mapped-store synchronize failed: {:?}", e);
                    false
                }
            },
            Err(e) => {
                println!("⚠ Cluster launch failed: {:?}", e);
                false
            }
        }
    };

    if store_pass {
        println!("✓ DSMEM mapped remote store PASSED\n");
    } else if !dsmem_context_error {
        println!("⚠ DSMEM mapped remote store failed\n");
    }

    // ====================================================================
    // Summary
    // ====================================================================

    println!("=== Summary ===");
    println!("Launch method: cuLaunchKernelEx with ClusterLaunchConfig\n");
    println!("Kernels with #[cluster_launch(4,1,1)] emit:");
    println!("  .explicitcluster");
    println!("  .reqnctapercluster 4, 1, 1\n");

    let all_pass = ct_pass && sync_pass && ring_pass && reduce_pass && store_pass;
    if all_pass {
        println!("All cluster + DSMEM tests PASSED!");
    } else {
        println!("Results:");
        println!(
            "  Test 0 (compile-time cluster): {}",
            if ct_pass { "PASS" } else { "FAIL" }
        );
        println!("  Test 1 (intrinsics, no cluster): completed");
        println!(
            "  Test 2 (cluster sync):           {}",
            if sync_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  Test 3 (DSMEM ring exchange):    {}",
            if ring_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  Test 4 (DSMEM reduction):        {}",
            if reduce_pass { "PASS" } else { "FAIL" }
        );
        println!(
            "  Test 5 (DSMEM mapped store):     {}",
            if store_pass { "PASS" } else { "FAIL" }
        );
        std::process::exit(1);
    }
}

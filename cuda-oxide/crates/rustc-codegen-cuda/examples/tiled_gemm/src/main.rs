/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Unified Tiled GEMM with Shared Memory
//!
//! Demonstrates a high-performance matrix multiplication using:
//! - SharedArray for on-chip shared memory (100x faster than global)
//! - Thread synchronization via sync_threads()
//! - Collaborative tile loading by all threads in a block
//!
//! Build and run with:
//!   cargo oxide run tiled_gemm

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

// =============================================================================
// KERNEL
// =============================================================================

const TILE_SIZE: usize = 16;
#[cuda_module]
mod kernels {
    use super::*;

    /// Tiled GEMM kernel using shared memory: C = alpha * A * B + beta * C
    ///
    /// Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
    /// Tiles are loaded cooperatively into shared memory, reducing global
    /// memory accesses by ~TILE_SIZE factor.
    #[kernel]
    pub fn sgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32], // M x K matrix
        b: &[f32], // K x N matrix
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>, // M x N matrix (output)
    ) {
        // Shared memory tiles for A and B (16x16 = 256 elements each)
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize; // column within tile
        let ty = thread::threadIdx_y() as usize; // row within tile

        // Global row and column this thread computes
        let row = thread::blockIdx_y() as usize * TILE_SIZE + ty;
        let col = thread::blockIdx_x() as usize * TILE_SIZE + tx;

        let m_size = m as usize;
        let n_size = n as usize;
        let k_size = k as usize;

        // Number of tiles along K dimension
        let num_tiles = k_size.div_ceil(TILE_SIZE);

        // Accumulator for dot product
        let mut sum = 0.0f32;

        // Iterate over tiles along K dimension
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;

            // Shared memory index for this thread (unique per thread)
            let smem_idx = ty * TILE_SIZE + tx;

            // Collaboratively load tiles into shared memory
            unsafe {
                // Load A[row, tile_start + tx] into shared memory
                let a_col = tile_start + tx;
                if row < m_size && a_col < k_size {
                    TILE_A[smem_idx] = a[row * k_size + a_col];
                } else {
                    TILE_A[smem_idx] = 0.0;
                }

                // Load B[tile_start + ty, col] into shared memory
                let b_row = tile_start + ty;
                if b_row < k_size && col < n_size {
                    TILE_B[smem_idx] = b[b_row * n_size + col];
                } else {
                    TILE_B[smem_idx] = 0.0;
                }
            }

            // Wait for all threads to finish loading
            thread::sync_threads();

            // Compute partial dot product for this tile
            unsafe {
                let mut i = 0usize;
                while i < TILE_SIZE {
                    sum += TILE_A[ty * TILE_SIZE + i] * TILE_B[i * TILE_SIZE + tx];
                    i += 1;
                }
            }

            // Wait before loading next tile
            thread::sync_threads();

            tile += 1;
        }

        // Write result to global memory
        // The row width comes from `c`, bound on the host to this same `n`.
        if let Some(c_idx) = thread::index_2d_runtime(&c) {
            // col < n_size guaranteed by index_2d_runtime returning Some
            if row < m_size
                && let Some(c_elem) = c.get_mut(c_idx)
            {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    println!("=== Unified Tiled GEMM with Shared Memory ===");
    println!("Matrix dimensions: {}x{} * {}x{} = {}x{}", M, K, K, N, M, N);
    println!("alpha = {}, beta = {}\n", ALPHA, BETA);

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Initialize matrices
    println!("Initializing matrices...");
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    let c = vec![0.0f32; M * N];

    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = ((i + j) % 10) as f32 * 0.1;
        }
    }
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = ((i * j) % 10) as f32 * 0.1;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // Configure launch: 16x16 threads per block (matches TILE_SIZE)
    let block_size = 16u32;
    let grid_x = (N as u32).div_ceil(block_size);
    let grid_y = (M as u32).div_ceil(block_size);

    println!(
        "Grid: ({}, {}), Block: ({}, {})",
        grid_x, grid_y, block_size, block_size
    );
    println!("Tile size: 16x16 = 256 elements per tile");
    println!("Shared memory per block: {} bytes", 2 * 256 * 4);

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0, // Static shared memory
    };

    let m_arg = M as u32;
    let n_arg = N as u32;
    let k_arg = K as u32;

    // Warmup
    println!("\nWarmup...");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.sgemm_tiled(
            (stream).as_ref(),
            cfg,
            m_arg,
            n_arg,
            k_arg,
            ALPHA,
            &a_dev,
            &b_dev,
            BETA,
            cuda_host::RowWidth::new(&mut c_dev, n_arg),
        )
    }
    .unwrap();
    stream.synchronize().unwrap();

    // Timed runs
    const NUM_RUNS: u32 = 10;
    println!("Running {} iterations...", NUM_RUNS);
    let start = Instant::now();
    for _ in 0..NUM_RUNS {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.sgemm_tiled(
                (stream).as_ref(),
                cfg,
                m_arg,
                n_arg,
                k_arg,
                ALPHA,
                &a_dev,
                &b_dev,
                BETA,
                cuda_host::RowWidth::new(&mut c_dev, n_arg),
            )
        }
        .unwrap();
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / NUM_RUNS as f64;

    // GFLOPS calculation
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / (avg_ms / 1000.0) / 1e9;

    println!("\n=== Performance ===");
    println!("Average time: {:.3} ms", avg_ms);
    println!("Throughput:   {:.2} GFLOPS", gflops);

    let tile = 16.0;
    let naive_reads = 2.0 * M as f64 * K as f64 * N as f64 * 4.0;
    let tiled_reads = 2.0 * M as f64 * K as f64 * N as f64 * 4.0 / tile;
    println!(
        "\nMemory reduction: {:.1}x fewer global reads than naive",
        naive_reads / tiled_reads
    );

    // Verify results
    let c_result = c_dev.to_host_vec(&stream).unwrap();

    println!("\nVerifying (sampling 100 elements)...");
    let mut max_error = 0.0f32;
    for sample in 0..100 {
        let idx = sample * M * N / 100;
        let row = idx / N;
        let col = idx % N;

        let mut expected = 0.0f32;
        for kk in 0..K {
            expected += a[row * K + kk] * b[kk * N + col];
        }
        expected = ALPHA * expected + BETA * c[idx];

        let error = (c_result[idx] - expected).abs();
        if error > max_error {
            max_error = error;
        }
    }

    println!("Max error: {:.6e}", max_error);

    if max_error < 1e-3 {
        println!("\n✓ SUCCESS: Tiled GEMM computed correctly!");
    } else {
        println!("\n✗ FAILED: max error too large");
        std::process::exit(1);
    }
}

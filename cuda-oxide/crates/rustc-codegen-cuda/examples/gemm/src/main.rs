/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Unified GEMM Example (Naive Implementation)
//!
//! Demonstrates matrix multiplication: C = alpha * A * B + beta * C
//! Each thread computes one element of C.
//!
//! Build and run with:
//!   cargo oxide run gemm

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

// =============================================================================
// KERNEL - Naive SGEMM
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Naive GEMM kernel: C = alpha * A * B + beta * C
    ///
    /// Each thread computes ONE element of C.
    /// Matrix layout: Row-major
    /// - A is M x K
    /// - B is K x N
    /// - C is M x N
    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32], // M x K matrix
        b: &[f32], // K x N matrix
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>, // M x N matrix (output)
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        // The row width comes from `c`, bound on the host to this same `n`.
        if let Some(c_idx) = thread::index_2d_runtime(&c) {
            // col < n guaranteed by index_2d_runtime returning Some
            if row < m as usize {
                let n_size = n as usize;
                let k_size = k as usize;

                // Compute dot product of row of A and column of B
                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_size {
                    sum += a[row * k_size + i] * b[i * n_size + col];
                    i += 1;
                }

                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

// Matrix dimensions
const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

fn main() {
    println!("=== Unified GEMM Example (Naive Implementation) ===");
    println!("Matrix dimensions: {}x{} * {}x{} = {}x{}", M, K, K, N, M, N);
    println!("alpha = {}, beta = {}\n", ALPHA, BETA);

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    println!("Initialized CUDA context");

    // Initialize matrices
    println!("\nInitializing matrices...");
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

    // Copy to device
    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut c_dev = DeviceBuffer::from_host(&stream, &c).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // Configure launch: 16x16 threads per block
    let block_size = 16u32;
    let grid_x = (N as u32).div_ceil(block_size);
    let grid_y = (M as u32).div_ceil(block_size);

    println!(
        "Grid: ({}, {}), Block: ({}, {})",
        grid_x, grid_y, block_size, block_size
    );

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_size, block_size, 1),
        shared_mem_bytes: 0,
    };

    // Kernel scalar arguments
    let m_arg = M as u32;
    let n_arg = N as u32;
    let k_arg = K as u32;

    // Warmup
    println!("\nWarmup...");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.sgemm_naive(
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
    .expect("Kernel launch failed");
    stream.synchronize().unwrap();

    // Timed runs
    const NUM_RUNS: u32 = 5;
    println!("Running {} iterations...", NUM_RUNS);
    let start = Instant::now();
    for _ in 0..NUM_RUNS {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.sgemm_naive(
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
        .expect("Kernel launch failed");
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / NUM_RUNS as f64;

    // GFLOPS calculation
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let gflops = flops / (avg_ms / 1000.0) / 1e9;

    println!("\nPerformance: {:.3} ms, {:.2} GFLOPS", avg_ms, gflops);

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
        println!("\n✓ SUCCESS!");
    } else {
        println!("\n✗ FAILED! (max error too large)");
        std::process::exit(1);
    }
}

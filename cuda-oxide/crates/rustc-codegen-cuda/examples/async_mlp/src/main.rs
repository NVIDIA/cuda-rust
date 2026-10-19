/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! Async MLP Pipeline Example
//!
//! Demonstrates the full `cuda-async` execution model with a multi-kernel
//! forward pass: GEMM → MatVec → ReLU, run concurrently across multiple
//! simulated devices.
//!
//! # Async patterns showcased
//!
//! - `with_context`         — stream-aware memory operations as DeviceOperations
//! - `and_then`             — chaining dependent kernel launches on the same stream
//! - `zip!`                 — combining independent allocations into one operation
//! - `.arc()`               — sharing weights across batches via Arc
//! - `tokio::spawn`         — concurrent batch execution across the stream pool
//! - `.await`               — non-blocking completion
//! - `value()`              — lifting host data into the DeviceOperation graph
//!
//! # Pipeline
//!
//! ```text
//! For each batch:
//!   input [DIM×DIM] ──► GEMM(input, W0) ──► hidden [DIM×DIM]
//!                                               │
//!                              MatVec(hidden, W1) ──► output [DIM]
//!                                                        │
//!                                                  ReLU(output) ──► result [DIM]
//! ```
//!
//! Build and run with:
//!   cargo oxide run async_mlp

use cuda_async::simt::device_box::DeviceBox;
use cuda_async::simt::device_context::init_device_contexts;
use cuda_async::simt::device_operation::{self, DeviceOperation, Zippable, value};
use cuda_async::zip;
use cuda_core::simt::LaunchConfig;
use cuda_core::simt::memory::{
    malloc_async, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::future::IntoFuture;
use std::mem;
use std::sync::Arc;

// =============================================================================
// KERNELS -- compiled to PTX by rustc-codegen-cuda
// =============================================================================

/// Naive GEMM: C = alpha * A * B + beta * C
/// Each thread computes one element of C. Row-major layout.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        // The row width comes from `c`, bound on the host to this same `n`.
        if let Some(c_idx) = thread::index_2d_runtime(&c) {
            // col < n guaranteed by index_2d_runtime returning Some
            if row < m as usize {
                let n_sz = n as usize;
                let k_sz = k as usize;
                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_sz {
                    sum += a[row * k_sz + i] * b[i * n_sz + col];
                    i += 1;
                }
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = alpha * sum + beta * (*c_elem);
                }
            }
        }
    }

    /// Naive matrix-vector multiply: out = mat * vec_in
    /// Each thread computes one element of the output vector.
    #[kernel]
    pub fn matvec_naive(
        _m: u32,
        n: u32,
        mat: &[f32],
        vec_in: &[f32],
        mut vec_out: DisjointSlice<f32>,
    ) {
        let row = thread::index_1d();
        let row_raw = row.get();
        if let Some(out_elem) = vec_out.get_mut(row) {
            let n_sz = n as usize;
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < n_sz {
                sum += mat[row_raw * n_sz + j] * vec_in[j];
                j += 1;
            }
            *out_elem = sum;
        }
    }

    /// In-place ReLU: data[i] = max(0, data[i])
    #[kernel]
    pub fn relu(mut data: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(elem) = data.get_mut(idx) {
            let val = *elem;
            *elem = if val > 0.0f32 { val } else { 0.0f32 };
        }
    }
}

// =============================================================================
// DEVICE MEMORY HELPERS -- DeviceOperations for allocation and transfer
// =============================================================================

/// Allocates device memory and copies host data to it (H2D).
/// `ctx` is provided by the scheduler at execution time (not the call site)
/// and carries the assigned CUDA stream for this operation.
fn h2d(host_data: Vec<f32>) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = host_data.len();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(dptr, host_data.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}

/// Allocates zero-initialized device memory.
/// `ctx` is provided by the scheduler at execution time.
fn zeros(n: usize) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memset_d8_async(dptr, 0, num_bytes, stream.cu_stream()).unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}

/// Copies device memory back to host (D2H).
/// `ctx` is provided by the scheduler at execution time.
fn d2h(dev: DeviceBox<[f32]>) -> impl DeviceOperation<Output = Vec<f32>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = dev.len();
        let num_bytes = n * mem::size_of::<f32>();
        let mut host = vec![0.0f32; n];
        unsafe {
            memcpy_dtoh_async(
                host.as_mut_ptr(),
                dev.cu_deviceptr(),
                num_bytes,
                stream.cu_stream(),
            )
            .unwrap();
        }
        // dev is kept alive in this closure until the stream synchronizes
        let _ = &dev;
        value(host)
    })
}

// =============================================================================
// HOST CODE
// =============================================================================

const DIM: usize = 64;
const BLOCK: u32 = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Async MLP Pipeline ===\n");

    // 1. Initialize the async device context (round-robin pool of 4 streams).
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;

    // 2. Allocate model weights on device using zip! to compose independent ops.
    //    Both allocations are submitted together on the same stream.
    println!("Allocating model weights...");
    let w0_host: Vec<f32> = (0..DIM * DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let w1_host: Vec<f32> = (0..DIM).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();

    let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) =
        zip!(h2d(w0_host).arc(), h2d(w1_host).arc()).await?;
    println!(
        "  W0: {}x{} on device (Arc refcount={})",
        DIM,
        DIM,
        Arc::strong_count(&w0)
    );
    println!(
        "  W1: {} on device (Arc refcount={})\n",
        DIM,
        Arc::strong_count(&w1)
    );

    // 3. Build and launch concurrent forward passes for multiple batches.
    //    Each batch is a lazy DeviceOperation graph that gets spawned onto
    //    the Tokio runtime. The round-robin scheduling policy distributes
    //    work across 4 CUDA streams automatically.
    let num_batches = 4;
    let mut handles = vec![];

    for batch_idx in 0..num_batches {
        let w0 = w0.clone();
        let w1 = w1.clone();
        let module = module.clone();

        let batch_data: Vec<f32> = (0..DIM * DIM)
            .map(|i| ((i + batch_idx * 37) % 13) as f32 * 0.1)
            .collect();

        // ── Build the lazy DeviceOperation pipeline ──────────────────────
        //
        //   zip!(input, hidden, output)     allocate 3 buffers
        //     .and_then(...)                 launch GEMM
        //     .and_then(...)                 launch MatVec
        //     .and_then(...)                 launch ReLU
        //     .and_then(d2h)                 copy result to host
        //
        // Nothing executes until tokio::spawn polls the future.

        let gemm_cfg = LaunchConfig {
            grid_dim: (
                (DIM as u32).div_ceil(BLOCK),
                (DIM as u32).div_ceil(BLOCK),
                1,
            ),
            block_dim: (BLOCK, BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let matvec_cfg = LaunchConfig::for_num_elems(DIM as u32);
        let relu_cfg = LaunchConfig::for_num_elems(DIM as u32);

        let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
            // ── Stage 1: GEMM  hidden = input @ W0 ──────────────────────
            .and_then(
                move |(input, hidden, output): (
                    DeviceBox<[f32]>,
                    DeviceBox<[f32]>,
                    DeviceBox<[f32]>,
                )| {
                    // SAFETY: the 2D grid and block match `sgemm_naive`'s
                    // indexing, and all matrices are DIM x DIM allocations.
                    let launch = unsafe {
                        module.sgemm_naive_async_owned(
                            gemm_cfg,
                            DIM as u32,
                            DIM as u32,
                            DIM as u32,
                            1.0f32,
                            input,
                            w0,
                            0.0f32,
                            // C's row width, bound to the buffer for this launch.
                            cuda_host::RowWidthOwned::new(hidden, DIM as u32),
                        )
                    }
                    .expect("Failed to build sgemm_naive launch");
                    launch.and_then(move |(_input, _w0, hidden)| {
                        value((hidden.into_buffer(), output, w1, module))
                    })
                },
            )
            // ── Stage 2: MatVec  output = hidden @ W1 ───────────────────
            .and_then(
                move |(hidden, output, w1, module): (
                    DeviceBox<[f32]>,
                    DeviceBox<[f32]>,
                    Arc<DeviceBox<[f32]>>,
                    kernels::LoadedModule,
                )| {
                    // SAFETY: this is a 1D launch of DIM guarded threads over
                    // DIM-sized vectors and a DIM x DIM matrix.
                    let launch = unsafe {
                        module.matvec_naive_async_owned(
                            matvec_cfg, DIM as u32, DIM as u32, hidden, w1, output,
                        )
                    }
                    .expect("Failed to build matvec_naive launch");
                    launch.and_then(move |(_hidden, _w1, output)| value((output, module)))
                },
            )
            // ── Stage 3: ReLU  result = max(0, output) ──────────────────
            .and_then(
                move |(output, module): (DeviceBox<[f32]>, kernels::LoadedModule)| {
                    // SAFETY: this is a 1D launch and the ReLU kernel guards
                    // accesses using the owned output's DIM-element length.
                    unsafe { module.relu_async_owned(relu_cfg, output) }
                        .expect("Failed to build relu launch")
                },
            )
            // ── Stage 4: D2H  copy result back to host ──────────────────
            .and_then(d2h);

        // Spawn the pipeline as a Tokio task. The DeviceOperation is
        // converted to a Future via IntoFuture and polled by the runtime.
        handles.push(tokio::spawn(pipeline.into_future()));
    }

    // 4. Await all batches and verify results.
    println!(
        "Launched {} batches concurrently, awaiting results...\n",
        num_batches
    );
    for (i, handle) in handles.into_iter().enumerate() {
        let result: Vec<f32> = handle.await.expect("Tokio task panicked")?;
        let all_non_negative = result.iter().all(|&v| v >= 0.0);
        println!(
            "Batch {}: {} elements, first 8 = {:?}{}",
            i,
            result.len(),
            &result[..8.min(result.len())],
            if all_non_negative {
                " [ReLU OK]"
            } else {
                " [ReLU FAILED: negative values found]"
            }
        );
    }

    println!("\nSUCCESS: All batches completed.");
    Ok(())
}

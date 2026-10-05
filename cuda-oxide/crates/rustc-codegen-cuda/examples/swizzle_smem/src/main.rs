/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Swizzled Shared Memory Example
//!
//! A 32x32 f32 tile transpose through shared memory, indexed via
//! `Swizzle<5, 0, 5>` so both the row-wise store and the column-wise load
//! are bank-conflict-free without padding the stride.
//!
//! Build and run with:
//!   cargo oxide run swizzle_smem

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, swizzle::Swizzle, thread};
use cuda_host::cuda_module;

/// Tile edge; the tile is DIM x DIM, one element per thread of a DIM x DIM block.
const DIM: usize = 32;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Transpose one 32x32 tile through a swizzled shared-memory staging buffer.
    ///
    /// The store writes rows and the load reads columns; with a stride-32 tile
    /// the column read would serialise 32-way, so both sides go through
    /// `Swizzle<5, 0, 5>`. The swizzle is its own inverse, so as long as both
    /// sides use it, the data comes back out where it went in.
    #[kernel]
    pub fn swizzled_transpose(input: &[f32], mut output: DisjointSlice<f32, thread::Index2D<DIM>>) {
        static mut TILE: SharedArray<f32, { DIM * DIM }> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        // Row-major store: lane index varies along the row (conflict-free even
        // unswizzled; the swizzle must not spoil it).
        unsafe {
            TILE[Swizzle::<5, 0, 5>::apply(ty * DIM + tx)] = input[ty * DIM + tx];
        }

        thread::sync_threads();

        // Column read: without the swizzle every lane of a warp would hit the
        // same bank. Through the same swizzle it is conflict-free.
        unsafe {
            if let Some(idx) = thread::index_2d::<DIM>()
                && let Some(out_elem) = output.get_mut(idx)
            {
                *out_elem = TILE[Swizzle::<5, 0, 5>::apply(tx * DIM + ty)];
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Swizzled Shared Memory Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = DIM * DIM;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (DIM as u32, DIM as u32, 1),
        shared_mem_bytes: 0,
    };

    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.swizzled_transpose((stream).as_ref(), cfg, &input_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();

    // Verify: output is the transpose of input. Any swizzle mismatch between
    // the store and the load scrambles the tile, so this checks the involution
    // on device, not just the transpose.
    for row in 0..DIM {
        for col in 0..DIM {
            let expected = input_host[col * DIM + row];
            let got = out_result[row * DIM + col];
            if (got - expected).abs() > 1e-5 {
                eprintln!("Mismatch at ({row}, {col}): expected {expected}, got {got}");
                std::process::exit(1);
            }
        }
    }
    println!("Output[0..5] = {:?}", &out_result[0..5]);
    println!("✓ SUCCESS: swizzled shared-memory transpose is correct");
}

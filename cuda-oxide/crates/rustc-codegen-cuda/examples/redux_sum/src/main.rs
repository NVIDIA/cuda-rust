/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-instruction warp sum reduction (`redux.sync.add`, sm_80+).
//!
//! Whereas `warp_reduce` builds a log-tree sum out of 5 `shfl` + 5 `add`, this
//! example performs the whole warp reduction with one hardware instruction:
//! PTX `redux.sync.add.s32` (bit-identical wrapping sum for `u32`), lowered from `warp::redux_sync_add`.
//!
//! Build and run with:
//!   cargo oxide run redux_sum

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Each lane contributes its lane id; the full-warp sum (0+1+..+31 = 496)
    /// is broadcast back to every lane. Lane 0 writes the per-warp result.
    #[kernel]
    pub fn redux_lane_sum(mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        let sum = warp::redux_sync_add(FULL_MASK, lane);

        if lane == 0 {
            let warp_idx = gid.get() / 32;
            if warp_idx < out.len() {
                unsafe {
                    *out.get_unchecked_mut(warp_idx) = sum;
                }
            }
        }
    }

    /// General data reduction: every lane contributes `data[gid]`, the warp sum
    /// is written once per warp by lane 0.
    #[kernel]
    pub fn redux_data_sum(data: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        let val = if gid.in_bounds(out.len() * 32) {
            data[gid.get()]
        } else {
            0
        };

        let sum = warp::redux_sync_add(FULL_MASK, val);

        if lane == 0 {
            let warp_idx = gid.get() / 32;
            if warp_idx < out.len() {
                unsafe {
                    *out.get_unchecked_mut(warp_idx) = sum;
                }
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== redux.sync.add Warp Reduction (sm_80+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // `redux.sync` is an Ampere instruction; the PTX won't assemble below sm_80.
    if major < 8 {
        println!("\nskipping: redux.sync.add requires sm_80+ (Ampere)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    const N: usize = 256;
    const WARPS: usize = N / 32;
    const EXPECTED: u32 = 496; // 0 + 1 + ... + 31

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: lane-id reduction =====
    println!("\n--- Test 1: redux_sync_add over lane ids ---");
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.redux_lane_sum((stream).as_ref(), cfg, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    if out_result.iter().all(|&x| x == EXPECTED) {
        println!("✓ All {} warp sums correct (each = {})", WARPS, EXPECTED);
    } else {
        println!("✗ Some warp sums incorrect (expected {})!", EXPECTED);
        std::process::exit(1);
    }

    // ===== Test 2: data reduction =====
    println!("\n--- Test 2: redux_sync_add over input data ---");
    let data_host: Vec<u32> = (0..N).map(|i| (i % 32) as u32).collect();
    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.redux_data_sum((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    if out_result.iter().all(|&x| x == EXPECTED) {
        println!("✓ All {} warp sums correct (each = {})", WARPS, EXPECTED);
    } else {
        println!("✗ Some warp sums incorrect (expected {})!", EXPECTED);
        std::process::exit(1);
    }

    println!("\nSUCCESS: redux.sync.add produced correct warp sums");
}

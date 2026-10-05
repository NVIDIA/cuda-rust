/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-instruction warp leader election (`elect.sync`, sm_90+).
//!
//! Hopper's `elect.sync` collectively picks one participating lane as the warp
//! leader and returns its lane id plus a one-hot elected predicate. The choice
//! is deterministic for the same mask, but PTX does not promise the lowest
//! lane.
//!
//! Two kernels:
//!   1. `elect_full_warp` — full-warp election via `warp::elect_sync`; checks
//!      that the returned leader matches the one-hot predicate.
//!   2. `elect_subset` — election over a *subset* of lanes (the upper half of
//!      the warp) via `warp::is_elected_sync`, showing the elected lane belongs
//!      to the participating set.
//!
//! Build and run with:
//!   cargo oxide run elect_leader

use cuda_device::{DisjointSlice, kernel, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Full-warp election. Every lane records whether it was elected; lane 0
    /// records the elected leader's lane id.
    #[kernel]
    pub fn elect_full_warp(
        mut leader_out: DisjointSlice<u32>,
        mut elected_out: DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id();
        let (leader, elected) = warp::elect_sync(FULL_MASK);

        unsafe {
            *elected_out.get_unchecked_mut(lane as usize) = elected as u32;
        }
        if lane == 0 {
            unsafe {
                *leader_out.get_unchecked_mut(0) = leader;
            }
        }
    }

    /// Subset election: only the upper half of the warp participates. The
    /// elected lane writes its id.
    #[kernel]
    pub fn elect_subset(mut out: DisjointSlice<u32>) {
        let lane = warp::lane_id();
        if lane >= 16 {
            // Mask of the lanes converged in this branch (the upper half).
            let mask = warp::active_mask();
            if warp::is_elected_sync(mask) {
                unsafe {
                    *out.get_unchecked_mut(0) = lane;
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

    println!("=== elect.sync warp leader election (sm_90+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // `elect.sync` is a Hopper instruction; the PTX won't assemble below sm_90.
    if major < 9 {
        println!("\nskipping: elect.sync requires sm_90+ (Hopper)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // A single warp is all we need to demonstrate election semantics.
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failed = false;

    // ===== Test 1: full-warp election =====
    println!("\n--- Test 1: elect_sync (full warp) ---");
    let mut leader_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let mut elected_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.elect_full_warp((stream).as_ref(), cfg, &mut leader_dev, &mut elected_dev) }
        .expect("Kernel launch failed");

    let leader = leader_dev.to_host_vec(&stream).unwrap();
    let elected = elected_dev.to_host_vec(&stream).unwrap();

    let elected_lanes: Vec<_> = elected
        .iter()
        .enumerate()
        .filter_map(|(lane, &value)| (value == 1).then_some(lane as u32))
        .collect();
    let elected_lane = elected_lanes.first().copied().unwrap_or(u32::MAX);
    let mut expected_elected = vec![0u32; 32];
    if elected_lane < 32 {
        expected_elected[elected_lane as usize] = 1;
    }

    println!("leader[0]    = {} (expected {})", leader[0], elected_lane);
    println!("elected mask = {:?}", elected);
    println!("expected     = {:?}", expected_elected);

    if elected_lanes.len() == 1 && leader[0] == elected_lane && elected == expected_elected {
        println!(
            "✓ leader is lane {} and is_elected is one-hot",
            elected_lane
        );
    } else {
        println!("✗ full-warp election mismatch!");
        failed = true;
    }

    // ===== Test 2: subset election (upper half) =====
    println!("\n--- Test 2: is_elected_sync (upper-half subset) ---");
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.elect_subset((stream).as_ref(), cfg, &mut out_dev) }
        .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    if out[0] == 16 {
        println!(
            "out[0] = {} (expected 16 — lowest lane of the active subset)",
            out[0]
        );
        println!("✓ subset leader is the lowest participating lane");
    } else if (16..32).contains(&out[0]) {
        println!("out[0] = {} (valid participating lane)", out[0]);
        println!("✓ subset leader is a participating lane");
    } else {
        println!("out[0] = {} (expected a lane in 16..=31)", out[0]);
        println!("✗ subset election mismatch!");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: elect.sync produced correct leader-election results");
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal test for the MCAST_BAR protocol used in gemm_sol_clc.
//!
//! Tests: 4 CTAs in a cluster, each CTA's thread 0 does
//! `mbarrier_arrive_cluster` to rank 0's barrier, rank 0 waits
//! via `mbarrier_try_wait_parity`. Loops N times with 2-stage
//! double-buffering parity, exactly matching gemm_sol_clc's pattern.
//!
//! Build and run:
//!   cargo oxide run mcast_barrier_test

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_cluster, mbarrier_init,
    mbarrier_try_wait_parity, nanosleep,
};
use cuda_device::cluster;
use cuda_device::shared::cvta_generic_to_shared_offset;
use cuda_device::{DisjointSlice, cluster_launch, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNEL
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub unsafe fn mcast_barrier_loop(mut out: DisjointSlice<u32>, num_iters: u32) {
        unsafe {
            static mut MCAST_BAR0: Barrier = Barrier::UNINIT;
            static mut MCAST_BAR1: Barrier = Barrier::UNINIT;

            const CLUSTER_SIZE: u32 = 4;

            let tid = thread::threadIdx_x();
            let my_rank = cluster::cluster_ctaidX();

            if tid == 0 {
                mbarrier_init(&raw mut MCAST_BAR0, CLUSTER_SIZE);
                mbarrier_init(&raw mut MCAST_BAR1, CLUSTER_SIZE);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            let rank0_bar0_addr = cvta_generic_to_shared_offset(cluster::map_shared_rank(
                &raw const MCAST_BAR0,
                0,
            ) as *const u8);
            let rank0_bar1_addr = cvta_generic_to_shared_offset(cluster::map_shared_rank(
                &raw const MCAST_BAR1,
                0,
            ) as *const u8);

            let is_rank0 = my_rank == 0;

            cluster::cluster_sync();

            let mut k: u32 = 0;
            while k < num_iters {
                let stage = k & 1;
                let mcast_parity = (k >> 1) & 1;

                // All CTAs: arrive at rank 0's MCAST_BAR for this stage
                if tid == 0 {
                    fence_proxy_async_shared_cta();
                    if stage == 0 {
                        mbarrier_arrive_cluster(rank0_bar0_addr);
                    } else {
                        mbarrier_arrive_cluster(rank0_bar1_addr);
                    }
                }

                // Rank 0: wait for all 4 CTAs to arrive
                if is_rank0 {
                    if stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const MCAST_BAR0, mcast_parity) {
                            nanosleep(32);
                        }
                    } else {
                        while !mbarrier_try_wait_parity(&raw const MCAST_BAR1, mcast_parity) {
                            nanosleep(32);
                        }
                    }
                }

                cluster::cluster_sync();
                k += 1;
            }

            cluster::cluster_sync();

            // Write success: rank + iteration count
            if tid == 0 {
                let idx = my_rank as usize;
                if idx < out.len() {
                    *out.get_unchecked_mut(idx) = num_iters;
                }
            }
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const CLUSTER_SIZE: u32 = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MCAST Barrier Protocol Test ===\n");

    let ctx = CudaContext::new(0)?;

    // Cluster-scoped mbarrier arrival and multicast need sm_90 (Hopper) or
    // later.
    let (major, minor) = ctx.compute_capability()?;
    if major < 9 {
        println!("skipping: cluster mbarrier requires sm_90+ (device is sm_{major}{minor})");
        return Ok(());
    }

    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    println!("Embedded CUDA module loaded\n");

    for &num_iters in &[4u32, 8, 16, 32, 64, 256, 1024] {
        print!("  {} iters ... ", num_iters);

        let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, CLUSTER_SIZE as usize)?;

        let cfg = LaunchConfig {
            grid_dim: (CLUSTER_SIZE, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { module.mcast_barrier_loop((stream).as_ref(), cfg, &mut dev_out, num_iters) }?;

        stream.synchronize()?;

        let host_out = dev_out.to_host_vec(&stream)?;
        let all_ok = host_out.iter().all(|&v| v == num_iters);
        if all_ok {
            println!("PASSED (all {} CTAs completed)", CLUSTER_SIZE);
        } else {
            println!("FAILED: {:?}", host_out);
            return Err("MCAST barrier test failed".into());
        }
    }

    println!("\n=== All Tests Passed ===");
    Ok(())
}

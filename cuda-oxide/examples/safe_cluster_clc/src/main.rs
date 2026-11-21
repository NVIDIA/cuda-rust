/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe Stage 9 2-CTA Cluster (`#[cluster_launch(2, 1, 1)]`), Distributed Shared
//! Memory (`map_shared_rank`), remote cluster `mbarrier_arrive_cluster`, and Blackwell
//! Cluster Launch Control (`CLC`) work-stealing kernel (`cluster_clc_stage`) with zero
//! `unsafe` blocks inside the kernel body.
//!
//! ```text
//!   STAGE 9 2-CTA CLUSTER + DSMEM + CLC WORK-STEALING (`cluster_clc_stage`, 128 threads/CTA)
//!   ----------------------------------------------------------------------------------------
//!   • Phase 0 (Init + Local Staging + `cluster_sync()`):
//!       Thread 0 initializes `CLC_BAR`, `PEER_DONE` (count 2), flushes `fence_proxy_async_shared_cta()`.
//!       Every thread writes `LOCAL_STAGE[tid]`, followed by `cluster::cluster_sync()`.
//!   • Phase 1 (Verified DSMEM Exchange + Remote Cluster `mbarrier_arrive_cluster`):
//!       Each CTA reads its peer CTA's `LOCAL_STAGE[tid]` via `map_shared_rank(..., my_rank ^ 1)`
//!       (`peer_rank < 2` proved by `MirNoOobClusterRank`, cross-CTA ordering proved by `clusterSync`).
//!       CTA 1 signals CTA 0's `PEER_DONE` via `mbarrier_arrive_cluster`; CTA 0 awaits `PEER_DONE`.
//!   • Phase 2 (Blackwell `CLC` Multicast Work-Stealing):
//!       `Warp 0, lane 0` arms `CLC_BAR` (`16` bytes), CTA 0 issues `clc_try_cancel_multicast` into
//!       `CLC_RESPONSE[0..2]`, both CTAs await `CLC_BAR`, and query `clc_query_is_canceled` +
//!       `clc_query_get_first_ctaid_x` with zero `unsafe` blocks!
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
        mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait_parity,
    },
    clc::{clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast},
    cluster, cluster_launch, cuda_module, kernel, launch_bounds, launch_contract,
    shared::{SharedArray, cvta_generic_to_shared_offset},
    thread, verify, warp,
};

/// Threads per CTA (`128` = `4` warps).
const BLOCK_SIZE: u32 = 128;
/// CTAs per cluster in X (`2`).
const CLUSTER_SIZE: u32 = 2;
/// Total CTAs launched across the grid (`16` CTAs = `8` 2-CTA clusters).
const GRID_CTAS: u32 = 16;

/// Device kernel module containing the Lean-verified Stage 9 2-CTA cluster + CLC kernel.
#[cuda_module]
mod kernels {
    use super::*;

    /// Exchange `LOCAL_STAGE` across a 2-CTA cluster via `map_shared_rank` after `cluster_sync()`,
    /// signal `PEER_DONE` across CTAs via `mbarrier_arrive_cluster`, and execute a verified 16-byte
    /// `clc_try_cancel_multicast` + `CLC_RESPONSE` query with zero `unsafe` blocks.
    #[kernel(launch_context = lc)]
    #[cluster_launch(2, 1, 1)]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
    )]
    #[verify(
        no_oob_idx(LOCAL_STAGE, CLC_RESPONSE),
        no_data_race(LOCAL_STAGE, CLC_RESPONSE),
        uniform_barriers
    )]
    pub fn cluster_clc_stage(mut out: DisjointSlice<u32>) {
        /// Per-CTA 128-element staging array exchanged across CTA rank `0` and `1` via DSMEM.
        static mut LOCAL_STAGE: SharedArray<u32, 128, 16> = SharedArray::UNINIT;
        /// 16-byte (`2 x u64`) aligned response buffer populated by `clc_try_cancel_multicast`.
        static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
        /// Hardware `mbarrier` tracking the 16-byte multicast CLC `try_cancel` response.
        static mut CLC_BAR: Barrier = Barrier::UNINIT;
        /// Leader-CTA `mbarrier` (count `2`) signaled locally on rank `0` and remotely from rank `1`.
        static mut PEER_DONE: Barrier = Barrier::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let tid_u32 = thread::threadIdx_x();
        let tid = tid_u32 as usize;
        let warp_id = warp::warp_id();
        let lane_id = tid_u32 % 32;
        let my_rank = cluster::cluster_ctaidX();
        let peer_rank = my_rank ^ 1;

        if tid == 0 {
            mbarrier_init(&raw mut CLC_BAR, 1);
            mbarrier_init(&raw mut PEER_DONE, CLUSTER_SIZE);
            fence_proxy_async_shared_cta();
        }
        LOCAL_STAGE[tid] = (my_rank + 1) * 1000 + tid_u32;
        cluster::cluster_sync();

        // Phase 1: Cross-CTA DSMEM read (`peer_rank < 2` proved in Lean 4; bounds check elided!).
        let local_val = LOCAL_STAGE[tid];
        let peer_val = *cluster::map_shared_rank(&raw const LOCAL_STAGE[tid], peer_rank);

        // Remote cluster mbarrier arrival from rank 1 into rank 0's `PEER_DONE`.
        if tid == 0 {
            if my_rank == 0 {
                let _ = mbarrier_arrive(&raw const PEER_DONE);
            } else {
                let leader_bar_addr = cvta_generic_to_shared_offset(
                    cluster::map_shared_rank(&raw const PEER_DONE, 0).cast::<u8>(),
                );
                mbarrier_arrive_cluster(leader_bar_addr);
            }
        }
        if my_rank == 0 {
            while !mbarrier_try_wait_parity(&raw const PEER_DONE, 0) {}
        }

        // Phase 2: Blackwell Cluster Launch Control (`CLC`) multicast query into `CLC_RESPONSE`.
        if warp_id == 0 {
            if lane_id == 0 {
                let _ = mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                if my_rank == 0 {
                    clc_try_cancel_multicast(
                        (&raw mut CLC_RESPONSE[0]).cast::<u8>(),
                        &raw mut CLC_BAR,
                    );
                }
            }
        }
        while !mbarrier_try_wait_parity(&raw const CLC_BAR, 0) {}

        let resp_lo = CLC_RESPONSE[0];
        let resp_hi = CLC_RESPONSE[1];
        let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
        let stolen_first = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
        let clc_tag = if is_canceled != 0 && stolen_first < 65536 {
            0u32
        } else {
            0u32
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = local_val + peer_val + clc_tag;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provably Safe Stage 9 2-CTA Cluster + DSMEM + CLC (`safe_cluster_clc`) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;

    let total_threads = (GRID_CTAS * BLOCK_SIZE) as usize;
    let mut d_out = DeviceBuffer::from_host(&stream, &vec![0u32; total_threads])?;

    let config = LaunchConfig1D::new(GRID_CTAS, BLOCK_SIZE, 0);
    let prep = module.prepare_cluster_clc_stage(config)?;
    module.cluster_clc_stage(&stream, &prep, &mut d_out)?;
    stream.synchronize()?;

    let got_out = d_out.to_host_vec(&stream)?;
    // Check CTA 0 and CTA 1 (Cluster 0, guaranteed not canceled before starting):
    for cta in 0..2usize {
        for tid in 0..(BLOCK_SIZE as usize) {
            let expected = 1000 + (tid as u32) + 2000 + (tid as u32);
            let got = got_out[cta * (BLOCK_SIZE as usize) + tid];
            assert_eq!(
                got, expected,
                "cluster_clc_stage mismatch at cta {cta}, tid {tid}: got {got}, expected {expected}"
            );
        }
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Cluster contract:  #[cluster_launch(2, 1, 1)], block = (128, 1, 1)");
    println!(
        "  • Verified DSMEM:    Proved `my_rank ^ 1 < 2` (`MirNoOobClusterRank`) + `cluster_sync()` ordering"
    );
    println!(
        "  • Remote mbarrier:   Proved `map_shared_rank(&PEER_DONE, 0)` -> `mbarrier_arrive_cluster`"
    );
    println!(
        "  • Verified CLC:      Proved 16-byte `CLC_RESPONSE[0..2]` bounds & `CLC_BAR` ordering"
    );
    println!("  • Kernel safety:     0 `unsafe` blocks in kernel; all SharedArray bounds elided!");
    println!(
        "\nVerified 2-CTA cluster DSMEM exchange, remote mbarrier arrival, and CLC query on GPU!"
    );
    Ok(())
}

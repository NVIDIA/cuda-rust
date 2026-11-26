/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe warp butterfly reduction (`warp_reduce_sum`) and 2-phase
//! shared-memory block reduction (`block_reduce_sum`) checked by Lean 4.
//!
//! ```text
//!   KERNEL 1: `warp_reduce_sum` (1D Warp Butterfly Reduction)
//!   ─────────────────────────────────────────────────────────────────────────
//!   • Guard: `out.get_mut(idx)` + host `requires = (data.len() >= out.len())`
//!   • Lean 4 proves: `no_oob_idx(data, out)`, `disjoint_writes(out)`,
//!                    `writes_all_elems(out)` (erases `data[gid]` bounds check)
//!
//!   KERNEL 2: `block_reduce_sum` (2-Phase Shared-Memory Block Reduction)
//!   ─────────────────────────────────────────────────────────────────────────
//!   • Phase 0 (Warp Leader Stage): `if lane == 0 { WARP_SUMS[tid / 32] = warp_sum; }`
//!     Lean 4 proves `tid / 32 < 8` (`no_oob_tile_group_slot`) and that only
//!     `lane == 0` writes each warp slot (`no_data_race_group_leader_slot`).
//!   • Barrier: `thread::sync_threads()` (`uniform_barriers`)
//!   • Phase 1 (Block Scan): `while w < 8 { block_sum += WARP_SUMS[w]; w += 1; }`
//!     Lean 4 proves `w < 8` (`no_oob_tile_linear_scan`) and barrier separation
//!     (`no_data_race_diff_phase`), erasing both `SharedArray` bounds checks!
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
    verify, warp,
};

/// Threads per block (`256` threads = `8` warps of `32` lanes).
const BLOCK_SIZE: u32 = 256;
/// Number of 32-lane warps per 256-thread block (`8`).
const WARPS_PER_BLOCK: usize = (BLOCK_SIZE as usize) / 32;
/// Number of thread blocks launched in the 1D grid (`4`).
const NUM_BLOCKS: u32 = 4;
/// Total number of elements and launched threads across the grid (`1,024`).
const N: usize = (BLOCK_SIZE * NUM_BLOCKS) as usize;

/// Device kernel module containing the provably safe warp and 2-phase block reductions.
#[cuda_module]
mod kernels {
    use super::*;

    /// 1D warp butterfly reduction where every lane loads `data[gid]`, reduces
    /// across its 32-lane warp via `warp::reduce_sum_f32`, and writes its warp's
    /// sum to `out[idx]`.
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (data.len() >= out.len()),
    )]
    #[verify(no_oob_idx(data, out), disjoint_writes(out), writes_all_elems(out))]
    #[kernel]
    pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let gid = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let val = data[gid];
            *out_elem = warp::reduce_sum_f32(val);
        }
    }

    /// 2-phase 256-thread block reduction across 8 warps using `WARP_SUMS`:
    /// - Phase 0: Each thread reduces its 32-lane warp; only `lane == 0` stages
    ///   `WARP_SUMS[tid / 32] = warp_sum` (zero `unsafe`!).
    /// - Barrier: `thread::sync_threads()` separates Phase 0 writes from Phase 1 reads.
    /// - Phase 1: Each thread sums `WARP_SUMS[0..8]` and writes the 256-element
    ///   block sum to `out`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        requires = (data.len() >= out.len()),
    )]
    #[verify(no_oob_idx(WARP_SUMS), no_data_race(WARP_SUMS), uniform_barriers)]
    pub fn block_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        /// Shared-memory scratchpad holding one partial sum per warp (`8` floats).
        static mut WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let gid = idx.get();
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();

        let val = if gid < data.len() { data[gid] } else { 0.0 };
        let warp_sum = warp::reduce_sum_f32(val);

        // Phase 0 (provably safe `no_oob_idx` + `no_data_race` under `lane == 0`):
        if lane == 0 {
            WARP_SUMS[tid / 32] = warp_sum;
        }

        thread::sync_threads();

        // Phase 1 (provably safe `no_oob_idx` + `no_data_race` + `uniform_barriers`):
        let mut block_sum = 0.0f32;
        let mut w = 0usize;
        while w < WARPS_PER_BLOCK {
            block_sum += WARP_SUMS[w];
            w += 1;
        }

        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = block_sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provably Safe Warp & Block Reduction (`safe_warp_reduce`) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;

    let host_data: Vec<f32> = (0..N).map(|i| ((i % 32) + 1) as f32).collect();
    let d_data = DeviceBuffer::from_host(&stream, &host_data)?;
    let mut d_warp_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; N])?;
    let mut d_block_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; N])?;

    let config = LaunchConfig1D::new(NUM_BLOCKS, BLOCK_SIZE, 0);

    let prep_warp = module.prepare_warp_reduce_sum(config)?;
    module.warp_reduce_sum(&stream, &prep_warp, &d_data, &mut d_warp_out)?;

    let prep_block = module.prepare_block_reduce_sum(config)?;
    module.block_reduce_sum(&stream, &prep_block, &d_data, &mut d_block_out)?;
    stream.synchronize()?;

    let warp_out = d_warp_out.to_host_vec(&stream)?;
    let block_out = d_block_out.to_host_vec(&stream)?;

    // Sum of 1..=32 is 32 * 33 / 2 = 528.0 per warp, and 8 * 528.0 = 4224.0 per 256-thread block.
    let expected_warp_sum = 528.0f32;
    let expected_block_sum = (WARPS_PER_BLOCK as f32) * expected_warp_sum;

    for (i, &got) in warp_out.iter().enumerate() {
        assert!(
            (got - expected_warp_sum).abs() < 1e-4,
            "warp_reduce_sum mismatch at {i}: got {got}, expected {expected_warp_sum}"
        );
    }
    for (i, &got) in block_out.iter().enumerate() {
        assert!(
            (got - expected_block_sum).abs() < 1e-4,
            "block_reduce_sum mismatch at {i}: got {got}, expected {expected_block_sum}"
        );
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract:   domain = 1, block = (256, 1, 1)");
    println!(
        "  • warp_reduce_sum:   Proved no_oob_idx(data, out), disjoint_writes(out), writes_all_elems(out)"
    );
    println!(
        "                       Elided: 1 global slice `mir.assert` bounds check (`data[gid]`)"
    );
    println!(
        "  • block_reduce_sum:  Proved no_oob_idx(WARP_SUMS), no_data_race(WARP_SUMS), uniform_barriers"
    );
    println!(
        "                       Phase 0: `if lane == 0` -> `WARP_SUMS[tid / 32]` (`no_oob_tile_group_slot` + `no_data_race_group_leader_slot`)"
    );
    println!(
        "                       Phase 1: `while w < 8`  -> `WARP_SUMS[w]`        (`no_oob_tile_linear_scan` + `no_data_race_diff_phase`)"
    );
    println!(
        "                       Elided: 2 SharedArray `mir.assert` bounds checks (`0` unsafe blocks!)"
    );
    println!(
        "\nVerified warp reduction ({expected_warp_sum}) and 2-phase block reduction ({expected_block_sum}) across all {N} threads!"
    );
    Ok(())
}

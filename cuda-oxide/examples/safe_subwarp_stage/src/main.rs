/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe sub-warp bitwise coordinate splits (`subwarp_bit_split_stage`)
//! and 3-phase multi-buffer shared-memory staging (`multibuf_subwarp_stage`) checked by Lean 4.
//!
//! ```text
//!   KERNEL 1: `subwarp_bit_split_stage` (`m_local = tid >> 4`, `k_lane = tid & 15`)
//!   ─────────────────────────────────────────────────────────────────────────────
//!   • Phase 0: `SUBWARP_TILE[m_local * 16 + k_lane] = val`
//!     Lean 4 proves `(tid >> 4) * 16 + (tid & 15) < 256` (`no_oob_tile_bit_split_slot`)
//!     and bit-split bijection `(tid >> 4) * 16 + (tid & 15) = tid` (`no_data_race_bit_split_slot`).
//!   • Barrier: `thread::sync_threads()` (`uniform_barriers`)
//!   • Phase 1: `while j < 16 { acc += SUBWARP_TILE[k_lane * 16 + j]; j += 1; }`
//!     Lean 4 proves `(tid % 16) * 16 + j < 256` (`no_oob_tile_group_lane_scan`)
//!     and phase separation (`no_data_race_diff_phase`).
//!
//!   KERNEL 2: `multibuf_subwarp_stage` (3-Phase, 3-Buffer `nemotron3_moe` Direct Pattern)
//!   ─────────────────────────────────────────────────────────────────────────────
//!   • Phase 0 (`if tid < 64`):
//!       `S_XS_DIRECT[tid] = scale` (`no_oob_tile_1d_slot`, `no_data_race_thread_slot`)
//!       `while w < 4 { S_X_DIRECT[tid * 4 + w] = ... }` (`no_oob_tile_packed_slot`, `no_data_race_packed_slot`)
//!   • Barrier 0: `thread::sync_threads()`
//!   • Phase 1 (`m_local = tid >> 4`, `k_lane = tid & 15`):
//!       Reads `S_X_DIRECT[k_lane * 16 + g]` & `S_XS_DIRECT[k_lane * 4 + g]` (`no_oob_tile_group_lane_scan`)
//!       16-lane sub-warp reduction (`shuffle_xor_f32` by `8, 4, 2, 1`)
//!       `if k_lane == 0 { S_ACT_DIRECT[m_local] = reduced; }` (`no_data_race_group_leader_slot`)
//!   • Barrier 1: `thread::sync_threads()`
//!   • Phase 2 (`if tid < 16`):
//!       Reads `S_ACT_DIRECT[tid]` (`no_oob_tile_1d_slot`, `no_data_race_diff_phase`)
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
    verify, warp,
};

/// Threads per block (`256` threads = `16` sub-warps of `16` lanes).
const BLOCK_SIZE: u32 = 256;
/// Number of 16-lane sub-warps per 256-thread block (`16`).
const SUBWARP_SIZE: usize = 16;
/// Number of packed elements staged per active thread in Phase 0 (`4`).
const PACK_WIDTH: usize = 4;
/// Number of active staging threads in Phase 0 (`64` threads * `4` words = `256`).
const STAGE_THREADS: usize = 64;
/// Number of thread blocks launched in the 1D grid (`4`).
const NUM_BLOCKS: u32 = 4;
/// Total number of elements and launched threads across the grid (`1,024`).
const N: usize = (BLOCK_SIZE * NUM_BLOCKS) as usize;

/// Device kernel module containing the provably safe sub-warp bit-split and multi-buffer kernels.
#[cuda_module]
mod kernels {
    use super::*;

    /// 2-phase 256-thread sub-warp bit-split staging kernel:
    /// - Splits `tid` (`0..256`) into `m_local = tid >> 4` (`0..16`) and
    ///   `k_lane = tid & 15` (`0..16`).
    /// - Phase 0 writes `SUBWARP_TILE[m_local * 16 + k_lane]` using the bit-split
    ///   bijection `(tid >> 4) * 16 + (tid & 15) = tid`.
    /// - Phase 1 scans `SUBWARP_TILE[k_lane * 16 + j]` across `j = 0..16`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        requires = (data.len() >= out.len()),
    )]
    #[verify(no_oob_idx(SUBWARP_TILE), no_data_race(SUBWARP_TILE), uniform_barriers)]
    pub fn subwarp_bit_split_stage(data: &[f32], mut out: DisjointSlice<f32>) {
        /// Shared-memory 16x16 tile indexed via sub-warp bit-splits (`256` floats).
        static mut SUBWARP_TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let gid = idx.get();
        let tid = thread::threadIdx_x() as usize;
        let m_local = tid >> 4;
        let k_lane = tid & 15;

        let val = if gid < data.len() { data[gid] } else { 0.0 };

        // Phase 0 (provably safe bit-split bijection `(tid >> 4) * 16 + (tid & 15) = tid`):
        SUBWARP_TILE[m_local * SUBWARP_SIZE + k_lane] = val;

        thread::sync_threads();

        // Phase 1 (provably safe sub-warp lane scan `(tid & 15) * 16 + j < 256`):
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < SUBWARP_SIZE {
            acc += SUBWARP_TILE[k_lane * SUBWARP_SIZE + j];
            j += 1;
        }

        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = acc;
        }
    }

    /// 3-phase, 3-buffer (`S_X_DIRECT`, `S_XS_DIRECT`, `S_ACT_DIRECT`) sub-warp staging
    /// and reduction kernel modeled on `fc1_relu2_requant_direct` (`nemotron3_moe`):
    /// - Phase 0 (`if tid < 64`): stages 1 scale per thread into `S_XS_DIRECT[tid]`
    ///   and 4 packed elements per thread into `S_X_DIRECT[tid * 4 + w]`.
    /// - Phase 1 (`m_local = tid >> 4`, `k_lane = tid & 15`): scans `S_X_DIRECT`
    ///   and `S_XS_DIRECT`, reduces across each 16-lane sub-warp, and stages
    ///   `S_ACT_DIRECT[m_local]` under `if k_lane == 0`.
    /// - Phase 2 (`if tid < 16`): reads `S_ACT_DIRECT[tid]` after the second barrier.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
        requires = (data.len() >= out.len()),
    )]
    #[verify(
        no_oob_idx(S_X_DIRECT, S_XS_DIRECT, S_ACT_DIRECT),
        no_data_race(S_X_DIRECT, S_XS_DIRECT, S_ACT_DIRECT),
        uniform_barriers
    )]
    pub fn multibuf_subwarp_stage(data: &[f32], mut out: DisjointSlice<f32>) {
        /// Shared-memory buffer holding 256 packed input activations (`64` groups * `4` words).
        static mut S_X_DIRECT: SharedArray<f32, 256> = SharedArray::UNINIT;
        /// Shared-memory buffer holding 64 per-group scale factors (`1` per 4-word group).
        static mut S_XS_DIRECT: SharedArray<f32, 64> = SharedArray::UNINIT;
        /// Shared-memory buffer holding 16 sub-warp reduced outputs (`1` per 16-lane sub-warp).
        static mut S_ACT_DIRECT: SharedArray<f32, 16> = SharedArray::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let gid = idx.get();
        let tid = thread::threadIdx_x() as usize;
        let m_local = tid >> 4;
        let k_lane = tid & 15;

        let base_val = if gid < data.len() { data[gid] } else { 0.0 };

        // Phase 0 (provably safe under `if tid < 64`: `S_XS_DIRECT[tid]` + `S_X_DIRECT[tid * 4 + w]`):
        if tid < STAGE_THREADS {
            S_XS_DIRECT[tid] = 0.5f32;
            let mut w = 0usize;
            while w < PACK_WIDTH {
                S_X_DIRECT[tid * PACK_WIDTH + w] = base_val;
                w += 1;
            }
        }

        thread::sync_threads();

        // Phase 1 (provably safe sub-warp lane scan + `if k_lane == 0` leader write to `S_ACT_DIRECT`):
        let mut sum = 0.0f32;
        let mut g = 0usize;
        while g < PACK_WIDTH {
            let xv = S_X_DIRECT[k_lane * SUBWARP_SIZE + g];
            let xsv = S_XS_DIRECT[k_lane * PACK_WIDTH + g];
            sum += xv * xsv;
            g += 1;
        }
        sum += warp::shuffle_xor_f32(sum, 8);
        sum += warp::shuffle_xor_f32(sum, 4);
        sum += warp::shuffle_xor_f32(sum, 2);
        sum += warp::shuffle_xor_f32(sum, 1);

        if k_lane == 0 {
            S_ACT_DIRECT[m_local] = sum;
        }

        thread::sync_threads();

        // Phase 2 (provably safe under `if tid < 16`: `S_ACT_DIRECT[tid]` after 2nd barrier):
        let mut final_val = sum;
        if tid < SUBWARP_SIZE {
            final_val = S_ACT_DIRECT[tid];
        }

        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = final_val;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== Provably Safe Sub-Warp Bit-Splits & Multi-Buffer Staging (`safe_subwarp_stage`) ===\n"
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;

    let host_data: Vec<f32> = vec![1.0f32; N];
    let d_data = DeviceBuffer::from_host(&stream, &host_data)?;
    let mut d_split_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; N])?;
    let mut d_multi_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; N])?;

    let config = LaunchConfig1D::new(NUM_BLOCKS, BLOCK_SIZE, 0);

    let prep_split = module.prepare_subwarp_bit_split_stage(config)?;
    module.subwarp_bit_split_stage(&stream, &prep_split, &d_data, &mut d_split_out)?;

    let prep_multi = module.prepare_multibuf_subwarp_stage(config)?;
    module.multibuf_subwarp_stage(&stream, &prep_multi, &d_data, &mut d_multi_out)?;
    stream.synchronize()?;

    let split_out = d_split_out.to_host_vec(&stream)?;
    let multi_out = d_multi_out.to_host_vec(&stream)?;

    // In `subwarp_bit_split_stage`, each thread sums 16 ones -> 16.0.
    // In `multibuf_subwarp_stage`, each lane sums 4 * (1.0 * 0.5) = 2.0, and
    // 16-lane sub-warp reduction sums 16 * 2.0 = 32.0.
    let expected_split = 16.0f32;
    let expected_multi = 32.0f32;

    for (i, &got) in split_out.iter().enumerate() {
        assert!(
            (got - expected_split).abs() < 1e-4,
            "subwarp_bit_split_stage mismatch at {i}: got {got}, expected {expected_split}"
        );
    }
    for (i, &got) in multi_out.iter().enumerate() {
        assert!(
            (got - expected_multi).abs() < 1e-4,
            "multibuf_subwarp_stage mismatch at {i}: got {got}, expected {expected_multi}"
        );
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract:          domain = 1, block = (256, 1, 1)");
    println!(
        "  • subwarp_bit_split_stage:  Proved no_oob_idx(SUBWARP_TILE), no_data_race(SUBWARP_TILE), uniform_barriers"
    );
    println!(
        "                              Phase 0: `SUBWARP_TILE[(tid >> 4) * 16 + (tid & 15)]` (`no_oob_tile_bit_split_slot` + `no_data_race_bit_split_slot`)"
    );
    println!(
        "                              Phase 1: `SUBWARP_TILE[(tid & 15) * 16 + j]`          (`no_oob_tile_group_lane_scan` + `no_data_race_diff_phase`)"
    );
    println!(
        "                              Elided: 2 SharedArray `mir.assert` bounds checks (`0` unsafe blocks!)"
    );
    println!(
        "  • multibuf_subwarp_stage:   Proved no_oob_idx & no_data_race on (S_X_DIRECT, S_XS_DIRECT, S_ACT_DIRECT), uniform_barriers"
    );
    println!(
        "                              Phase 0 (`if tid < 64`): `S_XS_DIRECT[tid]` + `S_X_DIRECT[tid * 4 + w]` (`no_oob_tile_packed_slot` + `no_data_race_packed_slot`)"
    );
    println!(
        "                              Phase 1 (`k_lane = tid & 15`): scans `S_X_DIRECT`/`S_XS_DIRECT` + `if k_lane == 0` writes `S_ACT_DIRECT[tid >> 4]`"
    );
    println!(
        "                              Phase 2 (`if tid < 16`): reads `S_ACT_DIRECT[tid]` after 2nd `sync_threads()`"
    );
    println!(
        "                              Elided: 6 SharedArray `mir.assert` bounds checks across 3 buffers (`0` unsafe blocks!)"
    );
    println!(
        "\nVerified sub-warp bit-split stage ({expected_split}) and 3-phase multi-buffer stage ({expected_multi}) across all {N} threads!"
    );
    Ok(())
}

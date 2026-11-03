/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A warp reduction whose single writer is proven by the index space.
//!
//! Only lane 0 of each warp holds the reduced value, and warp indices are
//! unique by construction, but there is no thread index for "the warp I belong
//! to". Such kernels used to compute the warp index by hand and store it
//! through `get_unchecked_mut`, giving up the bounds check along with the
//! disjointness proof.
//!
//! `WarpIndex` makes the warp the index space. `thread::warp_index()` mints the
//! witness only for lane 0, so the write goes through the ordinary checked
//! `get_mut` with no `unsafe`.
//!
//! The block width below is deliberately not a multiple of the warp size, which
//! is the case where deriving the warp index as `index_1d() / 32` would give
//! two different warps the same index. The same partial tail warp also rules
//! out `warp::reduce_sum_f32`, whose contract needs all 32 lanes launched and
//! converged, so both kernels call `warp::reduce_sum_f32_partial` with the
//! live-lane count instead. See the `partial_warp_reduce` example for the
//! tail widths that no butterfly reaches.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};
use std::time::Instant;

/// Not a multiple of 32, so each block's last warp is partial.
const BLOCK: u32 = 48;
const BLOCKS: u32 = 7;
const THREADS: u32 = BLOCK * BLOCKS;
/// Two warps per block, the second one partial.
const WARPS_PER_BLOCK: u32 = BLOCK.div_ceil(32);
const WARPS: u32 = WARPS_PER_BLOCK * BLOCKS;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum each warp's contributions and store one value per warp.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(48)]
    #[launch_contract(domain = 1, coordinates = u32, block = (48, 1, 1))]
    pub fn warp_sums(input: &[f32], mut sums: DisjointSlice<f32, thread::WarpIndex>) {
        let gid = thread::index_1d().get();
        let contribution = if gid < input.len() { input[gid] } else { 0.0 };
        let total = warp::reduce_sum_f32_partial(contribution, warp::live_lanes_1d());

        // No `unsafe`, and the store is bounds-checked: `warp_index` yields a
        // witness only for lane 0, and the slice's index space is the warp.
        if let Some(warp_slot) = thread::warp_index()
            && let Some(slot) = sums.get_mut(warp_slot)
        {
            *slot = total;
        }
    }

    /// The raw twin, kept as the measurement baseline.
    ///
    /// This is what the kernel writes without the index space: derive the warp
    /// index by hand and store through the unchecked accessor, giving up the
    /// bounds check along with the proof.
    ///
    /// # Safety
    ///
    /// Only lane 0 of each warp writes, the warp index is unique across the
    /// launch, and `sums` holds one element per warp of this launch geometry.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(48)]
    #[launch_contract(domain = 1, coordinates = u32, block = (48, 1, 1))]
    pub unsafe fn warp_sums_raw(input: &[f32], mut sums: DisjointSlice<f32>) {
        let gid = thread::index_1d().get();
        let contribution = if gid < input.len() { input[gid] } else { 0.0 };
        let total = warp::reduce_sum_f32_partial(contribution, warp::live_lanes_1d());

        if warp::lane_id() == 0 {
            let warps_per_block = thread::blockDim_x().div_ceil(32);
            let warp = thread::blockIdx_x() * warps_per_block + thread::threadIdx_x() / 32;
            // SAFETY: as documented on the kernel.
            unsafe {
                *sums.get_unchecked_mut(warp as usize) = total;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // SAFETY: the embedded module is the one built from this crate's
    // `kernels`, so every generated launch method matches its kernel.
    let module = unsafe { kernels::load(&ctx) }?;
    let host: Vec<f32> = (0..THREADS).map(|i| (i % 17) as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &host)?;
    let mut sums = DeviceBuffer::from_host(&stream, &vec![-1.0f32; WARPS as usize])?;

    let config = LaunchConfig1D::new(BLOCKS, BLOCK, 0);
    let prepared = module.prepare_warp_sums(config)?;
    module.warp_sums(&stream, &prepared, &input, &mut sums)?;
    stream.synchronize()?;

    // Every warp of every block, including each block's partial second warp.
    let mut expected = vec![0.0f32; WARPS as usize];
    for block in 0..BLOCKS {
        for lane in 0..BLOCK {
            let gid = block * BLOCK + lane;
            let warp = block * WARPS_PER_BLOCK + lane / 32;
            expected[warp as usize] += host[gid as usize];
        }
    }

    let got = sums.to_host_vec(&stream)?;
    let mut worst = 0.0f32;
    for (i, value) in got.iter().enumerate() {
        let error = (value - expected[i]).abs();
        if error > worst {
            worst = error;
        }
    }
    if worst > 1e-4 {
        return Err(format!("warp sums disagree with the host reference by {worst:e}").into());
    }
    if got.iter().any(|&value| value == -1.0) {
        return Err("some warp never wrote its slot".into());
    }

    println!(
        "warp_sums: {WARPS} warps over {BLOCKS} blocks of {BLOCK} threads, max error {worst:e}"
    );
    println!("every warp wrote exactly one slot, bounds-checked and without unsafe");

    // The raw twin must agree element for element, not merely run.
    let mut raw_sums = DeviceBuffer::from_host(&stream, &vec![-1.0f32; WARPS as usize])?;
    let raw_prepared = module.prepare_warp_sums_raw(config)?;
    // SAFETY: the kernel's contract is documented on it, and `sums` holds one
    // element per warp of this geometry.
    unsafe { module.warp_sums_raw(&stream, &raw_prepared, &input, &mut raw_sums) }?;
    stream.synchronize()?;
    let raw_got = raw_sums.to_host_vec(&stream)?;
    if raw_got != got {
        return Err("the safe and raw warp kernels disagree".into());
    }
    println!("safe and raw agree bitwise across every warp");

    // Larger geometry, so the timing is not dominated by launch overhead.
    const BENCH_BLOCKS: u32 = 8192;
    const BENCH_RUNS: u32 = 50;
    let bench_threads = BLOCK * BENCH_BLOCKS;
    let bench_warps = WARPS_PER_BLOCK * BENCH_BLOCKS;
    let bench_host: Vec<f32> = (0..bench_threads).map(|i| (i % 17) as f32).collect();
    let bench_input = DeviceBuffer::from_host(&stream, &bench_host)?;
    let mut bench_sums = DeviceBuffer::from_host(&stream, &vec![0.0f32; bench_warps as usize])?;
    let bench_config = LaunchConfig1D::new(BENCH_BLOCKS, BLOCK, 0);
    let bench_safe = module.prepare_warp_sums(bench_config)?;
    let bench_raw = module.prepare_warp_sums_raw(bench_config)?;

    let mut time = |label: &str, raw: bool| -> Result<f64, Box<dyn std::error::Error>> {
        for _ in 0..3 {
            if raw {
                // SAFETY: as above.
                unsafe {
                    module.warp_sums_raw(&stream, &bench_raw, &bench_input, &mut bench_sums)
                }?;
            } else {
                module.warp_sums(&stream, &bench_safe, &bench_input, &mut bench_sums)?;
            }
        }
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..BENCH_RUNS {
            if raw {
                // SAFETY: as above.
                unsafe {
                    module.warp_sums_raw(&stream, &bench_raw, &bench_input, &mut bench_sums)
                }?;
            } else {
                module.warp_sums(&stream, &bench_safe, &bench_input, &mut bench_sums)?;
            }
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0 / BENCH_RUNS as f64;
        println!("  {label:<22} {ms:7.4} ms");
        Ok(ms)
    };

    println!("\n{bench_warps} warps, {BENCH_RUNS} timed runs:");
    let safe_ms = time("safe warp index", false)?;
    let raw_ms = time("raw get_unchecked_mut", true)?;
    println!(
        "  ratio safe / raw: {:.3}",
        safe_ms / raw_ms.max(f64::MIN_POSITIVE)
    );

    println!("\nSUCCESS");
    Ok(())
}

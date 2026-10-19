/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reducing over the live lanes of a partial warp.
//!
//! `warp::reduce_sum_f32` shuffles with the full 32-lane member mask, so every
//! lane must be launched and converged. A block whose width is not a multiple
//! of 32 leaves its last warp short, and the PTX ISA makes `shfl.sync`
//! undefined when a thread sources a lane that is inactive or outside the
//! member mask. The partial forms take the live-lane count and reduce over
//! exactly those lanes.
//!
//! Two block widths cover the two paths through the reduction:
//!
//! - 48 threads leave a tail warp of 16, a power of two, which takes the same
//!   butterfly as a full warp with the mask and the first offset cut down.
//! - 45 threads leave a tail warp of 13, which no butterfly reaches: `lane ^
//!   offset` walks outside the live lanes. That count folds the upper part of
//!   the span into the lower half instead, sourcing a clamped lane so no
//!   thread ever reads one that was never launched.
//!
//! The input is deliberately not uniform across a warp, so a reduction that
//! dropped the tail lanes, double-counted a lane, or read a lane belonging to
//! another warp disagrees with the host reference rather than passing.
//!
//! What this example does not show is a wrong answer from the full-warp form.
//! Substituting `warp::reduce_sum_f32` here still matches the reference
//! exactly on sm_120, so the hardware returns something harmless for a
//! shuffle that sources a never-launched lane. The ISA promises nothing about
//! that value, on this architecture or another, which is the whole reason to
//! name the live lanes in the mask rather than rely on it.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};

/// A power-of-two tail: two warps per block, the second with 16 live lanes.
const BLOCK_POW2: u32 = 48;
/// A tail that is not a power of two: 13 live lanes in the second warp.
const BLOCK_ODD: u32 = 45;
const BLOCKS: u32 = 7;

const WARPS_PER_BLOCK_POW2: u32 = BLOCK_POW2.div_ceil(32);
const WARPS_PER_BLOCK_ODD: u32 = BLOCK_ODD.div_ceil(32);
const WARPS_POW2: u32 = WARPS_PER_BLOCK_POW2 * BLOCKS;
const WARPS_ODD: u32 = WARPS_PER_BLOCK_ODD * BLOCKS;

#[cuda_module]
mod kernels {
    use super::*;

    /// One sum per warp, from blocks of 48.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(48)]
    #[launch_contract(domain = 1, coordinates = u32, block = (48, 1, 1))]
    pub fn sums_pow2_tail(input: &[f32], mut sums: DisjointSlice<f32, thread::WarpIndex>) {
        let gid = thread::index_1d().get();
        let contribution = if gid < input.len() { input[gid] } else { 0.0 };
        let total = warp::reduce_sum_f32_partial(contribution, warp::live_lanes_1d());

        if let Some(warp_slot) = thread::warp_index()
            && let Some(slot) = sums.get_mut(warp_slot)
        {
            *slot = total;
        }
    }

    /// One sum per warp, from blocks of 45, whose tail warp has 13 live lanes.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(45)]
    #[launch_contract(domain = 1, coordinates = u32, block = (45, 1, 1))]
    pub fn sums_odd_tail(input: &[f32], mut sums: DisjointSlice<f32, thread::WarpIndex>) {
        let gid = thread::index_1d().get();
        let contribution = if gid < input.len() { input[gid] } else { 0.0 };
        let total = warp::reduce_sum_f32_partial(contribution, warp::live_lanes_1d());

        if let Some(warp_slot) = thread::warp_index()
            && let Some(slot) = sums.get_mut(warp_slot)
        {
            *slot = total;
        }
    }

    /// One maximum per warp, from the same 45-thread geometry.
    ///
    /// Maximum is the reduction where a lane read from outside the live set
    /// shows up plainly: an uninitialised or foreign lane carrying a larger
    /// value replaces the answer rather than perturbing it.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(45)]
    #[launch_contract(domain = 1, coordinates = u32, block = (45, 1, 1))]
    pub fn maxima_odd_tail(input: &[f32], mut maxima: DisjointSlice<f32, thread::WarpIndex>) {
        let gid = thread::index_1d().get();
        let contribution = if gid < input.len() {
            input[gid]
        } else {
            f32::NEG_INFINITY
        };
        let total = warp::reduce_max_f32_partial(contribution, warp::live_lanes_1d());

        if let Some(warp_slot) = thread::warp_index()
            && let Some(slot) = maxima.get_mut(warp_slot)
        {
            *slot = total;
        }
    }
}

/// Host reference: fold each warp of the launch geometry over its own lanes.
fn reference(
    host: &[f32],
    block: u32,
    blocks: u32,
    warps_per_block: u32,
    combine: impl Fn(f32, f32) -> f32,
    identity: f32,
) -> Vec<f32> {
    let mut expected = vec![identity; (warps_per_block * blocks) as usize];
    for b in 0..blocks {
        for lane in 0..block {
            let gid = b * block + lane;
            let warp = b * warps_per_block + lane / 32;
            expected[warp as usize] = combine(expected[warp as usize], host[gid as usize]);
        }
    }
    expected
}

fn compare(name: &str, got: &[f32], expected: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    if got.iter().any(|&value| value == -1.0) {
        return Err(format!("{name}: some warp never wrote its slot").into());
    }
    let mut worst = 0.0f32;
    for (i, value) in got.iter().enumerate() {
        let error = (value - expected[i]).abs();
        if error > worst {
            worst = error;
        }
    }
    if worst > 1e-4 {
        return Err(format!("{name}: disagrees with the host reference by {worst:e}").into());
    }
    println!("{name}: {} warps, max error {worst:e}", got.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // SAFETY: the embedded module is the one built from this crate's
    // `kernels`, so every generated launch method matches its kernel.
    let module = unsafe { kernels::load(&ctx) }?;
    // Values vary within each warp, so a reduction that lost or duplicated a
    // lane lands on a different answer.
    let threads_pow2 = BLOCK_POW2 * BLOCKS;
    let host_pow2: Vec<f32> = (0..threads_pow2).map(|i| (i % 17) as f32).collect();
    let input_pow2 = DeviceBuffer::from_host(&stream, &host_pow2)?;
    let mut sums = DeviceBuffer::from_host(&stream, &vec![-1.0f32; WARPS_POW2 as usize])?;
    let prepared = module.prepare_sums_pow2_tail(LaunchConfig1D::new(BLOCKS, BLOCK_POW2, 0))?;
    module.sums_pow2_tail(&stream, &prepared, &input_pow2, &mut sums)?;
    stream.synchronize()?;
    compare(
        "sums_pow2_tail",
        &sums.to_host_vec(&stream)?,
        &reference(
            &host_pow2,
            BLOCK_POW2,
            BLOCKS,
            WARPS_PER_BLOCK_POW2,
            |a, b| a + b,
            0.0,
        ),
    )?;

    let threads_odd = BLOCK_ODD * BLOCKS;
    let host_odd: Vec<f32> = (0..threads_odd).map(|i| (i % 17) as f32).collect();
    let input_odd = DeviceBuffer::from_host(&stream, &host_odd)?;

    let mut odd_sums = DeviceBuffer::from_host(&stream, &vec![-1.0f32; WARPS_ODD as usize])?;
    let odd_prepared = module.prepare_sums_odd_tail(LaunchConfig1D::new(BLOCKS, BLOCK_ODD, 0))?;
    module.sums_odd_tail(&stream, &odd_prepared, &input_odd, &mut odd_sums)?;
    stream.synchronize()?;
    compare(
        "sums_odd_tail",
        &odd_sums.to_host_vec(&stream)?,
        &reference(
            &host_odd,
            BLOCK_ODD,
            BLOCKS,
            WARPS_PER_BLOCK_ODD,
            |a, b| a + b,
            0.0,
        ),
    )?;

    let mut odd_maxima = DeviceBuffer::from_host(&stream, &vec![-1.0f32; WARPS_ODD as usize])?;
    let max_prepared = module.prepare_maxima_odd_tail(LaunchConfig1D::new(BLOCKS, BLOCK_ODD, 0))?;
    module.maxima_odd_tail(&stream, &max_prepared, &input_odd, &mut odd_maxima)?;
    stream.synchronize()?;
    compare(
        "maxima_odd_tail",
        &odd_maxima.to_host_vec(&stream)?,
        &reference(
            &host_odd,
            BLOCK_ODD,
            BLOCKS,
            WARPS_PER_BLOCK_ODD,
            f32::max,
            f32::NEG_INFINITY,
        ),
    )?;

    println!("SUCCESS: partial warps reduce over exactly their live lanes");
    Ok(())
}

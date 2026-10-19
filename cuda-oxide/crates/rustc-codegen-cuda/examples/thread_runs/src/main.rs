/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A thread owning several contiguous elements, including the clipped tail.
//!
//! `tile_thread32` accepts only a whole tile, so the one thread whose run
//! straddles the end of the buffer gets `None` and has to fall back to raw
//! indexing. Launches round the grid up, so that thread almost always exists.
//!
//! `thread_run32` returns a run that is either whole or clipped, and
//! `grid_stride_runs32` walks every run a thread owns with the period read from
//! the launch rather than passed in. Both buffer lengths below are deliberately
//! not multiples of the run width, so the tail path runs in every kernel.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, LinearTiles, cuda_module, kernel, launch_bounds, launch_contract, thread,
};
use std::time::Instant;

/// Elements per thread run.
const RUN: usize = 4;
/// Not a multiple of RUN, so the last run is clipped.
const LEN: usize = 4093;
/// Fewer threads than runs, so the grid-stride kernel loops.
const GRID_STRIDE_THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// Double every element, one run per thread, tail included.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn double_runs(mut data: DisjointSlice<f32, LinearTiles<RUN>>) {
        let thread_index = thread::index_1d_u32(launch_context);
        let Some(mut run) = data.thread_run32(thread_index) else {
            return;
        };
        // `len()` is RUN for a whole run and the remainder for a clipped one,
        // so one loop covers both without raw indexing.
        for k in 0..run.len() {
            if let Some(mut slot) = run.at(k) {
                let value = slot.read();
                slot.write(value * 2.0);
            }
        }
    }

    /// The raw twin of `double_runs`, kept as the measurement baseline.
    ///
    /// This is what a coarsened kernel writes today: compute the base by hand,
    /// bounds-check each element, and store through the unchecked accessor.
    ///
    /// # Safety
    ///
    /// Each thread owns `base .. base + RUN`, which are disjoint across
    /// threads, and every index is checked against the length before use.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub unsafe fn double_runs_raw(mut data: DisjointSlice<f32, LinearTiles<RUN>>) {
        let thread_index = thread::index_1d_u32(launch_context);
        let base = thread_index.get() as usize * RUN;
        let len = data.len();
        let mut k = 0usize;
        while k < RUN {
            let index = base + k;
            if index < len {
                // SAFETY: the index was just checked, and distinct threads own
                // disjoint runs.
                unsafe {
                    let slot = data.get_unchecked_mut(index);
                    *slot *= 2.0;
                }
            }
            k += 1;
        }
    }

    /// Add one to every element through a grid-stride walk over runs.
    ///
    /// The launch has fewer threads than the buffer has runs, so each thread
    /// takes several. The period comes from the launch geometry inside
    /// `grid_stride_runs32`, so no stride crosses the call to disagree with it.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn increment_grid_stride(mut data: DisjointSlice<f32, LinearTiles<RUN>>) {
        let thread_index = thread::index_1d_u32(launch_context);
        let mut runs = data.grid_stride_runs32(thread_index);
        while let Some(mut run) = runs.next_run() {
            for k in 0..run.len() {
                if let Some(mut slot) = run.at(k) {
                    let value = slot.read();
                    slot.write(value + 1.0);
                }
            }
        }
    }

    /// Write each run's length, so the host can see exactly one clipped run.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn record_run_lengths(
        mut data: DisjointSlice<f32, LinearTiles<RUN>>,
        mut lengths: DisjointSlice<u32>,
    ) {
        let thread_index = thread::index_1d_u32(launch_context);
        let clipped_marker = match data.thread_run32(thread_index) {
            None => return,
            Some(run) => {
                if run.is_clipped() {
                    run.len()
                } else {
                    0
                }
            }
        };
        let index = thread::index_1d();
        if let Some(slot) = lengths.get_mut(index) {
            *slot = clipped_marker;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // SAFETY: the embedded module is the one built from this crate's
    // `kernels`, so every generated launch method matches its kernel.
    let module = unsafe { kernels::load(&ctx) }?;
    let host: Vec<f32> = (0..LEN).map(|i| i as f32).collect();
    let runs = LEN.div_ceil(RUN) as u32;

    // ---- one run per thread, tail included -------------------------------
    let mut data = DeviceBuffer::from_host(&stream, &host)?;
    let config = LaunchConfig1D::new(runs.div_ceil(64), 64, 0);
    let prepared = module.prepare_double_runs(config)?;
    module.double_runs(&stream, &prepared, &mut data)?;
    stream.synchronize()?;

    let doubled = data.to_host_vec(&stream)?;
    let mut mismatches = 0usize;
    for (i, value) in doubled.iter().enumerate() {
        if (value - host[i] * 2.0).abs() > 1e-6 {
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        return Err(format!("double_runs left {mismatches} elements untouched").into());
    }
    println!("double_runs: all {LEN} elements doubled, including the clipped tail");

    // ---- grid-stride walk over runs --------------------------------------
    let mut strided = DeviceBuffer::from_host(&stream, &host)?;
    let stride_config = LaunchConfig1D::new(GRID_STRIDE_THREADS.div_ceil(64), 64, 0);
    let stride_prepared = module.prepare_increment_grid_stride(stride_config)?;
    module.increment_grid_stride(&stream, &stride_prepared, &mut strided)?;
    stream.synchronize()?;

    let incremented = strided.to_host_vec(&stream)?;
    let mut wrong = 0usize;
    for (i, value) in incremented.iter().enumerate() {
        if (value - (host[i] + 1.0)).abs() > 1e-6 {
            wrong += 1;
        }
    }
    if wrong != 0 {
        return Err(format!(
            "increment_grid_stride touched {wrong} elements the wrong number of times"
        )
        .into());
    }
    println!(
        "increment_grid_stride: {GRID_STRIDE_THREADS} threads covered {runs} runs exactly once each"
    );

    // ---- exactly one clipped run -----------------------------------------
    let mut lengths_data = DeviceBuffer::from_host(&stream, &host)?;
    let mut lengths = DeviceBuffer::from_host(&stream, &vec![0u32; runs as usize])?;
    let lengths_prepared = module.prepare_record_run_lengths(config)?;
    module.record_run_lengths(&stream, &lengths_prepared, &mut lengths_data, &mut lengths)?;
    stream.synchronize()?;

    let recorded = lengths.to_host_vec(&stream)?;
    let clipped: Vec<u32> = recorded.into_iter().filter(|&len| len != 0).collect();
    let expected_tail = (LEN % RUN) as u32;
    if clipped != vec![expected_tail] {
        return Err(format!(
            "expected exactly one clipped run of {expected_tail}, got {clipped:?}"
        )
        .into());
    }
    println!("record_run_lengths: exactly one clipped run, of {expected_tail} elements");

    // ---- safe runs against the raw twin ----------------------------------
    // A larger buffer than the correctness checks use, so the timing is not
    // dominated by launch overhead.
    const BENCH_LEN: usize = 1 << 22;
    const BENCH_RUNS: u32 = 20;
    let bench_host: Vec<f32> = (0..BENCH_LEN).map(|i| (i % 1000) as f32).collect();
    let bench_runs = BENCH_LEN.div_ceil(RUN) as u32;
    let bench_config = LaunchConfig1D::new(bench_runs.div_ceil(64), 64, 0);
    let safe_prepared = module.prepare_double_runs(bench_config)?;
    let raw_prepared = module.prepare_double_runs_raw(bench_config)?;
    let mut bench = DeviceBuffer::from_host(&stream, &bench_host)?;

    let mut time = |label: &str, raw: bool| -> Result<f64, Box<dyn std::error::Error>> {
        for _ in 0..3 {
            if raw {
                // SAFETY: the kernel's own contract is documented on it, and
                // this launch covers exactly the buffer.
                unsafe { module.double_runs_raw(&stream, &raw_prepared, &mut bench) }?;
            } else {
                module.double_runs(&stream, &safe_prepared, &mut bench)?;
            }
        }
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..BENCH_RUNS {
            if raw {
                // SAFETY: as above.
                unsafe { module.double_runs_raw(&stream, &raw_prepared, &mut bench) }?;
            } else {
                module.double_runs(&stream, &safe_prepared, &mut bench)?;
            }
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0 / BENCH_RUNS as f64;
        println!("  {label:<20} {ms:7.4} ms");
        Ok(ms)
    };

    println!("\n{BENCH_LEN} elements, runs of {RUN}, {BENCH_RUNS} timed runs:");
    let safe_ms = time("safe runs", false)?;
    let raw_ms = time("raw get_unchecked_mut", true)?;
    println!(
        "  ratio safe / raw: {:.3}",
        safe_ms / raw_ms.max(f64::MIN_POSITIVE)
    );

    println!("\nSUCCESS: contiguous runs cover the buffer with no raw indexing");
    Ok(())
}

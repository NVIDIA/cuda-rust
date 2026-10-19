/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pinned host memory and transfer/compute overlap benchmark.
//!
//! Run with:
//!   cargo oxide run pinned_overlap

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;
const BANDWIDTH_SIZES: &[usize] = &[1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20];
const CHUNK_BYTES: usize = 32 << 20;
const CHUNKS: usize = 8;
const SLOTS: usize = 3;

#[cuda_module]
mod kernels {
    use super::*;

    /// A deliberately cheap in-place transform so the transfer path is visible.
    #[kernel]
    pub fn increment(mut data: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(elem) = data.get_mut(idx) {
            *elem += 1.0f32;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BandwidthRow {
    bytes: usize,
    pageable_htod: f64,
    pinned_htod: f64,
    pageable_dtoh: f64,
    pinned_dtoh: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Pinned host transfer and overlap demo ===\n");
    println!("The kernel only adds 1.0 to each element; the measured difference is the data path.");

    let ctx = CudaContext::new(0)?;
    let main_stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let bandwidth = benchmark_bandwidth(&ctx, &main_stream)?;
    print_bandwidth(&bandwidth);

    let input = make_pipeline_input();
    let reference = input.iter().map(|value| value + 1.0).collect::<Vec<_>>();

    // The overlapped path pre-allocates its buffers before timing. The
    // serialized path intentionally retains per-chunk allocation and
    // synchronization to model the simple pageable helper path. This is an
    // end-to-end comparison, not an allocation-free copy benchmark.
    let (serialized_time, serialized_output) = run_serialized(
        &module,
        &main_stream,
        &input,
        CHUNK_BYTES / size_of::<f32>(),
        CHUNKS,
    )?;
    let (overlapped_time, overlapped_output) = run_overlapped(
        &ctx,
        &main_stream,
        &module,
        &input,
        CHUNK_BYTES / size_of::<f32>(),
        CHUNKS,
    )?;

    verify_exact("serialized pipeline", &serialized_output, &reference)?;
    verify_exact("overlapped pipeline", &overlapped_output, &reference)?;

    let speedup = serialized_time.as_secs_f64() / overlapped_time.as_secs_f64();
    println!(
        "\nPipeline impact ({} x {} MiB chunks)",
        CHUNKS,
        CHUNK_BYTES >> 20
    );
    println!(
        "  serialized pageable pipeline: {:>9.3} ms",
        millis(serialized_time)
    );
    println!(
        "  overlapped pinned pipeline:   {:>9.3} ms",
        millis(overlapped_time)
    );
    println!("  measured speedup:             {:>9.2}x", speedup);
    println!("\nSUCCESS: pinned transfers and overlap pipeline verified exactly.");
    Ok(())
}

fn benchmark_bandwidth(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
) -> Result<Vec<BandwidthRow>, Box<dyn Error>> {
    let mut rows = Vec::with_capacity(BANDWIDTH_SIZES.len());
    for &bytes in BANDWIDTH_SIZES {
        let len = bytes / size_of::<f32>();
        let source = make_bandwidth_input(len);

        let pageable_htod = median_bandwidth(TIMED_ITERS, bytes, || {
            let start = stream.record_event(Some(timing_flags()))?;
            let device = DeviceBuffer::from_host(stream, &source)?;
            let end = stream.record_event(Some(timing_flags()))?;
            let elapsed = start.elapsed_ms(&end)?;
            drop(device);
            Ok(elapsed)
        })?;

        let pinned_source = PinnedHostBuffer::from_slice(ctx, &source)?;
        let mut pinned_device = DeviceBuffer::<f32>::zeroed(stream, len)?;
        // Drops before the two buffers above: an early `?` return drains the
        // stream before pinned or device memory is freed under a live copy.
        let _drain_htod = DrainOnDrop(std::slice::from_ref(stream));
        for _ in 0..WARMUP_ITERS {
            // SAFETY: `pinned_source` and `pinned_device` remain alive and
            // unchanged until the stream is synchronized below.
            unsafe { pinned_device.copy_from_pinned_host_async(stream, &pinned_source)? };
        }
        stream.synchronize()?;
        let pinned_htod = median_bandwidth(TIMED_ITERS, bytes, || {
            let start = stream.record_event(Some(timing_flags()))?;
            // SAFETY: the source and destination remain alive until
            // `elapsed_ms` synchronizes the recorded copy.
            unsafe { pinned_device.copy_from_pinned_host_async(stream, &pinned_source)? };
            let end = stream.record_event(Some(timing_flags()))?;
            Ok(start.elapsed_ms(&end)?)
        })?;

        let pageable_device = DeviceBuffer::from_host(stream, &source)?;
        for _ in 0..WARMUP_ITERS {
            let _ = pageable_device.to_host_vec(stream)?;
        }
        let pageable_dtoh = median_bandwidth(TIMED_ITERS, bytes, || {
            let start = stream.record_event(Some(timing_flags()))?;
            let output = pageable_device.to_host_vec(stream)?;
            let end = stream.record_event(Some(timing_flags()))?;
            debug_assert_eq!(output, source);
            Ok(start.elapsed_ms(&end)?)
        })?;

        let mut pinned_output = PinnedHostBuffer::<f32>::zeroed(ctx, len)?;
        // Same guard for the download destination: drops before `pinned_output`.
        let _drain_dtoh = DrainOnDrop(std::slice::from_ref(stream));
        for _ in 0..WARMUP_ITERS {
            // SAFETY: `pinned_output` is not read or dropped until the stream
            // synchronization completes the download.
            unsafe { pinned_device.copy_to_pinned_host_async(stream, &mut pinned_output)? };
            stream.synchronize()?;
        }
        let pinned_dtoh = median_bandwidth(TIMED_ITERS, bytes, || {
            let start = stream.record_event(Some(timing_flags()))?;
            // SAFETY: `pinned_output` is kept alive and is not read or aliased
            // until `elapsed_ms` synchronizes the in-flight download.
            unsafe { pinned_device.copy_to_pinned_host_async(stream, &mut pinned_output)? };
            let end = stream.record_event(Some(timing_flags()))?;
            Ok(start.elapsed_ms(&end)?)
        })?;
        assert_eq!(pinned_output.as_slice(), source.as_slice());

        rows.push(BandwidthRow {
            bytes,
            pageable_htod,
            pinned_htod,
            pageable_dtoh,
            pinned_dtoh,
        });
    }
    Ok(rows)
}

fn median_bandwidth<F>(
    iterations: usize,
    bytes: usize,
    mut sample: F,
) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Result<f32, Box<dyn Error>>,
{
    for _ in 0..WARMUP_ITERS {
        let _ = sample()?;
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        samples.push(f64::from(sample()?));
    }
    samples.sort_by(f64::total_cmp);
    let median_ms = samples[samples.len() / 2];
    Ok(bytes as f64 / (median_ms * 1_000_000.0))
}

fn run_serialized(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    input: &[f32],
    chunk_len: usize,
    chunks: usize,
) -> Result<(Duration, Vec<f32>), Box<dyn Error>> {
    let mut output = vec![0.0f32; input.len()];
    let start = Instant::now();
    for chunk in 0..chunks {
        let range = chunk * chunk_len..(chunk + 1) * chunk_len;
        let mut device = DeviceBuffer::from_host(stream, &input[range.clone()])?;
        // SAFETY: the launch is one-dimensional and covers exactly the device
        // buffer passed to the in-place transform.
        unsafe {
            module.increment(
                stream,
                LaunchConfig::for_num_elems(chunk_len as u32),
                &mut device,
            )?;
        }
        let result = device.to_host_vec(stream)?;
        output[range].copy_from_slice(&result);
    }
    Ok((start.elapsed(), output))
}

/// Synchronizes streams on drop so an early error return cannot free pinned
/// or device memory that an in-flight async transfer still references.
struct DrainOnDrop<'a>(&'a [Arc<CudaStream>]);

impl Drop for DrainOnDrop<'_> {
    fn drop(&mut self) {
        for stream in self.0 {
            let _ = stream.synchronize();
        }
    }
}

fn run_overlapped(
    ctx: &Arc<CudaContext>,
    main_stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    input: &[f32],
    chunk_len: usize,
    chunks: usize,
) -> Result<(Duration, Vec<f32>), Box<dyn Error>> {
    let streams = (0..SLOTS)
        .map(|_| main_stream.fork())
        .collect::<Result<Vec<_>, _>>()?;
    let mut devices = streams
        .iter()
        .map(|stream| DeviceBuffer::<f32>::zeroed(stream, chunk_len))
        .collect::<Result<Vec<_>, _>>()?;
    let mut stagers = (0..SLOTS)
        .map(|_| PinnedHostBuffer::<f32>::zeroed(ctx, chunk_len))
        .collect::<Result<Vec<_>, _>>()?;
    let mut completions: Vec<Option<(usize, CudaEvent)>> = (0..SLOTS).map(|_| None).collect();
    let mut output = vec![0.0f32; input.len()];
    // Declared after the buffers so it drops before them: an early `?` return
    // drains the forked streams before any stager or device buffer is freed
    // under an in-flight transfer.
    let _drain = DrainOnDrop(&streams);

    let start = Instant::now();
    for chunk in 0..chunks {
        let slot = chunk % SLOTS;
        let range = chunk * chunk_len..(chunk + 1) * chunk_len;

        if let Some((previous_chunk, completion)) = completions[slot].take() {
            // A final join protects destruction, but cannot make host reuse
            // safe. The slot's previous DtoH must finish before we read its
            // result and refill the same pinned allocation.
            completion.synchronize()?;
            let previous_range = previous_chunk * chunk_len..(previous_chunk + 1) * chunk_len;
            output[previous_range].copy_from_slice(stagers[slot].as_slice());
        }
        stagers[slot].as_mut_slice().copy_from_slice(&input[range]);

        let stream = &streams[slot];
        // This is the canonical async overlap pattern described alongside
        // commit 6c6c9485: rotating pinned host stagers refill persistent
        // device buffers across independent stream iterations.
        // SAFETY: the pinned stager and device buffer remain alive. The stager
        // is not mutated or read again until `completion` synchronizes the
        // HtoD -> kernel -> DtoH sequence on this stream.
        unsafe {
            devices[slot].copy_from_pinned_host_async(stream, &stagers[slot])?;
            module.increment(
                stream,
                LaunchConfig::for_num_elems(chunk_len as u32),
                &mut devices[slot],
            )?;
            devices[slot].copy_to_pinned_host_async(stream, &mut stagers[slot])?;
        }
        completions[slot] = Some((chunk, stream.record_event(None)?));
    }

    for slot in 0..SLOTS {
        if let Some((chunk, completion)) = completions[slot].take() {
            completion.synchronize()?;
            let range = chunk * chunk_len..(chunk + 1) * chunk_len;
            output[range].copy_from_slice(stagers[slot].as_slice());
        }
    }

    // Join all forked streams on the parent before timing ends. This is the
    // final drain; per-slot waits above are only for safe ring-buffer reuse.
    for stream in &streams {
        main_stream.join(stream)?;
    }
    main_stream.synchronize()?;
    Ok((start.elapsed(), output))
}

fn verify_exact(name: &str, actual: &[f32], expected: &[f32]) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        let mismatch = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(actual.len().min(expected.len()));
        return Err(format!(
            "{name} mismatch at index {mismatch}: got {:?}, expected {:?}",
            actual.get(mismatch),
            expected.get(mismatch)
        )
        .into());
    }
    Ok(())
}

fn make_bandwidth_input(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 257) as f32 * 0.25).collect()
}

fn make_pipeline_input() -> Vec<f32> {
    make_bandwidth_input(CHUNK_BYTES / size_of::<f32>() * CHUNKS)
}

fn print_bandwidth(rows: &[BandwidthRow]) {
    println!("\nTransfer bandwidth (GB/s, median of {TIMED_ITERS} timed iterations)");
    println!("  size       pageable HtoD  pinned HtoD  pageable DtoH  pinned DtoH");
    for row in rows {
        println!(
            "  {:>5} MiB       {:>8.2}      {:>8.2}       {:>8.2}      {:>8.2}",
            row.bytes >> 20,
            row.pageable_htod,
            row.pinned_htod,
            row.pageable_dtoh,
            row.pinned_dtoh
        );
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn timing_flags() -> cuda_core::sys::CUevent_flags {
    cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT
}

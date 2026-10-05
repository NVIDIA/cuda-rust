/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! f32 vs f16 global atomic-add microbenchmark.
//!
//! Run:
//!   ./crates/rustc-codegen-cuda/examples/atomic_f16/run-bench.sh

#![feature(f16)]

use std::sync::Arc;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF16, DeviceAtomicF32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const N: u32 = 1 << 22;
const WARMUP: usize = 5;
const ITERS: usize = 20;
const BINS: &[u32] = &[1, 4, 16, 64, 256, 1024, 4096];

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn f32_add(hist: &[f32], n: u32, nbins: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let bin = i % nbins as usize;
        let cell = unsafe { &*(hist.as_ptr().add(bin) as *const DeviceAtomicF32) };
        cell.fetch_add(1.0f32, AtomicOrdering::Relaxed);
    }

    #[kernel]
    pub fn f16_add(hist: &[f16], n: u32, nbins: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let bin = i % nbins as usize;
        let cell = unsafe { &*(hist.as_ptr().add(bin) as *const DeviceAtomicF16) };
        cell.fetch_add(1.0f16, AtomicOrdering::Relaxed);
    }

    #[kernel]
    pub fn f32_add_return(hist: &[f32], mut old: DisjointSlice<f32>, n: u32, nbins: u32) {
        let gid = thread::index_1d();
        let i = gid.get();
        if i >= n as usize {
            return;
        }
        let bin = i % nbins as usize;
        let cell = unsafe { &*(hist.as_ptr().add(bin) as *const DeviceAtomicF32) };
        let prev = cell.fetch_add(1.0f32, AtomicOrdering::Relaxed);
        if let Some(out) = old.get_mut(gid) {
            *out = prev;
        }
    }

    #[kernel]
    pub fn f16_add_return(hist: &[f16], mut old: DisjointSlice<f16>, n: u32, nbins: u32) {
        let gid = thread::index_1d();
        let i = gid.get();
        if i >= n as usize {
            return;
        }
        let bin = i % nbins as usize;
        let cell = unsafe { &*(hist.as_ptr().add(bin) as *const DeviceAtomicF16) };
        let prev = cell.fetch_add(1.0f16, AtomicOrdering::Relaxed);
        if let Some(out) = old.get_mut(gid) {
            *out = prev;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    println!("atomic_f16 bench");
    println!("n,mode,type,bins,avg_ms,mops");

    for &bins in BINS {
        bench_no_return(&module, &stream, bins)?;
        bench_return(&module, &stream, bins)?;
    }

    Ok(())
}

fn bench_no_return(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    bins: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let hist_f32 = DeviceBuffer::<f32>::zeroed(stream, bins as usize)?;
    let hist_f16 = DeviceBuffer::<f16>::zeroed(stream, bins as usize)?;
    let cfg = LaunchConfig::for_num_elems(N);

    let f32_ms = time_gpu_iters(stream, ITERS, || {
        reset(&hist_f32, stream)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.f32_add(stream, cfg, &hist_f32, N, bins) }?;
        Ok(())
    })?;
    let f16_ms = time_gpu_iters(stream, ITERS, || {
        reset(&hist_f16, stream)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.f16_add(stream, cfg, &hist_f16, N, bins) }?;
        Ok(())
    })?;

    print_row("unused", "f32", bins, f32_ms);
    print_row("unused", "f16", bins, f16_ms);
    Ok(())
}

fn bench_return(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    bins: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let hist_f32 = DeviceBuffer::<f32>::zeroed(stream, bins as usize)?;
    let hist_f16 = DeviceBuffer::<f16>::zeroed(stream, bins as usize)?;
    let mut old_f32 = DeviceBuffer::<f32>::zeroed(stream, N as usize)?;
    let mut old_f16 = DeviceBuffer::<f16>::zeroed(stream, N as usize)?;
    let cfg = LaunchConfig::for_num_elems(N);

    let f32_ms = time_gpu_iters(stream, ITERS, || {
        reset(&hist_f32, stream)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.f32_add_return(stream, cfg, &hist_f32, &mut old_f32, N, bins) }?;
        Ok(())
    })?;
    let f16_ms = time_gpu_iters(stream, ITERS, || {
        reset(&hist_f16, stream)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.f16_add_return(stream, cfg, &hist_f16, &mut old_f16, N, bins) }?;
        Ok(())
    })?;

    print_row("return", "f32", bins, f32_ms);
    print_row("return", "f16", bins, f16_ms);
    Ok(())
}

fn time_gpu_iters<F>(
    stream: &Arc<CudaStream>,
    iters: usize,
    mut f: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..WARMUP {
        f()?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        f()?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
}

fn reset<T>(
    buffer: &DeviceBuffer<T>,
    stream: &CudaStream,
) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        cuda_core::simt::memory::memset_d8_async(
            buffer.cu_deviceptr(),
            0,
            buffer.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

fn print_row(mode: &str, ty: &str, bins: u32, avg_ms: f64) {
    let mops = N as f64 / (avg_ms * 1000.0);
    println!("{N},{mode},{ty},{bins},{avg_ms:.6},{mops:.3}");
}

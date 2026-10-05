/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Half-precision atomic histogram checks.
//!
//! Run: cargo oxide run atomic_f16

#![feature(f16)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicF16, DeviceAtomicF16, SystemAtomicF16};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn device_hist_f16(hist: &[f16], mut old: DisjointSlice<f16>, n: u32, nbins: u32) {
        let gid = thread::index_1d();
        let i = gid.get();
        if i >= n as usize {
            return;
        }

        let bin = i % nbins as usize;
        let cell = unsafe { &*(hist.as_ptr().add(bin) as *const DeviceAtomicF16) };
        let prev = cell.fetch_add(1.0f16, AtomicOrdering::Relaxed);
        if let Some(out_elem) = old.get_mut(gid) {
            *out_elem = prev;
        }
    }

    #[kernel]
    pub fn block_hist_f16(hist: &[f16], mut old: DisjointSlice<f16>) {
        let gid = thread::index_1d();
        let block = thread::blockIdx_x() as usize;
        let cell = unsafe { &*(hist.as_ptr().add(block) as *const BlockAtomicF16) };
        let prev = cell.fetch_add(1.0f16, AtomicOrdering::Relaxed);
        if let Some(out_elem) = old.get_mut(gid) {
            *out_elem = prev;
        }
    }

    #[kernel]
    pub fn device_sub_f16(counter: &[f16], mut old: DisjointSlice<f16>) {
        let gid = thread::index_1d();
        let cell = unsafe { &*(counter.as_ptr() as *const DeviceAtomicF16) };
        let prev = cell.fetch_sub(1.0f16, AtomicOrdering::Relaxed);
        if let Some(out_elem) = old.get_mut(gid) {
            *out_elem = prev;
        }
    }

    #[kernel]
    pub fn system_add_f16(counter: &[f16], mut old: DisjointSlice<f16>) {
        let gid = thread::index_1d();
        let cell = unsafe { &*(counter.as_ptr() as *const SystemAtomicF16) };
        let prev = cell.fetch_add(1.0f16, AtomicOrdering::Relaxed);
        if let Some(out_elem) = old.get_mut(gid) {
            *out_elem = prev;
        }
    }

    #[kernel]
    pub fn device_swap_f16(cell: &[f16], vals: &[f16], mut old: DisjointSlice<f16>, n: u32) {
        let gid = thread::index_1d();
        let i = gid.get();
        if i >= n as usize {
            return;
        }
        let cell = unsafe { &*(cell.as_ptr() as *const DeviceAtomicF16) };
        let prev = cell.swap(vals[i], AtomicOrdering::Relaxed);
        if let Some(out_elem) = old.get_mut(gid) {
            *out_elem = prev;
        }
    }

    #[kernel]
    pub fn device_store_load_f16(cells: &[f16], mut out: DisjointSlice<f16>, n: u32) {
        let gid = thread::index_1d();
        let i = gid.get();
        if i >= n as usize {
            return;
        }
        let cell = unsafe { &*(cells.as_ptr().add(i) as *const DeviceAtomicF16) };
        let seen = cell.load(AtomicOrdering::Acquire);
        cell.store(seen + 1.0f16, AtomicOrdering::Release);
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = seen;
        }
    }
}

fn main() {
    println!("=== DeviceAtomicF16 / BlockAtomicF16 tests ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    run_device_hist(&module, &stream, 31, 16);
    run_device_hist(&module, &stream, 4096, 16);
    run_device_sub(&module, &stream, 256);
    run_system_add(&module, &stream, 256);
    run_device_swap(&module, &stream, 256);
    run_device_store_load(&module, &stream, 256);
    run_block_hist(&module, &stream, 4, 64);

    println!("\nSUCCESS: all f16 atomic checks passed");
}

fn run_device_hist(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    n: u32,
    nbins: u32,
) {
    println!("--- DeviceAtomicF16 histogram: n={n}, bins={nbins} ---");

    let hist = DeviceBuffer::<f16>::zeroed(stream, nbins as usize).unwrap();
    let mut old = DeviceBuffer::<f16>::zeroed(stream, n as usize).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.device_hist_f16(
            stream,
            LaunchConfig::for_num_elems(n),
            &hist,
            &mut old,
            n,
            nbins,
        )
    }
    .expect("Kernel launch failed");

    stream.synchronize().unwrap();

    let counts: Vec<u16> = hist
        .to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let old_values = old.to_host_vec(stream).unwrap();

    let expected_counts: Vec<u16> = (0..nbins)
        .map(|bin| expected_bin_count(n, nbins, bin).to_bits())
        .collect();
    check_slice("final bin counts", &counts, &expected_counts);
    check_device_old_values(&old_values, n, nbins);
}

fn run_device_sub(module: &kernels::LoadedModule, stream: &cuda_core::CudaStream, n: u32) {
    println!("--- DeviceAtomicF16 fetch_sub: n={n} ---");

    let initial = vec![n as f16];
    let counter = DeviceBuffer::from_host(stream, &initial).unwrap();
    let mut old = DeviceBuffer::<f16>::zeroed(stream, n as usize).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.device_sub_f16(stream, LaunchConfig::for_num_elems(n), &counter, &mut old) }
        .expect("Kernel launch failed");

    stream.synchronize().unwrap();

    let final_count = counter.to_host_vec(stream).unwrap()[0].to_bits();
    check_slice("fetch_sub final count", &[final_count], &[0.0f16.to_bits()]);
    check_old_range(
        "fetch_sub old values",
        &old.to_host_vec(stream).unwrap(),
        1,
        n,
    );
}

fn run_system_add(module: &kernels::LoadedModule, stream: &cuda_core::CudaStream, n: u32) {
    println!("--- SystemAtomicF16 fetch_add: n={n} ---");

    let counter = DeviceBuffer::<f16>::zeroed(stream, 1).unwrap();
    let mut old = DeviceBuffer::<f16>::zeroed(stream, n as usize).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.system_add_f16(stream, LaunchConfig::for_num_elems(n), &counter, &mut old) }
        .expect("Kernel launch failed");

    stream.synchronize().unwrap();

    let final_count = counter.to_host_vec(stream).unwrap()[0].to_bits();
    check_slice(
        "system fetch_add final count",
        &[final_count],
        &[(n as f16).to_bits()],
    );
    check_old_sequence(
        "system fetch_add old values",
        &old.to_host_vec(stream).unwrap(),
        n,
    );
}

fn run_device_swap(module: &kernels::LoadedModule, stream: &cuda_core::CudaStream, n: u32) {
    println!("--- DeviceAtomicF16 swap: n={n} ---");

    let cell = DeviceBuffer::<f16>::zeroed(stream, 1).unwrap();
    let host_vals: Vec<f16> = (1..=n).map(|v| v as f16).collect();
    let vals = DeviceBuffer::from_host(stream, &host_vals).unwrap();
    let mut old = DeviceBuffer::<f16>::zeroed(stream, n as usize).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.device_swap_f16(
            stream,
            LaunchConfig::for_num_elems(n),
            &cell,
            &vals,
            &mut old,
            n,
        )
    }
    .expect("Kernel launch failed");

    stream.synchronize().unwrap();

    // Each swap returns whatever the previous swap (or the initial zero) left
    // in the cell, so {old values} + {final} must equal {initial} + {vals}.
    let mut got: Vec<u16> = old
        .to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    got.push(cell.to_host_vec(stream).unwrap()[0].to_bits());
    got.sort();

    let mut expected: Vec<u16> = std::iter::once(0.0f16)
        .chain(host_vals.iter().copied())
        .map(|v| v.to_bits())
        .collect();
    expected.sort();

    check_slice("swap old values + final", &got, &expected);
}

fn run_device_store_load(module: &kernels::LoadedModule, stream: &cuda_core::CudaStream, n: u32) {
    println!("--- DeviceAtomicF16 load/store: n={n} ---");

    let host_init: Vec<f16> = (0..n).map(|v| v as f16).collect();
    let cells = DeviceBuffer::from_host(stream, &host_init).unwrap();
    let mut out = DeviceBuffer::<f16>::zeroed(stream, n as usize).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.device_store_load_f16(stream, LaunchConfig::for_num_elems(n), &cells, &mut out, n)
    }
    .expect("Kernel launch failed");

    stream.synchronize().unwrap();

    let loaded: Vec<u16> = out
        .to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected_loaded: Vec<u16> = host_init.iter().map(|v| v.to_bits()).collect();
    check_slice("atomic load values", &loaded, &expected_loaded);

    let stored: Vec<u16> = cells
        .to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected_stored: Vec<u16> = host_init.iter().map(|v| (*v + 1.0f16).to_bits()).collect();
    check_slice("atomic store values", &stored, &expected_stored);
}

fn run_block_hist(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    blocks: u32,
    threads_per_block: u32,
) {
    println!(
        "--- BlockAtomicF16 histogram: blocks={blocks}, threads/block={threads_per_block} ---"
    );

    let hist = DeviceBuffer::<f16>::zeroed(stream, blocks as usize).unwrap();
    let mut old =
        DeviceBuffer::<f16>::zeroed(stream, (blocks * threads_per_block) as usize).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.block_hist_f16(stream, cfg, &hist, &mut old) }.expect("Kernel launch failed");

    stream.synchronize().unwrap();

    let counts: Vec<u16> = hist
        .to_host_vec(stream)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let old_values = old.to_host_vec(stream).unwrap();
    let expected_counts = vec![(threads_per_block as f16).to_bits(); blocks as usize];

    check_slice("block final counts", &counts, &expected_counts);
    for block in 0..blocks as usize {
        let start = block * threads_per_block as usize;
        let end = start + threads_per_block as usize;
        check_old_sequence(
            "block old values",
            &old_values[start..end],
            threads_per_block,
        );
    }
}

fn expected_bin_count(n: u32, nbins: u32, bin: u32) -> f16 {
    let base = n / nbins;
    let remainder = n % nbins;
    (base + u32::from(bin < remainder)) as f16
}

fn check_device_old_values(old_values: &[f16], n: u32, nbins: u32) {
    for bin in 0..nbins {
        let got: Vec<f16> = old_values
            .iter()
            .enumerate()
            .filter(|(i, _)| *i % nbins as usize == bin as usize)
            .map(|(_, &value)| value)
            .collect();
        check_old_sequence(
            "device old values",
            &got,
            expected_bin_count(n, nbins, bin) as u32,
        );
    }
}

fn check_old_sequence(name: &str, got: &[f16], len: u32) {
    check_old_range(name, got, 0, len - 1);
}

fn check_old_range(name: &str, got: &[f16], start: u32, end: u32) {
    let mut got_bits: Vec<u16> = got.iter().map(|value| value.to_bits()).collect();
    got_bits.sort();

    let expected: Vec<u16> = (start..=end)
        .map(|value| (value as f16).to_bits())
        .collect();
    check_slice(name, &got_bits, &expected);
}

fn check_slice<T: Eq + std::fmt::Debug>(name: &str, got: &[T], expected: &[T]) {
    if got == expected {
        println!("  PASS {name}: {got:?}");
    } else {
        eprintln!("  FAIL {name}: got {got:?}, expected {expected:?}");
        std::process::exit(1);
    }
}

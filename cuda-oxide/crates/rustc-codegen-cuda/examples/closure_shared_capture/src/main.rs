/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-`move` closures that capture a shared-memory pointer by reference.
//!
//! Each kernel keeps a CTA-shared pointer in a local, builds a closure that
//! captures that local by reference, and calls it to fill two stages of shared
//! memory. The loop keeps rustc's MIR inliner from inlining the closure, so the
//! kernel really borrows the pointer local (`&s`) and hands the borrow to a
//! separately compiled closure body.
//!
//! 1. **dynamic_smem_closure** - `DynamicSharedArray::get()` and plain stores
//! 2. **static_smem_closure** - `addr_of_mut!` of a `static mut SharedArray`
//! 3. **cp_async_closure** - `cp_async_cg_zfill_16` into dynamic shared memory
//!
//! Build and run with:
//!   cargo oxide run closure_shared_capture

use core::ptr::addr_of_mut;
use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::async_copy::{cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_all};
use cuda_device::{DisjointSlice, DynamicSharedArray, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const BLOCK: usize = 256;
const STAGES: usize = 2;
/// `u32` words moved by one 16-byte `cp.async`
const WORDS: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread `t` stores `input[i * BLOCK + t]` into stage `i`, then reads its
    /// neighbour's slots back.
    #[kernel]
    pub fn dynamic_smem_closure(input: &[u16], mut out: DisjointSlice<u16>) {
        let s = DynamicSharedArray::<u16>::get();
        let tid = thread::threadIdx_x() as usize;

        let fill = |stage: usize| {
            for i in 0..STAGES {
                let slot = (stage + i) * BLOCK + tid;
                unsafe { *s.add(slot) = input[slot] };
            }
        };
        fill(0);
        thread::sync_threads();

        let neighbor = (tid + 1) % BLOCK;
        let sum = unsafe { *s.add(neighbor) + *s.add(BLOCK + neighbor) };
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = sum;
        }
    }

    /// Same as `dynamic_smem_closure`, with the pointer taken from a static
    /// `SharedArray`.
    #[kernel]
    pub fn static_smem_closure(input: &[u16], mut out: DisjointSlice<u16>) {
        static mut SMEM: SharedArray<u16, { STAGES * BLOCK }> = SharedArray::UNINIT;

        let s = addr_of_mut!(SMEM) as *mut u16;
        let tid = thread::threadIdx_x() as usize;

        let fill = |stage: usize| {
            for i in 0..STAGES {
                let slot = (stage + i) * BLOCK + tid;
                unsafe { *s.add(slot) = input[slot] };
            }
        };
        fill(0);
        thread::sync_threads();

        let neighbor = (tid + 1) % BLOCK;
        let sum = unsafe { *s.add(neighbor) + *s.add(BLOCK + neighbor) };
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = sum;
        }
    }

    /// Thread `t` copies 16 bytes per stage with `cp.async`, then reads the
    /// first word of its neighbour's chunk in each stage.
    #[kernel]
    pub fn cp_async_closure(input: &[u32], mut out: DisjointSlice<u32>) {
        let s = DynamicSharedArray::<u32>::get();
        let src = input.as_ptr();
        let tid = thread::threadIdx_x() as usize;

        let load = |stage: usize| {
            for i in 0..STAGES {
                let word = ((stage + i) * BLOCK + tid) * WORDS;
                unsafe { cp_async_cg_zfill_16(s.add(word), src.add(word) as *const u8, 16) };
            }
        };
        load(0);
        unsafe {
            cp_async_commit_group();
            cp_async_wait_all();
        }
        thread::sync_threads();

        let neighbor = (tid + 1) % BLOCK;
        let sum = unsafe { *s.add(neighbor * WORDS) + *s.add((BLOCK + neighbor) * WORDS) };
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = sum;
        }
    }
}

fn check<T: PartialEq + core::fmt::Debug>(name: &str, got: &[T], expected: impl Fn(usize) -> T) {
    for (t, value) in got.iter().enumerate() {
        let want = expected(t);
        if *value != want {
            eprintln!("{name}: mismatch at {t}: expected {want:?}, got {value:?}");
            std::process::exit(1);
        }
    }
    println!("✓ {name}");
}

fn main() {
    println!("=== Closures capturing shared-memory pointers by reference ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 8 {
        println!("skipping: cp.async requires sm_80+ (device is sm_{major}{minor})");
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let neighbor = |t: usize| (t + 1) % BLOCK;

    let input: Vec<u16> = (0..STAGES * BLOCK).map(|i| i as u16).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let expected_u16 = |t: usize| input[neighbor(t)] + input[BLOCK + neighbor(t)];

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: (STAGES * BLOCK * size_of::<u16>()) as u32,
    };
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, BLOCK).unwrap();
    // SAFETY: one block of BLOCK threads; dynamic shared memory covers both stages
    unsafe { module.dynamic_smem_closure(stream.as_ref(), cfg, &input_dev, &mut out_dev) }
        .expect("Kernel launch failed");
    check(
        "dynamic_smem_closure",
        &out_dev.to_host_vec(&stream).unwrap(),
        expected_u16,
    );

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, BLOCK).unwrap();
    // SAFETY: one block of BLOCK threads; the static array covers both stages
    unsafe { module.static_smem_closure(stream.as_ref(), cfg, &input_dev, &mut out_dev) }
        .expect("Kernel launch failed");
    check(
        "static_smem_closure",
        &out_dev.to_host_vec(&stream).unwrap(),
        expected_u16,
    );

    let words: Vec<u32> = (0..STAGES * BLOCK * WORDS).map(|i| i as u32).collect();
    let words_dev = DeviceBuffer::from_host(&stream, &words).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: (STAGES * BLOCK * WORDS * size_of::<u32>()) as u32,
    };
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, BLOCK).unwrap();
    // SAFETY: one block of BLOCK threads; dynamic shared memory covers both
    // stages and every 16-byte chunk is 16-byte aligned on both sides
    unsafe { module.cp_async_closure(stream.as_ref(), cfg, &words_dev, &mut out_dev) }
        .expect("Kernel launch failed");
    check(
        "cp_async_closure",
        &out_dev.to_host_vec(&stream).unwrap(),
        |t| words[neighbor(t) * WORDS] + words[(BLOCK + neighbor(t)) * WORDS],
    );

    println!("\nSUCCESS: all closure captures read back correctly");
}

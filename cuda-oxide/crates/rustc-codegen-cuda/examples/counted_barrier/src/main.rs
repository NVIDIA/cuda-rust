/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Counted CTA barrier example.
//!
//! Verifies the counted numbered-barrier intrinsics (`barrier.sync id,
//! count` and `barrier.arrive id, count`) with producer/consumer
//! handshakes between a *subset* of the CTA:
//!
//! - `counted_barrier_subset`: 128 threads (4 warps). Warp 0 produces
//!   values in shared memory, then syncs on numbered barrier 1 with a
//!   thread count of 64. Warp 1 syncs on the same barrier and consumes
//!   the values. Warps 2 and 3 never touch barrier 1, proving part of
//!   the block can synchronize on a counted barrier while the rest of
//!   the block runs free.
//! - `counted_barrier_arrive`: split arrive/sync. Warp 0 signals arrival
//!   at barrier 2 without waiting and keeps going; warp 1 syncs on
//!   barrier 2, which completes once warp 0's arrival plus warp 1's own
//!   32 threads account for all 64 expected threads.
//!
//! Both kernels verify the data actually crossed the barrier (values,
//! not just completion) and that the bystander warps were unaffected.
//!
//! Build and run with:
//!   cargo oxide run counted_barrier

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::barrier::{barrier_cta_arrive, barrier_cta_sync};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const SENTINEL: u32 = 0xFACE;

#[cuda_module]
mod kernels {
    use super::*;

    /// Producer/consumer handshake between warp 0 and warp 1 on counted
    /// barrier 1 (64 participating threads out of a 128-thread CTA).
    #[kernel]
    pub fn counted_barrier_subset(mut out: DisjointSlice<u32>) {
        static mut DATA: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        if tid < 32 {
            // Warp 0: producer.
            unsafe { DATA[tid as usize] = tid * 3 + 7 };
            // SAFETY: exactly 64 threads (warps 0 and 1) sync on barrier 1.
            unsafe { barrier_cta_sync(1, 64) };
            if let Some(slot) = out.get_mut(gid) {
                *slot = 1;
            }
        } else if tid < 64 {
            // Warp 1: consumer. The counted barrier orders it after the
            // producer writes.
            // SAFETY: exactly 64 threads (warps 0 and 1) sync on barrier 1.
            unsafe { barrier_cta_sync(1, 64) };
            let value = unsafe { DATA[(tid - 32) as usize] };
            if let Some(slot) = out.get_mut(gid) {
                *slot = value;
            }
        } else {
            // Warps 2 and 3: bystanders that never arrive at barrier 1.
            if let Some(slot) = out.get_mut(gid) {
                *slot = SENTINEL;
            }
        }
    }

    /// Split arrive/sync: warp 0 arrives (no wait) at barrier 2 and keeps
    /// going; warp 1 syncs on barrier 2 which completes once warp 0's
    /// arrival plus warp 1's own 32 threads account for all 64 expected.
    #[kernel]
    pub fn counted_barrier_arrive(mut out: DisjointSlice<u32>) {
        static mut FLAG: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        if tid < 32 {
            unsafe { FLAG[tid as usize] = 0x5EED + tid };
            // SAFETY: warp 0 signals arrival of its 32 threads at barrier 2
            // (64 expected in total) without blocking.
            unsafe { barrier_cta_arrive(2, 64) };
            if let Some(slot) = out.get_mut(gid) {
                *slot = 1;
            }
        } else if tid < 64 {
            // SAFETY: warp 1's 32 threads complete the 64-thread quota.
            unsafe { barrier_cta_sync(2, 64) };
            let value = unsafe { FLAG[(tid - 32) as usize] };
            if let Some(slot) = out.get_mut(gid) {
                *slot = value;
            }
        } else if let Some(slot) = out.get_mut(gid) {
            *slot = SENTINEL;
        }
    }
}

fn main() {
    println!("=== counted CTA barrier example ===");
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const N: usize = 128;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failures = 0;

    // counted_barrier_subset
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel contract (128 threads).
    unsafe { module.counted_barrier_subset(&stream, cfg, &mut out_dev) }
        .expect("counted_barrier_subset launch failed");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for i in 0..32 {
        ok &= out[i] == 1;
        ok &= out[32 + i] == (i as u32 * 3 + 7);
        ok &= out[64 + i] == SENTINEL && out[96 + i] == SENTINEL;
    }
    println!(
        "counted_barrier_subset: {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    if !ok {
        println!("  first 8 consumer slots: {:?}", &out[32..40]);
        failures += 1;
    }

    // counted_barrier_arrive
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel contract (128 threads).
    unsafe { module.counted_barrier_arrive(&stream, cfg, &mut out_dev) }
        .expect("counted_barrier_arrive launch failed");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for i in 0..32 {
        ok &= out[i] == 1;
        ok &= out[32 + i] == 0x5EED + i as u32;
        ok &= out[64 + i] == SENTINEL && out[96 + i] == SENTINEL;
    }
    println!(
        "counted_barrier_arrive: {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    if !ok {
        println!("  first 8 consumer slots: {:?}", &out[32..40]);
        failures += 1;
    }

    if failures == 0 {
        println!("SUCCESS");
    } else {
        println!("FAILED: {failures} kernels mismatched");
        std::process::exit(1);
    }
}

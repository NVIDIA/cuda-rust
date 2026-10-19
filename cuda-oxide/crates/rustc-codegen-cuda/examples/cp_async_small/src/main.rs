/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end test for `cp.async` 4-byte and 8-byte copy intrinsics.
//!
//! Demonstrates asynchronous global-to-shared memory copies using
//! `cp_async_ca_4` (4 bytes) and `cp_async_ca_8` (8 bytes), including their
//! generated commit and wait operations.
//!
//! Requires **sm_80+** (Ampere or later).
//!
//! Build and run with:
//!   cargo oxide run cp_async_small

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::async_copy::{
    cp_async_ca_4, cp_async_ca_8, cp_async_ca_16, cp_async_ca_zfill_4, cp_async_ca_zfill_8,
    cp_async_ca_zfill_16, cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group,
    cp_async_mbarrier_arrive, cp_async_mbarrier_arrive_noinc,
    cp_async_mbarrier_arrive_noinc_shared, cp_async_mbarrier_arrive_shared, cp_async_wait_all,
    cp_async_wait_group,
};
use cuda_device::barrier::{
    Barrier, mbarrier_arrive, mbarrier_init, mbarrier_inval, mbarrier_wait,
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread copies one `u32` from global to shared memory via
    /// `cp.async.ca.shared.global [...], [...], 4`, then writes it out.
    #[kernel]
    pub fn test_cp_async_4(input: &[u32], mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d();

        // Obtain shared-memory and global-memory pointers.
        let dst_ptr = unsafe { (core::ptr::addr_of_mut!(SMEM) as *mut u32).add(tid) };
        let src_ptr = unsafe { input.as_ptr().add(gid.get()) };

        // Initiate the 4-byte async copy, commit, and wait.
        unsafe {
            cp_async_ca_4(dst_ptr, src_ptr);
            cp_async_commit_group();
            cp_async_wait_all();
        }

        // Ensure every thread's copy has landed before reading.
        thread::sync_threads();

        // Read back from shared memory and write to global output.
        let val = unsafe { SMEM[tid] };
        if let Some(slot) = out.get_mut(gid) {
            *slot = val;
        }
    }

    /// Each thread copies 8 bytes (two consecutive `u32`s) from global to
    /// shared memory via `cp.async.ca.shared.global [...], [...], 8`, then
    /// writes both values out.
    #[kernel]
    pub fn test_cp_async_8(input: &[u32], mut out: DisjointSlice<u32>) {
        // 64 elements: thread i owns elements [2*i] and [2*i+1].
        static mut SMEM: SharedArray<u32, 64, 8> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        let smem_idx = tid * 2;
        let gmem_idx = gid * 2;

        let dst_ptr = unsafe { (core::ptr::addr_of_mut!(SMEM) as *mut u32).add(smem_idx) };
        let src_ptr = unsafe { input.as_ptr().add(gmem_idx) };

        // Initiate the 8-byte async copy, commit, and wait.
        unsafe {
            cp_async_ca_8(dst_ptr, src_ptr);
            cp_async_commit_group();
            cp_async_wait_all();
        }

        thread::sync_threads();

        // Write both copied values to the output buffer.
        // Each thread writes to two unique slots [2*gid] and [2*gid+1].
        unsafe {
            let lo = SMEM[smem_idx];
            let hi = SMEM[smem_idx + 1];
            let base = gid * 2;
            *out.get_unchecked_mut(base) = lo;
            *out.get_unchecked_mut(base + 1) = hi;
        }
    }

    /// Exercises every classic `cp.async` form without changing host output.
    #[kernel]
    pub fn compile_all_classic_cp_async(input: &[u32]) {
        static mut SMEM: SharedArray<u32, 24, 16> = SharedArray::UNINIT;

        if thread::threadIdx_x() == 0 {
            let dst = core::ptr::addr_of_mut!(SMEM) as *mut u32;
            let src = input.as_ptr();

            unsafe {
                cp_async_ca_4(dst, src);
                cp_async_ca_8(dst.add(2), src.add(2));
                cp_async_ca_16(dst.add(4), src.add(4));
                cp_async_ca_zfill_4(dst.add(8), src.add(8).cast(), 3);
                cp_async_ca_zfill_8(dst.add(10), src.add(10).cast(), 3);
                cp_async_ca_zfill_16(dst.add(12), src.add(12).cast(), 3);
                cp_async_cg_16(dst.add(16), src.add(16));
                cp_async_cg_zfill_16(dst.add(20), src.add(20).cast(), 3);
                cp_async_commit_group();
                cp_async_wait_group(0);
                cp_async_wait_all();
            }
        }
    }

    /// Checks all classic copy-to-mbarrier bridge forms without printing.
    #[kernel]
    pub fn test_cp_async_mbarrier_bridges(input: &[u32], mut out: DisjointSlice<u32>) {
        static mut BAR: Barrier = Barrier::UNINIT;
        static mut SMEM: SharedArray<u32, 4> = SharedArray::UNINIT;

        let barrier = core::ptr::addr_of_mut!(BAR);
        let shared = core::ptr::addr_of_mut!(SMEM) as *mut u32;
        let source = input.as_ptr();

        unsafe {
            SMEM[0] = 0;
            mbarrier_init(barrier, 1);
            cp_async_ca_4(shared, source);
            cp_async_mbarrier_arrive(barrier);
            let token = mbarrier_arrive(barrier);
            mbarrier_wait(barrier, token);
            *out.get_unchecked_mut(0) = SMEM[0];
            mbarrier_inval(barrier);

            SMEM[1] = 0;
            mbarrier_init(barrier, 1);
            cp_async_ca_4(shared.add(1), source.add(1));
            cp_async_mbarrier_arrive_shared(barrier);
            let token = mbarrier_arrive(barrier);
            mbarrier_wait(barrier, token);
            *out.get_unchecked_mut(1) = SMEM[1];
            mbarrier_inval(barrier);

            SMEM[2] = 0;
            mbarrier_init(barrier, 2);
            cp_async_ca_4(shared.add(2), source.add(2));
            cp_async_mbarrier_arrive_noinc(barrier);
            let token = mbarrier_arrive(barrier);
            mbarrier_wait(barrier, token);
            *out.get_unchecked_mut(2) = SMEM[2];
            mbarrier_inval(barrier);

            SMEM[3] = 0;
            mbarrier_init(barrier, 2);
            cp_async_ca_4(shared.add(3), source.add(3));
            cp_async_mbarrier_arrive_noinc_shared(barrier);
            let token = mbarrier_arrive(barrier);
            mbarrier_wait(barrier, token);
            *out.get_unchecked_mut(3) = SMEM[3];
            mbarrier_inval(barrier);
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== cp.async small-copy example (4-byte and 8-byte) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // cp.async requires sm_80 (Ampere) or later.
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 8 {
        println!(
            "skipping: cp.async requires sm_80+ (device is sm_{}{})",
            major, minor
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let cfg32 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let coverage_input_dev = DeviceBuffer::from_host(&stream, &[0u32; 24]).unwrap();
    // SAFETY: thread zero owns every copy and all source and destination ranges are valid.
    unsafe { module.compile_all_classic_cp_async(&stream, cfg32, &coverage_input_dev) }
        .expect("classic cp.async coverage launch failed");

    let barrier_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let barrier_input = [11u32, 22, 33, 44];
    let barrier_input_dev = DeviceBuffer::from_host(&stream, &barrier_input).unwrap();
    let mut barrier_output_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: one thread owns the barrier and all four source and destination slots.
    unsafe {
        module.test_cp_async_mbarrier_bridges(
            &stream,
            barrier_cfg,
            &barrier_input_dev,
            &mut barrier_output_dev,
        )
    }
    .expect("cp.async mbarrier coverage launch failed");
    assert_eq!(
        barrier_output_dev.to_host_vec(&stream).unwrap(),
        barrier_input
    );

    // ===== Test 1: cp.async.ca 4-byte copy =====
    println!("--- Test 1: cp.async.ca 4-byte copy ---");
    {
        let input: Vec<u32> = (100..132).collect(); // 32 distinct values
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cp_async_4(&stream, cfg32, &input_dev, &mut out_dev) }
            .expect("test_cp_async_4 launch failed");

        let out = out_dev.to_host_vec(&stream).unwrap();
        for i in 0..32 {
            if out[i] != input[i] {
                eprintln!(
                    "FAIL at [{}]: expected {}, got {} (4-byte copy)",
                    i, input[i], out[i]
                );
                std::process::exit(1);
            }
        }
        println!("  PASS: 32 elements copied correctly via cp.async 4-byte");
    }

    // ===== Test 2: cp.async.ca 8-byte copy =====
    println!("--- Test 2: cp.async.ca 8-byte copy ---");
    {
        let input: Vec<u32> = (200..264).collect(); // 64 values (32 threads x 2)
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 64).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cp_async_8(&stream, cfg32, &input_dev, &mut out_dev) }
            .expect("test_cp_async_8 launch failed");

        let out = out_dev.to_host_vec(&stream).unwrap();
        for i in 0..64 {
            if out[i] != input[i] {
                eprintln!(
                    "FAIL at [{}]: expected {}, got {} (8-byte copy)",
                    i, input[i], out[i]
                );
                std::process::exit(1);
            }
        }
        println!("  PASS: 64 elements copied correctly via cp.async 8-byte");
    }

    println!("\nPASS: cp.async.ca 4-byte and 8-byte copies correct");
}

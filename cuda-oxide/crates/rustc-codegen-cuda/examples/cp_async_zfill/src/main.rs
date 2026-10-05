/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end test for `cp.async` zero-fill intrinsics.
//!
//! Demonstrates asynchronous global-to-shared memory copies with
//! hardware zero-fill using the `src_size` parameter. When
//! `src_size < cp_size`, the remaining bytes are zero-filled.
//!
//! Tests the 4-byte variant by copying 2 of 4 bytes and verifying the
//! upper 2 bytes are zeroed.
//!
//! Requires **sm_80+** (Ampere or later).
//!
//! Build and run with:
//!   cargo oxide run cp_async_zfill

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::async_copy::{cp_async_ca_zfill_4, cp_async_commit_group, cp_async_wait_all};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread copies `src_size` bytes (out of 4) from global to shared
    /// memory via `cp.async.ca.shared.global [...], [...], 4, src_size;`.
    /// The remaining `4 - src_size` bytes are zero-filled by hardware.
    #[kernel]
    pub fn test_zfill_4(input: &[u32], src_size: u32, mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d();
        let dst_ptr = unsafe { (core::ptr::addr_of_mut!(SMEM) as *mut u32).add(tid) };

        // Pre-fill shared memory with a sentinel so we can detect zero-fill.
        unsafe {
            dst_ptr.write(0xDEAD_BEEF);
        }
        thread::sync_threads();

        let src_ptr = unsafe { input.as_ptr().add(gid.get()) as *const u8 };

        // Initiate the zero-fill copy, commit, and wait.
        unsafe {
            cp_async_ca_zfill_4(dst_ptr, src_ptr, src_size);
            cp_async_commit_group();
            cp_async_wait_all();
        }

        thread::sync_threads();

        let val = unsafe { dst_ptr.read() };
        if let Some(slot) = out.get_mut(gid) {
            *slot = val;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== cp.async zero-fill example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

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

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut all_pass = true;

    // ----- Test 1: full copy (src_size == 4), no zero-fill -----
    println!("--- Test 1: src_size=4 (full copy, no zero-fill) ---");
    {
        let input: Vec<u32> = (0xAAAA_0000..0xAAAA_0020).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_zfill_4(&stream, cfg, &input_dev, 4u32, &mut out_dev) }
            .expect("launch failed");

        let mut test_pass = true;
        let out = out_dev.to_host_vec(&stream).unwrap();
        for i in 0..32 {
            if out[i] != input[i] {
                eprintln!(
                    "  FAIL [{}]: expected 0x{:08X}, got 0x{:08X}",
                    i, input[i], out[i]
                );
                test_pass = false;
                all_pass = false;
            }
        }
        if test_pass {
            println!("  PASS: 32 elements copied with src_size=4");
        }
    }

    // ----- Test 2: partial copy (src_size == 2), upper 2 bytes zeroed -----
    println!("--- Test 2: src_size=2 (partial copy, upper bytes zeroed) ---");
    {
        // Input: each element is 0xAABBCCDD.
        // With src_size=2 on little-endian, bytes 0-1 (0xDD, 0xCC) are copied,
        // bytes 2-3 are zero-filled, giving 0x0000CCDD.
        let input: Vec<u32> = vec![0xAABBCCDD; 32];
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_zfill_4(&stream, cfg, &input_dev, 2u32, &mut out_dev) }
            .expect("launch failed");

        let mut test_pass = true;
        let out = out_dev.to_host_vec(&stream).unwrap();
        let expected = 0x0000_CCDD_u32;
        for (i, &actual) in out.iter().enumerate() {
            if actual != expected {
                eprintln!(
                    "  FAIL [{}]: expected 0x{:08X}, got 0x{:08X}",
                    i, expected, actual
                );
                test_pass = false;
                all_pass = false;
            }
        }
        if test_pass {
            println!("  PASS: upper 2 bytes zeroed with src_size=2");
        }
    }

    // ----- Test 3: zero copy (src_size == 0), entire word zeroed -----
    println!("--- Test 3: src_size=0 (zero copy, entire word zeroed) ---");
    {
        let input: Vec<u32> = vec![0xFFFF_FFFF; 32];
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_zfill_4(&stream, cfg, &input_dev, 0u32, &mut out_dev) }
            .expect("launch failed");

        let mut test_pass = true;
        let out = out_dev.to_host_vec(&stream).unwrap();
        for (i, &actual) in out.iter().enumerate() {
            if actual != 0 {
                eprintln!("  FAIL [{i}]: expected 0x00000000, got 0x{actual:08X}");
                test_pass = false;
                all_pass = false;
            }
        }
        if test_pass {
            println!("  PASS: entire word zeroed with src_size=0");
        }
    }

    if !all_pass {
        println!("\nFAIL: cp.async zero-fill, one or more checks failed");
        std::process::exit(1);
    }
    println!(
        "\nPASS: cp.async zero-fill, all 3 tests verified on sm_{}{}",
        major, minor
    );
}

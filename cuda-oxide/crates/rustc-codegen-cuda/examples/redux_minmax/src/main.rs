/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-instruction warp integer reductions (sm_80+).
//!
//! Companion to `redux_sum` (which covers `redux.sync.add`). This example
//! exercises the rest of the integer family — min/max (signed *and* unsigned)
//! and the bitwise and/or/xor — each lowered to one hardware `redux.sync.*`
//! instruction instead of a `shfl`-based log-tree.
//!
//! The min/max kernel is built so the signed and unsigned answers *differ*,
//! which is exactly why the two variants exist:
//!   - `min.s32(0xFFFFFFFF, 0)` = -1   but   `min.u32(0xFFFFFFFF, 0)` = 0
//!
//! Build and run with:
//!   cargo oxide run redux_minmax

use cuda_device::{DisjointSlice, kernel, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

/// Device helpers that return a reduction directly (#811): the result reaches
/// the helper's `return` without passing through a store, the shape a generic
/// fast path such as `warp_reduce`'s takes. Kept out of line so each helper
/// lowers as its own device function.
#[inline(never)]
fn min_i32_via_helper(value: i32) -> i32 {
    warp::redux_sync_min_i32(FULL_MASK, value)
}

#[inline(never)]
fn max_u32_via_helper(value: u32) -> u32 {
    warp::redux_sync_max_u32(FULL_MASK, value)
}

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Lane `l` contributes `l as i32 - 16`, i.e. the values `-16..=15`.
    ///
    /// Signed vs unsigned reductions disagree because the negative lanes look
    /// huge when reinterpreted as `u32`:
    ///   - signed   min = -16, max = 15
    ///   - unsigned min = 0 (lane 16), max = 0xFFFFFFFF (lane 15, value -1)
    ///
    /// Lane 0 writes `[signed_min, signed_max]` and `[unsigned_min, unsigned_max]`.
    #[kernel]
    pub fn redux_minmax_signedness(
        mut signed_out: DisjointSlice<i32>,
        mut unsigned_out: DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id();
        let v = lane as i32 - 16;

        let smin = warp::redux_sync_min_i32(FULL_MASK, v);
        let smax = warp::redux_sync_max_i32(FULL_MASK, v);
        let umin = warp::redux_sync_min_u32(FULL_MASK, v as u32);
        let umax = warp::redux_sync_max_u32(FULL_MASK, v as u32);

        if lane == 0 {
            unsafe {
                *signed_out.get_unchecked_mut(0) = smin;
                *signed_out.get_unchecked_mut(1) = smax;
                *unsigned_out.get_unchecked_mut(0) = umin;
                *unsigned_out.get_unchecked_mut(1) = umax;
            }
        }
    }

    /// Lane `l` contributes the single-bit value `1 << l`, so every bit `0..=31`
    /// is set by exactly one lane:
    ///   - and = 0          (no bit is common to all lanes)
    ///   - or  = 0xFFFFFFFF (every bit appears)
    ///   - xor = 0xFFFFFFFF (every bit appears an odd number of times — once)
    ///
    /// Lane 0 writes `[and, or, xor]`.
    #[kernel]
    pub fn redux_bitwise(mut out: DisjointSlice<u32>) {
        let lane = warp::lane_id();
        let v = 1u32 << lane;

        let and = warp::redux_sync_and(FULL_MASK, v);
        let or = warp::redux_sync_or(FULL_MASK, v);
        let xor = warp::redux_sync_xor(FULL_MASK, v);

        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = and;
                *out.get_unchecked_mut(1) = or;
                *out.get_unchecked_mut(2) = xor;
            }
        }
    }

    /// The Test 1 inputs, reduced through the out-of-line helpers above.
    /// Lane 0 writes the signed min and the unsigned max.
    #[kernel]
    pub fn redux_via_device_helpers(
        mut signed_out: DisjointSlice<i32>,
        mut unsigned_out: DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id();
        let v = lane as i32 - 16;

        let smin = min_i32_via_helper(v);
        let umax = max_u32_via_helper(v as u32);

        if lane == 0 {
            unsafe {
                *signed_out.get_unchecked_mut(0) = smin;
                *unsigned_out.get_unchecked_mut(0) = umax;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== redux.sync integer family (sm_80+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // `redux.sync` is an Ampere instruction; the PTX won't assemble below sm_80.
    if major < 8 {
        println!("\nskipping: redux.sync requires sm_80+ (Ampere)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // A single warp is all we need to demonstrate the reduction semantics.
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failed = false;

    // ===== Test 1: signed vs unsigned min/max =====
    println!("\n--- Test 1: redux.sync.min/max (signed vs unsigned) ---");
    let mut signed_dev = DeviceBuffer::<i32>::zeroed(&stream, 2).unwrap();
    let mut unsigned_dev = DeviceBuffer::<u32>::zeroed(&stream, 2).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.redux_minmax_signedness((stream).as_ref(), cfg, &mut signed_dev, &mut unsigned_dev)
    }
    .expect("Kernel launch failed");

    let signed = signed_dev.to_host_vec(&stream).unwrap();
    let unsigned = unsigned_dev.to_host_vec(&stream).unwrap();
    println!(
        "signed   [min, max] = {:?}        (expected [-16, 15])",
        signed
    );
    println!(
        "unsigned [min, max] = {:?} (expected [0, 4294967295])",
        unsigned
    );

    if signed == [-16, 15] && unsigned == [0, u32::MAX] {
        println!("✓ signed and unsigned min/max both correct (and distinct)");
    } else {
        println!("✗ min/max mismatch!");
        failed = true;
    }

    // ===== Test 2: bitwise and/or/xor =====
    println!("\n--- Test 2: redux.sync.and/or/xor ---");
    let mut bits_dev = DeviceBuffer::<u32>::zeroed(&stream, 3).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.redux_bitwise((stream).as_ref(), cfg, &mut bits_dev) }
        .expect("Kernel launch failed");

    let bits = bits_dev.to_host_vec(&stream).unwrap();
    println!("[and, or, xor] = {:#x?}", bits);
    println!("expected       = [0x0, 0xffffffff, 0xffffffff]");

    if bits == [0, u32::MAX, u32::MAX] {
        println!("✓ and/or/xor correct");
    } else {
        println!("✗ bitwise reduction mismatch!");
        failed = true;
    }

    // ===== Test 3: reductions returned from device helpers =====
    println!("\n--- Test 3: redux.sync results returned from device helpers ---");
    let mut helper_signed_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();
    let mut helper_unsigned_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.redux_via_device_helpers(
            (stream).as_ref(),
            cfg,
            &mut helper_signed_dev,
            &mut helper_unsigned_dev,
        )
    }
    .expect("Kernel launch failed");

    let helper_signed = helper_signed_dev.to_host_vec(&stream).unwrap();
    let helper_unsigned = helper_unsigned_dev.to_host_vec(&stream).unwrap();
    println!(
        "[signed min, unsigned max] = [{:?}, {:?}] (expected [[-16], [4294967295]])",
        helper_signed, helper_unsigned
    );

    if helper_signed == [-16] && helper_unsigned == [u32::MAX] {
        println!("✓ reductions returned from device helpers correct");
    } else {
        println!("✗ helper-returned reduction mismatch!");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: redux.sync integer family produced correct results");
}

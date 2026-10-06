/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `core::hint::select_unpredictable` lowering.
//!
//! `select_unpredictable(cond, a, b)` (stable since 1.88) is the branchless form
//! of `if cond { a } else { b }`, bottoming out in the
//! `core::intrinsics::select_unpredictable` intrinsic. The device backend did
//! not lower it — codegen failed with "rustc intrinsic `select_unpredictable`
//! is not yet supported on the device". libcore reaches it pervasively from
//! branchless helpers (slice sorting, `Ord` combinators). The fix lowers it to
//! an `llvm.select`.
//!
//! Each kernel computes an elementwise reduction two ways — once with
//! `select_unpredictable` and once with a plain `if` — and writes the
//! `select_unpredictable` result; the host asserts it matches the branchy
//! reference. `i32` exercises the scalar path; `usize` mirrors the index-typed
//! selects libcore's sort emits. A compile-only raw-pointer kernel also forces
//! libcore to instantiate the intrinsic for both `*mut T` and its internal
//! `MaybeUninit<*mut T>` carrier, guarding pointer-kind preservation.
//!
//! Usage:
//!   cargo oxide run select_unpredictable

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = max(a[i], b[i])` via `select_unpredictable` on `i32`.
    #[kernel]
    pub fn select_max_i32(a: &[i32], b: &[i32], mut out: DisjointSlice<i32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            *slot = core::hint::select_unpredictable(a[i] >= b[i], a[i], b[i]);
        }
    }

    /// `out[i] = min(a[i], b[i])` via `select_unpredictable` on `usize`,
    /// mirroring the index-typed selects libcore's sort emits.
    #[kernel]
    pub fn select_min_usize(a: &[u64], b: &[u64], mut out: DisjointSlice<u64>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            let (x, y) = (a[i] as usize, b[i] as usize);
            *slot = core::hint::select_unpredictable(x <= y, x, y) as u64;
        }
    }

    /// Compile-only provenance control. For `T = *mut u32`, libcore's
    /// implementation invokes the intrinsic on raw pointers while choosing
    /// its drop guards and again on `MaybeUninit<*mut u32>` for the value.
    #[kernel]
    pub fn select_raw_mut(a: *mut u32, b: *mut u32, selector: u32) {
        let selected = core::hint::select_unpredictable(selector & 1 == 0, a, b);
        // SAFETY: this kernel is a compile-only control; callers remain
        // responsible for passing valid writable device pointers.
        unsafe { selected.write(selector) };
    }
}

fn main() {
    println!("=== select_unpredictable ===");

    const N: usize = 256;
    let a_i32: Vec<i32> = (0..N as i32).map(|i| (i * 7 - 300) ^ 0x2c).collect();
    let b_i32: Vec<i32> = (0..N as i32).map(|i| (i * 5 - 111) ^ 0x51).collect();
    let a_u64: Vec<u64> = (0..N as u64)
        .map(|i| i.wrapping_mul(2654435761) & 0xffff)
        .collect();
    let b_u64: Vec<u64> = (0..N as u64)
        .map(|i| i.wrapping_mul(40503) & 0xffff)
        .collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let da = DeviceBuffer::from_host(&stream, &a_i32).unwrap();
    let db = DeviceBuffer::from_host(&stream, &b_i32).unwrap();
    let mut out_max = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.select_max_i32(&stream, cfg, &da, &db, &mut out_max) }
        .expect("select_max_i32 launch");
    let got_max = out_max.to_host_vec(&stream).unwrap();
    let want_max: Vec<i32> = (0..N)
        .map(|i| {
            if a_i32[i] >= b_i32[i] {
                a_i32[i]
            } else {
                b_i32[i]
            }
        })
        .collect();
    assert_eq!(got_max, want_max, "select_max_i32");

    let da = DeviceBuffer::from_host(&stream, &a_u64).unwrap();
    let db = DeviceBuffer::from_host(&stream, &b_u64).unwrap();
    let mut out_min = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.select_min_usize(&stream, cfg, &da, &db, &mut out_min) }
        .expect("select_min_usize launch");
    let got_min = out_min.to_host_vec(&stream).unwrap();
    let want_min: Vec<u64> = (0..N)
        .map(|i| {
            if a_u64[i] <= b_u64[i] {
                a_u64[i]
            } else {
                b_u64[i]
            }
        })
        .collect();
    assert_eq!(got_min, want_min, "select_min_usize");

    println!("PASS: select_unpredictable (i32 max, usize min)");
}

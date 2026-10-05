/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for issue #137.
//!
//! `u32::carrying_mul_add(self, rhs, carry, add)` computes
//! `self * rhs + carry + add` exactly and returns the double-wide result as
//! a `(low, high)` pair. In MIR the method body is a call to the rustc
//! intrinsic `core::intrinsics::carrying_mul_add`, which the importer had
//! no dispatch arm for. The call fell through to a regular `mir.call`
//! against the intrinsic's mangled symbol, which the collector never
//! defines (it skips `InstanceKind::Intrinsic` by design), so the build
//! failed late with "Symbol ... not found" from the LLVM dialect verifier.
//!
//! The fix lowers the intrinsic the same way core's fallback does: widen
//! the four operands to double width, compute `a * b + c + d` (which
//! cannot overflow 2N bits), and split the wide value into `(low, high)`.
//! NVPTX folds that idiom into `mul.lo` / `mul.hi` / `mad` instructions.
//!
//! The stable methods `carrying_mul` and `carrying_mul_add` (unsigned) and
//! the unstable signed `carrying_mul_add` all funnel into this single
//! intrinsic, so they are all exercised here, with literal arguments (the
//! exact shape from the issue) and with host-supplied runtime arguments
//! (which cannot be const-folded away before reaching the importer).
//!
//! Build and run with:
//!   cargo oxide run carrying_mul_add

#![feature(signed_bigint_helpers)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Exercises the bigint helper intrinsic across widths and signedness.
    ///
    /// Results are written as `(low, high)` pairs into `out`:
    ///
    /// | slots  | case                                            |
    /// |--------|-------------------------------------------------|
    /// | 0, 1   | `4u32.carrying_mul_add(5, 0, 0)` (issue repro)  |
    /// | 2, 3   | `a.carrying_mul_add(b, c, d)` (u32, runtime)    |
    /// | 4, 5   | `a.carrying_mul(b, c)` (u32, runtime)           |
    /// | 6, 7   | `x.carrying_mul_add(y, z, w)` (u64, runtime)    |
    /// | 8, 9   | `s.carrying_mul_add(t, 0, 0)` (i32, runtime)    |
    ///
    /// u64 results are stored across two slots each (low then high), so the
    /// output buffer is `u64` and narrower results are widened on store.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn bigint_helpers(
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        x: u64,
        y: u64,
        z: u64,
        w: u64,
        s: i32,
        t: i32,
        mut out: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() == 0 {
            unsafe {
                // Literal arguments: the exact repro shape from issue #137.
                let (low, high) = 4u32.carrying_mul_add(5, 0, 0);
                *out.get_unchecked_mut(0) = low as u64;
                *out.get_unchecked_mut(1) = high as u64;

                // u32, runtime arguments (zero-extension path, i64 wide).
                let (low, high) = a.carrying_mul_add(b, c, d);
                *out.get_unchecked_mut(2) = low as u64;
                *out.get_unchecked_mut(3) = high as u64;

                // carrying_mul funnels into the same intrinsic.
                let (low, high) = a.carrying_mul(b, c);
                *out.get_unchecked_mut(4) = low as u64;
                *out.get_unchecked_mut(5) = high as u64;

                // u64 (zero-extension path, i128 wide).
                let (low, high) = x.carrying_mul_add(y, z, w);
                *out.get_unchecked_mut(6) = low;
                *out.get_unchecked_mut(7) = high;

                // i32 (sign-extension path). Low half is u32, high is i32;
                // store the high half's bit pattern.
                let (low, high) = s.carrying_mul_add(t, 0, 0);
                *out.get_unchecked_mut(8) = low as u64;
                *out.get_unchecked_mut(9) = high as u32 as u64;
            }
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const SLOTS: usize = 10;

fn main() {
    println!("=== carrying_mul_add (issue #137) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Runtime inputs, including the all-MAX case whose exact result is
    // 2^64 - 1 for u32 (the largest value carrying_mul_add can produce).
    let (a, b, c, d) = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    let (x, y, z, w) = (u64::MAX, 1_000_000_007_u64, u64::MAX, 12_345_u64);
    let (s, t) = (-7_i32, 9_i32);

    let mut out = DeviceBuffer::<u64>::zeroed(&stream, SLOTS).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.bigint_helpers(
            &stream,
            LaunchConfig::for_num_elems(1),
            a,
            b,
            c,
            d,
            x,
            y,
            z,
            w,
            s,
            t,
            &mut out,
        )
    }
    .expect("Kernel launch failed");

    let got = out.to_host_vec(&stream).unwrap();

    // Host reference values, computed with the same methods.
    let mut expected = [0_u64; SLOTS];
    let (low, high) = 4u32.carrying_mul_add(5, 0, 0);
    (expected[0], expected[1]) = (low as u64, high as u64);
    let (low, high) = a.carrying_mul_add(b, c, d);
    (expected[2], expected[3]) = (low as u64, high as u64);
    let (low, high) = a.carrying_mul(b, c);
    (expected[4], expected[5]) = (low as u64, high as u64);
    let (low, high) = x.carrying_mul_add(y, z, w);
    (expected[6], expected[7]) = (low, high);
    let (low, high) = s.carrying_mul_add(t, 0, 0);
    (expected[8], expected[9]) = (low as u64, high as u32 as u64);

    let mut errors = 0;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if g != e {
            eprintln!("  Error at slot [{i}]: expected {e:#x}, got {g:#x}");
            errors += 1;
        }
    }

    println!("device = {got:#x?}");
    if errors == 0 {
        println!("\n✓ SUCCESS: all {SLOTS} slots correct");
    } else {
        println!("\n✗ FAILED: {errors} mismatches");
        std::process::exit(1);
    }
}

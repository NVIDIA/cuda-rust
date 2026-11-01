/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end check that `f32::to_int_unchecked` / `f64::to_int_unchecked`
//! work in device code.
//!
//! Both methods call `core::intrinsics::float_to_int_unchecked`. Before that
//! intrinsic was supported, a kernel using either failed to translate (#1595):
//!
//! ```text
//! Translation failed: f64_to_i64: [".../core/src/convert/num.rs": line: 20, column: 30]
//!   Compilation error: invalid input program.
//!   Unsupported construct: rustc intrinsic `std::intrinsics::float_to_int_unchecked`
//!   is not yet supported on the device
//! ```
//!
//! The intrinsic lowers to a plain `fptosi` / `fptoui`, picked from the
//! destination's signedness. Each kernel below feeds values only one of the
//! two instructions handles: negative values for the signed arms, and values
//! above the signed maximum for the unsigned arms. Every input is finite and
//! in range for its destination type, as the method's contract requires.
//!
//! Run: `cargo oxide run float_to_int_unchecked --arch sm_86`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// `f32 -> i32`, the case reported in #1595.
    #[kernel]
    pub fn f32_to_i32(x: &[f32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: the host inputs are finite and inside the i32 range.
            *o = unsafe { x[i].to_int_unchecked::<i32>() };
        }
    }

    /// `f32 -> u32`, with inputs above `i32::MAX`.
    #[kernel]
    pub fn f32_to_u32(x: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: the host inputs are finite and inside the u32 range.
            *o = unsafe { x[i].to_int_unchecked::<u32>() };
        }
    }

    /// `f64 -> i64`, with inputs outside the i32 range.
    #[kernel]
    pub fn f64_to_i64(x: &[f64], mut out: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: the host inputs are finite and inside the i64 range.
            *o = unsafe { x[i].to_int_unchecked::<i64>() };
        }
    }

    /// `f64 -> u64`, with inputs above `i64::MAX`.
    #[kernel]
    pub fn f64_to_u64(x: &[f64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            // SAFETY: the host inputs are finite and inside the u64 range.
            *o = unsafe { x[i].to_int_unchecked::<u64>() };
        }
    }
}

/// Count mismatches against the host conversion and print the first few.
fn check<T: PartialEq + std::fmt::Display>(label: &str, got: &[T], want: &[T]) -> usize {
    let mut bad = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            if bad < 5 {
                println!("  {label} mismatch at {i}: got {g} want {w}");
            }
            bad += 1;
        }
    }
    println!("{label}: {} / {N} correct", N - bad);
    bad
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // Negative and positive values with fractional parts, so truncation
    // toward zero (-12345.678 -> -12345) is checked against rounding down.
    let x_i32: Vec<f32> = (0..N).map(|i| (i as f32 - 128.0) * 12345.678).collect();
    // Up to ~4.08e9: above i32::MAX, below u32::MAX.
    let x_u32: Vec<f32> = (0..N).map(|i| i as f32 * 1.6e7 + 0.5).collect();
    // -8.96e18 to 8.89e18: outside the i32 range, inside the i64 range.
    let x_i64: Vec<f64> = (0..N).map(|i| (i as f64 - 128.0) * 7.0e16 + 0.5).collect();
    // Up to ~1.79e19: above i64::MAX, below u64::MAX.
    let x_u64: Vec<f64> = (0..N).map(|i| i as f64 * 7.0e16 + 0.25).collect();

    // The host's `as` cast gives the same value as `to_int_unchecked` for
    // in-range inputs; it differs only where `to_int_unchecked` is undefined.
    let want_i32: Vec<i32> = x_i32.iter().map(|&v| v as i32).collect();
    let want_u32: Vec<u32> = x_u32.iter().map(|&v| v as u32).collect();
    let want_i64: Vec<i64> = x_i64.iter().map(|&v| v as i64).collect();
    let want_u64: Vec<u64> = x_u64.iter().map(|&v| v as u64).collect();

    let in_i32 = DeviceBuffer::from_host(&stream, &x_i32).unwrap();
    let in_u32 = DeviceBuffer::from_host(&stream, &x_u32).unwrap();
    let in_i64 = DeviceBuffer::from_host(&stream, &x_i64).unwrap();
    let in_u64 = DeviceBuffer::from_host(&stream, &x_u64).unwrap();
    let mut out_i32 = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
    let mut out_u32 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut out_i64 = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();
    let mut out_u64 = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    // SAFETY: each launch is 1-D with one thread per element, matching
    // `index_1d()`, and every buffer holds N elements.
    unsafe {
        module
            .f32_to_i32(&stream, cfg, &in_i32, &mut out_i32)
            .expect("f32_to_i32 launch");
        module
            .f32_to_u32(&stream, cfg, &in_u32, &mut out_u32)
            .expect("f32_to_u32 launch");
        module
            .f64_to_i64(&stream, cfg, &in_i64, &mut out_i64)
            .expect("f64_to_i64 launch");
        module
            .f64_to_u64(&stream, cfg, &in_u64, &mut out_u64)
            .expect("f64_to_u64 launch");
    }

    let got_i32 = out_i32.to_host_vec(&stream).unwrap();
    let got_u32 = out_u32.to_host_vec(&stream).unwrap();
    let got_i64 = out_i64.to_host_vec(&stream).unwrap();
    let got_u64 = out_u64.to_host_vec(&stream).unwrap();
    let bad = check("f32 -> i32", &got_i32, &want_i32)
        + check("f32 -> u32", &got_u32, &want_u32)
        + check("f64 -> i64", &got_i64, &want_i64)
        + check("f64 -> u64", &got_u64, &want_u64);

    if bad == 0 {
        println!("\nPASS");
    } else {
        println!("\nFAIL");
        std::process::exit(1);
    }
}

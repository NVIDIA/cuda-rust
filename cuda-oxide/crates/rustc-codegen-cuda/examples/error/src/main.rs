/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error Test Example - Tests compiler error handling
//!
//! This example contains both valid and intentionally broken kernels.
//! The broken kernel should cause compilation to FAIL with helpful error messages.
//!
//! Usage:
//!   cargo oxide run error
//!
//! Expected: Compilation should FAIL with error messages

use cuda_device::{DisjointSlice, kernel, thread};

/// VALID: f64 → f32 cast example (this compiles correctly)
///
/// This kernel demonstrates a VALID f64 to f32 conversion using `as f32`.
/// The compiler correctly generates: load f64 → add f64 → cvt.f32.f64 → store f32
#[kernel]
pub fn valid_f64_to_f32_kernel(a: &[f64], b: &[f64], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();

    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = (a[idx_raw] + b[idx_raw]) as f32;
    }
}

/// ERROR: Runs the core formatting machinery, which isn't supported on GPU
///
/// The compiler should fail with an error about unsupported operations.
///
/// Note the formatted value must actually be USED: an unused
/// `format_args!` binding is dead code, its `str` locals translate fine,
/// and the kernel compiles. `core::fmt::write` drives the real
/// formatting engine (trait objects, dynamic dispatch), which is the
/// part that is genuinely unsupported on the device.
#[kernel]
pub fn unsupported_format_kernel(a: &[f32], mut c: DisjointSlice<f32>) {
    struct Sink;
    impl core::fmt::Write for Sink {
        fn write_str(&mut self, _s: &str) -> core::fmt::Result {
            Ok(())
        }
    }

    let idx = thread::index_1d();
    let idx_raw = idx.get();

    if let Some(c_elem) = c.get_mut(idx) {
        let mut sink = Sink;
        let _ = core::fmt::write(&mut sink, core::format_args!("{}", a[idx_raw]));
        *c_elem = a[idx_raw];
    }
}
fn main() {
    println!("=== Error Test Example (Unified) ===");
    println!();
    println!("This example is intentionally broken to test error handling.");
    println!("It should NOT compile successfully.");
    println!();
    println!("If you see this message, something went wrong!");
    println!("The kernel compilation should have failed.");
}

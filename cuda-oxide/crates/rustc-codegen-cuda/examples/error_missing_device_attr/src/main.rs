/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: helper function used in device code without `#[device]`
//! (issue #76).
//!
//! `thread::index_1d()` only works inside `#[kernel]` / `#[device]`
//! bodies, where the proc macro rewrites the call to the real device
//! intrinsic. In a plain (un-annotated) helper the call resolves to the
//! public host-side stub, whose body is `unreachable!(...)`. That stub
//! gets MIR-inlined into the helper, so the helper's body collapses into
//! panic-formatting machinery instead of an index computation. Without a
//! dedicated diagnostic the collector silently skipped the collapsed
//! helper and the user got an opaque "Symbol ... not found" verifier
//! error (or, in inlined shapes, a string-constant translation error
//! pointing into `core`), with no hint about the real mistake.
//!
//! Usage:
//!   cargo oxide run error_missing_device_attr
//!
//! Expected: the build FAILS with this exact diagnostic (pinned):
//!
//! ```text
//! error: `thread::index_1d` only works inside `#[kernel]` / `#[device]`
//!        functions; here it resolves to a host-only stub that panics
//!        instead of reading the thread index
//! ```
//!
//! pointing at `helper`'s definition, with a note marking the call site
//! inside the kernel and a help line suggesting `#[device]`.

use cuda_device::{DisjointSlice, kernel, thread};

/// BUG UNDER TEST: this helper is missing `#[device]`, so
/// `thread::index_1d()` inside it resolves to the host-only panicking
/// stub instead of the device intrinsic.
fn helper(out: &mut DisjointSlice<f32>, scale: f32) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = scale;
    }
}

#[kernel]
pub fn missing_attr_kernel(out: DisjointSlice<f32>) {
    let mut out = out;
    helper(&mut out, 2.0);
}

fn main() {
    println!("=== error_missing_device_attr ===");
    println!("This example is intentionally broken: the kernel calls a helper");
    println!("that uses thread::index_1d() without a #[device] annotation.");
    println!("The build must FAIL at codegen time with a diagnostic pointing");
    println!("at the helper and suggesting #[device].");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("diagnostic regression has returned.");
}

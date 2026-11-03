/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `#[target_feature]` function reachable from a kernel.
//!
//! cuda-oxide compiles kernels from the host target's MIR, so a helper that
//! opts into host CPU features (`avx2` on x86_64, `neon` on aarch64) is
//! legal Rust as far as rustc is concerned, and a kernel can call it. It has
//! no PTX lowering. Without a dedicated diagnostic the failure surfaces deep
//! in the translator, or not at all.
//!
//! Usage:
//!   cargo oxide run error_host_target_feature
//!
//! Expected: the build FAILS, and the smoketest checks for this diagnostic:
//!
//! ```text
//! error: `host_only_scale` requires host CPU target features (`avx2`)
//!        and cannot run on the GPU
//! ```
//!
//! (`neon` instead of `avx2` on an aarch64 host) pointing at the call
//! below, with notes naming the kernel and explaining why host-CPU code is
//! visible to device code.

use cuda_device::{DisjointSlice, kernel, thread};

/// A host-only helper. The attribute is the bug under test: it declares a
/// dependency on host CPU features, which the GPU does not have.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn host_only_scale(x: f32) -> f32 {
    x * 2.0
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn host_only_scale(x: f32) -> f32 {
    x * 2.0
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("this fixture needs an x86_64 or aarch64 host");

#[kernel]
pub fn target_feature_kernel(a: &[f32], out: DisjointSlice<f32>) {
    let mut out = out;
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        // BUG UNDER TEST: calling a `#[target_feature]` function from device
        // code. The `unsafe` is what rustc requires to call a feature-gated
        // function from a caller without that feature; it is not the bug.
        *slot = unsafe { host_only_scale(a[idx_raw]) };
    }
}

fn main() {
    println!("=== error_host_target_feature ===");
    println!("This example is intentionally broken: the kernel calls a");
    println!("#[target_feature] helper. The build must FAIL at codegen time");
    println!("with a diagnostic naming the helper, its features, and the kernel.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("host-CPU guard in the collector is broken.");
}

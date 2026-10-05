/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `core::arch::<host>` intrinsic reachable from a kernel.
//!
//! cuda-oxide compiles kernels from the host target's MIR, so
//! `core::arch::x86_64` is fully visible to device code. Most of its
//! intrinsics carry `#[target_feature]` and are caught by that attribute.
//! `_rdtsc` does not (the instruction is baseline on every x86_64), so it
//! exercises the second signal in the collector's guard: the definition
//! path under `core_arch::<arch>`. After MIR inlining the kernel calls the
//! private foreign declaration behind `_rdtsc`, so the path check has to
//! recognise `::core_arch::x86::rdtsc::rdtsc`.
//!
//! This is an x86_64-only fixture. On other hosts it refuses to build
//! through the `compile_error!` below, and the smoketest accepts that.
//!
//! Usage:
//!   cargo oxide run error_host_arch_intrinsic
//!
//! Expected on an x86_64 host: the build FAILS, and the smoketest checks for
//! this diagnostic:
//!
//! ```text
//! error: `core::core_arch::x86::rdtsc::rdtsc` is a `core::arch` intrinsic
//!        for the host CPU (`x86_64`) and cannot run on the GPU
//! ```
//!
//! pointing at the `_rdtsc()` call below, with notes naming the kernel and
//! explaining why host-CPU code is visible to device code.

use cuda_device::{DisjointSlice, kernel, thread};

#[cfg(not(target_arch = "x86_64"))]
compile_error!("error_host_arch_intrinsic is an x86_64-only fixture; see the module docs");

#[kernel]
pub fn arch_intrinsic_kernel(out: DisjointSlice<u64>) {
    let mut out = out;
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        // BUG UNDER TEST: `_rdtsc` reads the x86 time-stamp counter. It has
        // no `#[target_feature]`, so only the `core_arch` path check refuses
        // it. The `unsafe` is what `_rdtsc` requires; it is not the bug.
        #[cfg(target_arch = "x86_64")]
        {
            *slot = unsafe { core::arch::x86_64::_rdtsc() };
        }
    }
}

fn main() {
    println!("=== error_host_arch_intrinsic ===");
    println!("This example is intentionally broken: the kernel reaches a");
    println!("core::arch intrinsic for the host CPU. The build must FAIL at");
    println!("codegen time with a diagnostic naming the intrinsic and the kernel.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("host-CPU guard in the collector is broken.");
}

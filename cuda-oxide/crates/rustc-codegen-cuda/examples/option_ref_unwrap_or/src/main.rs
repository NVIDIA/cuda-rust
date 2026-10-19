/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for issue #132: `Option<&T>::unwrap_or(&literal)` faulted
//! with an illegal memory access (CUDA error 700).
//!
//! With both a `Some` and a `None` of `Option<&T>` live in one function,
//! `-O` const-folds the `None` arm's `unwrap_or(&literal)` into a
//! reference-to-scalar constant: `ConstantKind::Allocated` with pointer
//! placeholder bytes plus a provenance entry naming the literal's
//! allocation. The importer's pointer-constant arm followed that
//! provenance only for struct pointees; for scalar pointees it fell
//! through to the raw-pointer path and emitted `inttoptr 0`, so the
//! dereference became `load i32, ptr null`.
//!
//! Covered here:
//! - the `Some` path (must keep working),
//! - the `None` path with an integer `&literal` default,
//! - the `None` path with a float `&literal` default,
//! - the `None` path with a named-`const` default (promoted to its own
//!   allocation, same provenance shape).
//!
//! A `static` default is deliberately NOT covered: `&STATIC` constants
//! are intercepted earlier by the device-global path, which currently
//! requires zero initializers, so they never reach the provenance branch
//! under test.
//!
//! Run with:
//!   cargo oxide run option_ref_unwrap_or

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Named-const default: `&DEFAULT_U32` promotes to a separate
    /// allocation, exercising the same provenance-following path as the
    /// inline literal.
    pub const DEFAULT_U32: u32 = 123;

    /// Integer case. Each thread evaluates all three unwrap_or results
    /// and writes the one selected by `tid % 3`.
    ///
    /// The literal `Some`/`None` + `unwrap_or` shapes are the test
    /// subject: -O const-folds them into provenance-carrying pointer
    /// constants, the exact pattern issue #132 miscompiled.
    #[allow(clippy::unnecessary_literal_unwrap)]
    #[kernel]
    pub fn opt_ref_unwrap_or_u32(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let r: u32 = 5;
            let a: Option<&u32> = Some(&r);
            let b: Option<&u32> = None; // keeping BOTH a Some and a None live
            // triggers the const-fold of the None arm's unwrap_or
            let v0: u32 = *a.unwrap_or(&77); // Some path: 5
            let v1: u32 = *b.unwrap_or(&77); // None path, literal default: 77
            let v2: u32 = *b.unwrap_or(&DEFAULT_U32); // None path, const default: 123
            *out_elem = match i % 3 {
                0 => v0,
                1 => v1,
                _ => v2,
            };
        }
    }

    /// Float case: same shape with an `&f32` literal default.
    #[allow(clippy::unnecessary_literal_unwrap)]
    #[kernel]
    pub fn opt_ref_unwrap_or_f32(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let i = tid.get();
        if let Some(out_elem) = out.get_mut(tid) {
            let r: f32 = 1.5;
            let a: Option<&f32> = Some(&r);
            let b: Option<&f32> = None;
            let v0: f32 = *a.unwrap_or(&2.5); // Some path: 1.5
            let v1: f32 = *b.unwrap_or(&2.5); // None path, literal default: 2.5
            *out_elem = if i.is_multiple_of(2) { v0 } else { v1 };
        }
    }
}

fn expected_u32(i: usize) -> u32 {
    match i % 3 {
        0 => 5,
        1 => 77,
        _ => kernels::DEFAULT_U32,
    }
}

fn expected_f32(i: usize) -> f32 {
    if i.is_multiple_of(2) { 1.5 } else { 2.5 }
}

fn main() {
    println!("=== option_ref_unwrap_or regression (issue #132) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_u32 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.opt_ref_unwrap_or_u32(stream.as_ref(), cfg, &mut d_u32) }
        .expect("launch opt_ref_unwrap_or_u32");
    let got_u32 = d_u32.to_host_vec(&stream).unwrap();

    let mut d_f32 = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.opt_ref_unwrap_or_f32(stream.as_ref(), cfg, &mut d_f32) }
        .expect("launch opt_ref_unwrap_or_f32");
    let got_f32 = d_f32.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    for tid in 0..N {
        let want_u = expected_u32(tid);
        let got_u = got_u32[tid];
        if got_u != want_u {
            println!("FAIL u32 tid={tid}: got={got_u} expected={want_u}");
            failures += 1;
        }
        let want_f = expected_f32(tid);
        let got_f = got_f32[tid];
        if got_f != want_f {
            println!("FAIL f32 tid={tid}: got={got_f} expected={want_f}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "option_ref_unwrap_or: PASS ({N} threads; Some=5/1.5, literal None=77/2.5, const None=123)"
        );
    } else {
        println!("option_ref_unwrap_or: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}

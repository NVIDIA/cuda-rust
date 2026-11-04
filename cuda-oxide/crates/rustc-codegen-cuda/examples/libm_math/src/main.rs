/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernels calling `libm` crate float functions (PR #142).
//!
//! This is the path glam's `nostd-libm` feature emits on nvptx: float
//! math goes through `libm::sinf`, `libm::expf`, ... instead of the
//! `f32`/`f64` inherent methods. Each call site is intercepted by
//! mir-importer's float-math dispatch and lowered to the matching
//! libdevice intrinsic (`__nv_sinf`, `__nv_expf`, ...); libm's
//! software-float bodies are never translated.
//!
//! Exercises the f32 ("...f") and f64 (bare) entry points, the
//! tuple-returning `sincosf`, and the fmax/fmin/rint lanes. The host
//! computes the same expressions with stdlib float methods and compares.
//!
//! Also pins the interception anchor: `libm_lookalike::expf` below is a
//! user function whose path contains "libm", and it must compile as a
//! regular device call, never be rerouted to `__nv_expf`.
//!
//! `sqrt`/`sqrtf` and `fma`/`fmaf` pin the libm functions that dispatch to a
//! per-host `arch` module (x86 `sqrtss` inline asm, aarch64 `fsqrt`). They are
//! small enough that rustc would MIR-inline them into the kernel, asm and all,
//! before the call could be rerouted; the backend keeps calls to libm's free
//! functions so they reach libdevice like the rest. `libm_arch_f32` also pins
//! the `Libm::<f32>` helper methods, whose f32 names lack the `f` suffix: they
//! must inline down to the free `sqrtf`/`sinf` rather than reach the importer
//! as `sqrt`/`sin` calls. It has no `+1000.0` guard lane, so its tolerance is
//! tight enough to notice a lost or wrong term.
//!
//! Run with:
//!   cargo oxide run libm_math

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Adversarial guard for the libm interception anchor: a user function
/// named like a libm entry point, in a module path containing "libm".
mod libm_lookalike {
    /// The +1000.0 offset makes any wrong rerouting to `__nv_expf` show
    /// up in the host comparison. `inline(never)` keeps the call visible
    /// to the importer's dispatch instead of being MIR-inlined away.
    #[inline(never)]
    pub fn expf(x: f32) -> f32 {
        x + 1000.0
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// f32 libm entry points.
    #[kernel]
    pub fn libm_f32(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let trig = libm::sinf(v) + libm::cosf(v);
            let e = libm::expf(v);
            let p = libm::powf(v, 1.5f32);
            let r = libm::floorf(v) + libm::fabsf(v) + libm::rintf(v * 2.5);
            let mm = libm::fmaxf(v, 1.0) + libm::fminf(v, 2.0);
            let (sn, cs) = libm::sincosf(v);
            let guard = libm_lookalike::expf(v);
            *o = trig + e + p + r + mm + sn * cs + guard;
        }
    }

    /// f64 libm entry points (bare names must dispatch to the f64
    /// libdevice symbols, not the f32 ones).
    #[kernel]
    pub fn libm_f64(x: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let trig = libm::sin(v) + libm::cos(v);
            let e = libm::exp(v);
            let p = libm::pow(v, 1.5f64);
            let t = libm::atan2(v, 2.0f64);
            let r = libm::rint(v * 2.5);
            let mm = libm::fmax(v, 1.0) + libm::fmin(v, 2.0);
            let arch = libm::sqrt(v) + libm::fma(v, 1.5, 0.25);
            *o = trig + e + p + t + r + mm + arch;
        }
    }

    /// The f32 libm entry points that dispatch to a per-host `arch` module,
    /// and the `Libm::<f32>` helpers that forward to the free functions.
    #[kernel]
    pub fn libm_arch_f32(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let arch = libm::sqrtf(v) + libm::fmaf(v, 1.5, 0.25);
            let helper = libm::Libm::<f32>::sqrt(v) + libm::Libm::<f32>::sin(v);
            *o = arch + helper;
        }
    }
}

fn main() {
    println!("=== libm_math: libm crate float ops on the device ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let x32: Vec<f32> = (0..N).map(|i| 0.25 + i as f32 * 0.1).collect();
    let x64: Vec<f64> = (0..N).map(|i| 0.25 + i as f64 * 0.1).collect();

    let x32_dev = DeviceBuffer::from_host(&stream, &x32).unwrap();
    let mut o32_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let x64_dev = DeviceBuffer::from_host(&stream, &x64).unwrap();
    let mut o64_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.libm_f32(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x32_dev,
            &mut o32_dev,
        )
    }
    .expect("libm_f32 launch failed");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.libm_f64(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x64_dev,
            &mut o64_dev,
        )
    }
    .expect("libm_f64 launch failed");

    let o32 = o32_dev.to_host_vec(&stream).unwrap();
    let o64 = o64_dev.to_host_vec(&stream).unwrap();

    let mut ok = true;
    for i in 0..N {
        let v = x32[i];
        let (sn, cs) = (v.sin(), v.cos());
        let want = (v.sin() + v.cos())
            + v.exp()
            + v.powf(1.5)
            + (v.floor() + v.abs() + (v * 2.5).round_ties_even())
            + (v.max(1.0) + v.min(2.0))
            + sn * cs
            + (v + 1000.0);
        // a NaN from a broken lowering compares false and would otherwise pass
        if o32[i].is_nan() || (o32[i] - want).abs() > 1e-3 * want.abs().max(1.0) {
            println!("f32 mismatch at {i}: got {} want {want}", o32[i]);
            ok = false;
        }
        let v = x64[i];
        let want = (v.sin() + v.cos())
            + v.exp()
            + v.powf(1.5)
            + v.atan2(2.0)
            + (v * 2.5).round_ties_even()
            + (v.max(1.0) + v.min(2.0))
            + (v.sqrt() + v.mul_add(1.5, 0.25));
        if o64[i].is_nan() || (o64[i] - want).abs() > 1e-9 * want.abs().max(1.0) {
            println!("f64 mismatch at {i}: got {} want {want}", o64[i]);
            ok = false;
        }
    }

    let mut oa_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.libm_arch_f32(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x32_dev,
            &mut oa_dev,
        )
    }
    .expect("libm_arch_f32 launch failed");
    let oa = oa_dev.to_host_vec(&stream).unwrap();
    for (i, (&v, &got)) in x32.iter().zip(&oa).enumerate() {
        let want = (v.sqrt() + v.mul_add(1.5, 0.25)) + (v.sqrt() + v.sin());
        if got.is_nan() || (got - want).abs() > 1e-5 * want.abs().max(1.0) {
            println!("f32 arch mismatch at {i}: got {got} want {want}");
            ok = false;
        }
    }

    if ok {
        println!("SUCCESS: all libm lanes match host math");
    } else {
        println!("FAIL: device results diverge from host math");
        std::process::exit(1);
    }
}

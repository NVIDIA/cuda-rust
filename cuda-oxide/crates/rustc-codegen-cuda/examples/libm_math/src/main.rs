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
//! `erf_fill_loop` and `erf_pair_loop` call `libm::erff` from small counted
//! loops that fill a per-thread array by the loop counter, with no `#[unroll]`.
//! nvcc and clang unroll such loops and keep the array in registers; if LLVM
//! leaves one rolled, the array moves to local memory. `prec_sqrt_f32` and
//! `prec_sqrt_f64` pin libdevice's correctly rounded sqrt, which `opt` selects
//! only when it folds `__nvvm_reflect` with nvcc's `__CUDA_PREC_SQRT` value.
//! `verify-code-shape.sh` asserts both; GPU-less CI runs it through
//! `scripts/smoketest.sh --compile-only`.
//!
//! Run with:
//!   cargo oxide run libm_math

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// GELU through `libm::erff`, the body of the counted-loop kernels.
#[inline(always)]
fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + libm::erff(v * core::f32::consts::FRAC_1_SQRT_2))
}

/// Whether a counted-loop result matches its host reference; a NaN never does.
fn close(got: f32, want: f32) -> bool {
    let error = (got - want).abs();
    !error.is_nan() && error <= 1e-4 * want.abs().max(1.0)
}

/// The host reference for the counted-loop kernels: their per-thread arrays
/// are folded with weights `1..=4`, so a misplaced lane changes the result.
fn weighted(lanes: [f32; 4]) -> f32 {
    lanes[0] + 2.0 * lanes[1] + 3.0 * lanes[2] + 4.0 * lanes[3]
}

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

    /// f32 libm entry points (including `libm::sqrtf`, which rustc MIR-inlines
    /// to an `arch` `InlineAsm` terminator before call-site interception).
    #[kernel]
    pub fn libm_f32(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let trig = libm::sinf(v) + libm::cosf(v);
            let e = libm::expf(v);
            let p = libm::powf(v, 1.5f32) + libm::sqrtf(v);
            let r = libm::floorf(v) + libm::fabsf(v) + libm::rintf(v * 2.5);
            let mm = libm::fmaxf(v, 1.0) + libm::fminf(v, 2.0);
            let (sn, cs) = libm::sincosf(v);
            let guard = libm_lookalike::expf(v);
            *o = trig + e + p + r + mm + sn * cs + guard;
        }
    }

    /// f64 libm entry points (bare names must dispatch to the f64
    /// libdevice symbols, not the f32 ones, and `libm::sqrt` / `libm::rint`
    /// exercise the MIR-inlined `InlineAsm` path).
    #[kernel]
    pub fn libm_f64(x: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i];
            let trig = libm::sin(v) + libm::cos(v);
            let e = libm::exp(v);
            let p = libm::pow(v, 1.5f64) + libm::sqrt(v);
            let t = libm::atan2(v, 2.0f64);
            let r = libm::rint(v * 2.5);
            let mm = libm::fmax(v, 1.0) + libm::fmin(v, 2.0);
            *o = trig + e + p + t + r + mm;
        }
    }

    /// A four-step fill loop whose body inlines libdevice's `erff`. Each
    /// inlined copy must be priced after `__nvvm_reflect` folds, or the loop
    /// looks twice its size, stays rolled, and `r` goes to local memory.
    #[kernel]
    pub fn erf_fill_loop(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut r = [0.0f32; 4];
            for e in 0..4 {
                r[e] = gelu(x[4 * i + e]);
            }
            *o = weighted(r);
        }
    }

    /// The bias+GELU shape: two `erff` calls per step. Small enough for the
    /// full-unroll threshold clang `-O3` and nvcc use, not for O2's.
    #[kernel]
    pub fn erf_pair_loop(x: &[f32], bias: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut lo = [0.0f32; 4];
            let mut hi = [0.0f32; 4];
            for e in 0..4 {
                lo[e] = gelu(x[8 * i + e] + bias[e]);
                hi[e] = gelu(x[8 * i + 4 + e] + bias[e]);
            }
            *o = weighted(lo) - weighted(hi);
        }
    }

    /// `f32::sqrt` lowers to libdevice's `__nv_sqrtf`, which branches on
    /// `__CUDA_PREC_SQRT` between `sqrt.rn.f32` and `sqrt.approx.f32`.
    #[kernel]
    pub fn prec_sqrt_f32(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i].sqrt();
        }
    }

    /// `f64::sqrt` lowers to libdevice's `__nv_sqrt`.
    #[kernel]
    pub fn prec_sqrt_f64(x: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i].sqrt();
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

    let xs: Vec<f32> = (0..8 * N).map(|i| i as f32 * 0.05 - 6.0).collect();
    let bias = [0.5f32, -0.25, 1.0, -1.5];
    let xs_dev = DeviceBuffer::from_host(&stream, &xs).unwrap();
    let bias_dev = DeviceBuffer::from_host(&stream, &bias).unwrap();
    let mut fill_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut pair_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut sqrt32_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut sqrt64_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.erf_fill_loop(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &xs_dev,
            &mut fill_dev,
        )
    }
    .expect("erf_fill_loop launch failed");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.erf_pair_loop(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &xs_dev,
            &bias_dev,
            &mut pair_dev,
        )
    }
    .expect("erf_pair_loop launch failed");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.prec_sqrt_f32(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x32_dev,
            &mut sqrt32_dev,
        )
    }
    .expect("prec_sqrt_f32 launch failed");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.prec_sqrt_f64(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x64_dev,
            &mut sqrt64_dev,
        )
    }
    .expect("prec_sqrt_f64 launch failed");

    let o32 = o32_dev.to_host_vec(&stream).unwrap();
    let o64 = o64_dev.to_host_vec(&stream).unwrap();
    let fill = fill_dev.to_host_vec(&stream).unwrap();
    let pair = pair_dev.to_host_vec(&stream).unwrap();
    let sqrt32 = sqrt32_dev.to_host_vec(&stream).unwrap();
    let sqrt64 = sqrt64_dev.to_host_vec(&stream).unwrap();

    let mut ok = true;
    for i in 0..N {
        let v = x32[i];
        let (sn, cs) = (v.sin(), v.cos());
        let want = (v.sin() + v.cos())
            + v.exp()
            + (v.powf(1.5) + v.sqrt())
            + (v.floor() + v.abs() + (v * 2.5).round_ties_even())
            + (v.max(1.0) + v.min(2.0))
            + sn * cs
            + (v + 1000.0);
        if (o32[i] - want).abs() > 1e-3 * want.abs().max(1.0) {
            println!("f32 mismatch at {i}: got {} want {want}", o32[i]);
            ok = false;
        }
        let v = x64[i];
        let want = (v.sin() + v.cos())
            + v.exp()
            + (v.powf(1.5) + v.sqrt())
            + v.atan2(2.0)
            + (v * 2.5).round_ties_even()
            + (v.max(1.0) + v.min(2.0));
        if (o64[i] - want).abs() > 1e-9 * want.abs().max(1.0) {
            println!("f64 mismatch at {i}: got {} want {want}", o64[i]);
            ok = false;
        }
        let want = weighted(std::array::from_fn(|e| gelu(xs[4 * i + e])));
        if !close(fill[i], want) {
            println!("erf_fill_loop mismatch at {i}: got {} want {want}", fill[i]);
            ok = false;
        }
        let want = weighted(std::array::from_fn(|e| gelu(xs[8 * i + e] + bias[e])))
            - weighted(std::array::from_fn(|e| gelu(xs[8 * i + 4 + e] + bias[e])));
        if !close(pair[i], want) {
            println!("erf_pair_loop mismatch at {i}: got {} want {want}", pair[i]);
            ok = false;
        }
        // correctly rounded sqrt matches the host bit for bit
        if sqrt32[i] != x32[i].sqrt() || sqrt64[i] != x64[i].sqrt() {
            println!(
                "sqrt mismatch at {i}: got {} / {}, want {} / {}",
                sqrt32[i],
                sqrt64[i],
                x32[i].sqrt(),
                x64[i].sqrt()
            );
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

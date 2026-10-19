/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates the hyperbolic + extended `f32`/`f64` math methods working
//! in a kernel: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
//! `exp_m1`, `ln_1p`, and `hypot`.
//!
//! Six of these (`sinh`, `cosh`, `tanh`, `exp_m1`, `ln_1p`, `hypot`) lower
//! to a `std::sys::cmath::*` shim and previously failed the "FORBIDDEN CRATE
//! IN DEVICE CODE" guard; the float-math dispatch now intercepts each and
//! emits the matching `__nv_*` libdevice call (`__nv_sinh`, `__nv_hypot`,
//! ...). The inverse hyperbolics (`asinh`, `acosh`, `atanh`) are pure-Rust
//! formulas in `std` (compositions of `ln`/`sqrt`/`ln_1p`), so they need no
//! new interception; `atanh` in particular only works once `ln_1p` is
//! intercepted, which this change does. They are exercised here as a
//! regression guard for that composition path.
//!
//! The host recomputes each expression with stdlib float methods and checks
//! the GPU result against a small relative tolerance (transcendental
//! libdevice and host libm differ by a few ULP, so a relative bound is more
//! robust than an exact ULP bound for this many functions).
//!
//! Run:
//!     cargo oxide run math_hyperbolic
//!
//! Exits 0 on PASS, 1 on FAIL.

// Each kernel takes 2 inputs + 9 result slices (one per function), and the
// host reference table is an array of fn pointers; both trip generic clippy
// style lints that do not improve a numeric smoke test.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, ltoir};

#[cuda_module]
mod kernels {
    use super::*;

    // `acosh` needs an argument >= 1 and `atanh` needs |arg| < 1, so we feed
    // them in-domain transforms of `x` (`1 + x*x` and `x*0.4`); the host
    // mirrors the same transforms.
    #[kernel]
    pub fn hyper_f64(
        xs: &[f64],
        ys: &[f64],
        mut o_sinh: DisjointSlice<f64>,
        mut o_cosh: DisjointSlice<f64>,
        mut o_tanh: DisjointSlice<f64>,
        mut o_asinh: DisjointSlice<f64>,
        mut o_acosh: DisjointSlice<f64>,
        mut o_atanh: DisjointSlice<f64>,
        mut o_expm1: DisjointSlice<f64>,
        mut o_ln1p: DisjointSlice<f64>,
        mut o_hypot: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            let y = ys[i];
            if let Some(o) = o_sinh.get_mut(thread::index_1d()) {
                *o = x.sinh();
            }
            if let Some(o) = o_cosh.get_mut(thread::index_1d()) {
                *o = x.cosh();
            }
            if let Some(o) = o_tanh.get_mut(thread::index_1d()) {
                *o = x.tanh();
            }
            if let Some(o) = o_asinh.get_mut(thread::index_1d()) {
                *o = x.asinh();
            }
            if let Some(o) = o_acosh.get_mut(thread::index_1d()) {
                *o = (1.0 + x * x).acosh();
            }
            if let Some(o) = o_atanh.get_mut(thread::index_1d()) {
                *o = (x * 0.4).atanh();
            }
            if let Some(o) = o_expm1.get_mut(thread::index_1d()) {
                *o = x.exp_m1();
            }
            if let Some(o) = o_ln1p.get_mut(thread::index_1d()) {
                *o = x.ln_1p();
            }
            if let Some(o) = o_hypot.get_mut(thread::index_1d()) {
                *o = x.hypot(y);
            }
        }
    }

    #[kernel]
    pub fn hyper_f32(
        xs: &[f32],
        ys: &[f32],
        mut o_sinh: DisjointSlice<f32>,
        mut o_cosh: DisjointSlice<f32>,
        mut o_tanh: DisjointSlice<f32>,
        mut o_asinh: DisjointSlice<f32>,
        mut o_acosh: DisjointSlice<f32>,
        mut o_atanh: DisjointSlice<f32>,
        mut o_expm1: DisjointSlice<f32>,
        mut o_ln1p: DisjointSlice<f32>,
        mut o_hypot: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            let y = ys[i];
            if let Some(o) = o_sinh.get_mut(thread::index_1d()) {
                *o = x.sinh();
            }
            if let Some(o) = o_cosh.get_mut(thread::index_1d()) {
                *o = x.cosh();
            }
            if let Some(o) = o_tanh.get_mut(thread::index_1d()) {
                *o = x.tanh();
            }
            if let Some(o) = o_asinh.get_mut(thread::index_1d()) {
                *o = x.asinh();
            }
            if let Some(o) = o_acosh.get_mut(thread::index_1d()) {
                *o = (1.0 + x * x).acosh();
            }
            if let Some(o) = o_atanh.get_mut(thread::index_1d()) {
                *o = (x * 0.4).atanh();
            }
            if let Some(o) = o_expm1.get_mut(thread::index_1d()) {
                *o = x.exp_m1();
            }
            if let Some(o) = o_ln1p.get_mut(thread::index_1d()) {
                *o = x.ln_1p();
            }
            if let Some(o) = o_hypot.get_mut(thread::index_1d()) {
                *o = x.hypot(y);
            }
        }
    }
}

/// `gpu` is close to `cpu` within a relative tolerance (with an absolute
/// floor for values near zero). Generous enough to absorb the few-ULP
/// libdevice-vs-host-libm disagreement, tight enough to catch a kernel wired
/// to the wrong function.
fn close(gpu: f64, cpu: f64, rtol: f64, atol: f64) -> bool {
    (gpu - cpu).abs() <= atol + rtol * cpu.abs()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== math_hyperbolic: f32/f64 hyperbolic + extended math via libdevice ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls force the NVVM-IR output flavor; the first launch builds
    // a cubin via libNVVM + nvJitLink.
    let module = ltoir::load_kernel_module(&ctx, "math_hyperbolic")?;
    let module = kernels::from_module(module)?;

    // `xs` in (-1, 2]: keeps `ln_1p` (needs x > -1) in domain; `acosh`/`atanh`
    // use the `1 + x*x` / `x*0.4` transforms above.
    let xs_f64: Vec<f64> = vec![
        -0.9, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 0.33, -0.7,
    ];
    let ys_f64: Vec<f64> = vec![
        0.5, 1.0, 2.0, 0.25, 3.0, 0.1, 1.5, 0.75, 2.5, 1.0, 0.6, 4.0, 0.2, 1.1, 0.9, 0.3,
    ];
    let xs_f32: Vec<f32> = xs_f64.iter().map(|&v| v as f32).collect();
    let ys_f32: Vec<f32> = ys_f64.iter().map(|&v| v as f32).collect();
    let n = xs_f64.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    // Names + host reference closures, applied to xs[i] (and ys[i] for hypot).
    let specs: [(&str, fn(f64, f64) -> f64); 9] = [
        ("sinh", |x, _| x.sinh()),
        ("cosh", |x, _| x.cosh()),
        ("tanh", |x, _| x.tanh()),
        ("asinh", |x, _| x.asinh()),
        ("acosh", |x, _| (1.0 + x * x).acosh()),
        ("atanh", |x, _| (x * 0.4).atanh()),
        ("exp_m1", |x, _| x.exp_m1()),
        ("ln_1p", |x, _| x.ln_1p()),
        ("hypot", |x, y| x.hypot(y)),
    ];

    // --- f64 launch ---
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;
    let ys64 = DeviceBuffer::from_host(&stream, &ys_f64)?;
    let mut out64: Vec<DeviceBuffer<f64>> = (0..9)
        .map(|_| DeviceBuffer::<f64>::zeroed(&stream, n))
        .collect::<Result<_, _>>()?;
    {
        let [a, b, c, d, e, f, g, h, i] = &mut out64[..] else {
            unreachable!()
        };
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.hyper_f64(&stream, cfg, &xs64, &ys64, a, b, c, d, e, f, g, h, i) }?;
    }
    let got64: Vec<Vec<f64>> = out64
        .iter()
        .map(|b| b.to_host_vec(&stream))
        .collect::<Result<_, _>>()?;

    // --- f32 launch ---
    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let ys32 = DeviceBuffer::from_host(&stream, &ys_f32)?;
    let mut out32: Vec<DeviceBuffer<f32>> = (0..9)
        .map(|_| DeviceBuffer::<f32>::zeroed(&stream, n))
        .collect::<Result<_, _>>()?;
    {
        let [a, b, c, d, e, f, g, h, i] = &mut out32[..] else {
            unreachable!()
        };
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.hyper_f32(&stream, cfg, &xs32, &ys32, a, b, c, d, e, f, g, h, i) }?;
    }
    let got32: Vec<Vec<f32>> = out32
        .iter()
        .map(|b| b.to_host_vec(&stream))
        .collect::<Result<_, _>>()?;

    let mut failures = 0usize;
    for (fidx, (name, host_fn)) in specs.iter().enumerate() {
        for i in 0..n {
            let exp = host_fn(xs_f64[i], ys_f64[i]);

            // f64: tight relative bound.
            if !close(got64[fidx][i], exp, 1e-11, 1e-12) {
                failures += 1;
                if failures <= 12 {
                    eprintln!(
                        "f64 {name}[{i}] x={:.4} gpu={:e} cpu={:e}",
                        xs_f64[i], got64[fidx][i], exp
                    );
                }
            }

            // f32: looser bound (single precision + recompute in f32).
            let exp_f32 = host_fn(xs_f32[i] as f64, ys_f32[i] as f64) as f32;
            if !close(got32[fidx][i] as f64, exp_f32 as f64, 2e-5, 1e-6) {
                failures += 1;
                if failures <= 12 {
                    eprintln!(
                        "f32 {name}[{i}] x={:.4} gpu={:e} cpu={:e}",
                        xs_f32[i], got32[fidx][i], exp_f32
                    );
                }
            }
        }
    }

    // Representative sample.
    println!(
        "sample: x={:.3} y={:.3}  sinh={:.6} cosh={:.6} tanh={:.6} asinh={:.6}",
        xs_f64[7], ys_f64[7], got64[0][7], got64[1][7], got64[2][7], got64[3][7],
    );
    println!(
        "        acosh={:.6} atanh={:.6} exp_m1={:.6} ln_1p={:.6} hypot={:.6}",
        got64[4][7], got64[5][7], got64[6][7], got64[7][7], got64[8][7],
    );

    if failures == 0 {
        println!("\nSUCCESS: {n} cases × 9 functions × 2 widths within tolerance of host libm");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} checks out of tolerance");
        std::process::exit(1);
    }
}

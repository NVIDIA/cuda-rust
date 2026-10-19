/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test: `std` float methods must compile in device code even when
//! rustc's MIR inliner is off.
//!
//! `x.atan()`, `x.sqrt()`, `x.sinh()`, ... are one-line `#[inline]` wrappers
//! in `std`. With MIR inlining on, the wrapper disappears into the kernel and
//! the collector only sees the function underneath (a `core` intrinsic or a
//! `std::sys::cmath` shim). With MIR inlining off, which is what every
//! `-C incremental` build does, the kernel calls the wrapper itself:
//!
//! ```text
//! inlining on    kernel -> std::sys::cmath::atanf         -> __nv_atanf
//! inlining off   kernel -> std::f32::<impl f32>::atan     -> std::sys::cmath::atanf -> __nv_atanf
//!                          ^ used to be rejected as "forbidden crate `std`"
//! ```
//!
//! This crate sets `-Zinline-mir=no` in its own `Cargo.toml` so the wrapper
//! calls survive in every build, and covers both wrapper families:
//!
//! - wrappers over `std::sys::cmath` shims: `atan`, `atan2`, `tan`, `sinh`,
//!   `exp_m1`, `hypot`
//! - wrappers over `core` intrinsics: `sqrt`, `sin`, `exp`, `ln`, `powf`
//! - a wrapper over other wrappers, returning a tuple: `sin_cos`
//! - one wrapper passed as a function item (`apply(f32::atan, x)`), so the
//!   collector's function-item path is exercised, not only direct calls
//!
//! The host evaluates the same expressions with `std` and compares within a
//! relative tolerance. Run:
//!
//!     cargo oxide run math_no_mir_inline
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, ltoir};

#[cuda_module]
mod kernels {
    use super::*;

    /// Calls `f` through a generic parameter, so the `f32::atan` passed below
    /// reaches the collector as a function item rather than a direct call.
    fn apply<F: Fn(f32) -> f32>(f: F, x: f32) -> f32 {
        f(x)
    }

    /// Wrappers whose bodies call `std::sys::cmath` shims (libdevice math).
    #[kernel]
    pub fn cmath_family_f32(xs: &[f32], ys: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            let (x, y) = (xs[i], ys[i]);
            *slot = apply(f32::atan, x) + x.atan2(y) + x.tan() + x.sinh() + x.exp_m1() + x.hypot(y);
        }
    }

    /// Wrappers whose bodies call `core` float intrinsics.
    #[kernel]
    pub fn core_family_f32(xs: &[f32], ys: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            let (x, y) = (xs[i], ys[i]);
            let (s, c) = x.sin_cos();
            *slot = x.sqrt() + x.sin() + x.exp() + (x + 1.0).ln() + x.powf(y) + (s + s + c);
        }
    }

    #[kernel]
    pub fn cmath_family_f64(xs: &[f64], ys: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            let (x, y) = (xs[i], ys[i]);
            *slot = x.atan() + x.atan2(y) + x.tan() + x.sinh() + x.exp_m1() + x.hypot(y);
        }
    }

    #[kernel]
    pub fn core_family_f64(xs: &[f64], ys: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            let (x, y) = (xs[i], ys[i]);
            let (s, c) = x.sin_cos();
            *slot = x.sqrt() + x.sin() + x.exp() + (x + 1.0).ln() + x.powf(y) + (s + s + c);
        }
    }
}

/// Same expression as `cmath_family_*`, evaluated on the host.
fn cmath_family_host<T: Float>(x: T, y: T) -> T {
    x.atan() + x.atan2(y) + x.tan() + x.sinh() + x.exp_m1() + x.hypot(y)
}

/// Same expression as `core_family_*`, evaluated on the host. `s + s + c` is
/// not symmetric in `s` and `c`, so a swapped `sin_cos` tuple would show up.
fn core_family_host<T: Float>(x: T, y: T) -> T {
    let (s, c) = x.sin_cos();
    x.sqrt() + x.sin() + x.exp() + x.add_one().ln() + x.powf(y) + (s + s + c)
}

/// The handful of `std` float methods the host reference needs, over both
/// widths, so the reference expressions are written once.
trait Float: Copy + core::ops::Add<Output = Self> {
    fn add_one(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn tan(self) -> Self;
    fn sinh(self) -> Self;
    fn exp_m1(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn rel_err(self, expected: Self) -> f64;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            fn add_one(self) -> Self {
                self + 1.0
            }
            fn atan(self) -> Self {
                <$t>::atan(self)
            }
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }
            fn tan(self) -> Self {
                <$t>::tan(self)
            }
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }
            fn exp_m1(self) -> Self {
                <$t>::exp_m1(self)
            }
            fn hypot(self, other: Self) -> Self {
                <$t>::hypot(self, other)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn powf(self, n: Self) -> Self {
                <$t>::powf(self, n)
            }
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }
            fn rel_err(self, expected: Self) -> f64 {
                ((self - expected).abs() / expected.abs()) as f64
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

/// Compare one GPU result vector against the host reference. Returns the
/// number of out-of-tolerance entries after printing the first few.
fn check<T: Float + std::fmt::Display>(
    label: &str,
    got: &[T],
    xs: &[T],
    ys: &[T],
    reference: fn(T, T) -> T,
    tolerance: f64,
) -> usize {
    let mut failures = 0;
    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        let expected = reference(x, y);
        let err = got[i].rel_err(expected);
        // A NaN result must count as a failure, so test the failing side.
        if err.is_nan() || err > tolerance {
            failures += 1;
            if failures <= 4 {
                eprintln!(
                    "[{label}] x={x} y={y} gpu={} cpu={expected} rel_err={err:e}",
                    got[i]
                );
            }
        }
    }
    failures
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== math_no_mir_inline: std float wrappers with MIR inlining off ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls in the kernels force the NVVM-IR output flavor; the
    // first launch builds a cubin via libNVVM + nvJitLink.
    let module = ltoir::load_kernel_module(&ctx, "math_no_mir_inline")?;
    let module = kernels::from_module(module)?;

    // Every term of both expressions is positive on these inputs (`ln` is
    // applied to x + 1, and x below pi/2 keeps `tan` positive), so the sums
    // have no cancellation and a relative tolerance is meaningful.
    // f32-representable so the same values double as the f64 inputs after a
    // widening cast.
    let xs_f32: Vec<f32> = vec![
        0.25, 0.5, 0.75, 1.0, 1.25, 1.4, 0.1, 0.9, 1.1, 0.3, 0.6, 1.3, 0.2, 0.8, 1.2, 0.45,
    ];
    let ys_f32: Vec<f32> = vec![
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.75, 1.25, 1.75, 2.25, 2.75, 0.6, 1.1, 1.6, 2.1, 2.6,
    ];
    let xs_f64: Vec<f64> = xs_f32.iter().map(|&v| v as f64).collect();
    let ys_f64: Vec<f64> = ys_f32.iter().map(|&v| v as f64).collect();
    let n = xs_f32.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let ys32 = DeviceBuffer::from_host(&stream, &ys_f32)?;
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;
    let ys64 = DeviceBuffer::from_host(&stream, &ys_f64)?;

    let mut out_cmath_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_core_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_cmath_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;
    let mut out_core_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cmath_family_f32(&stream, cfg, &xs32, &ys32, &mut out_cmath_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.core_family_f32(&stream, cfg, &xs32, &ys32, &mut out_core_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cmath_family_f64(&stream, cfg, &xs64, &ys64, &mut out_cmath_f64) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.core_family_f64(&stream, cfg, &xs64, &ys64, &mut out_core_f64) }?;

    let got_cmath_f32 = out_cmath_f32.to_host_vec(&stream)?;
    let got_core_f32 = out_core_f32.to_host_vec(&stream)?;
    let got_cmath_f64 = out_cmath_f64.to_host_vec(&stream)?;
    let got_core_f64 = out_core_f64.to_host_vec(&stream)?;

    // Each term is within a couple of ULP of host libm, so a sum of a few
    // positive terms stays far inside these bounds; a wrongly lowered
    // function, a swapped argument, or a swapped tuple is off by orders of
    // magnitude more.
    const TOL_F32: f64 = 1e-5;
    const TOL_F64: f64 = 1e-12;

    let failures = check(
        "cmath f32",
        &got_cmath_f32,
        &xs_f32,
        &ys_f32,
        cmath_family_host,
        TOL_F32,
    ) + check(
        "core f32",
        &got_core_f32,
        &xs_f32,
        &ys_f32,
        core_family_host,
        TOL_F32,
    ) + check(
        "cmath f64",
        &got_cmath_f64,
        &xs_f64,
        &ys_f64,
        cmath_family_host,
        TOL_F64,
    ) + check(
        "core f64",
        &got_core_f64,
        &xs_f64,
        &ys_f64,
        core_family_host,
        TOL_F64,
    );

    println!(
        "sample: x={} y={}  cmath_f32={}  core_f32={}  cmath_f64={}  core_f64={}",
        xs_f32[3], ys_f32[3], got_cmath_f32[3], got_core_f32[3], got_cmath_f64[3], got_core_f64[3]
    );

    if failures == 0 {
        println!("\nSUCCESS: {n} inputs x 12 std float wrappers x 2 widths match host libm");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} results out of tolerance");
        std::process::exit(1);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end device validation for special scalar math, plain `libm` aliases,
//! out-parameter tuple adapters (`frexp`, `modf`, `remquo`), composed `lgamma_r`,
//! IEEE-754-2019 `fmaximum*`/`fminimum*`, and Bessel `j1` in `f32` and `f64`.
//! Run with `cargo oxide run math_special`.
#![feature(float_gamma)]
// The width-generic host macros cast both f32 and f64 results to f64.
#![allow(clippy::too_many_arguments, clippy::unnecessary_cast)]

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Unrelated functions with `libm`-like names that must retain their own bodies.
mod libm_lookalike {
    /// Control for `f32` `tgammaf` namespace isolation.
    #[inline(never)]
    pub fn tgammaf(x: f32) -> f32 {
        x + 1000.0
    }

    /// Control for `f64` `tgamma` namespace isolation.
    #[inline(never)]
    pub fn tgamma(x: f64) -> f64 {
        x + 1000.0
    }

    /// Control for `f32` `frexpf` namespace isolation.
    #[inline(never)]
    pub fn frexpf(x: f32) -> (f32, i32) {
        (x, 12345)
    }

    /// Control for `f64` `frexp` namespace isolation.
    #[inline(never)]
    pub fn frexp(x: f64) -> (f64, i32) {
        (x, 12345)
    }

    /// Control for `f32` `erff` namespace isolation.
    #[inline(never)]
    pub fn erff(x: f32) -> f32 {
        x + 1000.0
    }

    /// Control for `f64` `erf` namespace isolation.
    #[inline(never)]
    pub fn erf(x: f64) -> f64 {
        x + 1000.0
    }

    /// Control for `f32` `modff` namespace isolation.
    #[inline(never)]
    pub fn modff(x: f32) -> (f32, f32) {
        (x + 1000.0, x)
    }

    /// Control for `f64` `modf` namespace isolation.
    #[inline(never)]
    pub fn modf(x: f64) -> (f64, f64) {
        (x + 1000.0, x)
    }

    /// Control for `f64` `lgamma_r` namespace isolation.
    #[inline(never)]
    pub fn lgamma_r(x: f64) -> (f64, i32) {
        (x + 10.0, 17)
    }

    /// Control for `f32` `fmaximumf` namespace isolation.
    #[inline(never)]
    pub fn fmaximumf(x: f32, y: f32) -> f32 {
        x + y + 1000.0
    }

    /// Control for `f64` `fmaximum` namespace isolation.
    #[inline(never)]
    pub fn fmaximum(x: f64, y: f64) -> f64 {
        x + y + 1000.0
    }

    /// Control for `f64` `j1` namespace isolation.
    #[inline(never)]
    pub fn j1(x: f64) -> f64 {
        x + 7.0
    }
}

/// Non-inlined device helpers exercising tuple and scalar return boundaries.
mod math_helpers {
    /// Call `libm::frexpf` across a non-inlined function boundary.
    #[inline(never)]
    pub fn frexp_f32(x: f32) -> (f32, i32) {
        libm::frexpf(x)
    }

    /// Call `libm::frexp` across a non-inlined function boundary.
    #[inline(never)]
    pub fn frexp_f64(x: f64) -> (f64, i32) {
        libm::frexp(x)
    }

    /// Call `libm::modff` across a non-inlined function boundary.
    #[inline(never)]
    pub fn modf_f32(x: f32) -> (f32, f32) {
        libm::modff(x)
    }

    /// Call `libm::modf` across a non-inlined function boundary.
    #[inline(never)]
    pub fn modf_f64(x: f64) -> (f64, f64) {
        libm::modf(x)
    }

    /// Call `libm::lgammaf_r` across a non-inlined function boundary.
    #[inline(never)]
    pub fn lgamma_r_f32(x: f32) -> (f32, i32) {
        libm::lgammaf_r(x)
    }

    /// Call `libm::lgamma_r` across a non-inlined function boundary.
    #[inline(never)]
    pub fn lgamma_r_f64(x: f64) -> (f64, i32) {
        libm::lgamma_r(x)
    }

    /// Call `libm::j1f` across a non-inlined function boundary.
    #[inline(never)]
    pub fn j1_f32(x: f32) -> f32 {
        libm::j1f(x)
    }

    /// Call `libm::j1` across a non-inlined function boundary.
    #[inline(never)]
    pub fn j1_f64(x: f64) -> f64 {
        libm::j1(x)
    }
}

/// Evaluate one of the 11 plain `f32` `libm` aliases or 2 control routes.
#[inline(always)]
fn alias_value_f32(op: u32, x: f32, y: f32) -> f32 {
    match op {
        0 => libm::erff(x),
        1 => libm::erfcf(x),
        2 => libm::exp10f(x),
        3 => libm::j0f(x),
        4 => libm::lgammaf(x),
        5 => libm::y0f(x),
        6 => libm::y1f(x),
        7 => libm::fdimf(x, y),
        8 => libm::fmodf(x, y),
        9 => libm::nextafterf(x, y),
        10 => libm::remainderf(x, y),
        11 => x + 1000.0,
        12 => libm::floorf(x),
        _ => x,
    }
}

/// Evaluate one of the 11 plain `f64` `libm` aliases or 2 control routes.
#[inline(always)]
fn alias_value_f64(op: u32, x: f64, y: f64) -> f64 {
    match op {
        0 => libm::erf(x),
        1 => libm::erfc(x),
        2 => libm::exp10(x),
        3 => libm::j0(x),
        4 => libm::lgamma(x),
        5 => libm::y0(x),
        6 => libm::y1(x),
        7 => libm::fdim(x, y),
        8 => libm::fmod(x, y),
        9 => libm::nextafter(x, y),
        10 => libm::remainder(x, y),
        11 => x + 1000.0,
        12 => libm::floor(x),
        _ => x,
    }
}

/// Evaluate one of the `f32` IEEE-754-2019 min/max operations or controls.
#[inline(always)]
fn minmax_value_f32(op: u32, x: f32, y: f32) -> f32 {
    match op {
        0 => libm::fmaximumf(x, y),
        1 => libm::fminimumf(x, y),
        2 => libm::fmaximum_numf(x, y),
        3 => libm::fminimum_numf(x, y),
        4 => libm::fmaxf(x, y),
        5 => libm::fminf(x, y),
        6 => libm_lookalike::fmaximumf(x, y),
        _ => x,
    }
}

/// Evaluate one of the `f64` IEEE-754-2019 min/max operations or controls.
#[inline(always)]
fn minmax_value_f64(op: u32, x: f64, y: f64) -> f64 {
    match op {
        0 => libm::fmaximum(x, y),
        1 => libm::fminimum(x, y),
        2 => libm::fmaximum_num(x, y),
        3 => libm::fminimum_num(x, y),
        4 => libm::fmax(x, y),
        5 => libm::fmin(x, y),
        6 => libm_lookalike::fmaximum(x, y),
        _ => x,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn special_f32(
        xs: &[f32],
        exponents: &[i32],
        mut gamma: DisjointSlice<f32>,
        mut std_gamma: DisjointSlice<f32>,
        mut logarithm: DisjointSlice<i32>,
        mut scaled: DisjointSlice<f32>,
        mut alias: DisjointSlice<f32>,
        mut control: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            if let Some(out) = gamma.get_mut(thread::index_1d()) {
                *out = libm::tgammaf(x);
            }
            if let Some(out) = std_gamma.get_mut(thread::index_1d()) {
                *out = x.gamma();
            }
            if let Some(out) = logarithm.get_mut(thread::index_1d()) {
                *out = libm::ilogbf(x);
            }
            if let Some(out) = scaled.get_mut(thread::index_1d()) {
                *out = libm::ldexpf(x, exponents[i]);
            }
            if let Some(out) = alias.get_mut(thread::index_1d()) {
                *out = libm::scalbnf(x, exponents[i]);
            }
            if let Some(out) = control.get_mut(thread::index_1d()) {
                *out = libm_lookalike::tgammaf(x);
            }
        }
    }

    #[kernel]
    pub fn special_f64(
        xs: &[f64],
        exponents: &[i32],
        mut gamma: DisjointSlice<f64>,
        mut std_gamma: DisjointSlice<f64>,
        mut logarithm: DisjointSlice<i32>,
        mut scaled: DisjointSlice<f64>,
        mut alias: DisjointSlice<f64>,
        mut control: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            if let Some(out) = gamma.get_mut(thread::index_1d()) {
                *out = libm::tgamma(x);
            }
            if let Some(out) = std_gamma.get_mut(thread::index_1d()) {
                *out = x.gamma();
            }
            if let Some(out) = logarithm.get_mut(thread::index_1d()) {
                *out = libm::ilogb(x);
            }
            if let Some(out) = scaled.get_mut(thread::index_1d()) {
                *out = libm::ldexp(x, exponents[i]);
            }
            if let Some(out) = alias.get_mut(thread::index_1d()) {
                *out = libm::scalbn(x, exponents[i]);
            }
            if let Some(out) = control.get_mut(thread::index_1d()) {
                *out = libm_lookalike::tgamma(x);
            }
        }
    }

    #[kernel]
    pub fn frexp_f32(
        xs: &[f32],
        mut direct_fraction: DisjointSlice<f32>,
        mut direct_exponent: DisjointSlice<i32>,
        mut projected_fraction: DisjointSlice<f32>,
        mut projected_exponent: DisjointSlice<i32>,
        mut loop_fraction: DisjointSlice<f32>,
        mut loop_exponent: DisjointSlice<i32>,
        mut control_fraction: DisjointSlice<f32>,
        mut control_exponent: DisjointSlice<i32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            let direct = math_helpers::frexp_f32(x);
            let mut projected = (17_u32, (x, 0_i32));
            projected.1 = libm::frexpf(projected.1.0);
            let mut repeated = (0.0, 0_i32);
            for step in 0..3 {
                let input = if step == 1 { -x } else { x };
                repeated = libm::frexpf(input);
            }
            let control = libm_lookalike::frexpf(x);
            if let Some(out) = direct_fraction.get_mut(thread::index_1d()) {
                *out = direct.0;
            }
            if let Some(out) = direct_exponent.get_mut(thread::index_1d()) {
                *out = direct.1;
            }
            if let Some(out) = projected_fraction.get_mut(thread::index_1d()) {
                *out = projected.1.0;
            }
            if let Some(out) = projected_exponent.get_mut(thread::index_1d()) {
                *out = projected.1.1;
            }
            if let Some(out) = loop_fraction.get_mut(thread::index_1d()) {
                *out = repeated.0;
            }
            if let Some(out) = loop_exponent.get_mut(thread::index_1d()) {
                *out = repeated.1;
            }
            if let Some(out) = control_fraction.get_mut(thread::index_1d()) {
                *out = control.0;
            }
            if let Some(out) = control_exponent.get_mut(thread::index_1d()) {
                *out = control.1;
            }
        }
    }

    #[kernel]
    pub fn frexp_f64(
        xs: &[f64],
        mut direct_fraction: DisjointSlice<f64>,
        mut direct_exponent: DisjointSlice<i32>,
        mut projected_fraction: DisjointSlice<f64>,
        mut projected_exponent: DisjointSlice<i32>,
        mut loop_fraction: DisjointSlice<f64>,
        mut loop_exponent: DisjointSlice<i32>,
        mut control_fraction: DisjointSlice<f64>,
        mut control_exponent: DisjointSlice<i32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            let direct = math_helpers::frexp_f64(x);
            let mut projected = (17_u32, (x, 0_i32));
            projected.1 = libm::frexp(projected.1.0);
            let mut repeated = (0.0, 0_i32);
            for step in 0..3 {
                let input = if step == 1 { -x } else { x };
                repeated = libm::frexp(input);
            }
            let control = libm_lookalike::frexp(x);
            if let Some(out) = direct_fraction.get_mut(thread::index_1d()) {
                *out = direct.0;
            }
            if let Some(out) = direct_exponent.get_mut(thread::index_1d()) {
                *out = direct.1;
            }
            if let Some(out) = projected_fraction.get_mut(thread::index_1d()) {
                *out = projected.1.0;
            }
            if let Some(out) = projected_exponent.get_mut(thread::index_1d()) {
                *out = projected.1.1;
            }
            if let Some(out) = loop_fraction.get_mut(thread::index_1d()) {
                *out = repeated.0;
            }
            if let Some(out) = loop_exponent.get_mut(thread::index_1d()) {
                *out = repeated.1;
            }
            if let Some(out) = control_fraction.get_mut(thread::index_1d()) {
                *out = control.0;
            }
            if let Some(out) = control_exponent.get_mut(thread::index_1d()) {
                *out = control.1;
            }
        }
    }

    #[kernel]
    pub fn aliases_f32(
        xs: &[f32],
        ys: &[f32],
        operations: &[u32],
        mut outputs: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let value = if operations[i] == 11 {
                libm_lookalike::erff(xs[i])
            } else {
                alias_value_f32(operations[i], xs[i], ys[i])
            };
            if let Some(out) = outputs.get_mut(thread::index_1d()) {
                *out = value;
            }
        }
    }

    #[kernel]
    pub fn aliases_f64(
        xs: &[f64],
        ys: &[f64],
        operations: &[u32],
        mut outputs: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let value = if operations[i] == 11 {
                libm_lookalike::erf(xs[i])
            } else {
                alias_value_f64(operations[i], xs[i], ys[i])
            };
            if let Some(out) = outputs.get_mut(thread::index_1d()) {
                *out = value;
            }
        }
    }

    #[kernel]
    pub fn out_f32(
        xs: &[f32],
        ys: &[f32],
        mut fraction: DisjointSlice<f32>,
        mut integral: DisjointSlice<f32>,
        mut remainder: DisjointSlice<f32>,
        mut quotient: DisjointSlice<i32>,
        mut control: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let (f, n) = math_helpers::modf_f32(xs[i]);
            let (r, q) = libm::remquof(xs[i], ys[i]);
            let (c, _) = libm_lookalike::modff(xs[i]);
            if let Some(out) = fraction.get_mut(thread::index_1d()) {
                *out = f;
            }
            if let Some(out) = integral.get_mut(thread::index_1d()) {
                *out = n;
            }
            if let Some(out) = remainder.get_mut(thread::index_1d()) {
                *out = r;
            }
            if let Some(out) = quotient.get_mut(thread::index_1d()) {
                *out = q;
            }
            if let Some(out) = control.get_mut(thread::index_1d()) {
                *out = c;
            }
        }
    }

    #[kernel]
    pub fn out_f64(
        xs: &[f64],
        ys: &[f64],
        mut fraction: DisjointSlice<f64>,
        mut integral: DisjointSlice<f64>,
        mut remainder: DisjointSlice<f64>,
        mut quotient: DisjointSlice<i32>,
        mut control: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let (f, n) = math_helpers::modf_f64(xs[i]);
            let (r, q) = libm::remquo(xs[i], ys[i]);
            let (c, _) = libm_lookalike::modf(xs[i]);
            if let Some(out) = fraction.get_mut(thread::index_1d()) {
                *out = f;
            }
            if let Some(out) = integral.get_mut(thread::index_1d()) {
                *out = n;
            }
            if let Some(out) = remainder.get_mut(thread::index_1d()) {
                *out = r;
            }
            if let Some(out) = quotient.get_mut(thread::index_1d()) {
                *out = q;
            }
            if let Some(out) = control.get_mut(thread::index_1d()) {
                *out = c;
            }
        }
    }

    #[kernel]
    pub fn gamma_f32(
        xs: &[f32],
        mut values: DisjointSlice<f32>,
        mut signs: DisjointSlice<i32>,
        mut scalar: DisjointSlice<f32>,
        mut control: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let (value, sign) = math_helpers::lgamma_r_f32(xs[i]);
            if let Some(slot) = values.get_mut(thread::index_1d()) {
                *slot = value;
            }
            if let Some(slot) = signs.get_mut(thread::index_1d()) {
                *slot = sign;
            }
            if let Some(slot) = scalar.get_mut(thread::index_1d()) {
                *slot = libm::lgammaf(xs[i]);
            }
            if let Some(slot) = control.get_mut(thread::index_1d()) {
                *slot = xs[i] + 10.0;
            }
        }
    }

    #[kernel]
    pub fn gamma_f64(
        xs: &[f64],
        mut values: DisjointSlice<f64>,
        mut signs: DisjointSlice<i32>,
        mut scalar: DisjointSlice<f64>,
        mut control: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let (value, sign) = math_helpers::lgamma_r_f64(xs[i]);
            let (lookalike_value, _) = libm_lookalike::lgamma_r(xs[i]);
            if let Some(slot) = values.get_mut(thread::index_1d()) {
                *slot = value;
            }
            if let Some(slot) = signs.get_mut(thread::index_1d()) {
                *slot = sign;
            }
            if let Some(slot) = scalar.get_mut(thread::index_1d()) {
                *slot = libm::lgamma(xs[i]);
            }
            if let Some(slot) = control.get_mut(thread::index_1d()) {
                *slot = lookalike_value;
            }
        }
    }

    #[kernel]
    pub fn minmax_f32(xs: &[f32], ys: &[f32], ops: &[u32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(thread::index_1d())
        {
            *slot = minmax_value_f32(ops[i], xs[i], ys[i]);
        }
    }

    #[kernel]
    pub fn minmax_f64(xs: &[f64], ys: &[f64], ops: &[u32], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d().get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(thread::index_1d())
        {
            *slot = minmax_value_f64(ops[i], xs[i], ys[i]);
        }
    }

    #[kernel]
    pub fn bessel_f32(
        xs: &[f32],
        mut values: DisjointSlice<f32>,
        mut controls: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            if let Some(slot) = values.get_mut(thread::index_1d()) {
                *slot = math_helpers::j1_f32(xs[i]);
            }
            if let Some(slot) = controls.get_mut(thread::index_1d()) {
                *slot = libm::j0f(xs[i]);
            }
        }
    }

    #[kernel]
    pub fn bessel_f64(
        xs: &[f64],
        mut values: DisjointSlice<f64>,
        mut controls: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            if let Some(slot) = values.get_mut(thread::index_1d()) {
                *slot = math_helpers::j1_f64(xs[i]);
            }
            if let Some(slot) = controls.get_mut(thread::index_1d()) {
                *slot = libm_lookalike::j1(xs[i]);
            }
        }
    }
}

/// Compare two floating-point values with relative tolerance and exact special-value checks.
fn close(got: f64, expected: f64, rtol: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if expected.is_infinite() || expected == 0.0 {
        got.to_bits() == expected.to_bits()
    } else {
        got.is_finite() && (got - expected).abs() <= rtol * expected.abs()
    }
}

/// Compare two floating-point values with combined relative/absolute tolerance or exact bits.
fn close_alias(got: f64, expected: f64, rtol: f64, atol: f64, exact: bool) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if exact || expected.is_infinite() || expected == 0.0 {
        got.to_bits() == expected.to_bits()
    } else {
        got.is_finite() && (got - expected).abs() <= atol + rtol * expected.abs()
    }
}

/// Check bitwise equality for finite, zero, and infinite values, or NaN equivalence.
fn same_bits_or_nan(got: f64, expected: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else {
        got.to_bits() == expected.to_bits()
    }
}

/// Compare `lgamma_r` or Bessel `j1` outputs with scaled tolerance and signed-zero checks.
fn close_scaled(got: f64, expected: f64, tolerance: f64, exact_zero: bool) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if exact_zero && expected == 0.0 {
        got.to_bits() == expected.to_bits()
    } else if expected.is_infinite() {
        got == expected
    } else {
        got.is_finite() && (got - expected).abs() <= tolerance * (1.0 + expected.abs())
    }
}

/// Execute all six device math validation suites across `f32` and `f64`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut failures = 0usize;
    let mut checks = 0usize;

    macro_rules! check_width {
        ($ty:ty, $kernel:ident, $gamma:ident, $ilogb:ident, $ldexp:ident, $rtol:expr, $extra:expr) => {{
            let mut xs: Vec<$ty> = vec![
                -3.5, -2.5, -1.25, -0.5, 0.125, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0,
                0.33,
            ];
            let mut exponents = vec![-12, -4, -1, 0, 1, 4, 8, 24, -24, 100, -100, 3, 2, -3, 5, -5];
            for (x, exponent) in $extra {
                xs.push(x);
                exponents.push(exponent);
            }
            let n = xs.len();
            let inputs = DeviceBuffer::from_host(&stream, &xs)?;
            let powers = DeviceBuffer::from_host(&stream, &exponents)?;
            let mut gamma = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut std_gamma = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut logarithm = DeviceBuffer::<i32>::zeroed(&stream, n)?;
            let mut scaled = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut alias = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut control = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            // SAFETY: every buffer has n elements; the kernel bounds-checks i.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &inputs,
                    &powers,
                    &mut gamma,
                    &mut std_gamma,
                    &mut logarithm,
                    &mut scaled,
                    &mut alias,
                    &mut control,
                )?;
            }
            let gamma = gamma.to_host_vec(&stream)?;
            let std_gamma = std_gamma.to_host_vec(&stream)?;
            let logarithm = logarithm.to_host_vec(&stream)?;
            let scaled = scaled.to_host_vec(&stream)?;
            let alias = alias.to_host_vec(&stream)?;
            let control = control.to_host_vec(&stream)?;
            for (i, &x) in xs.iter().enumerate() {
                let expected_gamma = libm::$gamma(x);
                let expected_logarithm = libm::$ilogb(x);
                let expected_scaled = libm::$ldexp(x, exponents[i]);
                for (name, valid) in [
                    (
                        "libm gamma",
                        close(gamma[i] as f64, expected_gamma as f64, $rtol),
                    ),
                    (
                        "std gamma",
                        close(std_gamma[i] as f64, expected_gamma as f64, $rtol),
                    ),
                    ("ilogb", logarithm[i] == expected_logarithm),
                    ("ldexp", scaled[i].to_bits() == expected_scaled.to_bits()),
                    ("scalbn", alias[i].to_bits() == expected_scaled.to_bits()),
                    ("control", control[i].to_bits() == (x + 1000.0).to_bits()),
                ] {
                    checks += 1;
                    if !valid {
                        failures += 1;
                        eprintln!(
                            "{} {name}[{i}]: x={x:?}, exponent={}",
                            stringify!($ty),
                            exponents[i]
                        );
                    }
                }
            }
        }};
    }
    check_width!(
        f32,
        special_f32,
        tgammaf,
        ilogbf,
        ldexpf,
        4e-6,
        [
            (f32::from_bits(1), 149),
            (f32::MIN_POSITIVE, -1),
            (1.0, -149),
            (-1.0, -150),
            (1.0, 128),
            (-1.0, 128),
        ]
    );
    check_width!(
        f64,
        special_f64,
        tgamma,
        ilogb,
        ldexp,
        2e-13,
        [
            (f64::from_bits(1), 1074),
            (f64::MIN_POSITIVE, -1),
            (1.0, -1074),
            (-1.0, -1075),
            (1.0, 1024),
            (-1.0, 1024),
        ]
    );

    macro_rules! check_frexp_width {
        ($ty:ty, $kernel:ident, $frexp:ident, $extra:expr) => {{
            let mut xs: Vec<$ty> = vec![
                0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.25, -3.25, 0.75, -0.75,
                42.0, -42.0, 0.125, -0.125,
            ];
            xs.extend($extra);
            let n = xs.len();
            let inputs = DeviceBuffer::from_host(&stream, &xs)?;
            let mut direct_fraction = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut direct_exponent = DeviceBuffer::<i32>::zeroed(&stream, n)?;
            let mut projected_fraction = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut projected_exponent = DeviceBuffer::<i32>::zeroed(&stream, n)?;
            let mut loop_fraction = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut loop_exponent = DeviceBuffer::<i32>::zeroed(&stream, n)?;
            let mut control_fraction = DeviceBuffer::<$ty>::zeroed(&stream, n)?;
            let mut control_exponent = DeviceBuffer::<i32>::zeroed(&stream, n)?;
            // SAFETY: every buffer contains n elements and each output is
            // written only by its matching thread, with an input bounds check.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(n as u32),
                    &inputs,
                    &mut direct_fraction,
                    &mut direct_exponent,
                    &mut projected_fraction,
                    &mut projected_exponent,
                    &mut loop_fraction,
                    &mut loop_exponent,
                    &mut control_fraction,
                    &mut control_exponent,
                )?;
            }
            let fractions = [
                direct_fraction.to_host_vec(&stream)?,
                projected_fraction.to_host_vec(&stream)?,
                loop_fraction.to_host_vec(&stream)?,
                control_fraction.to_host_vec(&stream)?,
            ];
            let exponents = [
                direct_exponent.to_host_vec(&stream)?,
                projected_exponent.to_host_vec(&stream)?,
                loop_exponent.to_host_vec(&stream)?,
                control_exponent.to_host_vec(&stream)?,
            ];
            for (i, &x) in xs.iter().enumerate() {
                let expected = libm::$frexp(x);
                for (route, name) in ["tuple return", "projected tuple", "loop", "user control"]
                    .iter()
                    .enumerate()
                {
                    let (fraction, exponent) = if route == 3 { (x, 12345) } else { expected };
                    checks += 2;
                    if fractions[route][i].to_bits() != fraction.to_bits() {
                        failures += 1;
                        eprintln!(
                            "{} frexp {name}[{i}] fraction: got={:#x} expected={:#x}",
                            stringify!($ty),
                            fractions[route][i].to_bits(),
                            fraction.to_bits()
                        );
                    }
                    if exponents[route][i] != exponent {
                        failures += 1;
                        eprintln!(
                            "{} frexp {name}[{i}] exponent: got={} expected={exponent}",
                            stringify!($ty),
                            exponents[route][i]
                        );
                    }
                }
            }
        }};
    }
    check_frexp_width!(
        f32,
        frexp_f32,
        frexpf,
        [
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::from_bits(0x007f_ffff),
            f32::from_bits(0x807f_ffff),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc1_2345),
            f32::from_bits(0xffc1_2345),
            f32::from_bits(0x7f81_2345),
            f32::from_bits(0xff81_2345),
        ]
    );
    check_frexp_width!(
        f64,
        frexp_f64,
        frexp,
        [
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
            f64::from_bits(1),
            f64::from_bits(0x8000_0000_0000_0001),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::from_bits(0x800f_ffff_ffff_ffff),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0001_2345),
            f64::from_bits(0xfff8_0000_0001_2345),
            f64::from_bits(0x7ff0_0000_0001_2345),
            f64::from_bits(0xfff0_0000_0001_2345),
        ]
    );

    macro_rules! check_aliases_width {
        ($ty:ty, $kernel:ident, $reference:ident, $rtol:expr, $atol:expr) => {{
            let cases: [($ty, $ty); 16] = [
                (0.0, 1.0),
                (-0.0, -1.0),
                (0.125, 0.5),
                (0.5, -0.75),
                (1.0, 0.5),
                (1.5, 1.0),
                (2.5, 1.0),
                (3.5, -2.0),
                (-1.25, 0.5),
                (8.0, 3.0),
                (<$ty>::INFINITY, 1.0),
                (<$ty>::NEG_INFINITY, <$ty>::INFINITY),
                (<$ty>::NAN, 2.0),
                (1.0, <$ty>::NAN),
                (0.75, 0.0),
                (4.0, <$ty>::INFINITY),
            ];
            let mut xs = Vec::with_capacity(13 * cases.len());
            let mut ys = Vec::with_capacity(xs.capacity());
            let mut operations = Vec::with_capacity(xs.capacity());
            for op in 0..13u32 {
                for &(x, y) in &cases {
                    xs.push(x);
                    ys.push(y);
                    operations.push(op);
                }
            }
            let count = xs.len();
            let device_xs = DeviceBuffer::from_host(&stream, &xs)?;
            let device_ys = DeviceBuffer::from_host(&stream, &ys)?;
            let device_ops = DeviceBuffer::from_host(&stream, &operations)?;
            let mut outputs = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            // SAFETY: all input and output buffers have count elements; each
            // thread checks its index and writes only its own output element.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(count as u32),
                    &device_xs,
                    &device_ys,
                    &device_ops,
                    &mut outputs,
                )?;
            }
            let outputs = outputs.to_host_vec(&stream)?;
            for i in 0..count {
                let op = operations[i];
                let expected = $reference(op, xs[i], ys[i]);
                let exact = (7..=10).contains(&op) || op >= 11;
                let valid = close_alias(outputs[i] as f64, expected as f64, $rtol, $atol, exact);
                checks += 1;
                if !valid {
                    failures += 1;
                    eprintln!(
                        "{} alias op={op} case={} x={:?} y={:?} got={:?} expected={:?}",
                        stringify!($ty),
                        i % cases.len(),
                        xs[i],
                        ys[i],
                        outputs[i],
                        expected,
                    );
                }
            }
        }};
    }
    check_aliases_width!(f32, aliases_f32, alias_value_f32, 4e-5, 4e-7);
    check_aliases_width!(f64, aliases_f64, alias_value_f64, 2e-11, 2e-14);

    macro_rules! check_out_params_width {
        ($ty:ty, $kernel:ident, $split:ident, $remquo:ident) => {{
            let cases: [($ty, $ty); 18] = [
                (0.0, 1.0),
                (-0.0, -1.0),
                (1.25, 0.5),
                (-1.25, 0.5),
                (2.5, 2.0),
                (-2.5, -2.0),
                (7.0, 2.0),
                (3.0, 2.0),
                (5.0, -2.0),
                (8.0, 3.0),
                (-12.5, 4.0),
                (<$ty>::MIN_POSITIVE, 1.0),
                (<$ty>::INFINITY, 1.0),
                (<$ty>::NEG_INFINITY, 1.0),
                (<$ty>::NAN, 1.0),
                (7.0, 0.0),
                (7.0, <$ty>::NAN),
                (-0.0, <$ty>::INFINITY),
            ];
            let xs: Vec<$ty> = cases.iter().map(|&(x, _)| x).collect();
            let ys: Vec<$ty> = cases.iter().map(|&(_, y)| y).collect();
            let count = cases.len();
            let device_xs = DeviceBuffer::from_host(&stream, &xs)?;
            let device_ys = DeviceBuffer::from_host(&stream, &ys)?;
            let mut fraction = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut integral = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut remainder = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut quotient = DeviceBuffer::<i32>::zeroed(&stream, count)?;
            let mut control = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            // SAFETY: each output has count elements and only thread i writes slot i.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(count as u32),
                    &device_xs,
                    &device_ys,
                    &mut fraction,
                    &mut integral,
                    &mut remainder,
                    &mut quotient,
                    &mut control,
                )?;
            }
            let fraction = fraction.to_host_vec(&stream)?;
            let integral = integral.to_host_vec(&stream)?;
            let remainder = remainder.to_host_vec(&stream)?;
            let quotient = quotient.to_host_vec(&stream)?;
            let control = control.to_host_vec(&stream)?;
            for (i, &(x, y)) in cases.iter().enumerate() {
                let (f, n) = libm::$split(x);
                let (r, q) = libm::$remquo(x, y);
                let valid_remainder = same_bits_or_nan(remainder[i] as f64, r as f64);
                let valid_quotient = if r.is_nan() {
                    quotient[i] == 0
                } else {
                    quotient[i].signum() == q.signum()
                        && quotient[i].unsigned_abs() & 7 == q.unsigned_abs() & 7
                };
                let results = [
                    same_bits_or_nan(fraction[i] as f64, f as f64),
                    same_bits_or_nan(integral[i] as f64, n as f64),
                    valid_remainder,
                    valid_quotient,
                    same_bits_or_nan(control[i] as f64, (x + 1000.0) as f64),
                ];
                for (field, valid) in results.into_iter().enumerate() {
                    checks += 1;
                    if !valid {
                        failures += 1;
                        eprintln!(
                            "{} out_params case={i} field={field} x={x:?} y={y:?} got=({:?}, {:?}, {:?}, {}, {:?}) expected=({f:?}, {n:?}, {r:?}, {q}, {:?})",
                            stringify!($ty),
                            fraction[i],
                            integral[i],
                            remainder[i],
                            quotient[i],
                            control[i],
                            x + 1000.0
                        );
                    }
                }
            }
        }};
    }
    check_out_params_width!(f32, out_f32, modff, remquof);
    check_out_params_width!(f64, out_f64, modf, remquo);

    macro_rules! check_lgamma_r_width {
        ($ty:ty, $kernel:ident, $reference:ident, $scalar:ident, $tolerance:expr) => {{
            let xs: Vec<$ty> = vec![
                0.0,
                -0.0,
                1.0,
                2.0,
                0.5,
                1.5,
                10.25,
                -0.5,
                -1.5,
                -2.5,
                -3.5,
                -4.5,
                -5.25,
                -6.125,
                -1.0,
                -2.0,
                -3.0,
                -10.0,
                <$ty>::MIN_POSITIVE,
                -<$ty>::MIN_POSITIVE,
                <$ty>::from_bits(1),
                -<$ty>::from_bits(1),
                <$ty>::INFINITY,
                <$ty>::NEG_INFINITY,
                <$ty>::NAN,
                if core::mem::size_of::<$ty>() == 4 {
                    -4_194_304.5
                } else {
                    -2_251_799_813_685_247.5
                },
            ];
            let count = xs.len();
            let input = DeviceBuffer::from_host(&stream, &xs)?;
            let mut values = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut signs = DeviceBuffer::<i32>::zeroed(&stream, count)?;
            let mut scalar = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut control = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            // SAFETY: each thread writes a unique in-bounds output slot.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(count as u32),
                    &input,
                    &mut values,
                    &mut signs,
                    &mut scalar,
                    &mut control,
                )?;
            }
            let values = values.to_host_vec(&stream)?;
            let signs = signs.to_host_vec(&stream)?;
            let scalar = scalar.to_host_vec(&stream)?;
            let control = control.to_host_vec(&stream)?;
            for (i, &x) in xs.iter().enumerate() {
                let (expected_value, expected_sign) = libm::$reference(x);
                let validations = [
                    close_scaled(values[i] as f64, expected_value as f64, $tolerance, false),
                    signs[i] == expected_sign,
                    if x == <$ty>::NEG_INFINITY {
                        scalar[i].is_infinite()
                    } else {
                        close_scaled(
                            scalar[i] as f64,
                            libm::$scalar(x) as f64,
                            $tolerance,
                            false,
                        )
                    },
                    close_scaled(control[i] as f64, (x + 10.0) as f64, $tolerance, false),
                ];
                for (field, valid) in validations.into_iter().enumerate() {
                    checks += 1;
                    if !valid {
                        failures += 1;
                        eprintln!(
                            "{} lgamma_r case={i} field={field} x={x:?} got=({:?}, {}, {:?}) expected=({expected_value:?}, {expected_sign}, {:?})",
                            stringify!($ty),
                            values[i],
                            signs[i],
                            scalar[i],
                            libm::$scalar(x)
                        );
                    }
                }
            }
        }};
    }
    check_lgamma_r_width!(f32, gamma_f32, lgammaf_r, lgammaf, 5e-5);
    check_lgamma_r_width!(f64, gamma_f64, lgamma_r, lgamma, 2e-11);

    macro_rules! check_minmax_width {
        ($ty:ty, $kernel:ident, $reference:ident) => {{
            let values: [$ty; 12] = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                <$ty>::INFINITY,
                <$ty>::NEG_INFINITY,
                <$ty>::NAN,
                -<$ty>::NAN,
                <$ty>::MIN_POSITIVE,
                -<$ty>::MIN_POSITIVE,
                <$ty>::from_bits(1),
                -<$ty>::from_bits(1),
            ];
            let mut cases: Vec<($ty, $ty, u32)> = Vec::new();
            for op in 0..4 {
                for &x in &values {
                    for &y in &values {
                        cases.push((x, y, op));
                    }
                }
            }
            cases.extend([(3.5, -1.25, 4), (3.5, -1.25, 5), (3.5, -1.25, 6)]);
            let xs: Vec<$ty> = cases.iter().map(|case| case.0).collect();
            let ys: Vec<$ty> = cases.iter().map(|case| case.1).collect();
            let ops: Vec<u32> = cases.iter().map(|case| case.2).collect();
            let count = cases.len();
            let input_x = DeviceBuffer::from_host(&stream, &xs)?;
            let input_y = DeviceBuffer::from_host(&stream, &ys)?;
            let input_ops = DeviceBuffer::from_host(&stream, &ops)?;
            let mut output = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            // SAFETY: each thread writes one distinct in-bounds output element.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(count as u32),
                    &input_x,
                    &input_y,
                    &input_ops,
                    &mut output,
                )?;
            }
            let output = output.to_host_vec(&stream)?;
            for (i, &(x, y, op)) in cases.iter().enumerate() {
                let expected = $reference(op, x, y);
                let got = output[i];
                let valid = if expected.is_nan() {
                    got.is_nan()
                } else {
                    got.to_bits() == expected.to_bits()
                };
                checks += 1;
                if !valid {
                    failures += 1;
                    eprintln!(
                        "{} minmax case={i} op={op} x={x:?} y={y:?} got={got:?} ({:x}) expected={expected:?} ({:x})",
                        stringify!($ty),
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }};
    }
    check_minmax_width!(f32, minmax_f32, minmax_value_f32);
    check_minmax_width!(f64, minmax_f64, minmax_value_f64);

    macro_rules! check_bessel_j1_width {
        ($ty:ty, $kernel:ident, $function:ident, $control:expr, $tolerance:expr) => {{
            let xs: Vec<$ty> = vec![
                0.0,
                -0.0,
                0.125,
                -0.125,
                0.5,
                -0.5,
                1.0,
                -1.0,
                1.5,
                -1.5,
                2.0,
                -2.0,
                4.0,
                -4.0,
                8.0,
                -8.0,
                <$ty>::MIN_POSITIVE,
                -<$ty>::MIN_POSITIVE,
                <$ty>::INFINITY,
                <$ty>::NEG_INFINITY,
                <$ty>::NAN,
            ];
            let len = xs.len();
            let input = DeviceBuffer::from_host(&stream, &xs)?;
            let mut values = DeviceBuffer::<$ty>::zeroed(&stream, len)?;
            let mut controls = DeviceBuffer::<$ty>::zeroed(&stream, len)?;
            // SAFETY: every thread writes at most one in-bounds element.
            unsafe {
                module.$kernel(
                    &stream,
                    LaunchConfig::for_num_elems(len as u32),
                    &input,
                    &mut values,
                    &mut controls,
                )?;
            }
            let values = values.to_host_vec(&stream)?;
            let controls = controls.to_host_vec(&stream)?;
            for (i, &x) in xs.iter().enumerate() {
                let expected = libm::$function(x);
                let control = ($control)(x);
                for (field, good) in [
                    close_scaled(values[i] as f64, expected as f64, $tolerance, true),
                    close_scaled(controls[i] as f64, control as f64, $tolerance, true),
                ]
                .into_iter()
                .enumerate()
                {
                    checks += 1;
                    if !good {
                        failures += 1;
                        eprintln!(
                            "{} bessel_j1 i={i} field={field} x={x:?}: got=({:?}, {:?}) expected=({expected:?}, {control:?})",
                            stringify!($ty),
                            values[i],
                            controls[i]
                        );
                    }
                }
            }
        }};
    }
    check_bessel_j1_width!(f32, bessel_f32, j1f, libm::j0f, 2e-4);
    check_bessel_j1_width!(f64, bessel_f64, j1, libm_lookalike::j1, 2e-9);

    if failures != 0 {
        return Err(format!("FAILED: {failures} of {checks} checks failed").into());
    }
    println!("SUCCESS: math_special passed {checks} checks across f32 and f64");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify host reference helpers for new `libm` alias families and controls.
    #[test]
    fn host_oracle_covers_new_families_and_controls() {
        assert_eq!(
            alias_value_f32(9, 0.0, -1.0).to_bits(),
            (-f32::from_bits(1)).to_bits()
        );
        assert_eq!(alias_value_f64(10, 7.0, 2.0), -1.0);
        assert!(alias_value_f64(5, -1.0, 0.0).is_nan());
        assert_eq!(alias_value_f32(11, 2.0, 0.0), 1002.0);
        assert_eq!(
            alias_value_f64(12, -0.0, 0.0).to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    /// Check special-value and tolerance behavior of `close_alias`.
    #[test]
    fn comparison_checks_special_values_and_accuracy() {
        assert!(close_alias(f64::NAN, f64::NAN, 0.0, 0.0, true));
        assert!(!close_alias(1.0, f64::NAN, 1.0, 1.0, false));
        assert!(!close_alias(0.0, -0.0, 1.0, 1.0, false));
        assert!(!close_alias(1.0, f64::INFINITY, 1.0, 1.0, false));
        assert!(!close_alias(1.0, 1.0 + 1e-3, 1e-6, 1e-6, false));
    }
}

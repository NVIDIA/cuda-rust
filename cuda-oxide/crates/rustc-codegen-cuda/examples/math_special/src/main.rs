/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! tgamma, ilogb, ldexp and scalbn in f32/f64 device code.
//! Run with `cargo oxide run math_special`.
#![feature(float_gamma)]
// The width-generic host macro casts both f32 and f64 results to f64.
#![allow(clippy::too_many_arguments, clippy::unnecessary_cast)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// An unrelated function with a libm-like path must retain its own body.
mod libm_lookalike {
    #[inline(never)]
    pub fn tgammaf(x: f32) -> f32 {
        x + 1000.0
    }

    #[inline(never)]
    pub fn tgamma(x: f64) -> f64 {
        x + 1000.0
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
}

fn close(got: f64, expected: f64, rtol: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if expected.is_infinite() || expected == 0.0 {
        got.to_bits() == expected.to_bits()
    } else {
        got.is_finite() && (got - expected).abs() <= rtol * expected.abs()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut failures = 0usize;
    let mut checks = 0usize;

    macro_rules! check_width {
        ($ty:ty, $kernel:ident, $gamma:ident, $ilogb:ident, $ldexp:ident, $rtol:expr, $extra:expr) => {{
            // Nonzero finite inputs keep ilogb away from C's domain-error
            // sentinels. Both gamma signs and both exponent signs are covered.
            let mut xs: Vec<$ty> = vec![
                -3.5, -2.5, -1.25, -0.5, 0.125, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0,
                0.33,
            ];
            let mut exponents = vec![-12, -4, -1, 0, 1, 4, 8, 24, -24, 100, -100, 3, 2, -3, 5, -5];
            // Width-specific subnormal inputs and scaling overflow/underflow.
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
    if failures != 0 {
        return Err(format!("FAILED: {failures} of {checks} checks failed").into());
    }
    println!("SUCCESS: math_special passed {checks} checks across f32 and f64");
    Ok(())
}

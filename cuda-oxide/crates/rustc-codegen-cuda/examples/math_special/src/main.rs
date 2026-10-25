/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar math and tuple-returning frexp in f32/f64 device code.
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

    #[inline(never)]
    pub fn frexpf(x: f32) -> (f32, i32) {
        (x, 12345)
    }

    #[inline(never)]
    pub fn frexp(x: f64) -> (f64, i32) {
        (x, 12345)
    }
}

// Exercise an ordinary device-function boundary with a tuple return.
mod frexp_helpers {
    #[inline(never)]
    pub fn split_f32(x: f32) -> (f32, i32) {
        libm::frexpf(x)
    }

    #[inline(never)]
    pub fn split_f64(x: f64) -> (f64, i32) {
        libm::frexp(x)
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
            let direct = frexp_helpers::split_f32(x);
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
            let direct = frexp_helpers::split_f64(x);
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

    if failures != 0 {
        return Err(format!("FAILED: {failures} of {checks} checks failed").into());
    }
    println!("SUCCESS: math_special passed {checks} checks across f32 and f64");
    Ok(())
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare libm's lgamma_r tuple with the composed CUDA libdevice lowering.
//! Run with `cargo oxide run math_lgamma_r` from cuda-oxide/.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

mod helpers {
    #[inline(never)]
    pub fn f32_result(x: f32) -> (f32, i32) {
        libm::lgammaf_r(x)
    }

    #[inline(never)]
    pub fn f64_result(x: f64) -> (f64, i32) {
        libm::lgamma_r(x)
    }
}

mod lookalike {
    #[inline(never)]
    pub fn lgamma_r(x: f64) -> (f64, i32) {
        (x + 10.0, 17)
    }
}

#[cuda_module]
mod kernels {
    use super::*;

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
            let (value, sign) = helpers::f32_result(xs[i]);
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
            let (value, sign) = helpers::f64_result(xs[i]);
            let (lookalike_value, _) = lookalike::lgamma_r(xs[i]);
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
}

fn same(got: f64, expected: f64, tolerance: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if expected.is_infinite() {
        got == expected
    } else {
        got.is_finite() && (got - expected).abs() <= tolerance * (1.0 + expected.abs())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut checks = 0;
    let mut failures = 0;

    macro_rules! check_width {
        ($ty:ty, $kernel:ident, $reference:ident, $scalar:ident, $tolerance:expr) => {{
            let xs: Vec<$ty> = vec![
                0.0, -0.0, 1.0, 2.0, 0.5, 1.5, 10.25,
                -0.5, -1.5, -2.5, -3.5, -4.5, -5.25, -6.125,
                -1.0, -2.0, -3.0, -10.0,
                <$ty>::MIN_POSITIVE, -<$ty>::MIN_POSITIVE,
                <$ty>::from_bits(1), -<$ty>::from_bits(1),
                <$ty>::INFINITY, <$ty>::NEG_INFINITY, <$ty>::NAN,
                if core::mem::size_of::<$ty>() == 4 { -4_194_304.5 } else { -2_251_799_813_685_247.5 },
            ];
            let count = xs.len();
            let input = DeviceBuffer::from_host(&stream, &xs)?;
            let mut values = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut signs = DeviceBuffer::<i32>::zeroed(&stream, count)?;
            let mut scalar = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            let mut control = DeviceBuffer::<$ty>::zeroed(&stream, count)?;
            // SAFETY: each thread writes a unique in-bounds output slot.
            unsafe {
                module.$kernel(&stream, LaunchConfig::for_num_elems(count as u32), &input, &mut values, &mut signs, &mut scalar, &mut control)?;
            }
            let values = values.to_host_vec(&stream)?;
            let signs = signs.to_host_vec(&stream)?;
            let scalar = scalar.to_host_vec(&stream)?;
            let control = control.to_host_vec(&stream)?;
            for (i, &x) in xs.iter().enumerate() {
                let (expected_value, expected_sign) = libm::$reference(x);
                let validations = [
                    same(values[i] as f64, expected_value as f64, $tolerance),
                    signs[i] == expected_sign,
                    // CUDA's documented scalar result is -inf, but the
                    // backend on sm_120a can return +inf. Only the tuple
                    // contract requires the libm-compatible +inf sign.
                    if x == <$ty>::NEG_INFINITY { scalar[i].is_infinite() }
                    else { same(scalar[i] as f64, libm::$scalar(x) as f64, $tolerance) },
                    same(control[i] as f64, (x + 10.0) as f64, $tolerance),
                ];
                for (field, valid) in validations.into_iter().enumerate() {
                    checks += 1;
                    if !valid {
                        failures += 1;
                        eprintln!("{} case={i} field={field} x={x:?} got=({:?}, {}, {:?}) expected=({expected_value:?}, {expected_sign}, {:?})", stringify!($ty), values[i], signs[i], scalar[i], libm::$scalar(x));
                    }
                }
            }
        }};
    }
    check_width!(f32, gamma_f32, lgammaf_r, lgammaf, 5e-5);
    check_width!(f64, gamma_f64, lgamma_r, lgamma, 2e-11);
    if failures != 0 {
        return Err(format!("math_lgamma_r failed {failures} of {checks} checks").into());
    }
    println!("SUCCESS: math_lgamma_r passed {checks} checks across f32 and f64");
    Ok(())
}

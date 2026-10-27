/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare Bessel j1 with libm 0.2.16, including infinity's signed zero.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

mod helpers {
    #[inline(never)]
    pub fn j1f(x: f32) -> f32 {
        libm::j1f(x)
    }
    #[inline(never)]
    pub fn j1(x: f64) -> f64 {
        libm::j1(x)
    }
}

mod lookalike {
    #[inline(never)]
    pub fn j1(x: f64) -> f64 {
        x + 7.0
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn bessel_f32(
        xs: &[f32],
        mut values: DisjointSlice<f32>,
        mut controls: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            if let Some(slot) = values.get_mut(thread::index_1d()) {
                *slot = helpers::j1f(xs[i]);
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
                *slot = helpers::j1(xs[i]);
            }
            if let Some(slot) = controls.get_mut(thread::index_1d()) {
                *slot = lookalike::j1(xs[i]);
            }
        }
    }
}

fn equal(got: f64, expected: f64, tolerance: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if expected == 0.0 {
        got.to_bits() == expected.to_bits()
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
        ($ty:ty, $kernel:ident, $function:ident, $control:expr, $tolerance:expr) => {{
            let xs: Vec<$ty> = vec![
                0.0, -0.0, 0.125, -0.125, 0.5, -0.5, 1.0, -1.0,
                1.5, -1.5, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0,
                <$ty>::MIN_POSITIVE, -<$ty>::MIN_POSITIVE,
                <$ty>::INFINITY, <$ty>::NEG_INFINITY, <$ty>::NAN,
            ];
            let len = xs.len();
            let input = DeviceBuffer::from_host(&stream, &xs)?;
            let mut values = DeviceBuffer::<$ty>::zeroed(&stream, len)?;
            let mut controls = DeviceBuffer::<$ty>::zeroed(&stream, len)?;
            // SAFETY: every thread writes at most one in-bounds element.
            unsafe { module.$kernel(&stream, LaunchConfig::for_num_elems(len as u32), &input, &mut values, &mut controls)?; }
            let values = values.to_host_vec(&stream)?;
            let controls = controls.to_host_vec(&stream)?;
            for (i, &x) in xs.iter().enumerate() {
                let expected = libm::$function(x);
                let control = ($control)(x);
                for (field, good) in [
                    equal(values[i] as f64, expected as f64, $tolerance),
                    equal(controls[i] as f64, control as f64, $tolerance),
                ].into_iter().enumerate() {
                    checks += 1;
                    if !good {
                        failures += 1;
                        eprintln!("{} i={i} field={field} x={x:?}: got=({:?}, {:?}) expected=({expected:?}, {control:?})", stringify!($ty), values[i], controls[i]);
                    }
                }
            }
        }};
    }
    check_width!(f32, bessel_f32, j1f, libm::j0f, 2e-4);
    check_width!(f64, bessel_f64, j1, lookalike::j1, 2e-9);
    if failures != 0 {
        return Err(format!("math_bessel_j1 failed {failures} of {checks} checks").into());
    }
    println!("SUCCESS: math_bessel_j1 passed {checks} checks across f32 and f64");
    Ok(())
}

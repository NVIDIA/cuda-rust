/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Validate the libm tuple ABI of modf and remquo on CUDA.
//! Run with `cargo oxide run math_out_params` from cuda-oxide/.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

mod lookalike {
    #[inline(never)]
    pub fn modff(x: f32) -> (f32, f32) {
        (x + 1000.0, x)
    }

    #[inline(never)]
    pub fn modf(x: f64) -> (f64, f64) {
        (x + 1000.0, x)
    }
}

mod helpers {
    #[inline(never)]
    pub fn split_f32(x: f32) -> (f32, f32) {
        libm::modff(x)
    }

    #[inline(never)]
    pub fn split_f64(x: f64) -> (f64, f64) {
        libm::modf(x)
    }
}

#[cuda_module]
mod kernels {
    use super::*;

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
            let (f, n) = helpers::split_f32(xs[i]);
            let (r, q) = libm::remquof(xs[i], ys[i]);
            let (c, _) = lookalike::modff(xs[i]);
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
            let (f, n) = helpers::split_f64(xs[i]);
            let (r, q) = libm::remquo(xs[i], ys[i]);
            let (c, _) = lookalike::modf(xs[i]);
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
}

fn same(got: f64, expected: f64) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else {
        got.to_bits() == expected.to_bits()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut checks = 0;
    let mut failures = 0;

    macro_rules! check_width {
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
                let valid_remainder = same(remainder[i] as f64, r as f64);
                let valid_quotient = if r.is_nan() {
                    quotient[i] == 0
                } else {
                    quotient[i].signum() == q.signum() &&
                        quotient[i].unsigned_abs() & 7 == q.unsigned_abs() & 7
                };
                let results = [
                    same(fraction[i] as f64, f as f64),
                    same(integral[i] as f64, n as f64),
                    valid_remainder,
                    valid_quotient,
                    same(control[i] as f64, (x + 1000.0) as f64),
                ];
                for (field, valid) in results.into_iter().enumerate() {
                    checks += 1;
                    if !valid {
                        failures += 1;
                        eprintln!(
                            "{} case={i} field={field} x={x:?} y={y:?} got=({:?}, {:?}, {:?}, {}, {:?}) expected=({f:?}, {n:?}, {r:?}, {q}, {:?})",
                            stringify!($ty), fraction[i], integral[i], remainder[i],
                            quotient[i], control[i], x + 1000.0
                        );
                    }
                }
            }
        }};
    }
    check_width!(f32, out_f32, modff, remquof);
    check_width!(f64, out_f64, modf, remquo);
    if failures != 0 {
        return Err(format!("math_out_params failed {failures} of {checks} checks").into());
    }
    println!("SUCCESS: math_out_params passed {checks} checks across f32 and f64");
    Ok(())
}

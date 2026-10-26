/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare IEEE-754-2019 min/max against libm on CUDA.
//! Run with `cargo oxide run math_c23_minmax` from cuda-oxide/.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

mod lookalike {
    #[inline(never)]
    pub fn fmaximumf(x: f32, y: f32) -> f32 {
        x + y + 1000.0
    }

    #[inline(never)]
    pub fn fmaximum(x: f64, y: f64) -> f64 {
        x + y + 1000.0
    }
}

#[inline(always)]
fn value_f32(op: u32, x: f32, y: f32) -> f32 {
    match op {
        0 => libm::fmaximumf(x, y),
        1 => libm::fminimumf(x, y),
        2 => libm::fmaximum_numf(x, y),
        3 => libm::fminimum_numf(x, y),
        4 => libm::fmaxf(x, y),
        5 => libm::fminf(x, y),
        6 => lookalike::fmaximumf(x, y),
        _ => x,
    }
}

#[inline(always)]
fn value_f64(op: u32, x: f64, y: f64) -> f64 {
    match op {
        0 => libm::fmaximum(x, y),
        1 => libm::fminimum(x, y),
        2 => libm::fmaximum_num(x, y),
        3 => libm::fminimum_num(x, y),
        4 => libm::fmax(x, y),
        5 => libm::fmin(x, y),
        6 => lookalike::fmaximum(x, y),
        _ => x,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn minmax_f32(xs: &[f32], ys: &[f32], ops: &[u32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            if let Some(slot) = out.get_mut(thread::index_1d()) {
                *slot = value_f32(ops[i], xs[i], ys[i]);
            }
        }
    }

    #[kernel]
    pub fn minmax_f64(xs: &[f64], ys: &[f64], ops: &[u32], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            if let Some(slot) = out.get_mut(thread::index_1d()) {
                *slot = value_f64(ops[i], xs[i], ys[i]);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut checks = 0;
    let mut failures = 0;

    macro_rules! check_width {
        ($ty:ty, $kernel:ident, $reference:ident) => {{
            let values: [$ty; 12] = [
                0.0, -0.0, 1.0, -1.0, <$ty>::INFINITY, <$ty>::NEG_INFINITY,
                <$ty>::NAN, -<$ty>::NAN, <$ty>::MIN_POSITIVE,
                -<$ty>::MIN_POSITIVE, <$ty>::from_bits(1), -<$ty>::from_bits(1),
            ];
            let mut cases: Vec<($ty, $ty, u32)> = Vec::new();
            for op in 0..4 {
                for &x in &values {
                    for &y in &values {
                        cases.push((x, y, op));
                    }
                }
            }
            // Existing fmax/fmin routes and an unrelated namesake are controls.
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
                    &stream, LaunchConfig::for_num_elems(count as u32),
                    &input_x, &input_y, &input_ops, &mut output,
                )?;
            }
            let output = output.to_host_vec(&stream)?;
            for (i, &(x, y, op)) in cases.iter().enumerate() {
                let expected = $reference(op, x, y);
                let got = output[i];
                let valid = if expected.is_nan() { got.is_nan() }
                    else { got.to_bits() == expected.to_bits() };
                checks += 1;
                if !valid {
                    failures += 1;
                    eprintln!("{} case={i} op={op} x={x:?} y={y:?} got={got:?} ({:x}) expected={expected:?} ({:x})", stringify!($ty), got.to_bits(), expected.to_bits());
                }
            }
        }};
    }
    check_width!(f32, minmax_f32, value_f32);
    check_width!(f64, minmax_f64, value_f64);
    if failures != 0 {
        return Err(format!("math_c23_minmax failed {failures} of {checks} checks").into());
    }
    println!("SUCCESS: math_c23_minmax passed {checks} checks across f32 and f64");
    Ok(())
}

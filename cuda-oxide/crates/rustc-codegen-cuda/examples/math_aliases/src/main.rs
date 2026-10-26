/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exercises the 22 new libm aliases on both floating-point widths.
//! Run with `cargo oxide run math_aliases` from cuda-oxide/.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// These functions are used on the CPU as libm references and in the kernels.
// The device compiler alone replaces libm calls with their libdevice symbols.
#[inline(always)]
fn value_f32(op: u32, x: f32, y: f32) -> f32 {
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
        11 => x + 1000.0,      // unrelated function with a libm-like name
        12 => libm::floorf(x), // existing generated route
        _ => x,
    }
}

#[inline(always)]
fn value_f64(op: u32, x: f64, y: f64) -> f64 {
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

mod libm_lookalike {
    #[inline(never)]
    pub fn erff(x: f32) -> f32 {
        x + 1000.0
    }

    #[inline(never)]
    pub fn erf(x: f64) -> f64 {
        x + 1000.0
    }
}

#[cuda_module]
mod kernels {
    use super::*;

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
                value_f32(operations[i], xs[i], ys[i])
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
                value_f64(operations[i], xs[i], ys[i])
            };
            if let Some(out) = outputs.get_mut(thread::index_1d()) {
                *out = value;
            }
        }
    }
}

fn close(got: f64, expected: f64, rtol: f64, atol: f64, exact: bool) -> bool {
    if expected.is_nan() {
        got.is_nan()
    } else if exact || expected.is_infinite() || expected == 0.0 {
        got.to_bits() == expected.to_bits()
    } else {
        got.is_finite() && (got - expected).abs() <= atol + rtol * expected.abs()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut checks = 0;
    let mut failures = 0;

    macro_rules! check_width {
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
            // SAFETY: all input and output buffers have count elements, each
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
                let valid = close(outputs[i] as f64, expected as f64, $rtol, $atol, exact);
                checks += 1;
                if !valid {
                    failures += 1;
                    eprintln!(
                        "{} op={op} case={} x={:?} y={:?} got={:?} expected={:?}",
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
    check_width!(f32, aliases_f32, value_f32, 4e-5, 4e-7);
    check_width!(f64, aliases_f64, value_f64, 2e-11, 2e-14);
    if failures != 0 {
        return Err(format!("math_aliases failed {failures} of {checks} checks").into());
    }
    println!("SUCCESS: math_aliases passed {checks} checks across f32 and f64");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_oracle_covers_new_families_and_controls() {
        assert_eq!(
            value_f32(9, 0.0, -1.0).to_bits(),
            (-f32::from_bits(1)).to_bits()
        );
        assert_eq!(value_f64(10, 7.0, 2.0), -1.0);
        assert!(value_f64(5, -1.0, 0.0).is_nan());
        assert_eq!(value_f32(11, 2.0, 0.0), 1002.0);
        assert_eq!(value_f64(12, -0.0, 0.0).to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn comparison_checks_special_values_and_accuracy() {
        assert!(close(f64::NAN, f64::NAN, 0.0, 0.0, true));
        assert!(!close(1.0, f64::NAN, 1.0, 1.0, false));
        assert!(!close(0.0, -0.0, 1.0, 1.0, false));
        assert!(!close(1.0, f64::INFINITY, 1.0, 1.0, false));
        assert!(!close(1.0, 1.0 + 1e-3, 1e-6, 1e-6, false));
    }
}

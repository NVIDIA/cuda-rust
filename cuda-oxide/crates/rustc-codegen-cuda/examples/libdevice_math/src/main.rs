/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA libdevice math through the normal `#[cuda_module]` API.
//!
//! Calling these math methods automatically selects the NVVM IR path. The
//! example checks SwiGLU plus `sin`, `exp`, `sqrt`, `atan`, `atan2`, `acos`,
//! and `tan` on both legacy and modern NVVM targets.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn swiglu_libdevice(x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let xi = x[i];
            let yi = y[i];
            let sigmoid = 1.0f32 / (1.0f32 + (-xi).exp());
            *out_elem = xi * sigmoid * yi;
        }
    }

    /// One output row per input, ordered like `OP_NAMES` on the host.
    #[kernel]
    pub fn math_functions(x: &[f32], atan2_y: &[f32], mut out: DisjointSlice<[f32; 7]>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len()
            && i < atan2_y.len()
            && let Some(row) = out.get_mut(idx)
        {
            let xi = x[i];
            row[0] = xi.sin();
            row[1] = xi.exp();
            row[2] = (xi + 1.25).sqrt();
            row[3] = xi.atan();
            row[4] = atan2_y[i].atan2(xi);
            row[5] = xi.acos();
            row[6] = xi.tan();
        }
    }
}

const OP_NAMES: [&str; 7] = ["sin", "exp", "sqrt", "atan", "atan2", "acos", "tan"];

fn ulp_distance(a: f32, b: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            0x8000_0000 - (bits & 0x7fff_ffff)
        } else {
            0x8000_0000 + bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let host_x = vec![-2.0f32, -0.5, 0.0, 2.0];
    let host_y = vec![1.0f32, 0.5, -1.0, 3.0];
    let dev_x = DeviceBuffer::from_host(&stream, &host_x)?;
    let dev_y = DeviceBuffer::from_host(&stream, &host_y)?;
    let mut dev_out = DeviceBuffer::from_host(&stream, &[0.0f32; 4])?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.swiglu_libdevice(
            &stream,
            LaunchConfig::for_num_elems(4),
            &dev_x,
            &dev_y,
            &mut dev_out,
        )
    }?;

    let actual = dev_out.to_host_vec(&stream)?;
    let expected: Vec<f32> = host_x
        .iter()
        .zip(host_y.iter())
        .map(|(&x, &y)| {
            let s = 1.0f32 / (1.0f32 + (-x).exp());
            x * s * y
        })
        .collect();
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "SwiGLU mismatch at {i}: GPU={a:e}, CPU={e:e}"
        );
    }

    // Inputs stay inside acos' domain, away from tan's asymptotes, and keep
    // the translated sqrt operand positive. That makes a direct CPU/libm
    // comparison meaningful for every operation.
    let math_x = vec![-1.0f32, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    let atan2_y = vec![-1.0f32, 0.5, 1.0, -0.75, 1.0, 0.25, -1.0, 1.0, -0.5];
    let math_x_dev = DeviceBuffer::from_host(&stream, &math_x)?;
    let atan2_y_dev = DeviceBuffer::from_host(&stream, &atan2_y)?;
    let mut math_out = DeviceBuffer::<[f32; 7]>::zeroed(&stream, math_x.len())?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.math_functions(
            &stream,
            LaunchConfig::for_num_elems(math_x.len() as u32),
            &math_x_dev,
            &atan2_y_dev,
            &mut math_out,
        )
    }?;

    let actual = math_out.to_host_vec(&stream)?;
    const ULP_LIMIT: u32 = 4;
    let mut failures = 0usize;
    for (i, (&x, &y)) in math_x.iter().zip(atan2_y.iter()).enumerate() {
        let expected = [
            x.sin(),
            x.exp(),
            (x + 1.25).sqrt(),
            x.atan(),
            y.atan2(x),
            x.acos(),
            x.tan(),
        ];
        for op in 0..OP_NAMES.len() {
            let distance = ulp_distance(actual[i][op], expected[op]);
            if !actual[i][op].is_finite() || distance > ULP_LIMIT {
                eprintln!(
                    "{} mismatch at input {i} (x={x}, y={y}): GPU={:e}, CPU={:e}, ULP={distance}",
                    OP_NAMES[op], actual[i][op], expected[op]
                );
                failures += 1;
            }
        }
    }

    if failures != 0 {
        return Err(format!("{failures} math result(s) exceeded {ULP_LIMIT} ULP").into());
    }

    println!(
        "PASS: SwiGLU plus {} inputs x {} libdevice math operations",
        math_x.len(),
        OP_NAMES.len()
    );
    Ok(())
}

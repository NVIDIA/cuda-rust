/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for float `%` in device code.
//!
//! Rust defines `a % b` on floats as C `fmod`: the result is exact for every
//! finite pair, `a - n * b` for the truncated integer quotient `n`, with the
//! sign of `a`. Lowering `%` to LLVM `frem` is not enough on NVPTX, whose
//! backend expands `frem` to `a - trunc(a / b) * b` in floating point. Once
//! `a / b` is too large to be an exact integer in the operand width, the
//! expansion drifts by whole multiples of `b`: `1e20_f64 % 7.0` came back as
//! `-6144.0` instead of `2.0`.
//!
//! Each kernel computes the same expression the host computes, and the host
//! compares the bit patterns (NaN matches NaN). `%=` and `rem_euclid` are
//! covered because both reach the same remainder lowering.
//!
//! Run:
//!     cargo oxide run float_rem_smoke
//!
//! Exits 0 on PASS, 1 on FAIL.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, ltoir};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn rem_f32(xs: &[f32], ys: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            let mut r = xs[i];
            r %= ys[i];
            *slot = r;
        }
    }

    #[kernel]
    pub fn rem_f64(xs: &[f64], ys: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i] % ys[i];
        }
    }

    #[kernel]
    pub fn rem_euclid_f32(xs: &[f32], ys: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].rem_euclid(ys[i]);
        }
    }

    #[kernel]
    pub fn rem_euclid_f64(xs: &[f64], ys: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].rem_euclid(ys[i]);
        }
    }
}

/// Operand pairs. The host-side cast to `f32` keeps the two widths on the
/// same table; the f32 cases are still large-quotient cases after rounding.
fn operands_f64() -> Vec<(f64, f64)> {
    vec![
        // Large quotients: trunc(a / b) is not an exact integer anymore.
        (1e20, 7.0),
        (-1e20, 7.0),
        (1e20, -3.0),
        (-1e20, -3.0),
        (3e17, 0.7),
        (f64::from(0xCAFE_BABE_u32), 3.0),
        (f64::MAX, 3.0),
        (f64::MAX, f64::MIN_POSITIVE),
        (1e300, 1e-300),
        // Small quotients, where the naive expansion also happens to work.
        (5.5, 2.0),
        (-5.5, 2.0),
        (5.5, -2.0),
        (-5.5, -2.0),
        (0.0, 3.0),
        (-0.0, 3.0),
        (2.0, 8.0),
        (-2.0, 8.0),
        (7.0, 7.0),
        (-7.0, 7.0),
        // Subnormal divisor, infinite and NaN edges.
        (1.0, 5e-324),
        (1.0, f64::INFINITY),
        (-1.0, f64::INFINITY),
        (f64::INFINITY, 2.0),
        (2.0, 0.0),
        (f64::NAN, 2.0),
        (2.0, f64::NAN),
    ]
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== float_rem_smoke: f32/f64 % against host fmod ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ltoir::load_kernel_module(&ctx, "float_rem_smoke")?;
    let module = kernels::from_module(module)?;

    let pairs = operands_f64();
    let xs_f64: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys_f64: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let xs_f32: Vec<f32> = xs_f64.iter().map(|&v| v as f32).collect();
    let ys_f32: Vec<f32> = ys_f64.iter().map(|&v| v as f32).collect();
    let n = pairs.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let ys32 = DeviceBuffer::from_host(&stream, &ys_f32)?;
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;
    let ys64 = DeviceBuffer::from_host(&stream, &ys_f64)?;

    let mut out_rem_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_rem_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;
    let mut out_euclid_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_euclid_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rem_f32(&stream, cfg, &xs32, &ys32, &mut out_rem_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rem_f64(&stream, cfg, &xs64, &ys64, &mut out_rem_f64) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rem_euclid_f32(&stream, cfg, &xs32, &ys32, &mut out_euclid_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rem_euclid_f64(&stream, cfg, &xs64, &ys64, &mut out_euclid_f64) }?;

    let got_rem_f32 = out_rem_f32.to_host_vec(&stream)?;
    let got_rem_f64 = out_rem_f64.to_host_vec(&stream)?;
    let got_euclid_f32 = out_euclid_f32.to_host_vec(&stream)?;
    let got_euclid_f64 = out_euclid_f64.to_host_vec(&stream)?;

    let mut failures = 0usize;
    for i in 0..n {
        let exp_rem_f32 = xs_f32[i] % ys_f32[i];
        let exp_rem_f64 = xs_f64[i] % ys_f64[i];
        let exp_euclid_f32 = xs_f32[i].rem_euclid(ys_f32[i]);
        let exp_euclid_f64 = xs_f64[i].rem_euclid(ys_f64[i]);
        let checks: [(&str, bool, String); 4] = [
            (
                "rem_f32",
                same_f32(got_rem_f32[i], exp_rem_f32),
                format!("gpu={:e} cpu={:e}", got_rem_f32[i], exp_rem_f32),
            ),
            (
                "rem_f64",
                same_f64(got_rem_f64[i], exp_rem_f64),
                format!("gpu={:e} cpu={:e}", got_rem_f64[i], exp_rem_f64),
            ),
            (
                "rem_euclid_f32",
                same_f32(got_euclid_f32[i], exp_euclid_f32),
                format!("gpu={:e} cpu={:e}", got_euclid_f32[i], exp_euclid_f32),
            ),
            (
                "rem_euclid_f64",
                same_f64(got_euclid_f64[i], exp_euclid_f64),
                format!("gpu={:e} cpu={:e}", got_euclid_f64[i], exp_euclid_f64),
            ),
        ];
        for (name, ok, detail) in checks {
            if !ok {
                failures += 1;
                eprintln!(
                    "[{i}] {name}: x={:e} y={:e} | {detail}",
                    xs_f64[i], ys_f64[i]
                );
            }
        }
    }

    for &i in &[0usize, 4, 5] {
        println!(
            "[{i}] x={:e} y={:e}  rem_f64 gpu={} cpu={}  rem_f32 gpu={} cpu={}",
            xs_f64[i],
            ys_f64[i],
            got_rem_f64[i],
            xs_f64[i] % ys_f64[i],
            got_rem_f32[i],
            xs_f32[i] % ys_f32[i],
        );
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} operand pairs x 4 kernels match host fmod bit for bit");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} mismatches across {n} operand pairs x 4 kernels");
        std::process::exit(1);
    }
}

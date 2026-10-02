/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates the `f32::atan{,2}` / `f64::atan{,2}` → libdevice lowering.
//!
//! Each `.atan()` / `.atan2()` call site lowers to a `__nv_atan{,2}{,f}`
//! libdevice call. The host computes the same expression with stdlib
//! `f{32,64}::atan{,2}` and compares within a 2-ULP tolerance (matching
//! the bound `primitive_stress` uses for the other transcendentals).
//!
//! Run:
//!     cargo oxide run math_atan
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
    pub fn atan2_f32(ys: &[f32], xs: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = ys[i].atan2(xs[i]);
        }
    }

    #[kernel]
    pub fn atan_f32(ys: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = ys[i].atan();
        }
    }

    #[kernel]
    pub fn atan2_f64(ys: &[f64], xs: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = ys[i].atan2(xs[i]);
        }
    }

    #[kernel]
    pub fn atan_f64(ys: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < ys.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = ys[i].atan();
        }
    }
}

/// IEEE-754 ULP distance for finite operands of a given width.
/// `atan{,2}` of finite real inputs lands in `[-pi, pi]`, so we don't need
/// to handle NaN/Inf or extreme sign-cross edge cases here.
fn ulp_distance(a_bits: u64, b_bits: u64, sign_mask: u64, body_mask: u64) -> u64 {
    let map = |bits: u64| {
        if bits & sign_mask != 0 {
            sign_mask - (bits & body_mask)
        } else {
            sign_mask + (bits & body_mask)
        }
    };
    map(a_bits).abs_diff(map(b_bits))
}

fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    ulp_distance(
        a.to_bits() as u64,
        b.to_bits() as u64,
        0x8000_0000,
        0x7FFF_FFFF,
    )
}

fn ulp_diff_f64(a: f64, b: f64) -> u64 {
    ulp_distance(
        a.to_bits(),
        b.to_bits(),
        0x8000_0000_0000_0000,
        0x7FFF_FFFF_FFFF_FFFF,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== math_atan: f32/f64 atan{{,2}} via libdevice ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls in the kernel force the NVVM-IR output flavor; the
    // first launch builds a cubin via libNVVM + nvJitLink.
    let module = ltoir::load_kernel_module(&ctx, "math_atan")?;
    let module = kernels::from_module(module)?;

    // All four atan2 quadrants plus small / large / mixed-sign magnitudes.
    // Values are f32-representable so the same arrays double as f64 inputs
    // after a widening cast.
    let ys_f32: Vec<f32> = vec![
        0.0, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.25, -3.25, 1e-3, -1e-3, 100.0, -100.0,
        7.7,
    ];
    let xs_f32: Vec<f32> = vec![
        1.0, 1.0, 1.0, -1.0, -1.0, 3.0, 3.0, -2.0, -2.0, 1.0, -1.0, 1e-3, -1e-3, 0.001, -0.001,
        -2.2,
    ];
    let ys_f64: Vec<f64> = ys_f32.iter().map(|&v| v as f64).collect();
    let xs_f64: Vec<f64> = xs_f32.iter().map(|&v| v as f64).collect();
    let n = ys_f32.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    // Single upload per input; the typed launch methods borrow the device
    // buffers, so the same buffer is reused by both `atan` and `atan2`
    // launches of its width.
    let ys32 = DeviceBuffer::from_host(&stream, &ys_f32)?;
    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let ys64 = DeviceBuffer::from_host(&stream, &ys_f64)?;
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;

    let mut out_atan2_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_atan_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_atan2_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;
    let mut out_atan_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.atan2_f32(&stream, cfg, &ys32, &xs32, &mut out_atan2_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.atan_f32(&stream, cfg, &ys32, &mut out_atan_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.atan2_f64(&stream, cfg, &ys64, &xs64, &mut out_atan2_f64) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.atan_f64(&stream, cfg, &ys64, &mut out_atan_f64) }?;

    let got_atan2_f32 = out_atan2_f32.to_host_vec(&stream)?;
    let got_atan_f32 = out_atan_f32.to_host_vec(&stream)?;
    let got_atan2_f64 = out_atan2_f64.to_host_vec(&stream)?;
    let got_atan_f64 = out_atan_f64.to_host_vec(&stream)?;

    // libdevice transcendentals are typically within 1 ULP of host libm;
    // 2 ULP matches the `primitive_stress::test_float_math_intrinsics`
    // bound used for `sin` / `cos` / `log*`.
    const ULP_LIMIT: u64 = 2;

    let mut failures = 0usize;
    for i in 0..n {
        let exp_atan2_f32 = ys_f32[i].atan2(xs_f32[i]);
        let exp_atan_f32 = ys_f32[i].atan();
        let exp_atan2_f64 = ys_f64[i].atan2(xs_f64[i]);
        let exp_atan_f64 = ys_f64[i].atan();

        let d_atan2_f32 = ulp_diff_f32(got_atan2_f32[i], exp_atan2_f32);
        let d_atan_f32 = ulp_diff_f32(got_atan_f32[i], exp_atan_f32);
        let d_atan2_f64 = ulp_diff_f64(got_atan2_f64[i], exp_atan2_f64);
        let d_atan_f64 = ulp_diff_f64(got_atan_f64[i], exp_atan_f64);

        let ok = d_atan2_f32 <= ULP_LIMIT
            && d_atan_f32 <= ULP_LIMIT
            && d_atan2_f64 <= ULP_LIMIT
            && d_atan_f64 <= ULP_LIMIT;

        if !ok {
            failures += 1;
            if failures <= 8 {
                eprintln!(
                    "[{i}] y={:>8.4} x={:>8.4} | \
                     atan2_f32 ulp={d_atan2_f32} (gpu={:e} cpu={:e}) | \
                     atan_f32 ulp={d_atan_f32} | \
                     atan2_f64 ulp={d_atan2_f64} | \
                     atan_f64 ulp={d_atan_f64}",
                    ys_f32[i], xs_f32[i], got_atan2_f32[i], exp_atan2_f32,
                );
            }
        }
    }

    // A handful of representative samples printed regardless of pass/fail.
    for &i in &[1usize, 5, 9] {
        println!(
            "[{i}] y={y32:>6.3} x={x32:>6.3}  \
             atan2_f32 gpu={a2f32} cpu={ea2f32}  \
             atan_f32 gpu={af32} cpu={eaf32}",
            y32 = ys_f32[i],
            x32 = xs_f32[i],
            a2f32 = got_atan2_f32[i],
            ea2f32 = ys_f32[i].atan2(xs_f32[i]),
            af32 = got_atan_f32[i],
            eaf32 = ys_f32[i].atan(),
        );
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} cases × 4 variants within {ULP_LIMIT} ULP of host libm");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures}/{n} cases out of tolerance");
        std::process::exit(1);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates the `f32::tan()` / `f64::tan()` -> libdevice lowering.
//!
//! Regression test for a real compiler gap: on current nightlies
//! `core_float_math` moves `sin`/`cos` into `core::intrinsics`, but `tan`
//! is NOT in that feature, so `f{32,64}::tan()` lowers to
//! `std::sys::cmath::tan{,f}`. The float-math dispatch did not intercept
//! that path, so a kernel calling `.tan()` failed with the
//! "FORBIDDEN CRATE IN DEVICE CODE: std::sys::cmath::tan" guard instead of
//! lowering to `__nv_tan{,f}`. Reported on Discord (RTX 3090, sm_86).
//!
//! Each `.tan()` call site now lowers to a `__nv_tan{,f}` libdevice call.
//! The host computes the same expression with stdlib `f{32,64}::tan()` and
//! compares within a 2-ULP tolerance (matching the bound `primitive_stress`
//! uses for the other transcendentals).
//!
//! Run:
//!     cargo oxide run math_tan
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
    pub fn tan_f32(xs: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].tan();
        }
    }

    #[kernel]
    pub fn tan_f64(xs: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].tan();
        }
    }
}

/// IEEE-754 ULP distance for finite operands of a given width.
/// Inputs are kept away from the `±pi/2` asymptotes, so the results are
/// finite and modest in magnitude (no NaN/Inf handling needed).
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
    println!("=== math_tan: f32/f64 tan via libdevice ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls in the kernel force the NVVM-IR output flavor; the
    // first launch builds a cubin via libNVVM + nvJitLink.
    let module = ltoir::load_kernel_module(&ctx, "math_tan")?;
    let module = kernels::from_module(module)?;

    // Angles in radians kept clear of the `±pi/2` asymptotes (|x| <= ~1.3)
    // so `tan` stays finite and the ULP comparison is meaningful. Values are
    // f32-representable so the same array doubles as f64 input after a cast.
    let qpi = std::f32::consts::FRAC_PI_4; // pi/4: tan(pi/4) == 1, a clean check point
    let xs_f32: Vec<f32> = vec![
        0.0, 0.25, -0.25, 0.5, -0.5, qpi, -qpi, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.33, -0.33, 1.3,
    ];
    let xs_f64: Vec<f64> = xs_f32.iter().map(|&v| v as f64).collect();
    let n = xs_f32.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;

    let mut out_tan_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_tan_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.tan_f32(&stream, cfg, &xs32, &mut out_tan_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.tan_f64(&stream, cfg, &xs64, &mut out_tan_f64) }?;

    let got_tan_f32 = out_tan_f32.to_host_vec(&stream)?;
    let got_tan_f64 = out_tan_f64.to_host_vec(&stream)?;

    // libdevice transcendentals are typically within 1 ULP of host libm;
    // 2 ULP matches the `primitive_stress::test_float_math_intrinsics`
    // bound used for `sin` / `cos` / `log*`.
    const ULP_LIMIT: u64 = 2;

    let mut failures = 0usize;
    for i in 0..n {
        let exp_tan_f32 = xs_f32[i].tan();
        let exp_tan_f64 = xs_f64[i].tan();

        let d_tan_f32 = ulp_diff_f32(got_tan_f32[i], exp_tan_f32);
        let d_tan_f64 = ulp_diff_f64(got_tan_f64[i], exp_tan_f64);

        if d_tan_f32 > ULP_LIMIT || d_tan_f64 > ULP_LIMIT {
            failures += 1;
            if failures <= 8 {
                eprintln!(
                    "[{i}] x={:>9.4} | tan_f32 ulp={d_tan_f32} (gpu={:e} cpu={:e}) | \
                     tan_f64 ulp={d_tan_f64} (gpu={:e} cpu={:e})",
                    xs_f32[i], got_tan_f32[i], exp_tan_f32, got_tan_f64[i], exp_tan_f64,
                );
            }
        }
    }

    // A handful of representative samples printed regardless of pass/fail.
    for &i in &[1usize, 5, 7] {
        println!(
            "[{i}] x={x:>8.4}  tan_f32 gpu={gf32} cpu={ef32}  tan_f64 gpu={gf64} cpu={ef64}",
            x = xs_f32[i],
            gf32 = got_tan_f32[i],
            ef32 = xs_f32[i].tan(),
            gf64 = got_tan_f64[i],
            ef64 = xs_f64[i].tan(),
        );
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} cases × 2 variants within {ULP_LIMIT} ULP of host libm");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures}/{n} cases out of tolerance");
        std::process::exit(1);
    }
}

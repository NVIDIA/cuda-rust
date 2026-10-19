/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for `f32::cbrt` / `f64::cbrt` → libdevice lowering.
//!
//! On the pinned toolchain, `f32::cbrt` / `f64::cbrt` inline down to the
//! `core::num::imp::libm::cbrtf` / `cbrt` extern declarations, so that is
//! the call shape device codegen sees in MIR (older toolchains routed the
//! same calls through `std::sys::cmath` shims; the importer recognizes
//! both spellings). Before this example was added, cuda-oxide had no
//! handler for those paths and the calls would fall out of the pipeline
//! as an unresolved symbol. After the matching entries were added to:
//!
//! * `dialect-mir::rust_intrinsics`
//! * `mir-importer::translator::terminator::intrinsics::float_math`
//! * `mir-lower::convert::ops::call`
//! * `rustc-codegen-cuda::collector` (std cmath shim allowlist)
//!
//! the calls lower to libdevice `__nv_cbrtf` / `__nv_cbrt`, which the
//! auto-detected libNVVM + nvJitLink pipeline resolves transparently.
//!
//! The host computes the same expression with stdlib `f{32,64}::cbrt` and
//! compares within a 2-ULP tolerance (matching the bound `math_atan` and
//! `primitive_stress` use for the other libdevice transcendentals). Inputs
//! deliberately span negative, zero, small and large magnitudes: unlike
//! `sqrt`, `cbrt` is defined for negative operands, so the sign-cross cases
//! are the interesting part of the check.
//!
//! Run:
//!     cargo oxide run cbrt_smoke
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
    pub fn cbrt_f32(xs: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].cbrt();
        }
    }

    #[kernel]
    pub fn cbrt_f64(xs: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < xs.len()
            && let Some(slot) = out.get_mut(idx)
        {
            *slot = xs[i].cbrt();
        }
    }
}

/// IEEE-754 ULP distance for finite operands of a given width.
/// `cbrt` of finite real inputs is finite and sign-preserving, so we don't
/// need to handle NaN/Inf here.
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
    println!("=== cbrt_smoke: f32/f64 cbrt via libdevice ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls in the kernel force the NVVM-IR output flavor; the
    // first launch builds a cubin via libNVVM + nvJitLink.
    let module = ltoir::load_kernel_module(&ctx, "cbrt_smoke")?;
    let module = kernels::from_module(module)?;

    // Negative / zero / small / large magnitudes, plus perfect cubes whose
    // root is exactly representable (8 -> 2, 27 -> 3, -64 -> -4, 1000 -> 10).
    // Values are f32-representable so the same array doubles as f64 input
    // after a widening cast.
    let xs_f32: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 27.0, -64.0, 1000.0, 0.125, -0.125, 2.0, -2.0, 1e-3, -1e6,
        1e6,
    ];
    let xs_f64: Vec<f64> = xs_f32.iter().map(|&v| v as f64).collect();
    let n = xs_f32.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let xs32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let xs64 = DeviceBuffer::from_host(&stream, &xs_f64)?;

    let mut out_cbrt_f32 = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_cbrt_f64 = DeviceBuffer::<f64>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cbrt_f32(&stream, cfg, &xs32, &mut out_cbrt_f32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cbrt_f64(&stream, cfg, &xs64, &mut out_cbrt_f64) }?;

    let got_cbrt_f32 = out_cbrt_f32.to_host_vec(&stream)?;
    let got_cbrt_f64 = out_cbrt_f64.to_host_vec(&stream)?;

    // libdevice transcendentals are typically within 1 ULP of host libm;
    // 2 ULP matches the bound `math_atan` / `primitive_stress` use for the
    // other libdevice transcendentals.
    const ULP_LIMIT: u64 = 2;

    let mut failures = 0usize;
    for i in 0..n {
        let exp_cbrt_f32 = xs_f32[i].cbrt();
        let exp_cbrt_f64 = xs_f64[i].cbrt();

        let d_cbrt_f32 = ulp_diff_f32(got_cbrt_f32[i], exp_cbrt_f32);
        let d_cbrt_f64 = ulp_diff_f64(got_cbrt_f64[i], exp_cbrt_f64);

        let ok = d_cbrt_f32 <= ULP_LIMIT && d_cbrt_f64 <= ULP_LIMIT;

        if !ok {
            failures += 1;
            if failures <= 8 {
                eprintln!(
                    "[{i}] x={:>10.4} | \
                     cbrt_f32 ulp={d_cbrt_f32} (gpu={:e} cpu={:e}) | \
                     cbrt_f64 ulp={d_cbrt_f64} (gpu={:e} cpu={:e})",
                    xs_f32[i], got_cbrt_f32[i], exp_cbrt_f32, got_cbrt_f64[i], exp_cbrt_f64,
                );
            }
        }
    }

    // A handful of representative samples printed regardless of pass/fail.
    for &i in &[3usize, 4, 7] {
        println!(
            "[{i}] x={x32:>8.3}  cbrt_f32 gpu={c32} cpu={ec32}  \
             cbrt_f64 gpu={c64} cpu={ec64}",
            x32 = xs_f32[i],
            c32 = got_cbrt_f32[i],
            ec32 = xs_f32[i].cbrt(),
            c64 = got_cbrt_f64[i],
            ec64 = xs_f64[i].cbrt(),
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

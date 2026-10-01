/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hand-written `__nv_*` libdevice externs called straight from a kernel.
//!
//! Regression test for device-extern declaration paths and transparent ABI:
//!
//! - A plain `extern "C" { fn __nv_asinf(...) }` block (no `#[device]`).
//!   The foreign item has no MIR body, so the importer must emit the call
//!   under the link symbol and mir-lower must declare it at the call site.
//! - The `#[device] extern "C"` scalar form (`__nv_acosf`) exercises the
//!   pipeline's device-extern declaration path and declaration idempotence.
//! - `#[repr(transparent)]` scalar wrappers around `f32` exercise the same
//!   external `float -> float` ABI for nested and ZST-marked wrappers.
//!
//! All symbols resolve against `libdevice.10.bc` when the NVVM IR is linked
//! via libNVVM + nvJitLink (same flow as `math_atan`).
//!
//! Run:
//!     cargo oxide run extern_libdevice
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use core::marker::PhantomData;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::{cuda_module, ltoir};

#[repr(transparent)]
#[derive(Clone, Copy)]
struct InnerF32(f32);

#[repr(transparent)]
#[derive(Clone, Copy)]
struct OuterF32(InnerF32);

#[repr(transparent)]
#[derive(Clone, Copy)]
struct MarkedF32(f32, PhantomData<()>);

// Plain extern block: the original motivating shape. No macro support;
// the kernel calls libdevice directly.
unsafe extern "C" {
    fn __nv_asinf(x: f32) -> f32;
}

// #[device] extern route for libdevice symbols. The transparent declarations
// intentionally use Rust wrapper types whose C ABI must still be `float(float)`.
#[device]
unsafe extern "C" {
    fn __nv_acosf(x: f32) -> f32;
    fn __nv_sinf(x: OuterF32) -> OuterF32;
    fn __nv_cosf(x: MarkedF32) -> MarkedF32;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn asin_acos(
        xs: &[f32],
        mut out_asin: DisjointSlice<f32>,
        mut out_acos: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            // `ThreadIndex` is not `Copy`; mint one per write surface.
            if let Some(slot) = out_asin.get_mut(thread::index_1d()) {
                *slot = unsafe { __nv_asinf(x) };
            }
            if let Some(slot) = out_acos.get_mut(thread::index_1d()) {
                *slot = unsafe { __nv_acosf(x) };
            }
        }
    }

    #[kernel]
    pub fn transparent_sin_cos(
        xs: &[f32],
        mut out_sin: DisjointSlice<f32>,
        mut out_cos: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];

            if let Some(slot) = out_sin.get_mut(thread::index_1d()) {
                let wrapped = OuterF32(InnerF32(x));
                let result = unsafe { __nv_sinf(wrapped) };
                *slot = result.0.0;
            }

            if let Some(slot) = out_cos.get_mut(thread::index_1d()) {
                let wrapped = MarkedF32(x, PhantomData);
                let result = unsafe { __nv_cosf(wrapped) };
                *slot = result.0;
            }
        }
    }
}

/// IEEE-754 ULP distance for finite f32 operands. All tested transcendental
/// results are finite, so NaN/Inf handling is not needed here.
fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    const SIGN: u64 = 0x8000_0000;
    const BODY: u64 = 0x7FFF_FFFF;
    let map = |bits: u64| {
        if bits & SIGN != 0 {
            SIGN - (bits & BODY)
        } else {
            SIGN + (bits & BODY)
        }
    };
    map(a.to_bits() as u64).abs_diff(map(b.to_bits() as u64))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== extern_libdevice: direct and repr(transparent) device extern calls ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls force the NVVM-IR output flavor; the first launch
    // builds a cubin via libNVVM + nvJitLink (links libdevice.10.bc).
    let module = ltoir::load_kernel_module(&ctx, "extern_libdevice")?;
    let module = kernels::from_module(module)?;

    // Full asin/acos domain including both endpoints. The same inputs are also
    // well inside the finite sin/cos domain used by the transparent ABI cases.
    let xs: Vec<f32> = vec![
        -1.0, -0.9, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0,
    ];
    let n = xs.len();
    let cfg_asin_acos = LaunchConfig::for_num_elems(n as u32);
    let cfg_transparent = LaunchConfig::for_num_elems(n as u32);

    let xs_dev = DeviceBuffer::from_host(&stream, &xs)?;
    let mut out_asin = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_acos = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_sin = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_cos = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    // SAFETY: launch shapes/resources match the kernels; buffers cover their accesses.
    unsafe {
        module.asin_acos(
            &stream,
            cfg_asin_acos,
            &xs_dev,
            &mut out_asin,
            &mut out_acos,
        )
    }?;
    unsafe {
        module.transparent_sin_cos(
            &stream,
            cfg_transparent,
            &xs_dev,
            &mut out_sin,
            &mut out_cos,
        )
    }?;

    let got_asin = out_asin.to_host_vec(&stream)?;
    let got_acos = out_acos.to_host_vec(&stream)?;
    let got_sin = out_sin.to_host_vec(&stream)?;
    let got_cos = out_cos.to_host_vec(&stream)?;

    // libdevice transcendentals are typically within 1 ULP of host libm;
    // 2 ULP matches the bound math_atan / primitive_stress use.
    const ULP_LIMIT: u64 = 2;

    let mut failures = 0usize;
    for i in 0..n {
        let expected = [xs[i].asin(), xs[i].acos(), xs[i].sin(), xs[i].cos()];
        let actual = [got_asin[i], got_acos[i], got_sin[i], got_cos[i]];
        let labels = ["asin", "acos", "sin(transparent)", "cos(marked)"];

        for j in 0..actual.len() {
            let ulp = ulp_diff_f32(actual[j], expected[j]);
            if ulp > ULP_LIMIT {
                failures += 1;
                eprintln!(
                    "[{i}] x={:>6.3} | {} ulp={ulp} (gpu={:e} cpu={:e})",
                    xs[i], labels[j], actual[j], expected[j],
                );
            }
        }
    }

    for &i in &[0usize, 6, 12] {
        println!(
            "[{i}] x={x:>6.3}  asin={asin}  acos={acos}  \
             sin(transparent)={sin}  cos(marked)={cos}",
            x = xs[i],
            asin = got_asin[i],
            acos = got_acos[i],
            sin = got_sin[i],
            cos = got_cos[i],
        );
    }

    if failures == 0 {
        println!(
            "\nSUCCESS: {n} cases x 4 functions within {ULP_LIMIT} ULP; \
             repr(transparent) device extern ABI preserved"
        );
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} function results out of tolerance");
        std::process::exit(1);
    }
}

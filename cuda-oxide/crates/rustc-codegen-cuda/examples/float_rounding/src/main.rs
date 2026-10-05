/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rounding-only kernels on the self-contained PTX path.
//!
//! `floor` / `ceil` / `trunc` / `round` / `round_ties_even` lower to the
//! native LLVM rounding intrinsics (`llvm.floor.*`, `llvm.ceil.*`, ...)
//! under the LLVM NVPTX backend, so a kernel whose only "math library"
//! usage is rounding needs no libdevice at all: no `__nv_*` symbols in the
//! emitted IR, no `llvm-link` step, no libNVVM/nvJitLink dependency. The
//! generated `float_rounding.ll` must contain zero `__nv_` references.
//! (Under `--emit-nvvm-ir` the same source instead routes rounding through
//! libdevice `__nv_floorf`/..., because the legacy LLVM 7-based NVVM IR
//! dialect predates `llvm.roundeven`.)
//!
//! Each kernel evaluates all five rounding ops per input element and writes
//! the raw IEEE-754 bits, so the host comparison is exact and catches signed
//! zero (`round(-0.4)` must be `-0.0`, not `+0.0`). The interesting
//! semantics pinned here:
//!
//! * `round` is ties-away-from-zero: `round(2.5) == 3.0`,
//!   `round(-2.5) == -3.0`.
//! * `round_ties_even` is ties-to-even: `round_ties_even(2.5) == 2.0`,
//!   `round_ties_even(3.5) == 4.0`.
//! * `floor(-1.5) == -2.0`, `ceil(-1.5) == -1.0`, `trunc(-1.7) == -1.0`.
//!
//! Run with:
//!   cargo oxide run float_rounding
//!
//! Exits 0 on PASS, 1 on FAIL.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Rounding ops evaluated per element, in output-lane order.
const OPS: [&str; 5] = ["floor", "ceil", "trunc", "round", "round_ties_even"];

#[cuda_module]
mod kernels {
    use super::*;

    /// Writes the bits of all five f32 rounding results for `x[i]` to
    /// `out[5 * i ..][..5]`, in [`OPS`] order.
    #[kernel]
    pub fn rounding_f32(x: &[f32], mut out: DisjointSlice<u32>) {
        let i = thread::index_1d().get();
        if i < x.len() {
            let v = x[i];
            let base = 5 * i;
            // SAFETY: the host allocates `out` with `5 * x.len()` elements
            // and `i < x.len()`, so all five lanes are in bounds.
            unsafe {
                *out.get_unchecked_mut(base) = v.floor().to_bits();
                *out.get_unchecked_mut(base + 1) = v.ceil().to_bits();
                *out.get_unchecked_mut(base + 2) = v.trunc().to_bits();
                *out.get_unchecked_mut(base + 3) = v.round().to_bits();
                *out.get_unchecked_mut(base + 4) = v.round_ties_even().to_bits();
            }
        }
    }

    /// Same as [`rounding_f32`] for f64.
    #[kernel]
    pub fn rounding_f64(x: &[f64], mut out: DisjointSlice<u64>) {
        let i = thread::index_1d().get();
        if i < x.len() {
            let v = x[i];
            let base = 5 * i;
            // SAFETY: the host allocates `out` with `5 * x.len()` elements
            // and `i < x.len()`, so all five lanes are in bounds.
            unsafe {
                *out.get_unchecked_mut(base) = v.floor().to_bits();
                *out.get_unchecked_mut(base + 1) = v.ceil().to_bits();
                *out.get_unchecked_mut(base + 2) = v.trunc().to_bits();
                *out.get_unchecked_mut(base + 3) = v.round().to_bits();
                *out.get_unchecked_mut(base + 4) = v.round_ties_even().to_bits();
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== float_rounding: floor/ceil/trunc/round/round_ties_even on device ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // Rounding-only kernels stay on the self-contained PTX path, so the
    // module loads from the embedded PTX bundle with no libNVVM involved.
    let module = kernels::load(&ctx)?;

    // Halfway ties (both parities), signed zero, fractional magnitudes on
    // both sides of .5, exact integers, and values large enough that every
    // rounding mode is the identity.
    let xs_f32: Vec<f32> = vec![
        2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 1.5, -1.5, 0.4, -0.4, 1.7, -1.7, 0.0, -0.0, 2.0, -2.0,
        1e8, -1e8,
    ];
    let xs_f64: Vec<f64> = xs_f32.iter().map(|&v| v as f64).collect();
    let n = xs_f32.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let x32 = DeviceBuffer::from_host(&stream, &xs_f32)?;
    let x64 = DeviceBuffer::from_host(&stream, &xs_f64)?;
    let mut out32 = DeviceBuffer::<u32>::zeroed(&stream, 5 * n)?;
    let mut out64 = DeviceBuffer::<u64>::zeroed(&stream, 5 * n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rounding_f32(&stream, cfg, &x32, &mut out32) }?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.rounding_f64(&stream, cfg, &x64, &mut out64) }?;

    let got32 = out32.to_host_vec(&stream)?;
    let got64 = out64.to_host_vec(&stream)?;

    // Bit-exact comparison against the host stdlib, which shares the exact
    // IEEE-754 semantics of the five ops (including signed-zero results).
    let mut failures = 0usize;
    for i in 0..n {
        let v32 = xs_f32[i];
        let v64 = xs_f64[i];
        let want32: [u32; 5] = [
            v32.floor().to_bits(),
            v32.ceil().to_bits(),
            v32.trunc().to_bits(),
            v32.round().to_bits(),
            v32.round_ties_even().to_bits(),
        ];
        let want64: [u64; 5] = [
            v64.floor().to_bits(),
            v64.ceil().to_bits(),
            v64.trunc().to_bits(),
            v64.round().to_bits(),
            v64.round_ties_even().to_bits(),
        ];
        for k in 0..5 {
            if got32[5 * i + k] != want32[k] {
                failures += 1;
                eprintln!(
                    "FAIL f32 {}({v32}): gpu={:e} host={:e}",
                    OPS[k],
                    f32::from_bits(got32[5 * i + k]),
                    f32::from_bits(want32[k]),
                );
            }
            if got64[5 * i + k] != want64[k] {
                failures += 1;
                eprintln!(
                    "FAIL f64 {}({v64}): gpu={:e} host={:e}",
                    OPS[k],
                    f64::from_bits(got64[5 * i + k]),
                    f64::from_bits(want64[k]),
                );
            }
        }
    }

    // Named semantic pins, asserted against literal expectations so a host
    // stdlib regression cannot mask a device one. Lane order per [`OPS`]:
    // floor 0, ceil 1, trunc 2, round 3, round_ties_even 4.
    let lane32 = |value: f32, lane: usize| -> u32 {
        let i = xs_f32
            .iter()
            .position(|&x| x.to_bits() == value.to_bits())
            .expect("pinned input present");
        got32[5 * i + lane]
    };
    let named: [(&str, u32, u32); 8] = [
        ("round(2.5) == 3.0", lane32(2.5, 3), 3.0f32.to_bits()),
        ("round(-2.5) == -3.0", lane32(-2.5, 3), (-3.0f32).to_bits()),
        (
            "round_ties_even(2.5) == 2.0",
            lane32(2.5, 4),
            2.0f32.to_bits(),
        ),
        (
            "round_ties_even(3.5) == 4.0",
            lane32(3.5, 4),
            4.0f32.to_bits(),
        ),
        (
            "round(-0.4) == -0.0 (sign preserved)",
            lane32(-0.4, 3),
            (-0.0f32).to_bits(),
        ),
        ("floor(-1.5) == -2.0", lane32(-1.5, 0), (-2.0f32).to_bits()),
        ("ceil(-1.5) == -1.0", lane32(-1.5, 1), (-1.0f32).to_bits()),
        ("trunc(-1.7) == -1.0", lane32(-1.7, 2), (-1.0f32).to_bits()),
    ];
    for (name, got, want) in named {
        if got == want {
            println!("PASS {name}");
        } else {
            failures += 1;
            eprintln!(
                "FAIL {name}: gpu={:e} want={:e}",
                f32::from_bits(got),
                f32::from_bits(want),
            );
        }
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} inputs x 5 ops x 2 widths bit-exact against host");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures} mismatches");
        std::process::exit(1);
    }
}

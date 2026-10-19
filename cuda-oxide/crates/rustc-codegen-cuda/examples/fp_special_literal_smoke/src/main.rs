/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for floating-point special-value literals (`NAN`, `INFINITY`,
//! `NEG_INFINITY`) inside kernels.
//!
//! Before the fix to `format_float_literal` in
//! `crates/llvm-export/src/export/literals.rs`, embedding `f32::NAN` in a
//! kernel as a float SSA value produced invalid LLVM IR:
//!
//! ```text
//!   store float nan, ptr %v9
//! ```
//!
//! Both `llc` and libNVVM reject the bare `nan` token with
//! "parse expected value token". After the fix, NaN renders as
//! `0x7FF8000000000000` (canonical quiet NaN as a 16-hex-digit double bit
//! pattern) and the kernel compiles through the standard `llc -> PTX` path.
//!
//! The kernels store the float values directly through a
//! `DisjointSlice<f32>` / `DisjointSlice<f64>` write so the constants
//! survive into LLVM IR as `float` / `double` SSA values rather than
//! getting folded into integer `to_bits()` constants by rustc.
//!
//! The negative-zero and one-ULP cases additionally verify that the MIR
//! importer preserves exact scalar `f32` and `f64` bit patterns without
//! parsing rustc `Debug` output.
//!
//! Build and run with:
//!   cargo oxide run fp_special_literal_smoke

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, ltoir};

// =============================================================================
// KERNELS
// =============================================================================

const F32_ONE_PLUS_ULP: f32 = f32::from_bits(0x3f80_0001);
const F64_ONE_PLUS_ULP: f64 = f64::from_bits(0x3ff0_0000_0000_0001);

#[cuda_module]
mod kernels {
    use super::*;

    /// Stores special values and exact-bit finite values so scalar `f32`
    /// constants exercise both MIR constant import and LLVM literal export.
    #[kernel]
    pub fn fp_special_literal_f32_kernel(mut out: DisjointSlice<f32>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = f32::NAN;
                *out.get_unchecked_mut(1) = f32::INFINITY;
                *out.get_unchecked_mut(2) = f32::NEG_INFINITY;
                *out.get_unchecked_mut(3) = 1.5_f32;
                *out.get_unchecked_mut(4) = -0.0_f32;
                *out.get_unchecked_mut(5) = F32_ONE_PLUS_ULP;
            }
        }
    }

    /// Same coverage as `fp_special_literal_f32_kernel` for `f64`.
    #[kernel]
    pub fn fp_special_literal_f64_kernel(mut out: DisjointSlice<f64>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = f64::NAN;
                *out.get_unchecked_mut(1) = f64::INFINITY;
                *out.get_unchecked_mut(2) = f64::NEG_INFINITY;
                *out.get_unchecked_mut(3) = 1.5_f64;
                *out.get_unchecked_mut(4) = -0.0_f64;
                *out.get_unchecked_mut(5) = F64_ONE_PLUS_ULP;
            }
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== fp special literal smoke ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ltoir::load_kernel_module(&ctx, "fp_special_literal_smoke")?;
    let module = kernels::from_module(module)?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    // ---------- f32 ----------
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, 6)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.fp_special_literal_f32_kernel(&stream, cfg, &mut out) }?;
        let got = out.to_host_vec(&stream)?;
        check_f32_specials(&got, &mut passed, &mut failed);
    }

    // ---------- f64 ----------
    {
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, 6)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.fp_special_literal_f64_kernel(&stream, cfg, &mut out) }?;
        let got = out.to_host_vec(&stream)?;
        check_f64_specials(&got, &mut passed, &mut failed);
    }

    println!("\n--- summary ---");
    println!("passed: {passed}");
    println!("failed: {failed}");
    if failed != 0 {
        std::process::exit(1);
    }
    println!("\n✓ SUCCESS");
    Ok(())
}

fn check_f32_specials(got: &[f32], passed: &mut u32, failed: &mut u32) {
    let mut ok = true;
    if !got[0].is_nan() {
        println!(
            "FAIL f32::NAN: got {} (bits 0x{:08x})",
            got[0],
            got[0].to_bits()
        );
        ok = false;
    }
    if got[1] != f32::INFINITY {
        println!(
            "FAIL f32::INFINITY: got {} (bits 0x{:08x})",
            got[1],
            got[1].to_bits()
        );
        ok = false;
    }
    if got[2] != f32::NEG_INFINITY {
        println!(
            "FAIL f32::NEG_INFINITY: got {} (bits 0x{:08x})",
            got[2],
            got[2].to_bits()
        );
        ok = false;
    }
    if got[3] != 1.5_f32 {
        println!("FAIL f32 control 1.5: got {}", got[3]);
        ok = false;
    }
    if got[4].to_bits() != 0x8000_0000 {
        println!(
            "FAIL f32 negative zero: got bits 0x{:08x}, expected 0x80000000",
            got[4].to_bits()
        );
        ok = false;
    }

    if got[5].to_bits() != 0x3f80_0001 {
        println!(
            "FAIL f32 one-plus-ULP: got bits 0x{:08x}, expected 0x3f800001",
            got[5].to_bits()
        );
        ok = false;
    }
    if ok {
        println!(
            "PASS f32 specials: NaN={}, +Inf={}, -Inf={}, control={}",
            got[0].is_nan(),
            got[1],
            got[2],
            got[3]
        );
        *passed += 1;
    } else {
        *failed += 1;
    }
}

fn check_f64_specials(got: &[f64], passed: &mut u32, failed: &mut u32) {
    let mut ok = true;
    if !got[0].is_nan() {
        println!(
            "FAIL f64::NAN: got {} (bits 0x{:016x})",
            got[0],
            got[0].to_bits()
        );
        ok = false;
    }
    if got[1] != f64::INFINITY {
        println!(
            "FAIL f64::INFINITY: got {} (bits 0x{:016x})",
            got[1],
            got[1].to_bits()
        );
        ok = false;
    }
    if got[2] != f64::NEG_INFINITY {
        println!(
            "FAIL f64::NEG_INFINITY: got {} (bits 0x{:016x})",
            got[2],
            got[2].to_bits()
        );
        ok = false;
    }
    if got[3] != 1.5_f64 {
        println!("FAIL f64 control 1.5: got {}", got[3]);
        ok = false;
    }
    if got[4].to_bits() != 0x8000_0000_0000_0000 {
        println!(
            "FAIL f64 negative zero: got bits 0x{:016x}, expected 0x8000000000000000",
            got[4].to_bits()
        );
        ok = false;
    }

    if got[5].to_bits() != 0x3ff0_0000_0000_0001 {
        println!(
            "FAIL f64 one-plus-ULP: got bits 0x{:016x}, expected 0x3ff0000000000001",
            got[5].to_bits()
        );
        ok = false;
    }
    if ok {
        println!(
            "PASS f64 specials: NaN={}, +Inf={}, -Inf={}, control={}",
            got[0].is_nan(),
            got[1],
            got[2],
            got[3]
        );
        *passed += 1;
    } else {
        *failed += 1;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for scalar `max`, `min`, `abs`, and `copysign` operations that
//! lower directly through LLVM for both float widths.
//!
//! These lower to `core::intrinsics::maximum_number_nsz_f32` /
//! `minimum_number_nsz_f32` / ... in MIR. Before this example was added,
//! cuda-oxide had no handler for the `maximum_number_nsz_*` /
//! `minimum_number_nsz_*` intrinsics and `f32::max` / `f32::min` would fall
//! out of the pipeline as an unresolved call. After the matching entries
//! were added to:
//!
//! * `dialect-mir::rust_intrinsics`
//! * `mir-importer::translator::terminator::intrinsics::float_math`
//! * `mir-lower::convert::ops::call`
//!
//! max/min lower to ordered comparisons and selects, while abs/copysign use
//! their LLVM intrinsics. llc lowers all four directly to PTX. This smoke check
//! validates the full chain on a real GPU.
//!
//! NaN inputs are passed in from the host rather than being embedded as
//! `f32::NAN` literals in the kernel so the example stays focused on the
//! max/min intrinsic lowering and does not depend on how cuda-oxide
//! renders NaN constants in LLVM IR.
//!
//! Build and run with:
//!   cargo oxide run fmaxmin_smoke

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, ltoir};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Exercise finite, quiet/signaling NaN, signed-zero, sign, and payload
    /// behavior for the four LLVM-only scalar operations.
    #[kernel]
    pub fn fmaxmin_f32_kernel(a: f32, b: f32, qnan: f32, snan: f32, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = a.max(b).to_bits();
                *out.get_unchecked_mut(1) = a.min(b).to_bits();
                *out.get_unchecked_mut(2) = qnan.max(b).to_bits();
                *out.get_unchecked_mut(3) = b.max(qnan).to_bits();
                *out.get_unchecked_mut(4) = snan.max(b).to_bits();
                *out.get_unchecked_mut(5) = b.max(snan).to_bits();
                *out.get_unchecked_mut(6) = qnan.min(b).to_bits();
                *out.get_unchecked_mut(7) = b.min(qnan).to_bits();
                *out.get_unchecked_mut(8) = snan.min(b).to_bits();
                *out.get_unchecked_mut(9) = b.min(snan).to_bits();
                *out.get_unchecked_mut(10) = qnan.max(snan).to_bits();
                *out.get_unchecked_mut(11) = (-0.0_f32).max(0.0).to_bits();
                *out.get_unchecked_mut(12) = b.abs().to_bits();
                *out.get_unchecked_mut(13) = a.copysign(b).to_bits();
                *out.get_unchecked_mut(14) = (-0.0_f32).abs().to_bits();
                *out.get_unchecked_mut(15) = (-qnan).abs().to_bits();
                *out.get_unchecked_mut(16) = a.copysign(-b).to_bits();
                *out.get_unchecked_mut(17) = qnan.copysign(b).to_bits();
                *out.get_unchecked_mut(18) = (-qnan).copysign(a).to_bits();
                *out.get_unchecked_mut(19) = 0.0_f32.max(-0.0).to_bits();
                *out.get_unchecked_mut(20) = (-0.0_f32).min(0.0).to_bits();
                *out.get_unchecked_mut(21) = 0.0_f32.min(-0.0).to_bits();
            }
        }
    }

    /// Same as `fmaxmin_f32_kernel` for `f64::max` / `f64::min`.
    #[kernel]
    pub fn fmaxmin_f64_kernel(a: f64, b: f64, qnan: f64, snan: f64, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = a.max(b).to_bits();
                *out.get_unchecked_mut(1) = a.min(b).to_bits();
                *out.get_unchecked_mut(2) = qnan.max(b).to_bits();
                *out.get_unchecked_mut(3) = b.max(qnan).to_bits();
                *out.get_unchecked_mut(4) = snan.max(b).to_bits();
                *out.get_unchecked_mut(5) = b.max(snan).to_bits();
                *out.get_unchecked_mut(6) = qnan.min(b).to_bits();
                *out.get_unchecked_mut(7) = b.min(qnan).to_bits();
                *out.get_unchecked_mut(8) = snan.min(b).to_bits();
                *out.get_unchecked_mut(9) = b.min(snan).to_bits();
                *out.get_unchecked_mut(10) = qnan.max(snan).to_bits();
                *out.get_unchecked_mut(11) = (-0.0_f64).max(0.0).to_bits();
                *out.get_unchecked_mut(12) = b.abs().to_bits();
                *out.get_unchecked_mut(13) = a.copysign(b).to_bits();
                *out.get_unchecked_mut(14) = (-0.0_f64).abs().to_bits();
                *out.get_unchecked_mut(15) = (-qnan).abs().to_bits();
                *out.get_unchecked_mut(16) = a.copysign(-b).to_bits();
                *out.get_unchecked_mut(17) = qnan.copysign(b).to_bits();
                *out.get_unchecked_mut(18) = (-qnan).copysign(a).to_bits();
                *out.get_unchecked_mut(19) = 0.0_f64.max(-0.0).to_bits();
                *out.get_unchecked_mut(20) = (-0.0_f64).min(0.0).to_bits();
                *out.get_unchecked_mut(21) = 0.0_f64.min(-0.0).to_bits();
            }
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== fmaxmin smoke ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // `load_kernel_module` loads the PTX produced by llc directly here. The
    // same helper also handles NVVM IR for examples that genuinely need
    // libdevice.
    let module = ltoir::load_kernel_module(&ctx, "fmaxmin_smoke")?;
    let module = kernels::from_module(module)?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    // ---------- f32 ----------
    {
        let a = 1.5_f32;
        let b = -2.5_f32;
        let qnan = f32::from_bits(0x7fc1_2345);
        let snan = f32::from_bits(0x7f81_2345);
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, 22)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.fmaxmin_f32_kernel(&stream, cfg, a, b, qnan, snan, &mut out) }?;
        let mut result = out.to_host_vec(&stream)?;
        let expected: [u32; 22] = [
            a.max(b).to_bits(), // f32::max finite
            a.min(b).to_bits(), // f32::min finite
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            qnan.to_bits(),    // normalized below: both-NaN payload is unspecified
            0.0_f32.to_bits(), // normalized below: either signed zero is valid
            b.abs().to_bits(),
            a.copysign(b).to_bits(),
            0.0_f32.to_bits(),
            qnan.to_bits(),
            a.to_bits(),
            (-qnan).to_bits(),
            qnan.to_bits(),
            0.0_f32.to_bits(),
            0.0_f32.to_bits(),
            0.0_f32.to_bits(),
        ];
        assert!(f32::from_bits(result[10]).is_nan());
        assert_eq!(f32::from_bits(result[11]), 0.0);
        assert!(f32::from_bits(result[15]).is_nan());
        assert_eq!(result[15] >> 31, 0);
        assert!(f32::from_bits(result[17]).is_nan());
        assert_eq!(result[17] >> 31, 1);
        assert!(f32::from_bits(result[18]).is_nan());
        assert_eq!(result[18] >> 31, 0);
        for bits in &result[19..22] {
            assert_eq!(f32::from_bits(*bits), 0.0);
        }
        result[10] = expected[10];
        result[11] = expected[11];
        result[15] = expected[15];
        result[19..22].copy_from_slice(&expected[19..22]);
        check_f32(
            "f32::max / f32::min",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    // ---------- f64 ----------
    {
        let a = 1.5e10_f64;
        let b = -2.5e10_f64;
        let qnan = f64::from_bits(0x7ff8_0000_0001_2345);
        let snan = f64::from_bits(0x7ff0_0000_0001_2345);
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 22)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.fmaxmin_f64_kernel(&stream, cfg, a, b, qnan, snan, &mut out) }?;
        let mut result = out.to_host_vec(&stream)?;
        let expected: [u64; 22] = [
            a.max(b).to_bits(),
            a.min(b).to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            b.to_bits(),
            qnan.to_bits(),
            0.0_f64.to_bits(),
            b.abs().to_bits(),
            a.copysign(b).to_bits(),
            0.0_f64.to_bits(),
            qnan.to_bits(),
            a.to_bits(),
            (-qnan).to_bits(),
            qnan.to_bits(),
            0.0_f64.to_bits(),
            0.0_f64.to_bits(),
            0.0_f64.to_bits(),
        ];
        assert!(f64::from_bits(result[10]).is_nan());
        assert_eq!(f64::from_bits(result[11]), 0.0);
        assert!(f64::from_bits(result[15]).is_nan());
        assert_eq!(result[15] >> 63, 0);
        assert!(f64::from_bits(result[17]).is_nan());
        assert_eq!(result[17] >> 63, 1);
        assert!(f64::from_bits(result[18]).is_nan());
        assert_eq!(result[18] >> 63, 0);
        for bits in &result[19..22] {
            assert_eq!(f64::from_bits(*bits), 0.0);
        }
        result[10] = expected[10];
        result[11] = expected[11];
        result[15] = expected[15];
        result[19..22].copy_from_slice(&expected[19..22]);
        check_f64(
            "f64::max / f64::min",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
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

fn check_f32(name: &str, got: &[u32], expected: &[u32], passed: &mut u32, failed: &mut u32) {
    if got == expected {
        println!(
            "PASS {name}: {:?}",
            got.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>()
        );
        *passed += 1;
    } else {
        println!(
            "FAIL {name}: got {:?} expected {:?}\ngot bits: {got:x?}\nexpected bits: {expected:x?}",
            got.iter().map(|b| f32::from_bits(*b)).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|b| f32::from_bits(*b))
                .collect::<Vec<_>>()
        );
        *failed += 1;
    }
}

fn check_f64(name: &str, got: &[u64], expected: &[u64], passed: &mut u32, failed: &mut u32) {
    if got == expected {
        println!(
            "PASS {name}: {:?}",
            got.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>()
        );
        *passed += 1;
    } else {
        println!(
            "FAIL {name}: got {:?} expected {:?}\ngot bits: {got:x?}\nexpected bits: {expected:x?}",
            got.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|b| f64::from_bits(*b))
                .collect::<Vec<_>>()
        );
        *failed += 1;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

//! Rust primitive `f16` stress test.
//!
//! Exercises first-class Rust nightly `f16` through MIR import, lowering to LLVM
//! `half`, constants, memory traffic, arithmetic, comparisons, and casts.
//!
//! Run: cargo oxide run f16_stress

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Stores one `f16` value so we cover constants and device memory traffic.
    #[kernel]
    pub fn test_f16_store(mut out: DisjointSlice<f16>) {
        if thread::index_1d().get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = f16::from_bits(0x3e00); // 1.5
            }
        }
    }

    /// Exercises basic `f16` arithmetic, comparison, and casts through `f32`.
    #[kernel]
    pub fn test_f16_ops(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            let one = f16::from_bits(0x3c00);
            let two = f16::from_bits(0x4000);
            let half = f16::from_bits(0x3800);

            let sum = one + two;
            let product = sum * half;
            let is_gt = if product > one { 1u32 } else { 0u32 };
            let widened = product as f32;
            let narrowed = (widened + 1.0_f32) as f16;

            unsafe {
                *out.get_unchecked_mut(0) = sum.to_bits() as u32;
                *out.get_unchecked_mut(1) = product.to_bits() as u32;
                *out.get_unchecked_mut(2) = is_gt;
                *out.get_unchecked_mut(3) = narrowed.to_bits() as u32;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== f16 Stress Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    {
        let mut out = DeviceBuffer::<f16>::zeroed(&stream, 1)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_f16_store(&stream, cfg, &mut out) }?;
        let got = out.to_host_vec(&stream)?[0].to_bits();
        let expected = f16::from_bits(0x3e00).to_bits();
        check("f16 load/store", got, expected, &mut passed, &mut failed);
    }

    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_f16_ops(&stream, cfg, &mut out) }?;
        let got = out.to_host_vec(&stream)?;

        let one = f16::from_bits(0x3c00);
        let two = f16::from_bits(0x4000);
        let half = f16::from_bits(0x3800);
        let sum = one + two;
        let product = sum * half;
        let widened = product as f32;
        let narrowed = (widened + 1.0_f32) as f16;
        let expected = [
            sum.to_bits() as u32,
            product.to_bits() as u32,
            if product > one { 1 } else { 0 },
            narrowed.to_bits() as u32,
        ];

        check_slice(
            "f16 arithmetic/comparison/casts",
            &got,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    println!("\n=== Results ===");
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if failed == 0 {
        println!("\nPASS: f16 checks matched");
        Ok(())
    } else {
        eprintln!("\nFAIL: {failed} f16 checks failed");
        std::process::exit(1);
    }
}

fn check<T: Eq + std::fmt::Debug>(
    name: &str,
    got: T,
    expected: T,
    passed: &mut u32,
    failed: &mut u32,
) {
    if got == expected {
        println!("PASS {name}: {got:?}");
        *passed += 1;
    } else {
        println!("FAIL {name}: got {got:?}, expected {expected:?}");
        *failed += 1;
    }
}

fn check_slice<T: Eq + std::fmt::Debug>(
    name: &str,
    got: &[T],
    expected: &[T],
    passed: &mut u32,
    failed: &mut u32,
) {
    if got == expected {
        println!("PASS {name}: {got:?}");
        *passed += 1;
    } else {
        println!("FAIL {name}: got {got:?}, expected {expected:?}");
        *failed += 1;
    }
}

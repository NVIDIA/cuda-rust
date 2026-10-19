/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed cvt variants end-to-end example (sm_80+).
//!
//! Tests the four new packed conversion intrinsics added in PR #276 alongside
//! the baseline `cvt_f16x2_f32`, all on the same
//! `(lo = 1.0065, hi = -1.0065)` inputs:
//!
//!   - `cvt_f16x2_f32`          round-to-nearest f16 pack (baseline)
//!   - `cvt_rz_f16x2_f32`      round-toward-zero f16 pack
//!   - `cvt_rn_relu_f16x2_f32` round-to-nearest + ReLU f16 pack
//!   - `cvt_rn_relu_bf16x2_f32` round-to-nearest + ReLU bf16 pack
//!   - `cvt_rz_bf16x2_f32`     round-toward-zero bf16 pack
//!
//! The finite input sits on opposite sides of the nearest and toward-zero
//! results for both f16 and bf16. The host therefore checks the exact packed
//! bits instead of using a tolerance that could let the wrong rounding mode
//! pass. A second launch verifies that the ReLU variants produce CUDA's
//! canonical NaN for NaN inputs.
//!
//! A third launch exercises the hand-written unpackers: it packs with the
//! generated `cvt_f16x2_f32` / `cvt_rz_bf16x2_f32` and unpacks with
//! `cvt_f32x2_f16x2`, `cvt_f32_f16x2_{lo,hi}`, `cvt_f32x2_bf16x2`, and
//! `cvt_f32_bf16x2_{lo,hi}`, writing the widened `f32` values out. Widening
//! is exact in both formats, so the host again checks exact bits. This is
//! the only example that runs the runtime u16-to-f16 transmute + fpext path
//! on the device.
//!
//! Run: cargo oxide run cvt_packed

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::convert::{
    cvt_f16x2_f32, cvt_f32_bf16x2_hi, cvt_f32_bf16x2_lo, cvt_f32_f16x2_hi, cvt_f32_f16x2_lo,
    cvt_f32x2_bf16x2, cvt_f32x2_f16x2, cvt_rn_relu_bf16x2_f32, cvt_rn_relu_f16x2_f32,
    cvt_rz_bf16x2_f32, cvt_rz_f16x2_f32,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Number of conversion results produced by the packing kernel.
const NUM_VARIANTS: usize = 5;

/// Number of unpacked f32 values produced by the round-trip kernel.
const NUM_UNPACKED: usize = 8;

// =============================================================================
// KERNEL
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 calls all five conversion functions with the same (lo, hi) pair
    /// and writes the five packed u32 results into `out[0..5]`.
    #[kernel]
    pub fn cvt_packed_variants(lo: f32, hi: f32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if idx.get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = cvt_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(1) = cvt_rz_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(2) = cvt_rn_relu_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(3) = cvt_rn_relu_bf16x2_f32(lo, hi);
                *out.get_unchecked_mut(4) = cvt_rz_bf16x2_f32(lo, hi);
            }
        }
    }

    /// Thread 0 packs (lo, hi) with the generated packers, unpacks the words
    /// with the hand-written unpackers, and writes the widened f32 results:
    ///
    ///   out[0..2] = cvt_f32x2_f16x2(cvt_f16x2_f32(lo, hi))
    ///   out[2..4] = cvt_f32_f16x2_lo / cvt_f32_f16x2_hi of the same word
    ///   out[4..6] = cvt_f32x2_bf16x2(cvt_rz_bf16x2_f32(lo, hi))
    ///   out[6..8] = cvt_f32_bf16x2_lo / cvt_f32_bf16x2_hi of the same word
    #[kernel]
    pub fn cvt_packed_unpack(lo: f32, hi: f32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if idx.get() == 0 {
            let f16_word = cvt_f16x2_f32(lo, hi);
            let bf16_word = cvt_rz_bf16x2_f32(lo, hi);
            let (f16_lo, f16_hi) = cvt_f32x2_f16x2(f16_word);
            let (bf16_lo, bf16_hi) = cvt_f32x2_bf16x2(bf16_word);
            unsafe {
                *out.get_unchecked_mut(0) = f16_lo;
                *out.get_unchecked_mut(1) = f16_hi;
                *out.get_unchecked_mut(2) = cvt_f32_f16x2_lo(f16_word);
                *out.get_unchecked_mut(3) = cvt_f32_f16x2_hi(f16_word);
                *out.get_unchecked_mut(4) = bf16_lo;
                *out.get_unchecked_mut(5) = bf16_hi;
                *out.get_unchecked_mut(6) = cvt_f32_bf16x2_lo(bf16_word);
                *out.get_unchecked_mut(7) = cvt_f32_bf16x2_hi(bf16_word);
            }
        }
    }
}

// =============================================================================
// HOST VERIFICATION
// =============================================================================

fn main() {
    println!("=== Packed cvt Variants (sm_80+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // The packed cvt.rz and cvt.rn.relu variants require sm_80+ (Ampere).
    if major < 8 {
        println!("\nskipping: packed cvt variants require sm_80+ (Ampere)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // This value rounds up under round-to-nearest but truncates down under
    // round-toward-zero in both f16 and bf16.
    let lo = 1.0065_f32;
    let hi = -lo;

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, NUM_VARIANTS).unwrap();
    let cfg = LaunchConfig::for_num_elems(1);

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cvt_packed_variants(&stream, cfg, lo, hi, &mut out_dev) }
        .expect("Kernel launch failed");

    let results = out_dev.to_host_vec(&stream).unwrap();
    assert_eq!(results.len(), NUM_VARIANTS);

    let labels = [
        "cvt_f16x2_f32 (rn)",
        "cvt_rz_f16x2_f32",
        "cvt_rn_relu_f16x2_f32",
        "cvt_rn_relu_bf16x2_f32",
        "cvt_rz_bf16x2_f32",
    ];
    let expected = [
        0xbc07_3c07, // f16 rn:  (-1.0068359375,  1.0068359375)
        0xbc06_3c06, // f16 rz:  (-1.0058593750,  1.0058593750)
        0x0000_3c07, // f16 rn + ReLU
        0x0000_3f81, // bf16 rn + ReLU
        0xbf80_3f80, // bf16 rz:  (-1.0, 1.0)
    ];

    let mut failures = 0;
    for i in 0..NUM_VARIANTS {
        let got = results[i];
        let want = expected[i];
        if got == want {
            println!("[{i}] {}: PASS ({got:#010x})", labels[i]);
        } else {
            eprintln!(
                "[{i}] {}: FAIL, expected {want:#010x}, got {got:#010x}",
                labels[i]
            );
            failures += 1;
        }
    }

    // PTX specifies that `.relu` converts NaN results to canonical NaN rather
    // than clamping them to zero. Check both packed lanes for each format.
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cvt_packed_variants(&stream, cfg, f32::NAN, f32::NAN, &mut out_dev) }
        .expect("NaN kernel launch failed");
    let nan_results = out_dev.to_host_vec(&stream).unwrap();
    for (label, packed, expected_nan) in [
        ("cvt_rn_relu_f16x2_f32 NaN", nan_results[2], 0x7fff_7fff),
        ("cvt_rn_relu_bf16x2_f32 NaN", nan_results[3], 0x7fff_7fff),
    ] {
        if packed == expected_nan {
            println!("{label}: PASS ({packed:#010x})");
        } else {
            eprintln!(
                "{label}: FAIL, expected canonical NaNs {expected_nan:#010x}, got {packed:#010x}"
            );
            failures += 1;
        }
    }

    // --- Pack/unpack round trip ---
    // The generated packers narrow (lossy, rounding-mode dependent); the
    // hand-written unpackers widen exactly. So the round trip lands on the
    // narrowed value widened back to f32, and the host can check exact bits:
    //   f16 rn:  1.0065 -> 0x3C07 -> 1.0068359375
    //   bf16 rz: 1.0065 -> 0x3F80 -> 1.0
    let mut unpack_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_UNPACKED).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cvt_packed_unpack(&stream, cfg, lo, hi, &mut unpack_dev) }
        .expect("Unpack kernel launch failed");
    let unpacked = unpack_dev.to_host_vec(&stream).unwrap();
    assert_eq!(unpacked.len(), NUM_UNPACKED);

    // Exactly 1 + 7/1024: the f16 value 0x3C07 widened to f32 (see table above).
    let f16_widened = 1.0_f32 + 7.0 / 1024.0;
    let unpack_labels = [
        "cvt_f32x2_f16x2 lo",
        "cvt_f32x2_f16x2 hi",
        "cvt_f32_f16x2_lo",
        "cvt_f32_f16x2_hi",
        "cvt_f32x2_bf16x2 lo",
        "cvt_f32x2_bf16x2 hi",
        "cvt_f32_bf16x2_lo",
        "cvt_f32_bf16x2_hi",
    ];
    let unpack_expected = [
        f16_widened,
        -f16_widened,
        f16_widened,
        -f16_widened,
        1.0,
        -1.0,
        1.0,
        -1.0,
    ];
    for i in 0..NUM_UNPACKED {
        let got = unpacked[i];
        let want = unpack_expected[i];
        if got.to_bits() == want.to_bits() {
            println!("[{i}] {}: PASS ({got})", unpack_labels[i]);
        } else {
            eprintln!(
                "[{i}] {}: FAIL, expected {want}, got {got}",
                unpack_labels[i]
            );
            failures += 1;
        }
    }

    // --- Summary ---
    println!();
    if failures == 0 {
        println!(
            "SUCCESS: all {} packed cvt variants and {} unpacked values produced correct results",
            NUM_VARIANTS, NUM_UNPACKED
        );
    } else {
        eprintln!("{failures} check(s) failed");
        std::process::exit(1);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Stress fixtures: `a / a` and `a - a` are the test inputs, and the
// 15-digit hex literal is shaped that way deliberately.
#![allow(clippy::eq_op, clippy::unusual_byte_groupings)]
#![feature(core_float_math)]

//! Primitive scalar stress test.
//!
//! Exercises primitive types and methods that are easy to miss in MIR import
//! and lowering: `char`, `u128`/`i128`, pointer-sized integers, and rustc
//! compiler intrinsics used by primitive integer and float methods.
//!
//! # How this example loads code on the GPU
//!
//! The float-math kernel lowers to CUDA `__nv_*` libdevice calls (`__nv_sinf`,
//! `__nv_powf`, ...). `llc` cannot resolve those, so cuda-oxide auto-detects
//! them and emits NVVM IR (`primitive_stress.ll`) instead of `.ptx`.
//! `#[cuda_module]` then loads the embedded NVVM IR, runs the libNVVM
//! (with libdevice) + nvJitLink pipeline, and launches through the generated
//! typed API -- no external tools, no symlinks, no boilerplate.
//!
//! Run: `cargo oxide run primitive_stress`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Checks that `char` imports as a 32-bit scalar and casts cleanly to `u32`.
    #[kernel]
    pub fn test_char(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            let ascii: char = 'A';
            let wide: char = '\u{1f980}';
            let value = (ascii as u32)
                .wrapping_add(wide as u32)
                .wrapping_add(core::mem::size_of::<char>() as u32);

            unsafe {
                *out.get_unchecked_mut(0) = value;
            }
        }
    }

    /// Checks wide integer constants, arithmetic, shifts, and argument passing.
    #[kernel]
    pub fn test_u128_i128(a: u128, b: u128, c: i128, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            let unsigned = a
                .wrapping_mul(3)
                .wrapping_add(b)
                .wrapping_mul(0x1_0001_u128)
                .wrapping_add(b >> 111)
                ^ 0xfeed_face_cafe_beef_0123_4567_89ab_cdef_u128;
            let signed = c.wrapping_mul(-7).wrapping_add(0x1234_5678_9abc_def0_i128);
            let signed_bits = signed as u128;

            unsafe {
                *out.get_unchecked_mut(0) = unsigned as u64;
                *out.get_unchecked_mut(1) = (unsigned >> 64) as u64;
                *out.get_unchecked_mut(2) = signed_bits as u64;
                *out.get_unchecked_mut(3) = (signed_bits >> 64) as u64;
            }
        }
    }

    /// Checks target pointer-sized integer import and arithmetic.
    #[kernel]
    pub fn test_pointer_sized(a: usize, b: isize, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            let unsigned = a
                .wrapping_mul(5)
                .wrapping_add(core::mem::size_of::<usize>());
            let signed = b
                .wrapping_mul(-3)
                .wrapping_sub(core::mem::size_of::<isize>() as isize);

            unsafe {
                *out.get_unchecked_mut(0) = unsigned as u64;
                *out.get_unchecked_mut(1) = signed as u64;
            }
        }
    }

    /// Checks primitive integer methods that call rustc bit intrinsics in libcore.
    #[kernel]
    pub fn test_bit_intrinsics(a: u128, b: u64, c: u32, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            let wide = a.rotate_left(17) ^ a.rotate_right(29) ^ a.swap_bytes() ^ a.reverse_bits();
            let wide_counts = (a.count_ones() as u64) | ((a.leading_zeros() as u64) << 32);
            let wide_trailing = a.trailing_zeros() as u64;

            let mid = b.rotate_left(7) ^ b.rotate_right(13) ^ b.swap_bytes() ^ b.reverse_bits();
            let narrow = c.rotate_left(5) ^ c.rotate_right(11) ^ c.swap_bytes() ^ c.reverse_bits();
            let narrow_counts = (c.count_ones() as u64)
                | ((c.leading_zeros() as u64) << 32)
                | ((c.trailing_zeros() as u64) << 48);

            unsafe {
                *out.get_unchecked_mut(0) = wide as u64;
                *out.get_unchecked_mut(1) = (wide >> 64) as u64;
                *out.get_unchecked_mut(2) = wide_counts;
                *out.get_unchecked_mut(3) = wide_trailing;
                *out.get_unchecked_mut(4) = mid;
                *out.get_unchecked_mut(5) = narrow as u64;
                *out.get_unchecked_mut(6) = narrow_counts;
            }
        }
    }

    /// Checks integer methods that call rustc saturating arithmetic intrinsics.
    #[kernel]
    pub fn test_saturating_intrinsics(
        u8_val: u8,
        i8_val: i8,
        u64_val: u64,
        i64_val: i64,
        mut out: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() == 0 {
            let signed_small = i8_val.saturating_add(10) as i64 as u64;
            let signed_wide = i64_val.saturating_sub(10) as u64;

            unsafe {
                *out.get_unchecked_mut(0) = u8_val.saturating_add(10) as u64;
                *out.get_unchecked_mut(1) = 0_u8.saturating_sub(1) as u64;
                *out.get_unchecked_mut(2) = signed_small;
                *out.get_unchecked_mut(3) = (-120_i8).saturating_sub(20) as i64 as u64;
                *out.get_unchecked_mut(4) = u64_val.saturating_add(10);
                *out.get_unchecked_mut(5) = 1_u64.saturating_sub(10);
                *out.get_unchecked_mut(6) = signed_wide;
            }
        }
    }

    /// Checks float methods that call rustc math intrinsics.
    #[kernel]
    pub fn test_float_math_intrinsics(seed32: f32, seed64: f64, mut out: DisjointSlice<u64>) {
        if thread::index_1d().get() == 0 {
            let zero32 = seed32 - seed32;
            let one32 = seed32 / seed32;
            let two32 = seed32;
            let three32 = two32 + one32;
            let four32 = two32 * two32;
            let eight32 = four32 * two32;
            let ten32 = eight32 + two32;
            let hundred32 = ten32 * ten32;
            let half32 = one32 / two32;
            let round32 = two32 + half32;
            let neg32 = -(three32 + half32);

            let zero64 = seed64 - seed64;
            let one64 = seed64 / seed64;
            let two64 = seed64;
            let three64 = two64 + one64;
            let four64 = two64 * two64;
            let eight64 = four64 * two64;
            let ten64 = eight64 + two64;
            let hundred64 = ten64 * ten64;
            let half64 = one64 / two64;
            let round64 = two64 + half64;
            let neg64 = -(three64 + half64);

            unsafe {
                *out.get_unchecked_mut(0) = neg32.abs().to_bits() as u64;
                *out.get_unchecked_mut(1) = two32.copysign(neg32).to_bits() as u64;
                *out.get_unchecked_mut(2) = neg32.floor().to_bits() as u64;
                *out.get_unchecked_mut(3) = neg32.ceil().to_bits() as u64;
                *out.get_unchecked_mut(4) = round32.round().to_bits() as u64;
                *out.get_unchecked_mut(5) = neg32.trunc().to_bits() as u64;
                *out.get_unchecked_mut(6) = two32.mul_add(three32, four32).to_bits() as u64;
                *out.get_unchecked_mut(7) = two32.powi(3).to_bits() as u64;
                *out.get_unchecked_mut(8) = two32.powf(three32).to_bits() as u64;
                *out.get_unchecked_mut(9) = four32.sqrt().to_bits() as u64;
                *out.get_unchecked_mut(10) = zero32.exp().to_bits() as u64;
                *out.get_unchecked_mut(11) = three32.exp2().to_bits() as u64;
                *out.get_unchecked_mut(12) = one32.ln().to_bits() as u64;
                *out.get_unchecked_mut(13) = eight32.log2().to_bits() as u64;
                *out.get_unchecked_mut(14) = hundred32.log10().to_bits() as u64;
                *out.get_unchecked_mut(15) = zero32.sin().to_bits() as u64;
                *out.get_unchecked_mut(16) = zero32.cos().to_bits() as u64;

                *out.get_unchecked_mut(17) = neg64.abs().to_bits();
                *out.get_unchecked_mut(18) = two64.copysign(neg64).to_bits();
                *out.get_unchecked_mut(19) = neg64.floor().to_bits();
                *out.get_unchecked_mut(20) = neg64.ceil().to_bits();
                *out.get_unchecked_mut(21) = round64.round().to_bits();
                *out.get_unchecked_mut(22) = neg64.trunc().to_bits();
                *out.get_unchecked_mut(23) = two64.mul_add(three64, four64).to_bits();
                *out.get_unchecked_mut(24) = two64.powi(3).to_bits();
                *out.get_unchecked_mut(25) = two64.powf(three64).to_bits();
                *out.get_unchecked_mut(26) = four64.sqrt().to_bits();
                *out.get_unchecked_mut(27) = zero64.exp().to_bits();
                *out.get_unchecked_mut(28) = three64.exp2().to_bits();
                *out.get_unchecked_mut(29) = one64.ln().to_bits();
                *out.get_unchecked_mut(30) = eight64.log2().to_bits();
                *out.get_unchecked_mut(31) = hundred64.log10().to_bits();
                *out.get_unchecked_mut(32) = zero64.sin().to_bits();
                *out.get_unchecked_mut(33) = zero64.cos().to_bits();

                *out.get_unchecked_mut(34) = core::f32::math::floor(neg32).to_bits() as u64;
                *out.get_unchecked_mut(35) = core::f32::math::ceil(neg32).to_bits() as u64;
                *out.get_unchecked_mut(36) = core::f32::math::round(round32).to_bits() as u64;
                *out.get_unchecked_mut(37) =
                    core::f32::math::round_ties_even(round32).to_bits() as u64;
                *out.get_unchecked_mut(38) = core::f32::math::trunc(neg32).to_bits() as u64;
                *out.get_unchecked_mut(39) =
                    core::f32::math::mul_add(two32, three32, four32).to_bits() as u64;
                *out.get_unchecked_mut(40) = core::f32::math::powi(two32, 3).to_bits() as u64;
                *out.get_unchecked_mut(41) = core::f32::math::sqrt(four32).to_bits() as u64;

                *out.get_unchecked_mut(42) = core::f64::math::floor(neg64).to_bits();
                *out.get_unchecked_mut(43) = core::f64::math::ceil(neg64).to_bits();
                *out.get_unchecked_mut(44) = core::f64::math::round(round64).to_bits();
                *out.get_unchecked_mut(45) = core::f64::math::round_ties_even(round64).to_bits();
                *out.get_unchecked_mut(46) = core::f64::math::trunc(neg64).to_bits();
                *out.get_unchecked_mut(47) =
                    core::f64::math::mul_add(two64, three64, four64).to_bits();
                *out.get_unchecked_mut(48) = core::f64::math::powi(two64, 3).to_bits();
                *out.get_unchecked_mut(49) = core::f64::math::sqrt(four64).to_bits();
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Primitive Stress Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Loads the embedded artifact. In this example that is NVVM IR because
    // the float-math kernels need libdevice.
    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_char(&stream, cfg, &mut out) }?;
        let result = out.to_host_vec(&stream)?[0];
        let expected = ('A' as u32)
            .wrapping_add('\u{1f980}' as u32)
            .wrapping_add(core::mem::size_of::<char>() as u32);
        check(
            "char constants and casts",
            result,
            expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_0000_0000_0000_0000_0000_0000_0011_u128;
        let b = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
        let c = -0x1234_5678_9abc_def_i128;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 4)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u128_i128(&stream, cfg, a, b, c, &mut out) }?;
        let result = out.to_host_vec(&stream)?;
        let unsigned = a
            .wrapping_mul(3)
            .wrapping_add(b)
            .wrapping_mul(0x1_0001_u128)
            .wrapping_add(b >> 111)
            ^ 0xfeed_face_cafe_beef_0123_4567_89ab_cdef_u128;
        let signed_bits = c.wrapping_mul(-7).wrapping_add(0x1234_5678_9abc_def0_i128) as u128;
        let expected = [
            unsigned as u64,
            (unsigned >> 64) as u64,
            signed_bits as u64,
            (signed_bits >> 64) as u64,
        ];
        check_slice(
            "u128/i128 arithmetic",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_1234_usize;
        let b = -12345_isize;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 2)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_pointer_sized(&stream, cfg, a, b, &mut out) }?;
        let result = out.to_host_vec(&stream)?;
        let expected = [
            a.wrapping_mul(5)
                .wrapping_add(core::mem::size_of::<usize>()) as u64,
            b.wrapping_mul(-3)
                .wrapping_sub(core::mem::size_of::<isize>() as isize) as u64,
        ];
        check_slice(
            "usize/isize arithmetic",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_0000_0000_0000_0123_4567_89ab_cdef_u128;
        let b = 0x0123_4567_89ab_cdef_u64;
        let c = 0x8020_0401_u32;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 7)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_bit_intrinsics(&stream, cfg, a, b, c, &mut out) }?;
        let result = out.to_host_vec(&stream)?;
        let wide = a.rotate_left(17) ^ a.rotate_right(29) ^ a.swap_bytes() ^ a.reverse_bits();
        let wide_counts = (a.count_ones() as u64) | ((a.leading_zeros() as u64) << 32);
        let mid = b.rotate_left(7) ^ b.rotate_right(13) ^ b.swap_bytes() ^ b.reverse_bits();
        let narrow = c.rotate_left(5) ^ c.rotate_right(11) ^ c.swap_bytes() ^ c.reverse_bits();
        let narrow_counts = (c.count_ones() as u64)
            | ((c.leading_zeros() as u64) << 32)
            | ((c.trailing_zeros() as u64) << 48);
        let expected = [
            wide as u64,
            (wide >> 64) as u64,
            wide_counts,
            a.trailing_zeros() as u64,
            mid,
            narrow as u64,
            narrow_counts,
        ];
        check_slice(
            "bit intrinsic methods",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let u8_val = 250_u8;
        let i8_val = 120_i8;
        let u64_val = u64::MAX - 2;
        let i64_val = i64::MIN + 2;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 7)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_saturating_intrinsics(
                &stream, cfg, u8_val, i8_val, u64_val, i64_val, &mut out,
            )
        }?;
        let result = out.to_host_vec(&stream)?;
        let expected = [
            u8_val.saturating_add(10) as u64,
            0_u8.saturating_sub(1) as u64,
            i8_val.saturating_add(10) as i64 as u64,
            (-120_i8).saturating_sub(20) as i64 as u64,
            u64_val.saturating_add(10),
            1_u64.saturating_sub(10),
            i64_val.saturating_sub(10) as u64,
        ];
        check_slice(
            "saturating arithmetic intrinsics",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let seed32 = 2.0_f32;
        let seed64 = 2.0_f64;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 50)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_float_math_intrinsics(&stream, cfg, seed32, seed64, &mut out) }?;
        let result = out.to_host_vec(&stream)?;
        let expected = expected_float_math_bits(seed32, seed64);
        // Allow up to 2 ULPs for transcendentals: CUDA `libdevice` uses
        // approximation-based implementations of `log`, `log2`, `log10`,
        // `exp`, `pow`, `sin`, `cos`. Host `libm` is often bit-exact for
        // power-of-two inputs (e.g. `log2(8.0) = 3.0`), libdevice may be
        // 1 ULP off. Ops that are bit-exact in IEEE-754 (`abs`, `copysign`,
        // `floor`, `ceil`, `trunc`, `round`, `sqrt`, `fma`) still pass at 0
        // ULP because their results match exactly.
        check_float_bits_slice_ulp(
            "float math intrinsics",
            &result,
            &expected,
            2,
            &mut passed,
            &mut failed,
        );
    }

    println!("\n=== Results ===");
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if failed == 0 {
        println!("\nPASS: primitive scalar checks matched");
        Ok(())
    } else {
        eprintln!("\nFAIL: {failed} primitive scalar checks failed");
        std::process::exit(1);
    }
}

fn expected_float_math_bits(seed32: f32, seed64: f64) -> Vec<u64> {
    let zero32 = seed32 - seed32;
    let one32 = seed32 / seed32;
    let two32 = seed32;
    let three32 = two32 + one32;
    let four32 = two32 * two32;
    let eight32 = four32 * two32;
    let ten32 = eight32 + two32;
    let hundred32 = ten32 * ten32;
    let half32 = one32 / two32;
    let round32 = two32 + half32;
    let neg32 = -(three32 + half32);

    let zero64 = seed64 - seed64;
    let one64 = seed64 / seed64;
    let two64 = seed64;
    let three64 = two64 + one64;
    let four64 = two64 * two64;
    let eight64 = four64 * two64;
    let ten64 = eight64 + two64;
    let hundred64 = ten64 * ten64;
    let half64 = one64 / two64;
    let round64 = two64 + half64;
    let neg64 = -(three64 + half64);

    vec![
        neg32.abs().to_bits() as u64,
        two32.copysign(neg32).to_bits() as u64,
        neg32.floor().to_bits() as u64,
        neg32.ceil().to_bits() as u64,
        round32.round().to_bits() as u64,
        neg32.trunc().to_bits() as u64,
        two32.mul_add(three32, four32).to_bits() as u64,
        two32.powi(3).to_bits() as u64,
        two32.powf(three32).to_bits() as u64,
        four32.sqrt().to_bits() as u64,
        zero32.exp().to_bits() as u64,
        three32.exp2().to_bits() as u64,
        one32.ln().to_bits() as u64,
        eight32.log2().to_bits() as u64,
        hundred32.log10().to_bits() as u64,
        zero32.sin().to_bits() as u64,
        zero32.cos().to_bits() as u64,
        neg64.abs().to_bits(),
        two64.copysign(neg64).to_bits(),
        neg64.floor().to_bits(),
        neg64.ceil().to_bits(),
        round64.round().to_bits(),
        neg64.trunc().to_bits(),
        two64.mul_add(three64, four64).to_bits(),
        two64.powi(3).to_bits(),
        two64.powf(three64).to_bits(),
        four64.sqrt().to_bits(),
        zero64.exp().to_bits(),
        three64.exp2().to_bits(),
        one64.ln().to_bits(),
        eight64.log2().to_bits(),
        hundred64.log10().to_bits(),
        zero64.sin().to_bits(),
        zero64.cos().to_bits(),
        core::f32::math::floor(neg32).to_bits() as u64,
        core::f32::math::ceil(neg32).to_bits() as u64,
        core::f32::math::round(round32).to_bits() as u64,
        core::f32::math::round_ties_even(round32).to_bits() as u64,
        core::f32::math::trunc(neg32).to_bits() as u64,
        core::f32::math::mul_add(two32, three32, four32).to_bits() as u64,
        core::f32::math::powi(two32, 3).to_bits() as u64,
        core::f32::math::sqrt(four32).to_bits() as u64,
        core::f64::math::floor(neg64).to_bits(),
        core::f64::math::ceil(neg64).to_bits(),
        core::f64::math::round(round64).to_bits(),
        core::f64::math::round_ties_even(round64).to_bits(),
        core::f64::math::trunc(neg64).to_bits(),
        core::f64::math::mul_add(two64, three64, four64).to_bits(),
        core::f64::math::powi(two64, 3).to_bits(),
        core::f64::math::sqrt(four64).to_bits(),
    ]
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

/// Layout of `test_float_math_intrinsics`'s `out` slice. Each tuple is the
/// half-open index range and the bit width (32 = f32 bits in low half of u64,
/// 64 = full f64 bits).
const FLOAT_MATH_SECTIONS: &[(usize, usize, u32)] = &[
    (0, 17, 32),  // f32 inherent methods
    (17, 34, 64), // f64 inherent methods
    (34, 42, 32), // core::f32::math::*
    (42, 50, 64), // core::f64::math::*
];

/// Width in bits (32 or 64) of `out[i]` interpreted as float bits.
fn float_math_width(i: usize) -> u32 {
    for (lo, hi, w) in FLOAT_MATH_SECTIONS {
        if i >= *lo && i < *hi {
            return *w;
        }
    }
    panic!("index {i} out of range for FLOAT_MATH_SECTIONS");
}

/// Distance in ULPs between two finite floats stored as `to_bits()` u64.
///
/// Both values must have the same width (`32` for f32 in low bits, `64` for f64).
/// Returns `u64::MAX` if the values differ in sign (e.g. `-0` vs `+0` is `0`,
/// but `-1.0` vs `+1.0` is huge), which forces a failure on those.
fn ulp_distance(a_bits: u64, b_bits: u64, width: u32) -> u64 {
    fn map_to_monotonic(bits: u64, sign_mask: u64, body_mask: u64) -> u64 {
        if bits & sign_mask != 0 {
            sign_mask - (bits & body_mask)
        } else {
            sign_mask + (bits & body_mask)
        }
    }
    let (a, b, sign_mask, body_mask) = match width {
        32 => (
            a_bits & 0xFFFF_FFFF,
            b_bits & 0xFFFF_FFFF,
            0x8000_0000,
            0x7FFF_FFFF,
        ),
        64 => (a_bits, b_bits, 0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF),
        _ => unreachable!(),
    };
    let am = map_to_monotonic(a, sign_mask, body_mask);
    let bm = map_to_monotonic(b, sign_mask, body_mask);
    am.abs_diff(bm)
}

/// Like `check_slice` but treats each entry as IEEE-754 bits and accepts up
/// to `tol_ulps` of difference. Used for transcendentals where `libdevice`
/// may diverge from host `libm` by ~1 ULP.
fn check_float_bits_slice_ulp(
    name: &str,
    got: &[u64],
    expected: &[u64],
    tol_ulps: u64,
    passed: &mut u32,
    failed: &mut u32,
) {
    assert_eq!(got.len(), expected.len(), "length mismatch in {name}");
    let mut bad = Vec::new();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = ulp_distance(*g, *e, float_math_width(i));
        if d > tol_ulps {
            bad.push((i, *g, *e, d));
        }
    }
    if bad.is_empty() {
        println!(
            "PASS {name}: all {} values within {tol_ulps} ULP",
            got.len()
        );
        *passed += 1;
    } else {
        println!(
            "FAIL {name}: {} of {} values exceed {tol_ulps} ULP",
            bad.len(),
            got.len()
        );
        for (i, g, e, d) in &bad {
            println!("  [{i}] got 0x{g:016x}, expected 0x{e:016x}, ulp_dist={d}");
        }
        *failed += 1;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Numeric Stress Test Suite
//!
//! Stress-tests the compiler's numeric handling. Exercises unsigned
//! division/remainder with MSB-set values, unsigned comparisons, signed
//! shift right, mixed signedness, wrapping overflow, and width-crossing casts.
//!
//! This example serves as a regression test for signedness bugs in the
//! arithmetic lowering (sdiv vs udiv, srem vs urem, sgt vs ugt, lshr vs ashr).
//!
//! Run: cargo oxide run numeric_stress

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// 1. UNSIGNED DIVISION / REMAINDER (MSB-set values)
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// u32 division where MSB is set: 0xFFFF_FFFE / 2 = 0x7FFF_FFFF (2147483647)
    /// Bug: sdiv interprets 0xFFFF_FFFE as -2, giving -2/2 = -1 = 0xFFFF_FFFF
    #[kernel]
    pub fn test_u32_div_msb(a: u32, b: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a / b;
        }
    }

    /// u32 remainder: 0xFFFF_FFFF % 10 = 5
    /// Bug: srem interprets 0xFFFF_FFFF as -1, giving -1 % 10 = -1 (wrong)
    #[kernel]
    pub fn test_u32_rem_msb(a: u32, b: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a % b;
        }
    }

    /// u64 division with large value: 0x8000_0000_0000_0000 / 2 = 0x4000_0000_0000_0000
    #[kernel]
    pub fn test_u64_div_msb(a: u64, b: u64, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a / b;
        }
    }

    /// u64 remainder: u64::MAX % 7 = 1
    /// (18446744073709551615 = 7 * 2635249153387078802 + 1)
    #[kernel]
    pub fn test_u64_rem_msb(a: u64, b: u64, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a % b;
        }
    }

    // =============================================================================
    // 2. UNSIGNED COMPARISONS (MSB-set values)
    // =============================================================================

    /// u32 comparison: 0x8000_0000 > 0x7FFF_FFFF should be true (1)
    /// Bug: sgt interprets 0x8000_0000 as -2147483648, so -2147483648 > 2147483647 = false
    #[kernel]
    pub fn test_u32_cmp_gt_msb(a: u32, b: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = if a > b { 1 } else { 0 };
        }
    }

    /// u64 comparison: u64::MAX > 0 should be true
    #[kernel]
    pub fn test_u64_cmp_gt_msb(a: u64, b: u64, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = if a > b { 1 } else { 0 };
        }
    }

    /// u32 less-than: 0x7FFF_FFFF < 0x8000_0000 should be true
    #[kernel]
    pub fn test_u32_cmp_lt_msb(a: u32, b: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = if a < b { 1 } else { 0 };
        }
    }

    // =============================================================================
    // 3. SIGNED SHIFT RIGHT (arithmetic shift)
    // =============================================================================

    /// i32 arithmetic shift right: -8 >> 1 should be -4 (sign-extending)
    /// Bug: lshr gives 0x7FFF_FFFC = 2147483644 instead
    #[kernel]
    pub fn test_i32_shr_negative(a: i32, b: u32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a >> b;
        }
    }

    /// i64 arithmetic shift: -1024 >> 3 should be -128
    #[kernel]
    pub fn test_i64_shr_negative(a: i64, b: u32, mut out: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a >> b;
        }
    }

    // =============================================================================
    // 4. MIXED SIGNED / UNSIGNED IN SAME KERNEL
    // =============================================================================

    /// Verify signed division and unsigned division both work in same kernel
    #[kernel]
    pub fn test_mixed_div(
        signed_a: i32,
        signed_b: i32,
        unsigned_a: u32,
        unsigned_b: u32,
        mut out_signed: DisjointSlice<i32>,
        mut out_unsigned: DisjointSlice<u32>,
    ) {
        let signed_idx = thread::index_1d();
        if let Some(s) = out_signed.get_mut(signed_idx) {
            *s = signed_a / signed_b;
        }
        let unsigned_idx = thread::index_1d();
        if let Some(u) = out_unsigned.get_mut(unsigned_idx) {
            *u = unsigned_a / unsigned_b;
        }
    }

    // =============================================================================
    // 5. WRAPPING ARITHMETIC
    // =============================================================================

    /// u32 wrapping add: MAX + 1 = 0
    #[kernel]
    pub fn test_u32_wrapping_add(a: u32, b: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a.wrapping_add(b);
        }
    }

    /// i32 wrapping mul: i32::MAX * 2 wraps
    #[kernel]
    pub fn test_i32_wrapping_mul(a: i32, b: i32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a.wrapping_mul(b);
        }
    }

    // =============================================================================
    // 6. WIDTH-CROSSING CASTS
    // =============================================================================

    /// Widening chain: u8 -> u32 -> u64
    #[kernel]
    pub fn test_widening_chain(a: u8, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mid = a as u32;
            *out_elem = mid as u64;
        }
    }

    /// Narrowing chain: i64 -> i32 (with sign preservation)
    #[kernel]
    pub fn test_narrowing_signed(a: i64, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a as i32;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Numeric Stress Test Suite ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut errors: Vec<String> = Vec::new();

    // =========================================================================
    // 1. UNSIGNED DIVISION / REMAINDER (MSB-set values)
    // =========================================================================
    println!("--- Unsigned Division / Remainder (MSB-set values) ---");

    // u32 div: 0xFFFF_FFFE / 2 = 0x7FFF_FFFF
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u32_div_msb((stream).as_ref(), cfg, 0xFFFF_FFFEu32, 2u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 0x7FFF_FFFFu32 {
            println!("  [PASS] u32 div MSB: 0xFFFFFFFE / 2 = 0x{:08X}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "u32 div MSB: expected 0x7FFFFFFF (2147483647), got 0x{:08X} ({})",
                r[0], r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u32 rem: 0xFFFF_FFFF % 10 = 5
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_u32_rem_msb((stream).as_ref(), cfg, 0xFFFF_FFFFu32, 10u32, &mut out)
        }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 5u32 {
            println!("  [PASS] u32 rem MSB: 0xFFFFFFFF % 10 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("u32 rem MSB: expected 5, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u64 div: 0x8000_0000_0000_0000 / 2 = 0x4000_0000_0000_0000
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_u64_div_msb(
                (stream).as_ref(),
                cfg,
                0x8000_0000_0000_0000u64,
                2u64,
                &mut out,
            )
        }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 0x4000_0000_0000_0000u64 {
            println!(
                "  [PASS] u64 div MSB: 0x8000000000000000 / 2 = 0x{:016X}",
                r[0]
            );
            passed += 1;
        } else {
            let msg = format!(
                "u64 div MSB: expected 0x4000000000000000, got 0x{:016X} ({})",
                r[0], r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u64 rem: u64::MAX % 7 = 1
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u64_rem_msb((stream).as_ref(), cfg, u64::MAX, 7u64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 1u64 {
            println!("  [PASS] u64 rem MSB: u64::MAX % 7 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("u64 rem MSB: expected 1, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 2. UNSIGNED COMPARISONS (MSB-set values)
    // =========================================================================
    println!("\n--- Unsigned Comparisons (MSB-set values) ---");

    // u32 gt: 0x8000_0000 > 0x7FFF_FFFF = true (1)
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_u32_cmp_gt_msb(
                (stream).as_ref(),
                cfg,
                0x8000_0000u32,
                0x7FFF_FFFFu32,
                &mut out,
            )
        }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 1u32 {
            println!("  [PASS] u32 gt MSB: 0x80000000 > 0x7FFFFFFF = true");
            passed += 1;
        } else {
            let msg = format!(
                "u32 gt MSB: expected 1 (true), got {} (0x80000000 > 0x7FFFFFFF)",
                r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u64 gt: u64::MAX > 0 = true (1)
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u64_cmp_gt_msb((stream).as_ref(), cfg, u64::MAX, 0u64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 1u32 {
            println!("  [PASS] u64 gt MSB: u64::MAX > 0 = true");
            passed += 1;
        } else {
            let msg = format!("u64 gt MSB: expected 1 (true), got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u32 lt: 0x7FFF_FFFF < 0x8000_0000 = true (1)
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_u32_cmp_lt_msb(
                (stream).as_ref(),
                cfg,
                0x7FFF_FFFFu32,
                0x8000_0000u32,
                &mut out,
            )
        }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 1u32 {
            println!("  [PASS] u32 lt MSB: 0x7FFFFFFF < 0x80000000 = true");
            passed += 1;
        } else {
            let msg = format!(
                "u32 lt MSB: expected 1 (true), got {} (0x7FFFFFFF < 0x80000000)",
                r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 3. SIGNED SHIFT RIGHT (arithmetic shift)
    // =========================================================================
    println!("\n--- Signed Shift Right (arithmetic shift) ---");

    // i32 shr: -8 >> 1 = -4
    {
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_i32_shr_negative((stream).as_ref(), cfg, -8i32, 1u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -4i32 {
            println!("  [PASS] i32 shr: -8 >> 1 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("i32 shr: expected -4, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // i64 shr: -1024 >> 3 = -128
    {
        let mut out = DeviceBuffer::<i64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_i64_shr_negative((stream).as_ref(), cfg, -1024i64, 3u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -128i64 {
            println!("  [PASS] i64 shr: -1024 >> 3 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("i64 shr: expected -128, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 4. MIXED SIGNED / UNSIGNED IN SAME KERNEL
    // =========================================================================
    println!("\n--- Mixed Signed / Unsigned in Same Kernel ---");

    // signed: -7 / 2 = -3; unsigned: 0xFFFF_FFFE / 2 = 0x7FFF_FFFF
    {
        let mut out_signed = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        let mut out_unsigned = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_mixed_div(
                (stream).as_ref(),
                cfg,
                -7i32,
                2i32,
                0xFFFF_FFFEu32,
                2u32,
                &mut out_signed,
                &mut out_unsigned,
            )
        }?;
        let rs = out_signed.to_host_vec(&stream)?;
        let ru = out_unsigned.to_host_vec(&stream)?;
        let signed_ok = rs[0] == -3i32;
        let unsigned_ok = ru[0] == 0x7FFF_FFFFu32;
        if signed_ok && unsigned_ok {
            println!(
                "  [PASS] mixed div: signed -7/2={}, unsigned 0xFFFFFFFE/2=0x{:08X}",
                rs[0], ru[0]
            );
            passed += 1;
        } else {
            let msg = format!(
                "mixed div: signed expected -3 got {}, unsigned expected 0x7FFFFFFF got 0x{:08X}",
                rs[0], ru[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 5. WRAPPING ARITHMETIC
    // =========================================================================
    println!("\n--- Wrapping Arithmetic ---");

    // u32 wrapping add: MAX + 1 = 0
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u32_wrapping_add((stream).as_ref(), cfg, u32::MAX, 1u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 0u32 {
            println!("  [PASS] u32 wrapping add: MAX + 1 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("u32 wrapping add: expected 0, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // i32 wrapping mul: i32::MAX * 2 = -2
    {
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_i32_wrapping_mul((stream).as_ref(), cfg, i32::MAX, 2i32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -2i32 {
            println!("  [PASS] i32 wrapping mul: MAX * 2 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("i32 wrapping mul: expected -2, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 6. WIDTH-CROSSING CASTS
    // =========================================================================
    println!("\n--- Width-Crossing Casts ---");

    // widening chain: u8(0xFF) -> u32 -> u64 = 255
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_widening_chain((stream).as_ref(), cfg, 0xFFu8, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 255u64 {
            println!("  [PASS] widening chain: u8(0xFF) -> u32 -> u64 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("widening chain: expected 255, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // narrowing signed: i64(-42) -> i32 = -42
    {
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_narrowing_signed((stream).as_ref(), cfg, -42i64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -42i32 {
            println!("  [PASS] narrowing signed: i64(-42) -> i32 = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("narrowing signed: expected -42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n=========================================");
    println!("RESULTS: {} passed, {} failed", passed, failed);
    println!("=========================================");

    if !errors.is_empty() {
        println!("\nFailures:");
        for (i, e) in errors.iter().enumerate() {
            println!("  {}. {}", i + 1, e);
        }
    }

    if failed > 0 {
        std::process::exit(1);
    }

    println!("\n=== All {} numeric stress tests PASSED! ===", passed);

    Ok(())
}

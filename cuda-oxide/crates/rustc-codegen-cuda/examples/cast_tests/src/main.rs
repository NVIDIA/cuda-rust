/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::approx_constant)]

//! Cast Operations Test Suite
//!
//! Tests MIR cast kind handling in the codegen backend. Exercises each
//! `CastKind` variant that can appear in device code.
//!
//! Run: cargo oxide run cast_tests
//!
//! ## Test Categories
//!
//! 1. Numeric casts (IntToInt, IntToFloat, FloatToInt, FloatToFloat)
//! 2. Transmute (CastKind::Transmute) — bit reinterpretation via bitcast
//! 3. Pointer casts (PtrToPtr, PointerExposeProvenance, PointerWithExposedProvenance)
//! 4. ConstantIndex on slices (Bug 2 regression test)
//! 5. Unsizing coercions (PointerCoercion::Unsize) — fat pointer construction
//!
//! All casts dispatch on `MirCastKindAttr` (preserved from Rust MIR) to select
//! the correct LLVM instruction. Struct↔ptr conversions (fat pointers, newtypes)
//! are handled generically via `emit_pointer_cast` in mir-lower.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// 1. NUMERIC CASTS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// IntToInt: u32 → u64 (zero extension)
    #[kernel]
    pub fn test_cast_u32_to_u64(val: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u64;
        }
    }

    /// IntToInt: u64 → u32 (truncation)
    #[kernel]
    pub fn test_cast_u64_to_u32(val: u64, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u32;
        }
    }

    /// IntToInt: i32 → i64 (sign extension)
    #[kernel]
    pub fn test_cast_i32_to_i64(val: i32, mut out: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as i64;
        }
    }

    /// IntToFloat: u32 → f32 (value conversion)
    #[kernel]
    pub fn test_cast_u32_to_f32(val: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as f32;
        }
    }

    /// IntToFloat: i32 → f32 (signed value conversion)
    #[kernel]
    pub fn test_cast_i32_to_f32(val: i32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as f32;
        }
    }

    /// FloatToInt: f32 → u32 (truncating value conversion)
    #[kernel]
    pub fn test_cast_f32_to_u32(val: f32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u32;
        }
    }

    /// FloatToInt: f32 → i32 (truncating signed conversion)
    #[kernel]
    pub fn test_cast_f32_to_i32(val: f32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as i32;
        }
    }

    /// FloatToInt: f64 → u32
    #[kernel]
    pub fn test_cast_f64_to_u32(val: f64, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u32;
        }
    }

    /// FloatToInt: f64 → i32
    #[kernel]
    pub fn test_cast_f64_to_i32(val: f64, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as i32;
        }
    }

    /// FloatToInt: f64 → u64
    #[kernel]
    pub fn test_cast_f64_to_u64(val: f64, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u64;
        }
    }

    /// FloatToInt: f64 → i64
    #[kernel]
    pub fn test_cast_f64_to_i64(val: f64, mut out: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as i64;
        }
    }

    /// FloatToInt: f32 → u64 (mixed precision)
    #[kernel]
    pub fn test_cast_f32_to_u64(val: f32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as u64;
        }
    }

    /// FloatToInt: f32 → i64 (mixed precision)
    #[kernel]
    pub fn test_cast_f32_to_i64(val: f32, mut out: DisjointSlice<i64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as i64;
        }
    }

    /// FloatToFloat: f32 → f64 (precision extension)
    #[kernel]
    pub fn test_cast_f32_to_f64(val: f32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as f64;
        }
    }

    /// FloatToFloat: f64 → f32 (precision truncation)
    #[kernel]
    pub fn test_cast_f64_to_f32(val: f64, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val as f32;
        }
    }

    /// bool → u32 cast
    #[kernel]
    pub fn test_cast_bool_to_u32(flag: bool, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = flag as u32;
        }
    }

    // =============================================================================
    // 2. TRANSMUTE (CastKind::Transmute)
    // =============================================================================

    /// Transmute i32 → f32: bit pattern 0x3F800000 = 1.0f32
    #[allow(unnecessary_transmutes)]
    #[kernel]
    pub fn test_transmute_i32_to_f32(bits: i32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = unsafe { core::mem::transmute::<i32, f32>(bits) };
        }
    }

    /// Transmute f32 → u32: extract bit pattern from a float
    ///
    /// 1.0f32 has bit pattern 0x3F800000 = 1065353216
    #[allow(unnecessary_transmutes)]
    #[kernel]
    pub fn test_transmute_f32_to_u32(val: f32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = unsafe { core::mem::transmute::<f32, u32>(val) };
        }
    }

    /// Transmute u64 → f64: bit pattern 0x4000000000000000 = 2.0f64
    #[allow(unnecessary_transmutes)]
    #[kernel]
    pub fn test_transmute_u64_to_f64(bits: u64, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = unsafe { core::mem::transmute::<u64, f64>(bits) };
        }
    }

    /// Transmute same-size integers: u32 → i32 (should be identity/bitcast)
    #[allow(unnecessary_transmutes)]
    #[kernel]
    pub fn test_transmute_u32_to_i32(val: u32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = unsafe { core::mem::transmute::<u32, i32>(val) };
        }
    }

    // =============================================================================
    // 2b. SAFE BIT REINTERPRETATION (from_bits, to_bits, cast_signed, cast_unsigned)
    // =============================================================================

    /// f32::from_bits with i32::cast_unsigned — safe alternative to transmute i32→f32
    #[kernel]
    pub fn test_from_bits_i32_to_f32(bits: i32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f32::from_bits(bits.cast_unsigned());
        }
    }

    /// f32::to_bits — safe alternative to transmute f32→u32
    #[kernel]
    pub fn test_to_bits_f32(val: f32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f32::to_bits(val);
        }
    }

    /// f64::from_bits — safe alternative to transmute u64→f64
    #[kernel]
    pub fn test_from_bits_u64_to_f64(bits: u64, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f64::from_bits(bits);
        }
    }

    /// u32::cast_signed — safe alternative to transmute u32→i32
    #[kernel]
    pub fn test_cast_signed_u32_to_i32(val: u32, mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val.cast_signed();
        }
    }

    /// i32::cast_unsigned — safe alternative to transmute i32→u32
    #[kernel]
    pub fn test_cast_unsigned_i32_to_u32(val: i32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val.cast_unsigned();
        }
    }

    // =============================================================================
    // 3. POINTER CASTS (PtrToPtr, PointerExposeProvenance, etc.)
    // =============================================================================

    /// PtrToPtr: *const u32 → *const f32 (thin-to-thin reinterpret)
    /// Uses raw pointer input to avoid ConstantIndex projection issues.
    #[kernel]
    pub fn test_cast_ptr_reinterpret(ptr: *const u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let reinterp: *const f32 = ptr as *const f32;
            *out_elem = reinterp as u64;
        }
    }

    /// PointerExposeProvenance: *const u32 → usize → u64
    #[kernel]
    pub fn test_cast_ptr_to_usize(ptr: *const u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr: usize = ptr as usize;
            *out_elem = addr as u64;
        }
    }

    // =============================================================================
    // 3b. CONSTANTINDEX ON SLICE (Bug 2 reproduction)
    // =============================================================================

    /// ConstantIndex on a slice: data[0] where data is &[u32]
    ///
    /// This triggers ProjectionElem::ConstantIndex on a slice argument.
    #[kernel]
    pub fn test_slice_constant_index(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = data[0];
        }
    }

    // =============================================================================
    // 4. UNSIZING COERCIONS (PointerCoercion::Unsize)
    // =============================================================================

    /// Array to slice: [f32; 4] → &[f32]
    ///
    /// Triggers CastKind::PointerCoercion(Unsize) in MIR. At opt-level=3 the
    /// coercion is usually eliminated, but when it survives, the lowering handles
    /// struct↔ptr conversions via `emit_pointer_cast`.
    #[kernel]
    pub fn test_unsize_array_to_slice(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
            let slice: &[f32] = &arr;
            let i = idx_raw % 4;
            *out_elem = slice[i];
        }
    }

    /// Array to slice with .as_slice() -- same underlying unsize coercion
    #[kernel]
    pub fn test_unsize_array_as_slice(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [f32; 4] = [100.0, 200.0, 300.0, 400.0];
            let slice = arr.as_slice();
            let i = idx_raw % 4;
            *out_elem = slice[i];
        }
    }

    /// Array to slice with iteration -- common pattern
    #[kernel]
    pub fn test_unsize_array_iter_sum(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [f32; 3] = [1.0, 2.0, 3.0];
            let mut sum: f32 = 0.0;
            for val in arr.as_slice().iter() {
                sum += *val;
            }
            *out_elem = sum;
        }
    }

    // --- f64 array unsizing tests ---

    /// f64 array to slice via .as_slice() with untyped float literals
    ///
    /// Untyped float literals default to f64 in Rust. This pattern previously
    /// failed because `translate_byte_string_constant` treated all pointer-to-array
    /// constants as byte arrays. Fixed via `translate_ptr_to_array_constant`.
    #[kernel]
    pub fn test_unsize_f64_as_slice(mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let slice = [1., 2., 3.].as_slice();
            let i = idx_raw % 3;
            *out_elem = slice[i];
        }
    }

    /// f64 array to slice with explicit type annotation
    #[kernel]
    pub fn test_unsize_f64_explicit(mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [f64; 3] = [1.0, 2.0, 3.0];
            let slice: &[f64] = &arr;
            let i = idx_raw % 3;
            *out_elem = slice[i];
        }
    }

    /// f64 array iter sum -- matches our f32 version but with f64
    #[kernel]
    pub fn test_unsize_f64_iter_sum(mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr = [1., 2., 3.];
            let mut sum: f64 = 0.0;
            for val in arr.as_slice().iter() {
                sum += *val;
            }
            *out_elem = sum;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cast Operations Test Suite ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut errors: Vec<String> = Vec::new();

    // =========================================================================
    // 1. NUMERIC CASTS (baseline)
    // =========================================================================
    println!("--- Numeric Casts (IntToInt, IntToFloat, FloatToInt, FloatToFloat) ---");

    // u32 → u64
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_u32_to_u64((stream).as_ref(), cfg, 42u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42u64 {
            println!("  [PASS] u32 → u64: {} → {}", 42u32, r[0]);
            passed += 1;
        } else {
            let msg = format!("u32 → u64: expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u64 → u32 (truncation)
    {
        let val: u64 = 0x1_0000_002A; // upper bits should be dropped, leaving 42
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_u64_to_u32((stream).as_ref(), cfg, val, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42u32 {
            println!("  [PASS] u64 → u32 (trunc): 0x{:X} → {}", val, r[0]);
            passed += 1;
        } else {
            let msg = format!("u64 → u32 (trunc): expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // i32 → i64 (sign extension)
    {
        let mut out = DeviceBuffer::<i64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_i32_to_i64((stream).as_ref(), cfg, -7i32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -7i64 {
            println!("  [PASS] i32 → i64 (sext): -7 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("i32 → i64 (sext): expected -7, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u32 → f32
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_u32_to_f32((stream).as_ref(), cfg, 42u32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if (r[0] - 42.0f32).abs() < 0.001 {
            println!("  [PASS] u32 → f32: 42 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("u32 → f32: expected 42.0, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // i32 → f32 (signed)
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_i32_to_f32((stream).as_ref(), cfg, -7i32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if (r[0] - (-7.0f32)).abs() < 0.001 {
            println!("  [PASS] i32 → f32: -7 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("i32 → f32: expected -7.0, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32 → u32
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f32_to_u32((stream).as_ref(), cfg, 42.9f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42u32 {
            println!("  [PASS] f32 → u32: 42.9 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f32 → u32: expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32 → i32
    {
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f32_to_i32((stream).as_ref(), cfg, -7.8f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -7i32 {
            println!("  [PASS] f32 → i32: -7.8 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f32 → i32: expected -7, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64 → u32
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f64_to_u32((stream).as_ref(), cfg, 42.9f64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42u32 {
            println!("  [PASS] f64 → u32: 42.9 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f64 → u32: expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64 → i32
    {
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f64_to_i32((stream).as_ref(), cfg, -7.8f64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -7i32 {
            println!("  [PASS] f64 → i32: -7.8 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f64 → i32: expected -7, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64 → u64
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f64_to_u64((stream).as_ref(), cfg, 100.5f64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 100u64 {
            println!("  [PASS] f64 → u64: 100.5 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f64 → u64: expected 100, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64 → i64
    {
        let mut out = DeviceBuffer::<i64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f64_to_i64((stream).as_ref(), cfg, -100.5f64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -100i64 {
            println!("  [PASS] f64 → i64: -100.5 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f64 → i64: expected -100, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32 → u64 (mixed precision)
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f32_to_u64((stream).as_ref(), cfg, 42.9f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42u64 {
            println!("  [PASS] f32 → u64 (mixed): 42.9 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f32 → u64 (mixed): expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32 → i64 (mixed precision)
    {
        let mut out = DeviceBuffer::<i64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f32_to_i64((stream).as_ref(), cfg, -7.8f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -7i64 {
            println!("  [PASS] f32 → i64 (mixed): -7.8 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f32 → i64 (mixed): expected -7, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32 → f64
    {
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f32_to_f64((stream).as_ref(), cfg, 3.14f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if (r[0] - 3.14f64).abs() < 0.001 {
            println!("  [PASS] f32 → f64: 3.14 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f32 → f64: expected ~3.14, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64 → f32
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_f64_to_f32((stream).as_ref(), cfg, 3.14f64, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if (r[0] - 3.14f32).abs() < 0.01 {
            println!("  [PASS] f64 → f32: 3.14 → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("f64 → f32: expected ~3.14, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // bool → u32
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_bool_to_u32((stream).as_ref(), cfg, true, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 1u32 {
            println!("  [PASS] bool → u32: true → {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("bool → u32: expected 1, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 2. TRANSMUTE (CastKind::Transmute)
    // =========================================================================
    println!("\n--- Transmute (bit reinterpretation) ---");

    // transmute i32 → f32: 0x3F800000 should become 1.0f32
    {
        let bits: i32 = 0x3F800000_u32 as i32; // bit pattern for 1.0f32
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_transmute_i32_to_f32((stream).as_ref(), cfg, bits, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 1.0f32;
        if (r[0] - expected).abs() < f32::EPSILON {
            println!("  [PASS] transmute i32(0x3F800000) → f32: {}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "transmute i32(0x3F800000) → f32: expected {} (bit reinterpret), got {}",
                expected, r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // transmute f32 → u32: 1.0f32 should become 0x3F800000
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_transmute_f32_to_u32((stream).as_ref(), cfg, 1.0f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 0x3F800000u32;
        if r[0] == expected {
            println!("  [PASS] transmute f32(1.0) → u32: 0x{:08X}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "transmute f32(1.0) → u32: expected 0x{:08X} (bit pattern), got 0x{:08X} (decimal: {})",
                expected, r[0], r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // transmute u64 → f64: 0x4000000000000000 should become 2.0f64
    {
        let bits: u64 = 0x4000000000000000; // bit pattern for 2.0f64
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_transmute_u64_to_f64((stream).as_ref(), cfg, bits, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 2.0f64;
        if (r[0] - expected).abs() < f64::EPSILON {
            println!("  [PASS] transmute u64(0x4000...) → f64: {}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "transmute u64(0x4000000000000000) → f64: expected {}, got {}",
                expected, r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // transmute u32 → i32 (same-size integer reinterpret)
    {
        let val: u32 = 0xFFFFFFFF; // should become -1 as i32
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_transmute_u32_to_i32((stream).as_ref(), cfg, val, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -1i32 {
            println!("  [PASS] transmute u32(0xFFFFFFFF) → i32: {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("transmute u32(0xFFFFFFFF) → i32: expected -1, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 2b. SAFE BIT REINTERPRETATION
    // =========================================================================
    println!(
        "\n--- Safe Bit Reinterpretation (from_bits, to_bits, cast_signed, cast_unsigned) ---"
    );

    // f32::from_bits(i32::cast_unsigned(..)) — same as transmute i32→f32
    {
        let bits: i32 = 0x3F800000_u32 as i32;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_from_bits_i32_to_f32((stream).as_ref(), cfg, bits, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 1.0f32;
        if (r[0] - expected).abs() < f32::EPSILON {
            println!(
                "  [PASS] f32::from_bits(i32::cast_unsigned(0x3F800000)): {}",
                r[0]
            );
            passed += 1;
        } else {
            let msg = format!("from_bits i32→f32: expected {}, got {}", expected, r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f32::to_bits — same as transmute f32→u32
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_to_bits_f32((stream).as_ref(), cfg, 1.0f32, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 0x3F800000u32;
        if r[0] == expected {
            println!("  [PASS] f32::to_bits(1.0): 0x{:08X}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "to_bits f32: expected 0x{:08X}, got 0x{:08X}",
                expected, r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // f64::from_bits — same as transmute u64→f64
    {
        let bits: u64 = 0x4000000000000000;
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_from_bits_u64_to_f64((stream).as_ref(), cfg, bits, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        let expected = 2.0f64;
        if (r[0] - expected).abs() < f64::EPSILON {
            println!("  [PASS] f64::from_bits(0x4000...): {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("from_bits u64→f64: expected {}, got {}", expected, r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // u32::cast_signed — same as transmute u32→i32
    {
        let val: u32 = 0xFFFFFFFF;
        let mut out = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_signed_u32_to_i32((stream).as_ref(), cfg, val, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == -1i32 {
            println!("  [PASS] u32::cast_signed(0xFFFFFFFF): {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("cast_signed u32→i32: expected -1, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // i32::cast_unsigned — i32→u32 safe reinterpret
    {
        let val: i32 = -1;
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_unsigned_i32_to_u32((stream).as_ref(), cfg, val, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 0xFFFFFFFF {
            println!("  [PASS] i32::cast_unsigned(-1): 0x{:08X}", r[0]);
            passed += 1;
        } else {
            let msg = format!(
                "cast_unsigned i32→u32: expected 0xFFFFFFFF, got 0x{:08X}",
                r[0]
            );
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 3. POINTER CASTS
    // =========================================================================
    println!("\n--- Pointer Casts (PtrToPtr, PointerExposeProvenance) ---");

    // ptr reinterpret: *const u32 → *const f32 → u64 (address should be preserved)
    // Uses HMM host pointer (same pattern as abi_hmm example)
    {
        let val: u32 = 42;
        let host_ptr: *const u32 = &val;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_ptr_reinterpret((stream).as_ref(), cfg, host_ptr, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] != 0 {
            println!(
                "  [PASS] ptr reinterpret: got address 0x{:016X} (non-null)",
                r[0]
            );
            passed += 1;
        } else {
            let msg = "ptr reinterpret: got null address".to_string();
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // ptr → usize
    {
        let val: u32 = 42;
        let host_ptr: *const u32 = &val;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_cast_ptr_to_usize((stream).as_ref(), cfg, host_ptr, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] != 0 {
            println!(
                "  [PASS] ptr → usize: got address 0x{:016X} (non-null)",
                r[0]
            );
            passed += 1;
        } else {
            let msg = "ptr → usize: got 0".to_string();
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 3b. CONSTANTINDEX ON SLICE (Bug 2 reproduction)
    // =========================================================================
    println!("\n--- ConstantIndex on Slice (Bug 2) ---");

    {
        let input_data: Vec<u32> = vec![42, 99, 7, 255];
        let input_gpu = DeviceBuffer::from_host(&stream, &input_data)?;
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_constant_index((stream).as_ref(), cfg, &input_gpu, &mut out) }?;
        let r = out.to_host_vec(&stream)?;
        if r[0] == 42 {
            println!("  [PASS] slice constant index: data[0] = {}", r[0]);
            passed += 1;
        } else {
            let msg = format!("slice constant index: expected 42, got {}", r[0]);
            println!("  [FAIL] {}", msg);
            errors.push(msg);
            failed += 1;
        }
    }

    // =========================================================================
    // 4. UNSIZING COERCIONS (PointerCoercion::Unsize)
    // =========================================================================
    println!("\n--- Unsizing Coercions (&[T; N] → &[T]) ---");

    // array → slice
    {
        let n = 4usize;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe {
            module.test_unsize_array_to_slice(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &mut out,
            )
        };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = [10.0f32, 20.0, 30.0, 40.0];
                if r == expected {
                    println!("  [PASS] array → slice: {:?}", r);
                    passed += 1;
                } else {
                    let msg = format!("array → slice: expected {:?}, got {:?}", expected, r);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("array → slice: kernel launch/compilation failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
        }
    }

    // array.as_slice()
    {
        let n = 4usize;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe {
            module.test_unsize_array_as_slice(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &mut out,
            )
        };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = [100.0f32, 200.0, 300.0, 400.0];
                if r == expected {
                    println!("  [PASS] array.as_slice(): {:?}", r);
                    passed += 1;
                } else {
                    let msg = format!("array.as_slice(): expected {:?}, got {:?}", expected, r);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("array.as_slice(): launch/compilation failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
        }
    }

    // array iter sum via .as_slice()
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe { module.test_unsize_array_iter_sum((stream).as_ref(), cfg, &mut out) };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = 6.0f32; // 1+2+3
                if (r[0] - expected).abs() < 0.001 {
                    println!("  [PASS] array iter sum: {}", r[0]);
                    passed += 1;
                } else {
                    let msg = format!("array iter sum: expected {}, got {}", expected, r[0]);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("array iter sum: launch/compilation failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
        }
    }

    // --- f64 array unsizing ---
    println!("\n--- f64 Array Unsizing ---");

    // [1., 2., 3.].as_slice() -- f64 untyped literals
    {
        let n = 3usize;
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, n)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe {
            module.test_unsize_f64_as_slice(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &mut out,
            )
        };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = [1.0f64, 2.0, 3.0];
                if r == expected {
                    println!("  [PASS] f64 as_slice: {:?}", r);
                    passed += 1;
                } else {
                    let msg = format!("f64 as_slice: expected {:?}, got {:?}", expected, r);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("f64 as_slice: compilation/launch failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
        }
    }

    // f64 explicit array to slice
    {
        let n = 3usize;
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, n)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe {
            module.test_unsize_f64_explicit(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &mut out,
            )
        };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = [1.0f64, 2.0, 3.0];
                if r == expected {
                    println!("  [PASS] f64 explicit slice: {:?}", r);
                    passed += 1;
                } else {
                    let msg = format!("f64 explicit slice: expected {:?}, got {:?}", expected, r);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("f64 explicit slice: compilation/launch failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
        }
    }

    // f64 iter sum
    {
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        let result = unsafe { module.test_unsize_f64_iter_sum((stream).as_ref(), cfg, &mut out) };
        match result {
            Ok(_) => {
                let r = out.to_host_vec(&stream)?;
                let expected = 6.0f64;
                if (r[0] - expected).abs() < 0.001 {
                    println!("  [PASS] f64 iter sum: {}", r[0]);
                    passed += 1;
                } else {
                    let msg = format!("f64 iter sum: expected {}, got {}", expected, r[0]);
                    println!("  [FAIL] {}", msg);
                    errors.push(msg);
                    failed += 1;
                }
            }
            Err(e) => {
                let msg = format!("f64 iter sum: compilation/launch failed: {}", e);
                println!("  [FAIL] {}", msg);
                errors.push(msg);
                failed += 1;
            }
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

    Ok(())
}

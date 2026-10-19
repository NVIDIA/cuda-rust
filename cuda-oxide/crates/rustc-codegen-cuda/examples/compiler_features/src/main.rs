/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler Features Test - Testing multi-way match, enums, for loops, and more
//!
//! Tests:
//! - Multi-way match statements (integer switches)
//! - Enum support (Option<T>)
//! - For loops (range, break, continue, nested, iterators)
//! - Baseline tests (while loop, binary match, vecadd)
//! - Shared memory address casting
//! - 64-bit arithmetic
//! - Parallel for loop patterns
//! - Full-debug closure environments
//! - Full-debug Rust enum variants (direct and niche layouts)
//! - Full-debug static and dereference projections
//! - Full-debug runtime-indexed fixed-array references
//! - Full-debug enum payload source projections
//!
//! Run: cargo oxide run compiler_features

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::shared::cvta_generic_to_shared_offset;
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// PHASE 1: Multi-way Match (Integer Switches)
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Direct-tag enum used by the full-debug DWARF smoke test.
    #[repr(u8)]
    enum DebugDirectEnum {
        Small(u32) = 3,
        Wide(u64) = 9,
    }

    /// Aggregate used to force a non-zero `Field` debug projection.
    #[repr(C)]
    struct DebugProjectionStruct {
        prefix: u8,
        projected_field: u64,
    }

    /// Test multi-way match on u32
    #[kernel]
    pub fn test_multiway_match_u32(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let result = match val {
                0 => 10u32,
                1 => 20u32,
                2 => 30u32,
                _ => 99u32,
            };
            *out_elem = result;
        }
    }

    // =============================================================================
    // PHASE 2: Enum Support - Option<T>
    // =============================================================================

    /// Test Option<T> - fundamental for for-loops
    #[kernel]
    pub fn test_option(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            // `get_mut` has returned: this PC belongs only to the kernel.
            // CUDA_OXIDE_DEBUG_INLINE_CALLER_BREAKPOINT
            let maybe: Option<u32> = if val > 0 { Some(val) } else { None };
            let result = maybe.unwrap_or_default();
            *out_elem = result;
        }
    }

    /// Full-debug fixture for direct-tag and niche-layout Rust enums.
    ///
    /// The breakpoint is after all four locals are initialized so cuda-gdb can
    /// inspect both the active variant and its payload.
    // The explicit match on each enum is the fixture: all four variant
    // reads stay spelled out the same way for cuda-gdb inspection.
    #[allow(clippy::manual_unwrap_or, clippy::manual_unwrap_or_default)]
    #[kernel]
    pub fn test_enum_debug(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let option_value: Option<u32> = Some(seed + 1);
            let result_value: Result<u32, u64> = Err(0x1_0000_0009u64);
            let direct_value = if seed == 0 {
                DebugDirectEnum::Small(17)
            } else {
                DebugDirectEnum::Wide(0x2_0000_000Bu64)
            };
            let pointee = seed + 5;
            let niche_value: Option<&u32> = Some(&pointee);

            *out_elem = seed; // CUDA_OXIDE_DEBUG_ENUM_BREAKPOINT

            let option_part = match option_value {
                Some(value) => value,
                None => 0,
            };
            let result_part = match result_value {
                Ok(value) => value,
                Err(value) => value as u32,
            };
            let direct_part = match direct_value {
                DebugDirectEnum::Small(value) => value,
                DebugDirectEnum::Wide(projected_enum_payload) => {
                    let part = projected_enum_payload as u32; // CUDA_OXIDE_DEBUG_ENUM_PROJECTION_BREAKPOINT
                    part
                }
            };
            let niche_part = match niche_value {
                Some(value) => *value,
                None => 0,
            };

            *out_elem = option_part + result_part + direct_part + niche_part;
        }
    }

    /// Helper whose destructured arguments produce rustc MIR debug places with
    /// static `Field` and `ConstantIndex` projections.
    #[inline(never)]
    fn debug_projection_values(
        DebugProjectionStruct {
            projected_field, ..
        }: DebugProjectionStruct,
        (_, projected_tuple): (u32, u64),
        [_, _, projected_array, _]: [u32; 4],
    ) -> u32 {
        let field_part = projected_field as u32; // CUDA_OXIDE_DEBUG_PROJECTION_BREAKPOINT
        field_part
            .wrapping_add(projected_tuple as u32)
            .wrapping_add(projected_array)
    }

    /// Full-debug fixture for statically-addressable source projections.
    #[kernel]
    pub fn test_projection_debug(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = debug_projection_values(
                DebugProjectionStruct {
                    prefix: 0xA5,
                    projected_field: seed as u64 + 11,
                },
                (seed, 0x1_0000_0021u64),
                [3u32, 5, 37, 11],
            );
        }
    }

    /// Helper whose reference binding becomes a statement-level
    /// `AssignRef(values[runtime_index])` after `ReferencePropagation`.
    #[inline(never)]
    fn debug_runtime_index_value(values: [u32; 4], runtime_index: usize) -> u32 {
        let projected_runtime = &values[runtime_index];
        let value = *projected_runtime; // CUDA_OXIDE_DEBUG_RUNTIME_INDEX_BREAKPOINT
        value
    }

    /// Full-debug fixture for one runtime index into a fixed-size array.
    #[kernel]
    pub fn test_runtime_index_debug(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let runtime_index = (seed as usize) & 3;
            *out_elem = debug_runtime_index_value([13u32, 21, 34, 55], runtime_index);
        }
    }

    /// Helper whose destructured references produce MIR `Deref` and
    /// `Deref -> Field` debug projections.
    #[inline(never)]
    fn debug_deref_projection_values(
        &DebugProjectionStruct {
            projected_field: deref_field,
            ..
        }: &DebugProjectionStruct,
        &deref_value: &u32,
    ) -> u32 {
        let field_part = deref_field as u32; // CUDA_OXIDE_DEBUG_DEREF_BREAKPOINT
        field_part.wrapping_add(deref_value)
    }

    /// Full-debug fixture for one thin-reference dereference followed by fields.
    #[kernel]
    pub fn test_deref_projection_debug(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let aggregate = DebugProjectionStruct {
                prefix: 0xA5,
                projected_field: seed as u64 + 11,
            };
            let value = 41u32;
            *out_elem = debug_deref_projection_values(&aggregate, &value);
        }
    }

    /// Full-debug fixture for closure environment DWARF.
    ///
    /// The `move` closure forces two scalar captures into the environment so
    /// cuda-gdb can verify the generated composite type and inspect both
    /// `capture_0` and `capture_1`.
    #[kernel]
    pub fn test_closure_debug(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let captured_u32 = seed + 10;
            let captured_u64 = 0x1_0000_0020u64;
            let closure = move |x: u32| x + captured_u32 + captured_u64 as u32;
            *out_elem = seed; // CUDA_OXIDE_DEBUG_CLOSURE_BREAKPOINT
            let closure_result = closure(5u32);
            *out_elem = closure_result;
        }
    }

    // =============================================================================
    // PHASE 3: For Loops
    // =============================================================================

    /// Test simple for loop with range: sum of 0..8 = 28
    #[kernel]
    pub fn test_for_loop_sum(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..8 {
                sum += i;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with slice.iter()
    #[kernel]
    pub fn test_iter_sum(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for val in data.iter() {
                sum += *val;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with enumerate()
    #[kernel]
    pub fn test_enumerate(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for (i, val) in data.iter().enumerate() {
                sum += (i as u32) * (*val);
            }
            *out_elem = sum;
        }
    }

    /// Test nested for loops: sum of i*j for i,j in 0..4 = 36
    #[kernel]
    pub fn test_nested_for_loops(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..4 {
                for j in 0u32..4 {
                    sum += i * j;
                }
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with early break: sum 0+1+2+3+4 = 10
    #[kernel]
    pub fn test_for_loop_break(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..100 {
                if i >= 5 {
                    break;
                }
                sum += i;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with continue: sum of odd numbers 1+3+5+7 = 16
    #[kernel]
    pub fn test_for_loop_continue(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..8 {
                if i % 2 == 0 {
                    continue;
                }
                sum += i;
            }
            *out_elem = sum;
        }
    }

    // =============================================================================
    // BASELINE TESTS
    // =============================================================================

    /// Baseline while loop for comparison (sum 0..8 = 28)
    #[kernel]
    pub fn baseline_while_loop(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < 8 {
                sum += i;
                i += 1;
            }
            *out_elem = sum;
        }
    }

    /// Baseline: binary if-else
    #[kernel]
    pub fn baseline_binary_match(flag: bool, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let result = if flag { 100u32 } else { 0u32 };
            *out_elem = result;
        }
    }

    /// Baseline: simple arithmetic vecadd
    #[kernel]
    pub fn baseline_vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    // =============================================================================
    // SHARED MEMORY ADDRESS CASTING TESTS
    // =============================================================================

    /// Test DIRECT cast to u64 - no intermediate pointer cast
    #[kernel]
    pub unsafe fn test_smem_addr_direct_u64(mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr = &raw const SMEM as u64;
            *out_elem = addr;
        }
    }

    /// Test Via *const u8 (must agree with the direct cast)
    #[kernel]
    pub unsafe fn test_smem_addr_via_ptr_u8(mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr = &raw const SMEM as *const u8 as u64;
            *out_elem = addr;
        }
    }

    /// Test the explicit raw `.shared` offset path for hardware descriptors
    #[kernel]
    pub unsafe fn test_smem_addr_shared_offset(mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let offset = unsafe { cvta_generic_to_shared_offset(&raw const SMEM as *const u8) };
            *out_elem = offset;
        }
    }

    // =============================================================================
    // 64-BIT ARITHMETIC TESTS
    // =============================================================================

    /// Test 64-bit descriptor building - reproduces tcgen05 SMEM descriptor bug
    #[kernel]
    pub fn test_u64_descriptor_build(
        addr: u64,
        leading_dim_bytes: u32,
        stride_bytes: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr_enc = (addr >> 4) & 0x3FFF;
            let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
            let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
            let fixed_bit: u64 = 1u64 << 46;

            let desc = addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit;
            *out_elem = desc;
        }
    }

    /// Simpler test: Just test that (val << 32) works correctly for 64-bit
    #[kernel]
    pub fn test_u64_shift_by_32(val: u64, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let shifted = val << 32;
            *out_elem = shifted;
        }
    }

    /// Test: (1u64 << 46) - fixed bit at position 46
    #[kernel]
    pub fn test_u64_shift_by_46(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let fixed_bit: u64 = 1u64 << 46;
            *out_elem = fixed_bit;
        }
    }

    // =============================================================================
    // PHASE 4: Parallel For Loop Patterns
    // =============================================================================

    /// Parallel polynomial evaluation: p(x) = 1 + x + x^2 + ... + x^7
    #[kernel]
    pub fn parallel_polynomial_eval(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let x = input[idx_raw];
            let mut result: f32 = 0.0;
            let mut power: f32 = 1.0;
            for _ in 0u32..8 {
                result += power;
                power *= x;
            }
            *out_elem = result;
        }
    }

    /// Parallel chunked sum: each thread sums a contiguous chunk
    #[kernel]
    pub fn parallel_chunked_sum(data: &[u32], chunk_size: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let data_len = data.len() as u32;
            let mut sum: u32 = 0;

            for i in start..end {
                if i < data_len {
                    sum += data[i as usize];
                }
            }
            *out_elem = sum;
        }
    }

    /// Parallel local average: each thread computes average of a window
    #[kernel]
    pub fn parallel_local_average(data: &[f32], radius: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let pos = idx_raw as i32;
            let len = data.len() as i32;
            let r = radius as i32;

            let mut sum: f32 = 0.0;
            let mut count: u32 = 0;

            for offset in 0u32..(2 * radius + 1) {
                let sample_pos = pos - r + (offset as i32);
                if sample_pos >= 0 && sample_pos < len {
                    sum += data[sample_pos as usize];
                    count += 1;
                }
            }

            let avg = if count > 0 { sum / (count as f32) } else { 0.0 };
            *out_elem = avg;
        }
    }

    /// Parallel dot product contribution: each thread computes partial dot product
    #[kernel]
    pub fn parallel_dot_product_chunked(
        a: &[f32],
        b: &[f32],
        chunk_size: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let len = a.len() as u32;

            let mut partial_sum: f32 = 0.0;
            for i in start..end {
                if i < len {
                    partial_sum += a[i as usize] * b[i as usize];
                }
            }
            *out_elem = partial_sum;
        }
    }

    /// Parallel matrix row sum: each thread sums one row
    #[kernel]
    pub fn parallel_row_sum(matrix: &[u32], cols: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = idx_raw as u32;
            let row_start = row * cols;

            let mut sum: u32 = 0;
            for col in 0u32..cols {
                let elem_idx = row_start + col;
                if (elem_idx as usize) < matrix.len() {
                    sum += matrix[elem_idx as usize];
                }
            }
            *out_elem = sum;
        }
    }

    /// Parallel histogram counting: count occurrences in range [low, high)
    #[kernel]
    pub fn parallel_range_count(
        data: &[u32],
        chunk_size: u32,
        low: u32,
        high: u32,
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let len = data.len() as u32;

            let mut count: u32 = 0;
            for i in start..end {
                if i < len {
                    let val = data[i as usize];
                    if val >= low && val < high {
                        count += 1;
                    }
                }
            }
            *out_elem = count;
        }
    }

    /// Parallel partial product: each thread computes a factorial-like product
    #[kernel]
    pub fn parallel_partial_product(elements_per_thread: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let base = idx_raw as u64 * elements_per_thread as u64;

            let mut product: u64 = 1;
            for i in 1u32..=elements_per_thread {
                product *= base + (i as u64);
            }
            *out_elem = product;
        }
    }

    // =============================================================================
    // CONSTANT ASSERT OPERANDS (COMPILE-ONLY REGRESSION)
    // =============================================================================

    /// Keeps rustc's `assert(!const true, ...)` form (expected=false) in
    /// optimized MIR. This kernel is compiled to exercise importer lowering
    /// but is deliberately never launched by the host test.
    #[allow(unconditional_panic)]
    #[kernel]
    pub fn compile_constant_assert_expected_false(value: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = value / 0;
        }
    }

    /// Keeps rustc's `assert(const false, ...)` form (expected=true) in
    /// optimized MIR. Like the division case, this is a compile-only probe and
    /// must not be launched.
    #[allow(unconditional_panic)]
    // The out-of-bounds index is the point: it is what produces the constant
    // assert condition this probe exists to compile.
    #[allow(clippy::out_of_bounds_indexing)]
    #[kernel]
    pub fn compile_constant_assert_expected_true(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let empty: [u32; 0] = [];
            *out_elem = empty[0];
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Compiler Features Test (Unified) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx)?;
    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // Test baseline while loop
    println!("Testing: baseline_while_loop");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.baseline_while_loop((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 28, "baseline_while_loop failed");
        println!("  ✓ Result: {} (expected 28)", result[0]);
    }

    // Test binary match
    println!("Testing: baseline_binary_match");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.baseline_binary_match((stream).as_ref(), cfg, true, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 100, "baseline_binary_match(true) failed");
        println!("  ✓ flag=true: {} (expected 100)", result[0]);

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.baseline_binary_match((stream).as_ref(), cfg, false, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 0, "baseline_binary_match(false) failed");
        println!("  ✓ flag=false: {} (expected 0)", result[0]);
    }

    // Test vecadd
    println!("Testing: baseline_vecadd");
    {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();

        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.baseline_vecadd(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(n as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
        }?;
        let result = c_dev.to_host_vec(&stream)?;
        let expected = vec![11.0f32, 22.0, 33.0, 44.0];
        assert_eq!(result, expected, "baseline_vecadd failed");
        println!("  ✓ Result: {:?}", result);
    }

    // Test multi-way match
    println!("Testing: test_multiway_match_u32");
    {
        let test_cases = [(0u32, 10u32), (1, 20), (2, 30), (3, 99), (100, 99)];
        for (val, expected) in test_cases {
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_multiway_match_u32((stream).as_ref(), cfg, val, &mut out_dev) }?;
            let result = out_dev.to_host_vec(&stream)?;
            assert_eq!(
                result[0], expected,
                "test_multiway_match_u32({}) failed",
                val
            );
            println!("  ✓ val={}: {} (expected {})", val, result[0], expected);
        }
    }

    // Test Option<T> enum
    println!("Testing: test_option");
    {
        let test_cases = [(0u32, 0u32), (1u32, 1u32), (42u32, 42u32)];
        for (val, expected) in test_cases {
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_option((stream).as_ref(), cfg, val, &mut out_dev) }?;
            let result = out_dev.to_host_vec(&stream)?;
            assert_eq!(result[0], expected, "test_option({}) failed", val);
            println!("  ✓ val={}: {} (expected {})", val, result[0], expected);
        }
    }

    // Test enum lowering and keep deterministic direct/niche debug fixtures live.
    println!("Testing: test_enum_debug");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // seed=7: Some(8) + Err(...09) + Wide(...0B) + Some(&12) = 40.
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_enum_debug((stream).as_ref(), cfg, 7u32, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 40, "test_enum_debug failed");
        println!("  ✓ Result: {} (expected 40)", result[0]);
    }

    // Test projected debug bindings and keep deterministic Field/ConstantIndex values live.
    println!("Testing: test_projection_debug");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // seed=7: projected_field=18, projected_tuple low32=33, projected_array=37.
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_projection_debug((stream).as_ref(), cfg, 7u32, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 88, "test_projection_debug failed");
        println!("  ✓ Result: {} (expected 88)", result[0]);
    }

    // Test runtime-index debug bindings on a fixed-size array via the debug stack-home bridge.
    println!("Testing: test_runtime_index_debug");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // Keep index 3 first for CUDA-GDB, then exercise every other runtime index.
        for (seed, expected) in [(7u32, 55), (0, 13), (1, 21), (2, 34)] {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_runtime_index_debug((stream).as_ref(), cfg, seed, &mut out_dev) }?;
            let result = out_dev.to_host_vec(&stream)?;
            assert!(
                result.iter().all(|&value| value == expected),
                "test_runtime_index_debug failed for seed {seed}"
            );
            println!("  ✓ Runtime index {}: {expected}", seed & 3);
        }
    }

    // Test dereference debug bindings and keep deterministic values live.
    println!("Testing: test_deref_projection_debug");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // seed=7: deref_field=18 and deref_value=41.
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_deref_projection_debug((stream).as_ref(), cfg, 7u32, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 59, "test_deref_projection_debug failed");
        println!("  ✓ Result: {} (expected 59)", result[0]);
    }

    // Test closure lowering and keep a deterministic full-debug fixture live.
    println!("Testing: test_closure_debug");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // capture_0 = seed + 10 = 17, capture_1 low 32 bits = 32, x = 5.
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_debug((stream).as_ref(), cfg, 7u32, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 54, "test_closure_debug failed");
        println!("  ✓ Result: {} (expected 54)", result[0]);
    }

    // Test for loop sum
    println!("Testing: test_for_loop_sum");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_for_loop_sum((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 28, "test_for_loop_sum failed");
        println!("  ✓ Result: {} (expected 28)", result[0]);
    }

    // Test iter sum
    println!("Testing: test_iter_sum");
    {
        let data = vec![1u32, 2, 3, 4, 5];
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_iter_sum((stream).as_ref(), cfg, &data_dev, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 15, "test_iter_sum failed");
        println!("  ✓ Result: {} (expected 15)", result[0]);
    }

    // Test enumerate
    println!("Testing: test_enumerate");
    {
        let data = vec![10u32, 20, 30, 40]; // 0*10 + 1*20 + 2*30 + 3*40 = 0+20+60+120=200
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_enumerate((stream).as_ref(), cfg, &data_dev, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 200, "test_enumerate failed");
        println!("  ✓ Result: {} (expected 200)", result[0]);
    }

    // Test for loop break
    println!("Testing: test_for_loop_break");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_for_loop_break((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 10, "test_for_loop_break failed");
        println!("  ✓ Result: {} (expected 10)", result[0]);
    }

    // Test for loop continue
    println!("Testing: test_for_loop_continue");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_for_loop_continue((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 16, "test_for_loop_continue failed");
        println!("  ✓ Result: {} (expected 16)", result[0]);
    }

    // Test nested for loops
    println!("Testing: test_nested_for_loops");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_nested_for_loops((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 36, "test_nested_for_loops failed");
        println!("  ✓ Result: {} (expected 36)", result[0]);
    }

    // Test u64 shift by 32
    println!("Testing: test_u64_shift_by_32");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u64_shift_by_32((stream).as_ref(), cfg, 8u64, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 8u64 << 32;
        assert_eq!(result[0], expected, "test_u64_shift_by_32 failed");
        println!(
            "  ✓ Result: 0x{:016X} (expected 0x{:016X})",
            result[0], expected
        );
    }

    // Test u64 shift by 46
    println!("Testing: test_u64_shift_by_46");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_u64_shift_by_46((stream).as_ref(), cfg, &mut out_dev) }?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 1u64 << 46;
        assert_eq!(result[0], expected, "test_u64_shift_by_46 failed");
        println!(
            "  ✓ Result: 0x{:016X} (expected 0x{:016X})",
            result[0], expected
        );
    }

    // Test parallel polynomial eval
    println!("Testing: parallel_polynomial_eval");
    {
        let input = vec![2.0f32; 4]; // p(2) = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_polynomial_eval(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(4),
                &input_dev,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 255.0f32;
        assert!(
            (result[0] - expected).abs() < 0.01,
            "parallel_polynomial_eval failed"
        );
        println!("  ✓ Result: {} (expected {})", result[0], expected);
    }

    // Test parallel chunked sum
    println!("Testing: parallel_chunked_sum");
    {
        let data: Vec<u32> = (1..=16).collect(); // 1,2,3,...,16
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
        // 4 threads, chunk_size=4: thread 0 sums 1+2+3+4=10, thread 1 sums 5+6+7+8=26, etc.
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_chunked_sum(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(4),
                &data_dev,
                4u32,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 10, "parallel_chunked_sum[0] failed");
        assert_eq!(result[1], 26, "parallel_chunked_sum[1] failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel row sum
    println!("Testing: parallel_row_sum");
    {
        // 4x4 matrix with row i having values i*4+1, i*4+2, i*4+3, i*4+4
        let matrix: Vec<u32> = (1..=16).collect();
        let matrix_dev = DeviceBuffer::from_host(&stream, &matrix)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_row_sum(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(4),
                &matrix_dev,
                4u32,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        // Row 0: 1+2+3+4=10, Row 1: 5+6+7+8=26, Row 2: 9+10+11+12=42, Row 3: 13+14+15+16=58
        assert_eq!(result, vec![10, 26, 42, 58], "parallel_row_sum failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel partial product
    println!("Testing: parallel_partial_product");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 4)?;
        // Thread 0: product of 1,2,3 = 6
        // Thread 1: product of 4,5,6 = 120
        // Thread 2: product of 7,8,9 = 504
        // Thread 3: product of 10,11,12 = 1320
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_partial_product(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(4),
                3u32,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 6, "parallel_partial_product[0] failed");
        assert_eq!(result[1], 120, "parallel_partial_product[1] failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel local average
    println!("Testing: parallel_local_average");
    {
        const N: usize = 512;
        const RADIUS: u32 = 3;
        let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_local_average(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &data_dev,
                RADIUS,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        // At position 256 (middle), average of 253..259 = 256.0
        let mid = N / 2;
        let expected = mid as f32;
        let tol = 0.001;
        assert!(
            (result[mid] - expected).abs() < tol,
            "parallel_local_average[{}] failed: got {}, expected {}",
            mid,
            result[mid],
            expected
        );
        println!(
            "  ✓ result[{}]={:.2} (expected {:.2})",
            mid, result[mid], expected
        );
    }

    // Test parallel dot product chunked
    println!("Testing: parallel_dot_product_chunked");
    {
        const NUM_THREADS: usize = 128;
        const CHUNK_SIZE: u32 = 32;
        const TOTAL_SIZE: usize = NUM_THREADS * CHUNK_SIZE as usize;
        // a = [1, 1, 1, ...], b = [2, 2, 2, ...], each element contributes 2.0
        let a: Vec<f32> = vec![1.0f32; TOTAL_SIZE];
        let b: Vec<f32> = vec![2.0f32; TOTAL_SIZE];
        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_THREADS)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_dot_product_chunked(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(NUM_THREADS as u32),
                &a_dev,
                &b_dev,
                CHUNK_SIZE,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        // Each thread: chunk_size * 2.0 = 64.0, total = 128 * 64 = 8192.0
        let total: f32 = result.iter().sum();
        let expected_total = (TOTAL_SIZE as f32) * 2.0;
        let tol = 0.1;
        assert!(
            (total - expected_total).abs() < tol,
            "parallel_dot_product_chunked total failed: got {}, expected {}",
            total,
            expected_total
        );
        println!(
            "  ✓ Total dot product: {} (expected {}), each thread: {}",
            total, expected_total, result[0]
        );
    }

    // Test parallel range count
    println!("Testing: parallel_range_count");
    {
        const NUM_THREADS: usize = 256;
        const CHUNK_SIZE: u32 = 16;
        const TOTAL_SIZE: usize = NUM_THREADS * CHUNK_SIZE as usize;
        let data: Vec<u32> = (0..TOTAL_SIZE as u32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, NUM_THREADS)?;
        let low: u32 = 50;
        let high: u32 = 150;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.parallel_range_count(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(NUM_THREADS as u32),
                &data_dev,
                CHUNK_SIZE,
                low,
                high,
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        // Count values in [50, 150) = 100 values
        let total: u32 = result.iter().sum();
        let expected = 100u32;
        assert_eq!(
            total, expected,
            "parallel_range_count total failed: got {}, expected {}",
            total, expected
        );
        println!(
            "  ✓ Total count in [50, 150): {} (expected {})",
            total, expected
        );
    }

    // ==========================================================================
    // SHARED MEMORY ADDRESS CASTING TESTS
    // ==========================================================================
    println!("\n-----------------------------------------");
    println!("SHARED MEMORY ADDRESS CASTING TESTS");
    println!("-----------------------------------------");

    let mut smem_direct_ok = true;

    // Rust-observed pointer addresses are CUDA generic addresses (the nvcc
    // model): `ptr as u64` must yield the same nonzero generic address
    // whether or not an intermediate `*const u8` cast is involved. The raw
    // `.shared` window offset is available only through the explicit
    // `cvta_generic_to_shared_offset` intrinsic, which hardware SMEM descriptors
    // consume.
    println!("Testing: generic-address contract for shared statics");
    let direct = {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.test_smem_addr_direct_u64(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(1),
                &mut out_dev,
            )
        }?;
        out_dev.to_host_vec(&stream)?[0]
    };
    let via_ptr = {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.test_smem_addr_via_ptr_u8(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(1),
                &mut out_dev,
            )
        }?;
        out_dev.to_host_vec(&stream)?[0]
    };
    let shared_offset = {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.test_smem_addr_shared_offset(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(1),
                &mut out_dev,
            )
        }?;
        out_dev.to_host_vec(&stream)?[0]
    };

    println!("  DIRECT addr:   0x{direct:016x}");
    println!("  VIA PTR addr:  0x{via_ptr:016x}");
    println!("  SHARED offset: 0x{shared_offset:016x}");
    if direct == 0 {
        println!("  ✗ FAILED: generic address of a valid shared static is null");
        smem_direct_ok = false;
    }
    if direct != via_ptr {
        println!("  ✗ FAILED: the two Rust-level casts disagree on the address");
        smem_direct_ok = false;
    }
    if shared_offset >= 0x100000 {
        println!("  ✗ FAILED: cvta_generic_to_shared_offset must yield the raw .shared offset");
        smem_direct_ok = false;
    }
    if shared_offset % 128 != 0 {
        println!("  ✗ FAILED: shared offset ignores the array's 128-byte alignment");
        smem_direct_ok = false;
    }
    if smem_direct_ok {
        println!("  ✓ generic addresses agree and are non-null; raw offset via intrinsic");
    }

    if !smem_direct_ok {
        println!("\n=== FAILED: at least one test did not pass ===");
        std::process::exit(1);
    }

    println!("\n=== ALL TESTS PASSED ✓ ===");
    Ok(())
}

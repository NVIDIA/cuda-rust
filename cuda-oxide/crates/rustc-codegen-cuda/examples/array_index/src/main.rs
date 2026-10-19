/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::manual_swap
)]

//! Array Index Operations Test
//!
//! This example tests all combinations of array index operations:
//!
//! | Operation | Index Type | Expected Result          |
//! |-----------|------------|--------------------------|
//! | Read      | Constant   | ✓ PASS (extractvalue)    |
//! | Read      | Runtime    | ✓ PASS (alloca+gep+load) |
//! | Write     | Constant   | ✗ FAIL (not implemented) |
//! | Write     | Runtime    | ✗ FAIL (not implemented) |
//!
//! Run: cargo oxide run array_index
//!
//! Once array index writes are implemented, all tests should pass.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

// =============================================================================
// SECTION 1: Constant Index Reads (SHOULD WORK)
//
// These use ProjectionElem::ConstantIndex → MirExtractFieldOp → extractvalue
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test: Read array elements at constant indices
    /// Expected: PASS
    #[kernel]
    pub fn test_const_index_read(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            // Create a local array
            let arr: [u32; 4] = [10, 20, 30, 40];

            // Read at constant indices (should use extractvalue)
            let a = arr[0]; // ConstantIndex { offset: 0 }
            let b = arr[1]; // ConstantIndex { offset: 1 }
            let c = arr[2]; // ConstantIndex { offset: 2 }
            let d = arr[3]; // ConstantIndex { offset: 3 }

            // Output the sum: 10 + 20 + 30 + 40 = 100
            *out_elem = a + b + c + d;
        }
    }

    /// Test: Read from array with constant index in expression
    /// Expected: PASS
    #[kernel]
    pub fn test_const_index_read_expr(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [u32; 4] = [1, 2, 3, 4];

            // Direct use in expression
            let result = arr[0] * arr[1] + arr[2] * arr[3]; // 1*2 + 3*4 = 14
            *out_elem = result;
        }
    }

    // =============================================================================
    // SECTION 2: Runtime Index Reads (SHOULD WORK)
    //
    // These use ProjectionElem::Index → MirExtractArrayElementOp → alloca+gep+load
    // =============================================================================

    /// Test: Read array element at runtime index
    /// Expected: PASS (but inefficient - copies whole array per read)
    #[kernel]
    pub fn test_runtime_index_read(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [u32; 4] = [100, 200, 300, 400];

            // Read at runtime index (index is a kernel parameter)
            let i = index as usize;
            let val = arr[i]; // ProjectionElem::Index

            *out_elem = val;
        }
    }

    /// Test: Read array in a loop (runtime index from loop variable)
    /// Expected: PASS
    #[kernel]
    pub fn test_runtime_index_read_loop(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [u32; 4] = [1, 2, 3, 4];

            // Sum using runtime index in loop
            let mut sum: u32 = 0;
            for i in 0..4 {
                sum += arr[i]; // Each iteration: alloca + store whole array + gep + load
            }

            *out_elem = sum; // 1 + 2 + 3 + 4 = 10
        }
    }

    /// Test: Mixed constant and runtime reads
    /// Expected: PASS
    #[kernel]
    pub fn test_mixed_read(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [u32; 4] = [10, 20, 30, 40];

            // Constant index read
            let first = arr[0]; // extractvalue

            // Runtime index read
            let i = index as usize;
            let dynamic = arr[i]; // alloca+gep+load

            *out_elem = first + dynamic;
        }
    }

    // =============================================================================
    // SECTION 3: Constant Index Writes (CURRENTLY FAILS)
    //
    // Should use: ProjectionElem::ConstantIndex → MirInsertFieldOp → insertvalue
    // Currently: MirInsertFieldOp doesn't support arrays, only structs/tuples
    // =============================================================================

    /// Test: Write to array at constant index
    /// Expected: FAIL (until MirInsertFieldOp is extended for arrays)
    #[kernel]
    pub fn test_const_index_write(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [0, 0, 0, 0];

            // Write at constant indices
            arr[0] = val; // Should use insertvalue
            arr[1] = val + 1; // Should use insertvalue
            arr[2] = val + 2; // Should use insertvalue
            arr[3] = val + 3; // Should use insertvalue

            // Read back and sum
            *out_elem = arr[0] + arr[1] + arr[2] + arr[3];
        }
    }

    /// Test: Initialize array element by element (constant indices)
    /// Expected: FAIL
    #[kernel]
    pub fn test_const_index_write_init(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [0; 4]; // Zero-initialized

            // Set each element
            arr[0] = 10;
            arr[1] = 20;
            arr[2] = 30;
            arr[3] = 40;

            *out_elem = arr[0] + arr[3]; // 10 + 40 = 50
        }
    }

    // =============================================================================
    // SECTION 4: Runtime Index Writes (CURRENTLY FAILS)
    //
    // Should use: Memory approach with MirArrayElementAddrOp → gep + store
    // Currently: "Assignments to projections other than Deref and Field not yet implemented"
    // =============================================================================

    /// Test: Write to array at runtime index
    /// Expected: FAIL (until memory approach is implemented)
    #[kernel]
    pub fn test_runtime_index_write(index: u32, val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [0, 0, 0, 0];

            // Write at runtime index
            let i = index as usize;
            arr[i] = val; // ProjectionElem::Index - NOT IMPLEMENTED

            *out_elem = arr[i];
        }
    }

    /// Test: Fill array in a loop (THE MATHDX USE CASE)
    /// Expected: FAIL
    #[kernel]
    pub fn test_runtime_index_write_loop(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 8] = [0; 8];

            // Fill array with loop - this is the MathDx pattern
            for i in 0..8 {
                arr[i] = (i as u32) * 10; // Runtime index write in loop
            }

            // Sum the array
            let mut sum: u32 = 0;
            for i in 0..8 {
                sum += arr[i]; // Runtime index read
            }

            // Expected: 0 + 10 + 20 + 30 + 40 + 50 + 60 + 70 = 280
            *out_elem = sum;
        }
    }

    /// Test: Copy from input to local array (MathDx FFT pattern)
    /// Expected: FAIL
    #[kernel]
    pub fn test_copy_to_local_array(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut local: [u32; 4] = [0; 4];

            // Copy input to local array (the exact MathDx pattern)
            for i in 0..4 {
                local[i] = input[i]; // Runtime index write
            }

            // Process local array
            let sum = local[0] + local[1] + local[2] + local[3];
            *out_elem = sum;
        }
    }

    // =============================================================================
    // SECTION 5: Complex Patterns (CURRENTLY FAILS)
    // =============================================================================

    /// Test: Read-modify-write pattern
    /// Expected: FAIL (needs both read and write)
    #[kernel]
    pub fn test_read_modify_write(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [1, 2, 3, 4];

            // Double each element
            for i in 0..4 {
                arr[i] *= 2; // Read at runtime index, then write
            }

            *out_elem = arr[0] + arr[1] + arr[2] + arr[3]; // 2+4+6+8 = 20
        }
    }

    /// Test: Swap elements
    /// Expected: FAIL
    #[kernel]
    pub fn test_swap_elements(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [1, 2, 3, 4];

            // Swap arr[0] and arr[3]
            let tmp = arr[0];
            arr[0] = arr[3];
            arr[3] = tmp;

            // arr is now [4, 2, 3, 1]
            *out_elem = arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3]; // 4231
        }
    }

    /// Test: Accumulate into array
    /// Expected: FAIL
    #[kernel]
    pub fn test_accumulate(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut buckets: [u32; 4] = [0; 4];

            // Accumulate input values into buckets based on value mod 4
            for i in 0..input.len() {
                let val = input[i];
                let bucket = (val % 4) as usize;
                buckets[bucket] += 1; // Runtime index write
            }

            // Output bucket 0 count
            *out_elem = buckets[0];
        }
    }
}

// =============================================================================
// Host Test Infrastructure
// =============================================================================

fn main() {
    println!("=== Array Index Operations Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let module = match kernels::load(&ctx) {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to load PTX: {}", e);
            println!(
                "\nThis is expected if compilation failed due to unsupported array index writes."
            );
            println!("Run: cargo oxide run array_index");
            println!("to see which kernels compile successfully.\n");
            return;
        }
    };

    let stream = ctx.default_stream();

    println!("=== SECTION 1: Constant Index Reads ===\n");
    run_test_const_index_read(&ctx, &module, &stream);
    run_test_const_index_read_expr(&ctx, &module, &stream);

    println!("\n=== SECTION 2: Runtime Index Reads ===\n");
    run_test_runtime_index_read(&ctx, &module, &stream);
    run_test_runtime_index_read_loop(&ctx, &module, &stream);
    run_test_mixed_read(&ctx, &module, &stream);

    println!("\n=== SECTION 3: Constant Index Writes ===\n");
    run_test_const_index_write(&ctx, &module, &stream);

    println!("\n=== SECTION 4: Runtime Index Writes ===\n");
    run_test_runtime_index_write_loop(&ctx, &module, &stream);
    run_test_copy_to_local_array(&ctx, &module, &stream);

    println!("\n=== SECTION 5: Complex Patterns ===\n");
    run_test_read_modify_write(&ctx, &module, &stream);

    println!("\n=== Test Complete ===");
}

fn run_test_const_index_read(
    __ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: this test launches exactly one thread for one output element.
    unsafe { module.test_const_index_read((stream).as_ref(), config, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 100u32; // 10 + 20 + 30 + 40
    if result == expected {
        println!("test_const_index_read: PASS (result = {})", result);
    } else {
        println!(
            "test_const_index_read: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

fn run_test_const_index_read_expr(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: this test launches exactly one thread for one output element.
    unsafe { module.test_const_index_read_expr((stream).as_ref(), config, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 14u32; // 1*2 + 3*4
    if result == expected {
        println!("test_const_index_read_expr: PASS (result = {})", result);
    } else {
        println!(
            "test_const_index_read_expr: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

fn run_test_runtime_index_read(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let index = 2u32; // Read arr[2] = 300
    // SAFETY: this test launches exactly one thread, and `index` selects a
    // valid element of the kernel's four-element local array.
    unsafe { module.test_runtime_index_read((stream).as_ref(), config, index, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 300u32;
    if result == expected {
        println!("test_runtime_index_read: PASS (result = {})", result);
    } else {
        println!(
            "test_runtime_index_read: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

fn run_test_runtime_index_read_loop(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: this test launches exactly one thread for one output element.
    unsafe { module.test_runtime_index_read_loop((stream).as_ref(), config, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 10u32; // 1 + 2 + 3 + 4
    if result == expected {
        println!("test_runtime_index_read_loop: PASS (result = {})", result);
    } else {
        println!(
            "test_runtime_index_read_loop: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

fn run_test_mixed_read(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let index = 2u32; // arr[2] = 30
    // SAFETY: this test launches exactly one thread, and `index` selects a
    // valid element of the kernel's four-element local array.
    unsafe { module.test_mixed_read((stream).as_ref(), config, index, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 40u32; // 10 + 30
    if result == expected {
        println!("test_mixed_read: PASS (result = {})", result);
    } else {
        println!(
            "test_mixed_read: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

fn run_test_const_index_write(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let val = 5u32;
    // SAFETY: this test launches exactly one thread for one output element.
    match unsafe { module.test_const_index_write((stream).as_ref(), config, val, &mut d_out) } {
        Ok(_) => {
            let result = d_out.to_host_vec(stream).unwrap()[0];
            let expected = 26u32; // 5 + 6 + 7 + 8
            if result == expected {
                println!("test_const_index_write: PASS (result = {})", result);
            } else {
                println!(
                    "test_const_index_write: FAIL (expected {}, got {})",
                    expected, result
                );
            }
        }
        Err(e) => {
            println!("test_const_index_write: SKIPPED (kernel not found: {})", e);
        }
    }
}

fn run_test_runtime_index_write_loop(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: this test launches exactly one thread for one output element.
    match unsafe { module.test_runtime_index_write_loop((stream).as_ref(), config, &mut d_out) } {
        Ok(_) => {
            let result = d_out.to_host_vec(stream).unwrap()[0];
            let expected = 280u32; // 0+10+20+30+40+50+60+70
            if result == expected {
                println!("test_runtime_index_write_loop: PASS (result = {})", result);
            } else {
                println!(
                    "test_runtime_index_write_loop: FAIL (expected {}, got {})",
                    expected, result
                );
            }
        }
        Err(e) => {
            println!(
                "test_runtime_index_write_loop: SKIPPED (kernel not found: {})",
                e
            );
        }
    }
}

fn run_test_copy_to_local_array(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let h_input = vec![100u32, 200, 300, 400];
    let d_input = DeviceBuffer::from_host(stream, &h_input).unwrap();
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: one thread reads the four-element input and writes the single
    // output element allocated above.
    match unsafe {
        module.test_copy_to_local_array((stream).as_ref(), config, &d_input, &mut d_out)
    } {
        Ok(_) => {
            let result = d_out.to_host_vec(stream).unwrap()[0];
            let expected = 1000u32; // 100+200+300+400
            if result == expected {
                println!("test_copy_to_local_array: PASS (result = {})", result);
            } else {
                println!(
                    "test_copy_to_local_array: FAIL (expected {}, got {})",
                    expected, result
                );
            }
        }
        Err(e) => {
            println!(
                "test_copy_to_local_array: SKIPPED (kernel not found: {})",
                e
            );
        }
    }
}

fn run_test_read_modify_write(
    _ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
) {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: this test launches exactly one thread for one output element.
    match unsafe { module.test_read_modify_write((stream).as_ref(), config, &mut d_out) } {
        Ok(_) => {
            let result = d_out.to_host_vec(stream).unwrap()[0];
            let expected = 20u32; // 2+4+6+8
            if result == expected {
                println!("test_read_modify_write: PASS (result = {})", result);
            } else {
                println!(
                    "test_read_modify_write: FAIL (expected {}, got {})",
                    expected, result
                );
            }
        }
        Err(e) => {
            println!("test_read_modify_write: SKIPPED (kernel not found: {})", e);
        }
    }
}

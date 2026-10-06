/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for assigning through nested array indexes.
//!
//! MIR represents `local[i][j] = value` as a two-level `Index, Index`
//! projection. Fixing either level at a constant produces `ConstantIndex,
//! Index`, `Index, ConstantIndex`, or `ConstantIndex, ConstantIndex`. The
//! statement translator must lower every combination to an address and store
//! through it instead of rejecting the assignment before the generic
//! projection walker can handle it.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn nested_index_assignment_kernel(i: usize, j: usize, mut out: DisjointSlice<u32>) {
        let mut values = [[0u32; 4]; 4];
        values[i][j] = 0x5a00_0000 | ((i as u32) << 8) | (j as u32);

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[i][j];
        }
    }

    // Non-square nested array: 5 rows of 3 columns. A square array cannot
    // catch a row-stride bug (row count == column count makes i and j
    // interchangeable), so this case proves the outer index uses the inner
    // array's real stride.
    #[kernel]
    pub fn nested_index_assignment_nonsquare_kernel(
        i: usize,
        j: usize,
        mut out: DisjointSlice<u32>,
    ) {
        let mut values = [[0u32; 3]; 5];
        values[i][j] = 0x5a00_0000 | ((i as u32) << 8) | (j as u32);

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[i][j];
        }
    }

    #[kernel]
    pub fn nested_constant_runtime_index_assignment_kernel(j: usize, mut out: DisjointSlice<u32>) {
        let mut values = [[0u32; 3]; 5];
        values[4][j] = 0x6b00_0000 | j as u32;

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[4][j];
        }
    }

    // Runtime outer row followed by a constant inner column: the mirrored
    // `Index, ConstantIndex` projection pair.
    #[kernel]
    pub fn nested_runtime_constant_index_assignment_kernel(i: usize, mut out: DisjointSlice<u32>) {
        let mut values = [[0u32; 3]; 5];
        values[i][2] = 0x7c00_0000 | i as u32;

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[i][2];
        }
    }

    // Both indexes constant. rustc usually keeps this as a
    // `ConstantIndex, ConstantIndex` projection pair in optimized MIR, so it
    // exercises the remaining combination of the merged translator arm.
    #[kernel]
    pub fn nested_constant_constant_index_assignment_kernel(mut out: DisjointSlice<u32>) {
        let mut values = [[0u32; 3]; 5];
        values[1][2] = 0x8d00_0102;

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[1][2];
        }
    }
}

fn main() {
    println!("=== nested_index_assignment ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Square [[u32; 4]; 4], write [2][3].
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.nested_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            2usize,
            3usize,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");
    assert_eq!(out_dev.to_host_vec(&stream).unwrap(), vec![0x5a00_0203]);

    // Non-square [[u32; 3]; 5], write the last element [4][2].
    let mut out_ns = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.nested_index_assignment_nonsquare_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            4usize,
            2usize,
            &mut out_ns,
        )
    }
    .expect("Non-square kernel launch failed");
    assert_eq!(out_ns.to_host_vec(&stream).unwrap(), vec![0x5a00_0402]);

    // Constant outer row followed by a runtime inner column: write [4][2].
    let mut out_constant_runtime = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: one thread is launched, the output contains one element, and j=2
    // is within the inner array's length of 3.
    unsafe {
        module.nested_constant_runtime_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            2usize,
            &mut out_constant_runtime,
        )
    }
    .expect("Constant/runtime nested-index kernel launch failed");
    assert_eq!(
        out_constant_runtime.to_host_vec(&stream).unwrap(),
        vec![0x6b00_0002],
    );

    // Runtime outer row followed by a constant inner column: write [3][2].
    let mut out_runtime_constant = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: one thread is launched, the output contains one element, and i=3
    // is within the outer array's length of 5.
    unsafe {
        module.nested_runtime_constant_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            3usize,
            &mut out_runtime_constant,
        )
    }
    .expect("Runtime/constant nested-index kernel launch failed");
    assert_eq!(
        out_runtime_constant.to_host_vec(&stream).unwrap(),
        vec![0x7c00_0003],
    );

    // Both indexes constant: write [1][2].
    let mut out_constant_constant = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: one thread is launched and the output contains one element.
    unsafe {
        module.nested_constant_constant_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            &mut out_constant_constant,
        )
    }
    .expect("Constant/constant nested-index kernel launch failed");
    assert_eq!(
        out_constant_constant.to_host_vec(&stream).unwrap(),
        vec![0x8d00_0102],
    );

    println!(
        "PASS: nested index assignments (runtime and constant, in every combination) wrote and read back correctly"
    );
}

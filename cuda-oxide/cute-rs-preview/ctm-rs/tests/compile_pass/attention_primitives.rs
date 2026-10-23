/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-pass probe: type-check attention instruction primitives and their canonical storage adapters.
#![allow(dead_code, unused_imports)]

use ctm_rs::memory::{DisjointSlice, DynamicSharedArray};
use ctm_rs::mma::{Accumulator16x8, F16A, F16B, mma_f16};
use ctm_rs::{
    async_copy, convert, cuda_module, float, kernel, launch_bounds, launch_contract, warp,
};

// A valid instruction sequence type-checks without executing device operations
// on the host. A and B have different register counts and operand roles.
/// Type-check copy waits, FP16 MMA, warp reduction, softmax arithmetic and FP16 packing; never executed.
unsafe fn attention_step(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> u32 {
    unsafe {
        async_copy::cp_async_commit_group();
        async_copy::cp_async_wait_group::<0>();
        async_copy::cp_async_wait_group::<7>();
        let c = mma_f16(
            Accumulator16x8::from_registers(c),
            F16A::from_registers(a),
            F16B::from_registers(b),
        )
        .registers();
        let maximum = warp::shuffle_xor_f32(c[0], 2);
        let prefill = float::ex2_approx_ftz_f32(c[0] - maximum);
        let decode = float::ex2_approx_f32(c[1] - maximum);
        let reciprocal = float::rcp_approx_ftz_f32(prefill + decode);
        convert::cvt_f16x2_f32(prefill * reciprocal, decode * reciprocal)
    }
}

// Storage aliases must retain the canonical types used by importer and macros.
/// Verify the ctm-rs alias is exactly the canonical device disjoint-slice type.
fn canonical_slice<'s>(slice: DisjointSlice<'s, f32>) -> cuda_device::DisjointSlice<'s, f32> {
    slice
}
/// Verify the ctm-rs alias preserves the canonical dynamic-shared-storage type.
fn canonical_dynamic_storage(
    storage: DynamicSharedArray<u8, 1024>,
) -> cuda_device::DynamicSharedArray<u8, 1024> {
    storage
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only BF16 WGMMA integration example.
//!
//! This example exercises five public-API lowering paths:
//!
//! 1. an m64n64 accumulator with a final `wait_group<0>`;
//! 2. two m64n64 MMAs using separate reborrows of one accumulator;
//! 3. a counted m64n64 K-loop whose reborrow is created inside the loop;
//! 4. two independent m64n64 accumulator slots with two committed groups,
//!    `wait_group<1>`, and a mandatory final `wait_group<0>`;
//! 5. an m64n128 64-value accumulator with a final `wait_group<0>`.
//!
//! All kernels use zero descriptors intentionally. They validate Rust MIR
//! import, WGMMA region selection, LLVM lowering, and PTX generation only.
//! Do not launch any kernel in this example.
//!
//! Usage:
//!   cargo oxide build wgmma_mma_bf16 --arch sm_90a

use cuda_device::wgmma::{
    wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16, wgmma_mma_m64n128k16_f32_bf16,
    wgmma_wait_group,
};
use cuda_device::{DisjointSlice, kernel, thread};

/// Compile-only m64n64 full-drain WGMMA kernel.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_mma_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];

    unsafe {
        wgmma_fence();
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
        wgmma_commit_group();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

/// Compile-only repeated-reborrow m64n64 WGMMA kernel.
///
/// Each `&mut acc` expression is a distinct Rust reborrow, but both calls must
/// be recognized as operating on the same accumulator storage.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_reborrow_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];

    unsafe {
        wgmma_fence();
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
        wgmma_commit_group();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

/// Compile-only counted m64n64 K-loop WGMMA kernel.
///
/// The `&mut acc` reborrow is created in the loop body. Fusion must identify
/// the storage allocated before the loop so the accumulator can stay in
/// registers across all four iterations.
///
/// # Safety
///
/// The zero-based descriptors are not valid WGMMA shared-memory descriptors.
/// This kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_counted_reborrow_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];
    let mut desc_a = 0u64;
    let mut desc_b = 0u64;
    let mut k = 0u32;

    unsafe {
        wgmma_fence();
        while k < 4 {
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            k += 1;
            desc_a += 16;
            desc_b += 32;
        }
        wgmma_commit_group();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

/// Compile-only two-slot partial-wait WGMMA kernel.
///
/// Two independent accumulators allow two committed groups to coexist. The
/// partial wait leaves at most one group pending; the final full wait makes both
/// accumulators safe to observe.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_partial_wait_kernel(mut out: DisjointSlice<u32>) {
    let mut acc0: [[f32; 8]; 4] = [[0.0f32; 8]; 4];
    let mut acc1: [[f32; 8]; 4] = [[0.0f32; 8]; 4];

    unsafe {
        wgmma_fence();

        wgmma_mma_m64n64k16_f32_bf16(&mut acc0, 0u64, 0u64);
        wgmma_commit_group();

        wgmma_mma_m64n64k16_f32_bf16(&mut acc1, 0u64, 0u64);
        wgmma_commit_group();

        wgmma_wait_group::<1>();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc0[0][0].to_bits() ^ acc1[0][0].to_bits();
    }
}

/// Compile-only m64n128 full-drain WGMMA kernel.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_m64n128_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 8] = [[0.0f32; 8]; 8];

    unsafe {
        wgmma_fence();
        wgmma_mma_m64n128k16_f32_bf16(&mut acc, 0u64, 0u64);
        wgmma_commit_group();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits() ^ acc[7][7].to_bits();
    }
}

fn main() {
    println!(
        "SUCCESS: BF16 WGMMA value-threaded, reborrow, counted-loop, and partial-wait lowering compiled."
    );
}

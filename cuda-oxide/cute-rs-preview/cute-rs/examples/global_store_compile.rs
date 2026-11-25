// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-Rust compile-only coverage of the aligned global-store boundary.
//!
//! From `cuda-oxide/` (no GPU execution):
//! ```text
//! CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide build --arch sm_100a \
//!   --device-codegen-crate global_store_compile -- \
//!   -p cute-rs --example global_store_compile --release
//! ```

#![feature(f16)]
use cuda_device::{kernel, thread};
use cute_rs::{GlobalStoreTile, store_global_no_allocate};

/// # Safety
/// `out` must be 32-byte aligned and writable for `8 * blockDim.x()` elements.
/// Launch one one-dimensional block so threads write disjoint ranges.
#[kernel]
pub unsafe fn store_global_f32(out: *mut f32, bits: u64) {
    let x = f32::from_bits(bits as u32);
    let values = GlobalStoreTile([
        x,
        x + 1.0,
        x + 2.0,
        x + 3.0,
        x + 4.0,
        x + 5.0,
        x + 6.0,
        x + 7.0,
    ]);
    unsafe {
        store_global_no_allocate(out.add(thread::threadIdx_x() as usize * 8), &values);
    }
}

/// # Safety
/// `out` must be 32-byte aligned and writable for `8 * blockDim.x()` elements.
/// Launch one one-dimensional block so threads write disjoint ranges.
#[kernel]
pub unsafe fn store_global_u32(out: *mut u32, bits: u64) {
    let x = bits as u32;
    let values = GlobalStoreTile([x, x ^ 1, x ^ 2, x ^ 3, x ^ 4, x ^ 5, x ^ 6, x ^ 7]);
    unsafe {
        store_global_no_allocate(out.add(thread::threadIdx_x() as usize * 8), &values);
    }
}

/// # Safety
/// `out` must be 32-byte aligned and writable for `16 * blockDim.x()` elements.
/// Launch one one-dimensional block so threads write disjoint ranges.
#[kernel]
pub unsafe fn store_global_f16(out: *mut f16, bits: u64) {
    let x = f32::from_bits(bits as u32);
    let values = GlobalStoreTile([x as f16; 16]);
    unsafe {
        store_global_no_allocate(out.add(thread::threadIdx_x() as usize * 16), &values);
    }
}

/// Provide a host entry point so this device-store compilation probe is a valid example target.
fn main() {} // Kernels are compiled, never launched.

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: overlapping enum payloads cannot mix pointers and integers
//! until lowering can preserve LLVM pointer provenance through the shared slot.

use cuda_device::kernel;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum PointerOrBits {
    Bits(u64),
    Pointer(*const u8),
}

#[inline(never)]
fn value_bits(value: PointerOrBits) -> u64 {
    match value {
        PointerOrBits::Bits(bits) => bits,
        PointerOrBits::Pointer(pointer) => pointer as u64,
    }
}

/// # Safety
///
/// `input` must point to one initialized `PointerOrBits`, and `out` must point
/// to one writable `u64`. Neither location may be raced by another thread.
#[kernel]
pub unsafe fn enum_pointer_overlap(input: *const PointerOrBits, out: *mut u64) {
    let value = unsafe { input.read() };
    unsafe {
        out.write(value_bits(value));
    }
}

fn main() {
    // Keep both variants reachable in the host crate as well as device MIR.
    let values = [
        PointerOrBits::Bits(7),
        PointerOrBits::Pointer(core::ptr::null()),
    ];
    println!(
        "This negative example should fail during device compilation ({} variants).",
        values.len()
    );
}

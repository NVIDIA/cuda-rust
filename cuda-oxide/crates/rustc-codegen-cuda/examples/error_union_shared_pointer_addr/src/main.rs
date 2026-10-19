/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: an in-place write to a union field that holds a
//! shared-memory pointer must fail closed.
//!
//! The union stores the pointer as a generic pointer so that its byte image is
//! the same on every target: a shared-memory pointer is 4 bytes wide on the
//! modern NVVM path and 8 bytes wide elsewhere. Construction and extraction by
//! value convert at that boundary. `view.ptr = p` instead stores through the
//! field's address with the field's own type, the raw shared pointer, into
//! bytes every other reader treats as a generic pointer. The compiler refuses
//! to form that address; `&view.ptr` and `&mut view.ptr` are refused the same
//! way. Build the union by value (`SharedPtrBits { ptr: p }`) and read its
//! fields by value instead.

use cuda_device::{SharedArray, kernel};

static mut STAGE: SharedArray<u32, 32> = SharedArray::UNINIT;

#[allow(dead_code)]
union SharedPtrBits {
    ptr: *mut SharedArray<u32, 32>,
    bits: u64,
}

/// # Safety
///
/// `out` must point to writable device memory for one `u64`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn write_shared_pointer_in_place(out: *mut u64) {
    let mut view = SharedPtrBits { bits: 0 };
    view.ptr = &raw mut STAGE;
    unsafe {
        *out = view.bits;
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}

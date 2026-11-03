/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: enum storage containing an oversized array of shared-memory
//! pointers must fail closed when recursive reconstruction would exceed the
//! explicit code-shape bound.

use cuda_device::{SharedArray, kernel};

static mut SHARED: SharedArray<u32, 1> = SharedArray::UNINIT;

struct SharedPointerArrayWrapper {
    pointers: [&'static SharedArray<u32, 1>; 17],
}

#[inline(never)]
fn pointer_bits(value: Option<SharedPointerArrayWrapper>) -> u64 {
    value.map_or(0, |wrapper| {
        wrapper.pointers[0] as *const SharedArray<u32, 1> as u64
    })
}

/// # Safety
///
/// `out` must point to writable device memory for one `u64`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn shared_pointer_enum(out: *mut u64) {
    let shared_ptr: *const SharedArray<u32, 1> = &raw const SHARED;
    let shared: &'static SharedArray<u32, 1> = unsafe { &*shared_ptr };
    unsafe {
        *out = pointer_bits(Some(SharedPointerArrayWrapper {
            pointers: [shared; 17],
        }));
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}

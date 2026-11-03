/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a union field holding an oversized array of shared-memory
//! pointers must fail closed when the rewrite to generic-pointer storage
//! would exceed the explicit code-shape bound.

use cuda_device::{SharedArray, kernel};

static mut SHARED: SharedArray<u32, 1> = SharedArray::UNINIT;

#[allow(dead_code)]
union SharedPointerArrayBits {
    pointers: [*mut SharedArray<u32, 1>; 17],
    words: [u64; 17],
}

/// # Safety
///
/// `out` must point to writable device memory for one `u64`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn shared_pointer_array_union(out: *mut u64) {
    let shared: *mut SharedArray<u32, 1> = &raw mut SHARED;
    let view = SharedPointerArrayBits {
        pointers: [shared; 17],
    };
    unsafe {
        *out = view.words[16];
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}

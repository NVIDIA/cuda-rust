/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a static with two `SharedArray` fields of the same type
//! must be rejected.
//!
//! `SharedArray` is zero-sized, so both fields of `PAIR` sit at byte offset
//! zero and a constant pointer to either has the same type and offset. The
//! compiler cannot tell them apart, and giving them one global would make the
//! two arrays share storage.

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

static mut PAIR: (SharedArray<u32, 64>, SharedArray<u32, 64>) =
    (SharedArray::UNINIT, SharedArray::UNINIT);

const FIRST: *mut SharedArray<u32, 64> = unsafe { &raw mut PAIR.0 };
const SECOND: *mut SharedArray<u32, 64> = unsafe { &raw mut PAIR.1 };

/// Writes `1 + 2` to `out[0]` if the two fields were separate arrays.
///
/// Launch with one block of 64 threads; `out` must hold one element.
#[kernel]
pub fn same_type_fields(mut out: DisjointSlice<u32>) {
    let first = unsafe { SharedArray::as_raw_mut_ptr(FIRST) };
    let second = unsafe { SharedArray::as_raw_mut_ptr(SECOND) };
    let tid = thread::threadIdx_x() as usize;
    unsafe {
        *first.add(tid & 63) = 1;
        *second.add(tid & 63) = 2;
    }
    thread::sync_threads();
    if tid == 0 {
        unsafe { *out.get_unchecked_mut(0) = *first + *second };
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `SharedArray` field inside a union must not be confused
//! with a same-typed sibling at the same offset.
//!
//! `WRAP.direct` and `WRAP.hidden.tile` are both `SharedArray<u32, 64>` at
//! byte offset zero, since a `SharedArray` is zero-sized and every union
//! field starts at the union's base. Constant pointers to them have the same
//! type and offset, so the compiler cannot tell them apart, and one global
//! for both would make the two arrays share storage.

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

type Tile = SharedArray<u32, 64>;

#[repr(C)]
union Inner {
    tile: Tile,
}

#[repr(C)]
struct Wrap {
    direct: Tile,
    hidden: Inner,
}

static mut WRAP: Wrap = Wrap {
    direct: Tile::UNINIT,
    hidden: Inner { tile: Tile::UNINIT },
};

const DIRECT: *mut Tile = unsafe { &raw mut WRAP.direct };
const HIDDEN: *mut Tile = unsafe { &raw mut WRAP.hidden.tile };

/// Writes `1 + 2` to `out[0]` if the two arrays were separate.
///
/// Launch with one block of 64 threads; `out` must hold one element.
#[kernel]
pub fn union_field(mut out: DisjointSlice<u32>) {
    let direct = unsafe { SharedArray::as_raw_mut_ptr(DIRECT) };
    let hidden = unsafe { SharedArray::as_raw_mut_ptr(HIDDEN) };
    let tid = thread::threadIdx_x() as usize;
    unsafe {
        *direct.add(tid & 63) = 1;
        *hidden.add(tid & 63) = 2;
    }
    thread::sync_threads();
    if tid == 0 {
        unsafe { *out.get_unchecked_mut(0) = *direct + *hidden };
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}

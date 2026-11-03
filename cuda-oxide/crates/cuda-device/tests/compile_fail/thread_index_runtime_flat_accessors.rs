/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A `Runtime2DIndex` witness stores the thread's packed (row, col)
// coordinates, not a flat index: a flat position only exists relative to
// the row width of the slice being addressed. The flat accessors `get()` and
// `in_bounds()` therefore live behind the sealed `FlatIndexSpace` marker
// (implemented for `Index1D` and `Index2D<S>` only), so reading the packed
// word as if it were a flat index must be a hard type error; `row()` /
// `col()` are the accessors a runtime witness exposes instead.

use cuda_device::thread::Runtime2DIndex;
use cuda_device::{device, DisjointSlice};

#[device]
pub fn bad_flat_accessors(slice: DisjointSlice<u32, Runtime2DIndex>) {
    let idx = cuda_device::thread::index_2d_runtime(&slice).unwrap();
    let _flat = idx.get();
    let _within = idx.in_bounds(slice.len());
}

fn main() {}

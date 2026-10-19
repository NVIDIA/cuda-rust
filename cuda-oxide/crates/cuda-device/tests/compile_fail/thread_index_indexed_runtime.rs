/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// `get_mut_indexed` is gated by `IndexFormula`, which is implemented for
// `Index1D` and `Index2D<S>` but deliberately not for `Runtime2DIndex` —
// the row stride is a runtime value the type system can't see, so the
// only way to mint a `ThreadIndex<Runtime2DIndex>` is `index_2d_runtime`,
// which reads the row width from the slice it will address. Calling
// `get_mut_indexed` on a runtime-stride slice must therefore be a hard
// type error.

use cuda_device::thread::Runtime2DIndex;
use cuda_device::{device, DisjointSlice};

#[device]
pub fn bad_runtime_indexed(mut out: DisjointSlice<u32, Runtime2DIndex>) {
    if let Some((cell, _idx)) = out.get_mut_indexed() {
        *cell = 0;
    }
}

fn main() {}

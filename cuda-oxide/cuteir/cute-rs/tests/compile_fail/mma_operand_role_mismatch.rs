// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use core::ptr;

use cute_rs::{Mxf4E2M1, Mxfp4TiledMma, Nkl, SharedTensor};

fn main() {
    let shared_b = unsafe {
        SharedTensor::<Mxf4E2M1, f16, (), Nkl>::from_raw_parts(ptr::null_mut(), 0)
    };
    let mma = Mxfp4TiledMma::<()>::get_slice(0);
    let _ = unsafe { mma.load_a(&shared_b, (0, 0)) };
}

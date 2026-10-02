// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use core::ptr;

use cute_rs::{
    CpAsync, Mkl, Mxf4E2M1, Mxfp4BlockScaledTensor, Nkl, SharedTensor, TiledCopy,
};

fn main() {
    let values = [f16::from_bits(0); 32];
    let scales = [127u8; 2];
    let matrix = Mxfp4BlockScaledTensor::<Mkl>::from_slices(&values, &scales, 1, 64);
    let source = matrix.values_for_copy();
    let mut shared_b = unsafe {
        SharedTensor::<Mxf4E2M1, f16, (), Nkl>::from_raw_parts(ptr::null_mut(), 0)
    };
    let copy = TiledCopy::<CpAsync<16>, (), (), ()>::new();

    unsafe { copy.copy(&source, (0, 0), &mut shared_b, 0) };
}

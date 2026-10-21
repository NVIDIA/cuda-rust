// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject copying packed E2M1 values into a plain FP16 shared tensor.
#![feature(f16)]

use core::ptr;

use cute_rs::{CpAsync, Mkl, Mxfp4BlockScaledTensor, SharedTensor, TiledCopy};

/// Reject copying packed E2M1 values into a plain FP16 shared tensor.

fn main() {
    let values = [f16::from_bits(0); 32];
    let scales = [127u8; 2];
    let matrix = Mxfp4BlockScaledTensor::<Mkl>::from_slices(&values, &scales, 1, 64);
    let source = matrix.values_for_copy();
    let mut wrong_encoding = unsafe {
        SharedTensor::<f16, f16, (), Mkl>::from_raw_parts(ptr::null_mut(), 0)
    };
    let copy = TiledCopy::<CpAsync<16>, (), (), ()>::new();

    unsafe { copy.copy(&source, (0, 0), &mut wrong_encoding, 0) };
}

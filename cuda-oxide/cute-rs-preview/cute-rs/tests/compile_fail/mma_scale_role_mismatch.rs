// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject swapped A and B scale roles in block-scaled MMA.
#![feature(f16)]

use cute_rs::{AccC, FragA, FragB, Mkl, Mxfp4BlockScaledTensor, Mxfp4TiledMma, Nkl};

/// Reject swapped A and B scale roles in block-scaled MMA.

fn main() {
    let values = [f16::from_bits(0); 32];
    let scales = [127u8; 2];
    let matrix = Mxfp4BlockScaledTensor::<Mkl>::from_slices(&values, &scales, 1, 64);
    let vector = Mxfp4BlockScaledTensor::<Nkl>::from_slices(&values, &scales, 1, 64);
    let mma = Mxfp4TiledMma::<()>::get_slice(0);

    let _ = unsafe {
        mma.gemm(
            FragA([0; 4]),
            vector.load_scale_pair(0, 0),
            FragB([0; 2]),
            matrix.load_scale_pair(0, 0),
            AccC::ZERO,
        )
    };
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject an FP32 carrier for packed E2M1 tensor storage.
#![feature(f16)]

use cute_rs::{Float4E2M1FN, KMajor, Mkl, Tensor};

/// Reject an FP32 carrier for packed E2M1 tensor storage.

fn main() {
    let wrong_carrier = [0.0f32; 32];
    let _ = Tensor::<Float4E2M1FN, _, f32>::from_storage(
        &wrong_carrier,
        KMajor::<Mkl>::new(1, 64),
    );
}

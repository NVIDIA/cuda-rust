// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use cute_rs::{E2M1, KMajor, Mkl, Tensor};

fn main() {
    let wrong_carrier = [0.0f32; 32];
    let _ = Tensor::<E2M1, _, f32>::from_storage(
        &wrong_carrier,
        KMajor::<Mkl>::new(1, 64),
    );
}

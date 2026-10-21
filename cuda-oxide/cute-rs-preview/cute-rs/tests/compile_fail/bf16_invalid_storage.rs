/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![feature(f16)]
use cute_rs::{Bf16, Contiguous1D, Tensor};
fn main() {
    let values = [1.0_f16; 8];
    let _ = Tensor::<Bf16, _, f16>::from_storage(&values, Contiguous1D::new(8));
}

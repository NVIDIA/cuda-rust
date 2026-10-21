/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![feature(f16)]
use cute_rs::{
    Bf16,
    partition::{Strided1D, StridedTensor},
};
fn main() {
    let _invalid: Option<StridedTensor<'_, Bf16, Strided1D, f16>> = None;
}

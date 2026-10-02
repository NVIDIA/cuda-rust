// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use cute_rs::{Contiguous1D, TensorMut};

fn require_copy<T: Copy>() {}

fn main() {
    require_copy::<TensorMut<'static, f32, Contiguous1D>>();
}

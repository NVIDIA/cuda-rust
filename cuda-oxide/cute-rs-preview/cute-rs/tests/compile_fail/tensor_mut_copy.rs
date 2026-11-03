// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: keep writable tensor views non-Copy so their exclusive borrow cannot be duplicated.
#![feature(f16)]

use cute_rs::{Contiguous1D, TensorMut};

/// Require Copy so passing a writable tensor view exposes the missing bound.

fn require_copy<T: Copy>() {}

/// Keep writable tensor views non-Copy so their exclusive borrow cannot be duplicated.

fn main() {
    require_copy::<TensorMut<'static, f32, Contiguous1D>>();
}

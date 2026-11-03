// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: keep writable tensor views non-Clone so their exclusive borrow cannot be duplicated.
#![feature(f16)]

use cute_rs::{Contiguous1D, TensorMut};

/// Require Clone so passing a writable tensor view exposes the missing bound.

fn require_clone<T: Clone>() {}

/// Keep writable tensor views non-Clone so their exclusive borrow cannot be duplicated.

fn main() {
    require_clone::<TensorMut<'static, f32, Contiguous1D>>();
}

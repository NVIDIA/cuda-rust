// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// rustc gives `unused_attributes` for a second `inline` attribute.
#![deny(unused_attributes)]
#![allow(dead_code)]

use cuda_macros::device;

#[device]
#[inline(always)]
fn always<T>(value: T) -> T {
    value
}

#[device]
#[cfg_attr(all(), cfg_attr(all(), inline))]
fn conditional<T>(value: T) -> T {
    value
}

fn main() {}

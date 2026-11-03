// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a B-role tile in the shared/shared MMA A position.
#![feature(f16)]
use cute_rs::attention::sm100::{MmaSs, SharedLayout, SharedTile};
use cute_rs::OperandB;
/// Supported 128-by-128 layout shared by both role-checked operands.
type L = SharedLayout<128, 128>;
/// Reject a B-role tile in the shared/shared MMA A position.
fn main() {
    let a = unsafe { SharedTile::<L, OperandB>::from_raw(core::ptr::null_mut()) };
    let b = unsafe { SharedTile::<L, OperandB>::from_raw(core::ptr::null_mut()) };
    let mma = MmaSs::<L, L, 128, 128, 128, 1>::new();
    unsafe { mma.gemm(0, &a, &b, false) };
}

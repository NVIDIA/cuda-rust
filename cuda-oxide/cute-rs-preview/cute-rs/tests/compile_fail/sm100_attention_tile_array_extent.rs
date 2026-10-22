// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject empty and byte-overflowing shared tile arrays at compile time.
#![feature(f16)]
use cute_rs::OperandB;
use cute_rs::attention::sm100::{SharedLayout, SharedTileArray};
/// Small 8-by-32 shared layout whose stage-array extent is checked.
type L = SharedLayout<8, 32>;
/// Force const evaluation of an invalid zero-tile array.
const EMPTY: SharedTileArray<L, OperandB, 0> =
    unsafe { SharedTileArray::from_raw(core::ptr::null_mut()) };
/// Force const evaluation of an array whose byte extent exceeds u32.
const OVERFLOW: SharedTileArray<L, OperandB, { u32::MAX }> =
    unsafe { SharedTileArray::from_raw(core::ptr::null_mut()) };
/// Reject empty and byte-overflowing shared tile arrays at compile time.
fn main() {
    let _ = (EMPTY, OVERFLOW);
}

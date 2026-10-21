// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a TMA load whose destination has a different operand-major layout.
#![feature(f16)]
use cute_rs::sm100_attention::{SharedLayout, SharedTile, TmaCopy};
use cute_rs::{OperandB, TmaDesc};
/// K-major 128-by-128 descriptor layout.
type Qk = SharedLayout<128, 128>;
/// MN-major 128-by-128 destination layout.
type Pv = SharedLayout<128, 128, true>;
/// Reject a TMA load whose destination has a different operand-major layout.
fn mismatch(desc: &TmaDesc<f16, Qk>, tile: &mut SharedTile<Pv, OperandB>) {
    unsafe { TmaCopy::<Qk, 1>::load(desc, tile, 0, 0, core::ptr::null_mut(), 0) };
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}

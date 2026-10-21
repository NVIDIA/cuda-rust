/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject an FP32 descriptor element type paired with an FP16 physical tile.
#![feature(f16)]
use ctm_rs::{memory::F16Tile128,tma::TensorMap};
/// Reject an FP32 descriptor element type paired with an FP16 physical tile.
fn wrong(_: &TensorMap<f32,F16Tile128>) {}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}

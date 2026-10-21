/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject a shared output tile paired with a different TMA descriptor format.
#![feature(f16)]
use ctm_rs::{memory::*,tma::*};
/// Reject a shared output tile paired with a different TMA descriptor format.
fn wrong(src:&SharedRegion<F16TileC>,map:&TensorMap<f16,F16Tile128>) { unsafe { store_2d(src,map,Coordinates2D::new(0,0)) }; }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}

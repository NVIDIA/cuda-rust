// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every parameter and return form `#[tile_export]` accepts, expanded
//! against the real `cuda_device::device` so the generated wrapper
//! type-checks.

#![allow(dead_code)]
#![feature(f16)]

use cuda_device::tile_export;

#[tile_export(tile_shape = 128)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[tile_export(tile_shape = 128, name = "scale_in_place", symbol = "scale_in_place_impl")]
pub fn scale(x: &mut f32, factor: f32) {
    *x *= factor;
}

#[tile_export(tile_shape = 256, sync = "warp", arch = ["sm_100", "sm_120"])]
pub fn warp_sum(x: f32) -> f32 {
    x
}

// Every parameter of one export shares its footprint, so the bias is a
// 4-element window too.
#[tile_export(tile_shape = 64)]
pub fn window(acc: &mut [i32; 4], bias: &[i32; 4]) {
    for (value, bias) in acc.iter_mut().zip(bias) {
        *value += *bias;
    }
}

// A warp of 4-element footprints owns 128 elements; 256 holds two.
#[tile_export(tile_shape = 256, sync = "warp")]
pub fn warp_scale4(x: &mut [f32; 4], factor: &[f32; 4]) {
    for (value, factor) in x.iter_mut().zip(factor) {
        *value *= *factor;
    }
}

#[tile_export(tile_shape = 32)]
pub fn half_add(a: f16, b: f16) -> f16 {
    a + b
}

#[tile_export(tile_shape = 32)]
pub fn widen(x: u8, y: i64) -> u64 {
    (x as u64).wrapping_add(y as u64)
}

mod nested {
    use cuda_device::tile_export;

    #[tile_export(tile_shape = 128, name = "nested_negate")]
    pub fn negate(x: &mut f64) {
        *x = -*x;
    }
}

fn main() {
    // The original functions stay ordinary Rust functions.
    assert_eq!(relu(-1.0), 0.0);
    let mut value = 2.0;
    scale(&mut value, 3.0);
    assert_eq!(value, 6.0);
}

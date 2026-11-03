/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

use cute_rs::tensor::RegisterTile;

fn main() {
    let half = RegisterTile::from_array([1.0_f16; 4]);
    let single = RegisterTile::from_array([2.0_f32; 4]);
    let _ = half.zip_map(single, |a, b| a + b);
}

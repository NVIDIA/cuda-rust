/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::tensor::RegisterTile;

fn main() {
    let four = RegisterTile::from_array([1.0_f32; 4]);
    let eight = RegisterTile::from_array([2.0_f32; 8]);
    let _ = four.zip_map(eight, |a, b| a + b);
}

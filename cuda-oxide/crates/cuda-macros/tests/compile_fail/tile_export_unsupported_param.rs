// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 128)]
pub fn first(x: &[f32]) -> f32 {
    x[0]
}

fn main() {}

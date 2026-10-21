// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 100, sync = "warp")]
pub fn partial_warp(x: f32) -> f32 {
    x
}

fn main() {}

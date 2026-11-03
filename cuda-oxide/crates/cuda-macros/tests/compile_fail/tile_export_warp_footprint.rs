// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 64, sync = "warp")]
pub fn scale4(x: &mut [f32; 4]) {
    for value in x.iter_mut() {
        *value *= 2.0;
    }
}

fn main() {}

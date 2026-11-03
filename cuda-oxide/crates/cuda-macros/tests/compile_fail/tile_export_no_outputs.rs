// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 128)]
pub fn sink(x: f32, y: &f32) {
    let _ = (x, y);
}

fn main() {}

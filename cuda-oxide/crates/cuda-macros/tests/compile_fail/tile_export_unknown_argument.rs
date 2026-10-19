// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 128, shap = 64)]
pub fn relu(x: f32) -> f32 {
    x
}

fn main() {}

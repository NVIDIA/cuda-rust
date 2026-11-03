// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

#[tile_export(tile_shape = 64)]
pub fn window(acc: &mut [i32; 4], bias: &i32) {
    for value in acc.iter_mut() {
        *value += *bias;
    }
}

fn main() {}

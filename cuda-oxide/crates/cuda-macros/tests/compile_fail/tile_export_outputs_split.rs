// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::tile_export;

// [inout, in, out]: the appended return slot is separated from `a` by `b`.
#[tile_export(tile_shape = 128)]
pub fn split(a: &mut f32, b: f32) -> f32 {
    *a *= b;
    *a
}

fn main() {}

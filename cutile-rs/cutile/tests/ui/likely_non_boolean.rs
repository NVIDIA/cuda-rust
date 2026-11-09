/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::core::*;

fn scalar(value: i32) {
    let _ = value.likely::<true>();
}

fn tile(value: Tile_1<i32, 4>) {
    let _ = value.likely::<false>();
}

fn main() {}

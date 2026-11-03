/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::core::*;
fn bad(global: &Global_0<AtomicI32>, value: Tile_0<i32>) {
    global.load(ordering::Relaxed, scope::TileBlock);
    global.store(value, ordering::Release, scope::TileBlock);
    global.atomic_add(value, ordering::Relaxed, scope::TileBlock);
}
fn main() {}

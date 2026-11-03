/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};
use cuda_device::{DisjointSlice, RuntimeRowMajorTiles, RuntimeTileMut32};

static mut SAVED: Option<RuntimeTileMut32<'static, f32, 1, 1>> = None;

fn cannot_stash_a_tile_beyond_its_parent<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
    mut c: DisjointSlice<'_, f32, RuntimeRowMajorTiles<1, 1>>,
) {
    let coord = __internal::coord_2d_u32(launch_context);
    let tile = c.tile_2d32_rt(coord);
    unsafe {
        SAVED = tile;
    }
}

fn main() {}

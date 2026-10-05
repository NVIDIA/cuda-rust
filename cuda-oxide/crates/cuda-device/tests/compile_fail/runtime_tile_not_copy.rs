/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::RuntimeTileMut32;

fn cannot_reuse_tile(tile: RuntimeTileMut32<'_, u32, 1, 1>) {
    let _moved = tile;
    let _ = tile.rows();
}

fn main() {}

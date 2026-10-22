/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{BFloat16, RegisterTile};
fn main() {
    let tile = RegisterTile::from_array([BFloat16::from_bits(0); 4]);
    let _: RegisterTile<f32, 4> = tile;
}

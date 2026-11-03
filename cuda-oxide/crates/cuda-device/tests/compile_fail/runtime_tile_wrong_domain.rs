/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{DisjointSlice, RuntimeRowMajorTiles, __LaunchContractDisjointSlice};

type TileOutput = DisjointSlice<'static, f32, RuntimeRowMajorTiles<1, 1>>;

fn requires_one_dimensional_contract<T: __LaunchContractDisjointSlice<f32, 1>>() {}

fn main() {
    requires_one_dimensional_contract::<TileOutput>();
}

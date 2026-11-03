/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{DisjointSlice, RowMajorTiles, __LaunchContractDisjointSlice};

type TileOutput = DisjointSlice<'static, u32, RowMajorTiles<2, 4, 16>>;

fn requires_one_dimensional_contract<T: __LaunchContractDisjointSlice<u32, 1>>() {}

fn main() {
    requires_one_dimensional_contract::<TileOutput>();
}

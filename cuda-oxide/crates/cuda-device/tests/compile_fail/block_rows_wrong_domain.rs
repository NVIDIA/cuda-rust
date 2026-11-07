/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block rows are disjoint only under a 1D launch: the row is `blockIdx.x` and
//! the columns are residues of `threadIdx.x`, so an active Y axis would give
//! two threads the same element. A 2D contract cannot take the slice, and a
//! 2D launch context cannot mint a row.

use cuda_device::thread::{LaunchContextRef, __internal};
use cuda_device::{BlockRows, DisjointSlice, __LaunchContractDisjointSlice};

fn requires_two_dimensional_contract<T: __LaunchContractDisjointSlice<f32, 2>>() {}

fn two_dimensional_context_mints_no_row<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
    mut out: DisjointSlice<'_, f32, BlockRows>,
) {
    let _ = out.block_row32(launch_context);
}

fn main() {
    requires_two_dimensional_contract::<DisjointSlice<'static, f32, BlockRows>>();
}

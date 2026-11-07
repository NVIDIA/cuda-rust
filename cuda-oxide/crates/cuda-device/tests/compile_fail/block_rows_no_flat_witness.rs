/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A `BlockRows` slice gives thread `t` of block `b` the block-strided columns
//! of row `b`, so a flat thread index names another thread's element: at
//! width 512 with 256 threads per block, thread 1 of block 0 owns element 257
//! through its second column, and thread 1 of block 1 holds flat index 257.
//! The space therefore accepts no flat witness, neither the 32-bit one nor a
//! legacy `ThreadIndex`.

use cuda_device::thread::{LaunchContextRef, ThreadIndex, __internal};
use cuda_device::{BlockRows, DisjointSlice};

fn flat_witnesses_do_not_address_block_rows<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain1, __internal::U32Coordinates>,
    mut out: DisjointSlice<'_, f32, BlockRows>,
    legacy: ThreadIndex<'kernel>,
) {
    let _ = out.element_thread32(__internal::index_1d_u32(launch_context));
    let _ = out.get_mut(legacy);
}

fn main() {}

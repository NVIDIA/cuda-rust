/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};

fn cannot_reuse_thread_proof<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain1, __internal::U32Coordinates>,
) {
    let index = __internal::index_1d_u32(launch_context);
    let _moved = index;
    let _ = index.get();
}

fn main() {}

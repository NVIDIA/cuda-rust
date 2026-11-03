/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};

fn one_dimensional_launch_context<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain1, __internal::U32Coordinates>,
) {
    let _ = __internal::coord_2d_u32(launch_context);
}

fn main() {}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};

fn cannot_reuse_coordinate_proof<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
) {
    let coord = __internal::coord_2d_u32(launch_context);
    let _moved = coord;
    let _ = coord.row();
}

fn main() {}

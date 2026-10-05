/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};

fn missing_u32_coordinates<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain1, __internal::NativeCoordinates>,
) {
    let _ = __internal::index_1d_u32(launch_context);
}

fn main() {}

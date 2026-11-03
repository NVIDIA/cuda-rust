/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::device;

#[device]
pub fn helper_has_no_two_dimensional_launch_context() {
    let _ = cuda_device::thread::coord_2d_u32();
}

fn main() {}

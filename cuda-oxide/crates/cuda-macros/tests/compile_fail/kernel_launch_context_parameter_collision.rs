// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::kernel;

#[kernel(launch_context = launch_context)]
pub fn collision(launch_context: u32) {
    let _ = launch_context;
}

fn main() {}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, launch_contract, thread};

#[kernel(launch_context = launch_context)]
#[launch_contract(domain = 1, block = (64, 1, 1))]
pub fn native_coordinates() {
    let _ = thread::index_1d_u32(launch_context);
}

fn main() {}

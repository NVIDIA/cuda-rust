// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, thread};

#[kernel(launch_context = launch_context)]
pub fn missing_contract() {
    let _ = thread::index_1d_u32(launch_context);
}

fn main() {}

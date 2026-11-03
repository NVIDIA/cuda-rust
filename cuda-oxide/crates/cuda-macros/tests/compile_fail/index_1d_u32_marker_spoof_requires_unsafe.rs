// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, thread};

#[kernel(launch_context = launch_context)]
pub fn handwritten_marker_is_not_a_safe_contract() {
    thread::__launch_contract_config::<1, true>();
    let _ = thread::index_1d_u32(launch_context);
}

fn main() {}

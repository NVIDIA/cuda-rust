// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, thread};

#[kernel]
pub fn handwritten_marker_is_not_a_safe_abi_declaration(value: &u32) {
    thread::__grid_constant_config::<0>();
    let _ = value;
}

fn main() {}

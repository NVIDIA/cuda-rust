// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::kernel;

#[kernel(u32)]
fn invalid(value: u32) {
    let _ = value;
}

fn main() {}

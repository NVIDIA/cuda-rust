// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::device;

#[device]
unsafe extern "Rust" {
    fn wrong_abi(value: *mut f32);
}

fn main() {}

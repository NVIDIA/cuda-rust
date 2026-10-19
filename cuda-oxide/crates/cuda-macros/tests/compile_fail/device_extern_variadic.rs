// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::device;

#[device]
unsafe extern "C" {
    fn variadic_helper(x: f32, ...) -> f32;
}

fn main() {}

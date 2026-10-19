// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::kernel;

#[kernel]
fn invalid(value: impl Copy) {
    let _ = value;
}

fn main() {}

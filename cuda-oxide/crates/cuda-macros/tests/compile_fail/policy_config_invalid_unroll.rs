// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::kernel;

struct Invalid;

impl Invalid {
    const UNROLL: u32 = 1;
}

#[kernel]
fn invalid_unroll() {
    let mut index = 0;
    #[unroll(Invalid::UNROLL)]
    while index < 8 {
        index += 1;
    }
}

fn main() {}

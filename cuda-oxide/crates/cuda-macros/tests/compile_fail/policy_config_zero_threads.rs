// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, launch_bounds};

struct Invalid;

impl Invalid {
    const MAX_THREADS: u32 = 0;
}

#[kernel]
#[launch_bounds(Invalid::MAX_THREADS)]
fn zero_threads() {}

fn main() {}

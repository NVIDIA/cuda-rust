/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The escape hatch that asserts uniformity is `unsafe`, so wrapping a
//! per-thread value is never silent. `threadIdx.x` differs across a block and
//! is the exact value the witness must exclude.

use cuda_device::Uniform;
use cuda_device::thread;

fn main() {
    let _from_thread_index = Uniform::new_unchecked(thread::threadIdx_x());
}

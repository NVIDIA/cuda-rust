/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::device;

#[device]
pub fn bad_copy() {
    let idx = cuda_device::thread::index_1d();
    let _moved = idx;
    let _ = idx.get();
}

fn main() {}

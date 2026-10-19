/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::device;

#[device]
pub fn bad_device_return<'a>() -> cuda_device::thread::ThreadIndex<'a> {
    cuda_device::thread::index_1d()
}

fn main() {}

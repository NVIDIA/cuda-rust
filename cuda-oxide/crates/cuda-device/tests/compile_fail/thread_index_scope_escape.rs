/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::device;

static mut SAVED: Option<cuda_device::thread::ThreadIndex<'static>> = None;

#[device]
pub fn bad_scope_escape() {
    let idx = cuda_device::thread::index_1d();
    unsafe {
        SAVED = Some(idx);
    }
}

fn main() {}

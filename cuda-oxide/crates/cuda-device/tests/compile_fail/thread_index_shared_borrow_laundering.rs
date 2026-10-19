/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{device, DisjointSlice, SharedArray};

static mut COURIER: SharedArray<cuda_device::thread::ThreadIndex<'static>, 1> =
    SharedArray::UNINIT;

#[device]
pub fn bad_shared_borrow_laundering(mut out: DisjointSlice<u32>) {
    unsafe {
        COURIER[0] = cuda_device::thread::index_1d();
        let _ = out.get_mut(COURIER[0]);
    }
}

fn main() {}

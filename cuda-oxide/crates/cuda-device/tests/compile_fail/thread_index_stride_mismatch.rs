/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::{device, DisjointSlice};

#[device]
pub fn bad_stride(mut out: DisjointSlice<u32, cuda_device::thread::Index2D<100>>) {
    let idx = cuda_device::thread::index_2d::<200>().unwrap();
    let _ = out.get_mut(idx);
}

fn main() {}

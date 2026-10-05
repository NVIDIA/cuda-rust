// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

#[derive(Copy, Clone, DeviceCopy)]
struct Named {
    id: u32,
    values: [f32; 2],
}

#[derive(Copy, Clone, DeviceCopy)]
struct Tuple(u16, Named);

#[derive(Copy, Clone, DeviceCopy)]
struct Generic<T> {
    value: T,
}

fn assert_device_copy<T: DeviceCopy>() {}

fn main() {
    assert_device_copy::<Named>();
    assert_device_copy::<Tuple>();
    assert_device_copy::<Generic<Named>>();
}

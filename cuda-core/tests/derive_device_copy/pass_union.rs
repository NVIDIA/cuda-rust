// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

// Unions are untagged, so any bit pattern (including all-zero) is valid as
// long as every field is itself `DeviceCopy`. The derive accepts them.
#[derive(Copy, Clone, DeviceCopy)]
union Word {
    bits: u32,
    scalar: f32,
}

fn assert_device_copy<T: DeviceCopy>() {}

fn main() {
    assert_device_copy::<Word>();
}

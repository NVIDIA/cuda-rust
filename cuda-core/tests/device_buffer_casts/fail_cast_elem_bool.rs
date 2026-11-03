// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceBuffer;

// Only 0 and 1 are valid `bool` values, so a byte buffer must not become a
// `DeviceBuffer<bool>` through the safe reinterpreting cast.
fn forge(bytes: DeviceBuffer<u8>) -> DeviceBuffer<bool> {
    bytes.cast_elem()
}

fn main() {
    let _ = forge;
}

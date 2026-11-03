// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceBuffer;

// `bool` is not a `DeviceRepr<u8>`, and no downstream crate can make
// it one, so the wrapper path cannot forge a `bool` buffer either.
fn forge(bytes: DeviceBuffer<u8>) -> DeviceBuffer<bool> {
    bytes.wrap_elem()
}

fn main() {
    let _ = forge;
}

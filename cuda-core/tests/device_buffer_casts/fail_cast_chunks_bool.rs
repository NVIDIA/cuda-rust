// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceBuffer;

// The length-adjusting cast has the same bound as `cast_elem`: `u32` words
// must not be regrouped into `bool` bytes.
fn forge(words: DeviceBuffer<u32>) -> Result<DeviceBuffer<bool>, DeviceBuffer<u32>> {
    words.cast_chunks()
}

fn main() {
    let _ = forge;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use cute_rs::Tensor;

fn main() {
    let storage = [0.0f32; 4];
    let tile = Tensor::from_slice(&storage)
        .zipped_divide::<4>()
        .slice(0);
    let values = unsafe { tile.load() };
    unsafe { tile.store(values) };
}

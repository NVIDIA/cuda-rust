// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::Wrapping;

use cuda_core::bytemuck::TransparentWrapper;
use cuda_core::DeviceBuffer;

// A wrapper that is not `Copy`, like the atomic types in cuda-device: it can
// be wrapped and peeled, but not reinterpreted into with `cast_elem`.
#[repr(transparent)]
struct Counter(u32);

// SAFETY: `#[repr(transparent)]` over `u32` with no other fields.
unsafe impl TransparentWrapper<u32> for Counter {}

// Every bit pattern is a valid `f32`, so bytes written as `u32` are fine.
fn reinterpret(words: DeviceBuffer<u32>) -> DeviceBuffer<f32> {
    words.cast_elem()
}

// `Wrapping<u32>` is `Pod`, so it can be cast into as well as wrapped.
fn reinterpret_wrapping(words: DeviceBuffer<u32>) -> DeviceBuffer<Wrapping<u32>> {
    words.cast_elem()
}

fn group(floats: DeviceBuffer<f32>) -> Result<DeviceBuffer<[f32; 4]>, DeviceBuffer<f32>> {
    floats.cast_chunks()
}

fn wrap(words: DeviceBuffer<u32>) -> DeviceBuffer<Counter> {
    words.wrap_elem()
}

fn peel(counters: DeviceBuffer<Counter>) -> DeviceBuffer<u32> {
    counters.peel_elem()
}

fn wrap_std(words: DeviceBuffer<u32>) -> DeviceBuffer<Wrapping<u32>> {
    words.wrap_elem()
}

fn main() {
    // Type-checking is the test; the functions are never called.
    let _ = (reinterpret, reinterpret_wrapping, group, wrap, peel, wrap_std);
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::Wrapping;

use cuda_core::{DeviceBuffer, DeviceRepr};

// A wrapper that is not `Copy`, like the atomic types in cuda-device: it can
// be wrapped and peeled, but not reinterpreted into with `cast_elem`.
#[repr(transparent)]
struct Counter(std::cell::UnsafeCell<u32>);

// SAFETY: transparent over UnsafeCell<u32>, which has u32's layout and valid
// representations, and neither representation owns resources or needs Drop.
unsafe impl DeviceRepr<u32> for Counter {}

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

// NoUninit does not require every bit pattern to be valid: bool is still a
// supported source, even though the opposite byte-to-bool casts are forbidden.
fn bool_bytes(flags: DeviceBuffer<bool>) -> DeviceBuffer<u8> {
    flags.cast_elem()
}

fn bool_chunks(flags: DeviceBuffer<bool>) -> Result<DeviceBuffer<u8>, DeviceBuffer<bool>> {
    flags.cast_chunks()
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
    let _ = (
        reinterpret,
        reinterpret_wrapping,
        group,
        bool_bytes,
        bool_chunks,
        wrap,
        peel,
        wrap_std,
    );
}

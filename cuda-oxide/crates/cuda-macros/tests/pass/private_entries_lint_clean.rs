// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A host build instantiates device entries without exporting them. Private
// entries, and the device functions only they call, must still count as used
// there, or every crate that denies warnings fails to build. trybuild passes
// `-A dead_code`, which only an explicit deny overrides.
#![deny(warnings, dead_code)]

use cuda_device::{device, kernel};

#[device]
fn twice(x: u32) -> u32 {
    x * 2
}

#[kernel]
fn simple(limit: u32) {
    let _ = twice(limit);
}

#[kernel(f32, u32)]
fn instantiated<T: Copy>(value: T) {
    let _ = value;
}

fn main() {}

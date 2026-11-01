// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytemuck::TransparentWrapper;
use cuda_device::atomic::{DeviceAtomicF32, DeviceAtomicU32};

fn wrap(integer: &u32, float: &f32) {
    // Shared scalar references do not permit atomic mutation.
    let _ = DeviceAtomicU32::wrap_ref(integer);
    let _ = DeviceAtomicF32::wrap_ref(float);
}

fn peel(integer: &DeviceAtomicU32, float: &DeviceAtomicF32) {
    // Atomic mutation must not coexist with ordinary shared references.
    let _ = DeviceAtomicU32::peel_ref(integer);
    let _ = DeviceAtomicF32::peel_ref(float);
}

fn main() {}

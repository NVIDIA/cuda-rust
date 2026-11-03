// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::mem::MaybeUninit;

use cuda_core::{DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
struct Padded {
    tag: u8,
    value: u32,
}

// SAFETY: the fields are byte-copyable, zero is valid, and the type owns no
// resources. DeviceCopy permits padding; it does not imply NoUninit.
unsafe impl DeviceCopy for Padded {}

fn uninitialized_element(buffer: DeviceBuffer<MaybeUninit<u32>>) -> DeviceBuffer<u32> {
    buffer.cast_elem()
}

fn uninitialized_chunks(
    buffer: DeviceBuffer<MaybeUninit<u32>>,
) -> Result<DeviceBuffer<u8>, DeviceBuffer<MaybeUninit<u32>>> {
    buffer.cast_chunks()
}

fn padded_element(buffer: DeviceBuffer<Padded>) -> DeviceBuffer<[u32; 2]> {
    buffer.cast_elem()
}

fn padded_chunks(buffer: DeviceBuffer<Padded>) -> Result<DeviceBuffer<u8>, DeviceBuffer<Padded>> {
    buffer.cast_chunks()
}

fn main() {
    fn is_device_copy<T: DeviceCopy>() {}
    is_device_copy::<MaybeUninit<u32>>();
    is_device_copy::<Padded>();
}

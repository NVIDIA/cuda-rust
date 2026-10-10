// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(feature = "f16", feature(f16))]

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::Wrapping;

use cuda_core::DeviceCopy;

fn assert_device_copy<T: DeviceCopy>() {}

#[test]
fn device_copy_supports_half_types() {
    assert_device_copy::<half::f16>();
    assert_device_copy::<half::bf16>();
}

#[cfg(feature = "f16")]
#[test]
fn device_copy_supports_native_f16() {
    assert_device_copy::<f16>();
}

#[test]
fn device_copy_covers_core_parity_types() {
    // `bool` and `char` are intentionally NOT `DeviceCopy`: they have validity
    // holes (only 0/1 for `bool`, only valid Unicode scalars for `char`), so a
    // device-written byte outside that set would be UB on readback. Only the
    // representation-preserving wrappers below are sound parity additions.
    assert_device_copy::<PhantomData<String>>();
    assert_device_copy::<MaybeUninit<u32>>();
    assert_device_copy::<Wrapping<u64>>();
}

#[test]
fn device_copy_covers_thin_raw_pointers_only() {
    // Thin pointers to any sized pointee, `DeviceCopy` or not: the pointee is
    // never dereferenced, and every address is a valid pointer value.
    assert_device_copy::<*const u8>();
    assert_device_copy::<*mut f32>();
    assert_device_copy::<*const String>();
    assert_device_copy::<*mut [u32; 4]>();
    // Fat pointers (`*const dyn Trait`, `*const [T]`) are intentionally NOT
    // `DeviceCopy`; the compile_fail doctest on the trait pins the `dyn` case.
}

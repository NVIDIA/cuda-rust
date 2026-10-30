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
    // The representation-preserving wrappers are sound parity additions.
    assert_device_copy::<PhantomData<String>>();
    assert_device_copy::<MaybeUninit<u32>>();
    assert_device_copy::<Wrapping<u64>>();
}

#[test]
fn device_copy_supports_bool() {
    // `bool` is `DeviceCopy` although only 0 and 1 are valid: a typed buffer
    // holds valid values by construction, and the reinterpreting casts that
    // could forge a `bool` from arbitrary bytes require `AnyBitPattern`, which
    // `bool` does not implement (see tests/device_buffer_casts, #1506).
    assert_device_copy::<bool>();
}

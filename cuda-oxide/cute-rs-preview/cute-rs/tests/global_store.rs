// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = "Checks global-store type contracts and guaranteed carrier alignment."]
/// Store function pointers must preserve scalar type and compile-time vector width.
#[test]
fn global_store_preserves_element_and_vector_width_in_its_signature() {
    let _: unsafe fn(*mut f32, &cute_rs::GlobalStoreTile<f32, 8>) =
        cute_rs::store_global_no_allocate::<f32, 8>;
    let _: unsafe fn(*mut u32, &cute_rs::GlobalStoreTile<u32, 8>) =
        cute_rs::store_global_no_allocate::<u32, 8>;
}

/// Every supported register-tile store carrier must provide its full vector alignment.
#[test]
fn global_store_carrier_guarantees_the_full_source_alignment() {
    use core::mem::{align_of, size_of};
    assert_eq!(align_of::<cute_rs::GlobalStoreTile<f32, 8>>(), 32);
    assert_eq!(size_of::<cute_rs::GlobalStoreTile<f32, 8>>(), 32);
    let tile = cute_rs::GlobalStoreTile([0x7fc0_0001u32, 1, 2, 3, 4, 5, 6, 0x8000_0000]);
    assert_eq!(tile.0[0], 0x7fc0_0001);
    assert_eq!(tile.0.as_ptr() as usize % 32, 0);
    assert_eq!((&tile as *const _ as usize), tile.0.as_ptr() as usize);
}

/// Compile checks must reject unsupported store elements and calls outside unsafe blocks.
#[test]
fn global_store_rejects_unsupported_storage_and_requires_unsafe() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/global_store_unsafe.rs");
    tests.compile_fail("tests/compile_fail/global_store_storage.rs");
}

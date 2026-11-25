// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: require an unsafe block when issuing a raw-pointer global store.
/// Require an unsafe block when issuing a raw-pointer global store.
fn main() {
    let values = cute_rs::GlobalStoreTile([0.0f32; 8]);
    cute_rs::store_global_no_allocate(core::ptr::null_mut(), &values);
}

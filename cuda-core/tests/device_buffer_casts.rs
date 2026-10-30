// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The element casts on `DeviceBuffer` must not forge invalid values (#1506).
//!
//! `cast_elem` and `cast_chunks` reinterpret bytes, so their source must be
//! `bytemuck::NoUninit` and their target `bytemuck::AnyBitPattern`;
//! `wrap_elem` and `peel_elem` relabel through a `cuda_core::DeviceRepr`.
//! No safe cast can turn byte storage into a `DeviceBuffer<bool>`, or turn
//! uninitialized source bytes into initialized values. These cases only need to
//! compile or fail to compile; none touches a GPU. Regenerate the
//! expectations with `TRYBUILD=overwrite cargo test -p cuda-core --test
//! device_buffer_casts` after an intentional diagnostic change.

#[test]
fn device_buffer_cast_bounds() {
    let t = trybuild::TestCases::new();
    t.pass("tests/device_buffer_casts/pass_bounded_casts.rs");
    t.compile_fail("tests/device_buffer_casts/fail_cast_elem_bool.rs");
    t.compile_fail("tests/device_buffer_casts/fail_cast_chunks_bool.rs");
    t.compile_fail("tests/device_buffer_casts/fail_cast_uninitialized_sources.rs");
    t.compile_fail("tests/device_buffer_casts/fail_wrap_elem_bool.rs");
}

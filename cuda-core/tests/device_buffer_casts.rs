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
//! AArch64 cases include the same source with separate expectations for
//! rustc's target-dependent list of `Pod` implementors.

#[test]
fn device_buffer_cast_bounds() {
    let t = trybuild::TestCases::new();
    t.pass("tests/device_buffer_casts/pass_bounded_casts.rs");
    let casts = if cfg!(target_arch = "aarch64") {
        "tests/device_buffer_casts/aarch64"
    } else {
        "tests/device_buffer_casts"
    };
    for case in [
        "fail_cast_elem_bool.rs",
        "fail_cast_chunks_bool.rs",
        "fail_cast_uninitialized_sources.rs",
    ] {
        t.compile_fail(format!("{casts}/{case}"));
    }
    t.compile_fail("tests/device_buffer_casts/fail_wrap_elem_bool.rs");
}

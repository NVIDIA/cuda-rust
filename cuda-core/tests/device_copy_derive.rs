// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn device_copy_derive_compile_pass() {
    let t = trybuild::TestCases::new();
    t.pass("tests/derive_device_copy/pass_structs.rs");
    t.pass("tests/derive_device_copy/pass_union.rs");
    t.compile_fail("tests/derive_device_copy/fail_non_device_copy_field.rs");
    t.compile_fail("tests/derive_device_copy/fail_enum.rs");
}

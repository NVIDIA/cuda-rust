// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = "Checks the unsafe u32 contract of the warp-uniform compiler hint."]
/// Compile checks must reject unchecked calls and non-u32 warp-uniform arguments.
#[test]
fn warp_uniform_hint_requires_an_explicit_unsafe_u32_contract() {
    let _: unsafe fn(u32) -> u32 = cute_rs::arch::assume_warp_uniform;
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/warp_uniform_unsafe.rs");
    tests.compile_fail("tests/compile_fail/warp_uniform_width.rs");
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(feature = "async"))]
#[test]
fn grid_constant_source_contract() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/pass/grid_constant_arguments.rs");
    tests.pass("tests/pass/grid_constant_prepared_launches.rs");
    tests.compile_fail("tests/compile_fail/grid_constant_named_lifetime.rs");
    tests.compile_fail("tests/compile_fail/grid_constant_marker_requires_unsafe.rs");
    tests.compile_fail("tests/compile_fail/grid_constant_prepared_requires_unsafe.rs");
}

#[cfg(feature = "async")]
#[test]
fn grid_constant_async_host_launch_contract() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/pass/grid_constant_prepared_launches.rs");
    tests.compile_fail("tests/compile_fail/grid_constant_prepared_async_requires_unsafe.rs");
}

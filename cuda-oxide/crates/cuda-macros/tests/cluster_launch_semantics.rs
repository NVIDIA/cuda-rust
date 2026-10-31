// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn cluster_launch_compile_contracts() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/cluster_launch_valid.rs");
    t.pass("tests/pass/cluster_config_direct_valid.rs");
    t.pass("tests/pass/cluster_config_generic_valid.rs");
    t.compile_fail("tests/compile_fail/cluster_config_direct_zero_dimension.rs");
    t.compile_fail("tests/compile_fail/cluster_config_generic_zero_dimension.rs");
    t.compile_fail("tests/compile_fail/cluster_launch_zero_dimension.rs");
    t.compile_fail("tests/compile_fail/cluster_launch_wrong_arity.rs");
}

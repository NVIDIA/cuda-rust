/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[test]
fn launch_contract_types_fail_closed() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/launch_contract/fail_wrong_rank.rs");
    tests.compile_fail("tests/launch_contract/fail_wrong_brand.rs");
    tests.compile_fail("tests/launch_contract/fail_private_construction.rs");
    tests.compile_fail("tests/launch_contract/fail_private_mutation.rs");
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// trybuild's generated crate does not mirror feature flags onto the separate
// cuda-host dev dependency. Async expansion is covered by the macro unit tests
// and cuda-host integration tests instead.
#[cfg(not(feature = "async"))]
#[test]
fn launch_contract_types_are_resolved_semantically() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/launch_contract_disjoint_aliases.rs");
    t.pass("tests/pass/launch_contract_uniform_scalar.rs");
    t.pass("tests/pass/kernel_launch_context_api.rs");
    t.compile_fail("tests/compile_fail/launch_contract_misleading_index_alias.rs");
    t.compile_fail("tests/compile_fail/launch_contract_fake_disjoint_slice.rs");
    t.compile_fail("tests/compile_fail/launch_contract_fake_uniform.rs");
    t.compile_fail("tests/compile_fail/launch_contract_alias_hides_row_width.rs");
    t.compile_fail("tests/compile_fail/launch_contract_alias_fakes_row_width.rs");
    t.compile_fail("tests/compile_fail/launch_contract_untrusted_loaders.rs");
    t.compile_fail("tests/compile_fail/launch_contract_wrong_const_brand.rs");
    t.compile_fail("tests/compile_fail/launch_contract_reordered_disjoint_alias.rs");
    t.compile_fail("tests/compile_fail/index_1d_u32_requires_contract.rs");
    t.compile_fail("tests/compile_fail/index_1d_u32_requires_u32_coordinates.rs");
    t.compile_fail("tests/compile_fail/index_1d_u32_requires_1d_domain.rs");
    t.compile_fail("tests/compile_fail/index_1d_u32_marker_spoof_requires_unsafe.rs");
    t.compile_fail(
        "tests/compile_fail/index_1d_u32_generic_helper_requires_entry_launch_context.rs",
    );
    t.compile_fail("tests/compile_fail/kernel_launch_context_duplicate.rs");
    t.compile_fail("tests/compile_fail/kernel_launch_context_unknown_argument.rs");
    t.compile_fail("tests/compile_fail/kernel_launch_context_parameter_collision.rs");
    t.compile_fail("tests/compile_fail/launch_contract_requires_unknown_ident.rs");
    t.compile_fail("tests/compile_fail/launch_contract_requires_len_on_scalar.rs");
    t.compile_fail("tests/compile_fail/launch_contract_requires_bare_slice.rs");
    t.compile_fail("tests/compile_fail/launch_contract_requires_bad_operator.rs");
    t.compile_fail("tests/compile_fail/launch_contract_requires_signed_scalar.rs");
    t.pass("tests/pass/launch_contract_standalone_requires.rs");
    t.compile_fail("tests/compile_fail/launch_contract_standalone_requires_unknown_ident.rs");
    t.pass("tests/pass/launch_contract_standalone_generic_requires.rs");
    t.compile_fail(
        "tests/compile_fail/launch_contract_standalone_generic_requires_unknown_ident.rs",
    );
}

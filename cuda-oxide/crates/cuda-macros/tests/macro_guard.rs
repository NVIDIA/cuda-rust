// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for `#[kernel]`, `#[device]`, and low-level launch API
//! contracts. These keep invalid signatures and reserved names on clear macro
//! diagnostics instead of allowing confusing generated-code failures.

#[test]
fn macro_guards() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/const_generic_hygiene.rs");
    t.pass("tests/pass/cuda_module_inline_namespaces.rs");
    t.compile_fail("tests/compile_fail/kernel_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_extern_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_extern_wrong_abi.rs");
    t.compile_fail("tests/compile_fail/device_extern_variadic.rs");
    t.compile_fail("tests/compile_fail/kernel_legacy_const_instantiation.rs");
    t.compile_fail("tests/compile_fail/kernel_legacy_lifetime_instantiation.rs");
    t.compile_fail("tests/compile_fail/kernel_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/cuda_module_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/device_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/kernel_instantiation_on_non_generic.rs");
    t.compile_fail("tests/compile_fail/cuda_module_duplicate_nested_kernel.rs");
    t.compile_fail("tests/compile_fail/cuda_module_raw_duplicate_kernel.rs");
    t.compile_fail("tests/compile_fail/cuda_module_raw_loaded_module.rs");
    t.compile_fail("tests/compile_fail/cuda_module_reserved_from_parent.rs");
    t.compile_fail("tests/compile_fail/cuda_module_nested_type_mismatch.rs");
    t.compile_fail("tests/compile_fail/cuda_module_pub_super_scope.rs");
    t.compile_fail("tests/compile_fail/cuda_module_file_kernel_boundary.rs");
    t.compile_fail("tests/compile_fail/cuda_module_include_kernel_boundary.rs");
}

/// Raw launch APIs leave their safety obligation at the call site. A bare
/// `cuda_launch!`, `cuda_launch_async!`, or generated raw module launch must
/// therefore fail to compile with an unsafe-required error (E0133).
#[test]
fn cuda_launch_requires_unsafe() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/launch_requires_unsafe.rs");
    t.compile_fail("tests/compile_fail/async_launch_macro_requires_unsafe.rs");
    t.pass("tests/pass/async_launch_macro_in_unsafe.rs");

    // This case expands `#[cuda_module]`. The trybuild fixture depends on the
    // non-async `cuda-host` API, so only compile it with the matching macro
    // feature set. Async contract expansion is covered by the macro unit tests
    // and the `cuda-host --all-features` integration tests.
    #[cfg(not(feature = "async"))]
    {
        t.compile_fail("tests/compile_fail/contract_unchecked_requires_unsafe.rs");
        t.compile_fail("tests/compile_fail/uncontracted_launch_requires_unsafe.rs");
        t.pass("tests/pass/uncontracted_launch_in_unsafe.rs");
    }
}

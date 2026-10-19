// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time coverage for `ptx_asm!`.

#[test]
fn ptx_asm_accepts_cuda_doc_shape() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/ptx_asm_cuda_doc_shape.rs");
}

#[test]
fn ptx_asm_accepts_multi_output() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/ptx_asm_multi_output.rs");
}

#[test]
fn ptx_asm_accepts_compile_time_strings() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/ptx_asm_c_string.rs");
}

#[test]
fn ptx_asm_accepts_inout() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/ptx_asm_inout.rs");
    t.pass("tests/pass/ptx_asm_marker_arity_contract.rs");
}

#[test]
fn ptx_asm_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/ptx_asm_requires_unsafe.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_unescaped_register.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_output_limit.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_input_limit.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_constraint_comma.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_unsupported_constraint.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_c_constraint_not_bytes.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_input_output_constraint.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_placeholder_out_of_range.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_register_only_requires_output.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_register_only_clobber.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_may_diverge_requires_register_only.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_unknown_option.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_inout_constraint.rs");
    t.compile_fail("tests/compile_fail/ptx_asm_inout_after_input.rs");
}

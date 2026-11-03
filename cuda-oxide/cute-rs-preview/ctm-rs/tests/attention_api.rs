/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Compile-pass coverage for attention primitives and canonical storage adapters."]
/// Compile the attention primitive probe with typed instruction operands and canonical storage.
#[test]
fn attention_primitives_accept_instruction_operands_and_canonical_storage() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/compile_pass/attention_primitives.rs");
    tests.pass("tests/compile_pass/sm100_attention_primitives.rs");
}

/// The importer identifies loop markers by their defining crate before device inlining.
#[test]
fn unroll_marker_retains_its_canonical_definition() {
    assert_eq!(
        core::any::type_name_of_val(&ctm_rs::thread::__unroll_config::<0>),
        core::any::type_name_of_val(&cuda_device::thread::__unroll_config::<0>),
    );
}

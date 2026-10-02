/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[test]
fn tensor_and_block_scaled_types_fail_closed() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/invalid_storage.rs");
    tests.compile_fail("tests/compile_fail/immutable_store.rs");
    tests.compile_fail("tests/compile_fail/tensor_mut_copy.rs");
    tests.compile_fail("tests/compile_fail/tensor_mut_clone.rs");
    tests.compile_fail("tests/compile_fail/mma_operand_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/mma_scale_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tiled_copy_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tiled_copy_encoding_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tma_store_layout_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_operand_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_tma_layout_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_mma_shape_mismatch.rs");
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Compile-fail coverage for tensor storage, operand roles and writable-view ownership."]
/// Compile checks must reject invalid storage encodings, operand roles and mutable-view duplication.
#[test]
fn tensor_and_block_scaled_types_fail_closed() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/compile_pass/tma_descriptor_prefetch.rs");
    tests.compile_fail("tests/compile_fail/tma_descriptor_prefetch_requires_unsafe.rs");
    tests.compile_fail("tests/compile_fail/invalid_storage.rs");
    tests.compile_fail("tests/compile_fail/immutable_store.rs");
    tests.compile_fail("tests/compile_fail/tensor_mut_copy.rs");
    tests.compile_fail("tests/compile_fail/tensor_mut_clone.rs");
    tests.compile_fail("tests/compile_fail/register_tile_lane_mismatch.rs");
    tests.compile_fail("tests/compile_fail/register_tile_implicit_conversion.rs");
    tests.compile_fail("tests/compile_fail/bf16_implicit_conversion.rs");
    tests.compile_fail("tests/compile_fail/bf16_invalid_storage.rs");
    tests.compile_fail("tests/compile_fail/warp_sum_requires_unsafe.rs");
    tests.compile_fail("tests/compile_fail/mma_operand_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/mma_scale_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tiled_copy_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tiled_copy_encoding_mismatch.rs");
    tests.compile_fail("tests/compile_fail/tma_store_layout_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_operand_role_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_tma_layout_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_mma_shape_mismatch.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_operand_role.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_tma_layout.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_invalid_layout.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_ts_k16.rs");
}

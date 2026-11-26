/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[path = "../examples/support/variants.rs"]
mod variants;

fn run(f: fn() -> variants::Result) {
    std::thread::Builder::new()
        .stack_size(32_000_000)
        .spawn(move || f().unwrap())
        .unwrap()
        .join()
        .unwrap();
}
#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn gemm_memory_variants_and_tails() {
    run(variants::gemm);
}
#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn causal_attention_variants_and_tails() {
    run(variants::attention);
}
#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn moe_routing_and_materialization() {
    run(variants::moe);
}

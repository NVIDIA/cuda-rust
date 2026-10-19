#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running CPU tests"

run_step \
    "cutile-ir tests" \
    cargo test -p cutile-ir

run_step \
    "example capability checks (driver-free)" \
    cargo test -p cutile-examples --lib requirements::

run_step \
    "example runner accounting tests" \
    bash "$REPO_ROOT/scripts/test_example_runner.sh"

run_step \
    "cutile-compiler CPU unit tests" \
    cargo test -p cutile-compiler --lib

run_step \
    "cutile-compiler doc tests" \
    cargo test -p cutile-compiler --doc

run_step \
    "cutile-compiler Tile IR to cubin lowering tests" \
    cargo test -p cutile-compiler --test compiler2_e2e

run_step \
    "cutile-compiler JIT disk store tests" \
    cargo test -p cutile-compiler --test jit_cache

run_step \
    "cutile library tests" \
    cargo test -p cutile --lib --features experimental-tune

run_step \
    "cutile doc tests" \
    cargo test -p cutile --doc

for test_target in \
    basics_and_inlining \
    binary_math_ops \
    bitwise_and_bitcast_ops \
    compile_error_quality \
    compile_only \
    debug_info \
    element_type_zero \
    element_type_zero_jit \
    error_quality \
    flash_attention_compile \
    global_memory \
    integer_ops \
    kernel_compiler \
    load_tile_like_examples \
    macro_submodule \
    memory_and_atomic_ops \
    num_tiles_jit \
    optimization_hints \
    partition_index_schedules \
    reduce_scan_ops \
    registry_phase_a \
    span_source_location \
    trait_dispatch_probe \
    tile_ir_capabilities \
    token_threading \
    two_cga_trait_impl \
    type_inference_sanity \
    unary_math_ops \
    use_classifier
do
    run_step \
        "cutile CPU integration test ${test_target}" \
        cargo test -p cutile --test "$test_target"
done

run_step \
    "cutile compile-only control-flow regressions" \
    cargo test -p cutile --test control_flow_ops compile_

run_step \
    "cutile compile-only tensor and matrix regressions" \
    cargo test -p cutile --test tensor_and_matrix_ops compile_

run_step \
    "cutile compile-only type conversion regressions" \
    cargo test -p cutile --test type_conversion_ops compile_

run_step \
    "cutile compile-only specialization regressions" \
    cargo test -p cutile --test specialization_bits -- --skip raw_pointer_launch

run_step \
    "cutile warmup/cache-key CPU tests" \
    cargo test -p cutile --test warmup

# Shared host-side crates: the CPU-safe portion of their suites. Unit tests
# never invoke a live driver call (a missing/driverless libcuda makes the
# generated shims return error codes, which these tests tolerate), and the
# three cuda-core integration targets below are compile-time contracts
# (trybuild derive suite, DeviceCopy impl coverage, launch type contracts).
# The GPU-dependent integration tests live in run_gpu_tests.sh.
# `cuda_tests` is the one GPU-requiring module in the cuda-bindings suite
# (its `init()` asserts a live `cuInit`); it runs unfiltered in
# run_gpu_tests.sh.
run_step \
    "cuda-bindings unit and toolkit-discovery tests" \
    host_cargo test -p cuda-bindings -- --skip cuda_tests::

run_step \
    "cuda-core unit tests" \
    host_cargo test -p cuda-core --lib

run_step \
    "cuda-core compile-time contract tests" \
    host_cargo test -p cuda-core \
    --test simt_device_copy_derive \
    --test simt_device_copy_impls \
    --test simt_launch_contract_types

# Builds the compile_fail soundness contracts (non-Send launch storage,
# brand/rank rejection, buffer aliasing) that no other target covers.
run_step \
    "cuda-core doc tests" \
    host_cargo test -p cuda-core --doc

run_step \
    "cuda-async unit tests" \
    host_cargo test -p cuda-async --lib

run_step \
    "cuda-async doc tests" \
    host_cargo test -p cuda-async --doc

# Driver-free by construction: its one driver-touching case asserts the
# Err path, which driverless shims also produce.
run_step \
    "cuda-async error-handling integration test" \
    host_cargo test -p cuda-async --test error_handling

print_summary_and_exit \
    "All CPU tests passed!" \
    "Some CPU checks failed. See output above for details."

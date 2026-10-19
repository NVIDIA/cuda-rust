#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running GPU tests"

for test_target in \
    arange \
    autotune \
    do_bench \
    dtype_float_ops \
    gpu_execution_ops \
    nested_partition_mut \
    slice_non_divisible \
    tensor_reinterpret \
    tensor_views
do
    run_step \
        "cutile GPU integration test ${test_target}" \
        cargo test -p cutile --features experimental-tune --test "$test_target"
done

run_step \
    "cutile GPU integration test control_flow_ops runtime cases" \
    cargo test -p cutile --test control_flow_ops -- --skip compile_

run_step \
    "cutile GPU integration test tensor_and_matrix_ops runtime cases" \
    cargo test -p cutile --test tensor_and_matrix_ops execute_

run_step \
    "cutile GPU integration test type_conversion_ops runtime cases" \
    cargo test -p cutile --test type_conversion_ops execute_

run_step \
    "cutile GPU integration test specialization_bits runtime cases" \
    cargo test -p cutile --test specialization_bits raw_pointer_launch

run_step \
    "cutile GPU warmup and disk-cache tests" \
    cargo test -p cutile --test warmup_suite

run_step \
    "cutile GPU aggregate tests" \
    cargo test -p cutile --test gpu

# Submission-lifetime tests: their own binary, single-threaded. Each test gates
# a stream with a blocking host function, and any context-wide synchronize in
# the same process (a first-use module load, a cache eviction's module unload,
# pool growth) waits on that gate while the gate owner waits on the context;
# only the Gate's safety bound breaks the cycle (60 s stalls and spurious
# "completed early" failures in the aggregate binary on DGX Spark). Run at both
# completion mechanisms: the default 20 us budget lets short pipelines complete
# in the inline cuStreamQuery spin; 0 forces the reactor path, a large budget
# forces the spin path to resolve against the tests' blocking Gates. Under a
# Gate, `Pending` is the only correct first-poll outcome on either path.
run_step \
    "cutile GPU submission-lifetime tests" \
    cargo test -p cutile --test submission_lifetimes -- --test-threads=1

run_step \
    "cutile GPU submission-lifetime tests (reactor path, CUDA_ASYNC_SPIN_BUDGET_US=0)" \
    env CUDA_ASYNC_SPIN_BUDGET_US=0 cargo test -p cutile --test submission_lifetimes -- --test-threads=1

run_step \
    "cutile GPU submission-lifetime tests (spin path, CUDA_ASYNC_SPIN_BUDGET_US=200000)" \
    env CUDA_ASYNC_SPIN_BUDGET_US=200000 cargo test -p cutile --test submission_lifetimes -- --test-threads=1

run_step \
    "cutile cross-file debug source kernel" \
    cargo test -p cutile --test debug_info kernel_executes_in_all_debug_modes -- --ignored

run_step \
    "cuda-core GPU integration test vmm" \
    host_cargo test -p cuda-core --test vmm

run_step \
    "cuda-core GPU integration test vmm_multicast" \
    host_cargo test -p cuda-core --test vmm_multicast

# The SIMT integration tests carried over from cuda-oxide. simt_vmm_p2p
# exercises the real two-GPU P2P path (it self-skips on single-device
# machines or when peer access is unavailable); simt_vmm_multicast
# self-skips without NVLink switch multicast support.
for test_target in \
    borrow_raw \
    simt_context_limits \
    simt_context_sync_policy \
    simt_device_buffer \
    simt_device_buffer_leaks \
    simt_pinned_host_buffer \
    simt_stream_priority \
    simt_stream_query \
    simt_vmm_multicast \
    simt_vmm_p2p
do
    run_step \
        "cuda-core GPU integration test ${test_target}" \
        host_cargo test -p cuda-core --test "$test_target"
done

run_step \
    "cuda-async unit tests (with live driver)" \
    host_cargo test -p cuda-async --lib

# Unfiltered: includes the GPU-requiring `cuda_tests` module that the CPU
# path skips.
run_step \
    "cuda-bindings tests (with live driver)" \
    host_cargo test -p cuda-bindings

for test_target in \
    concurrent_capture \
    cuda_graph \
    device_fault \
    drop_in_flight \
    execute_once \
    execution_lock \
    pool_allocation \
    reactor_correctness
do
    run_step \
        "cuda-async GPU integration test ${test_target}" \
        host_cargo test -p cuda-async --test "$test_target"
done

print_summary_and_exit \
    "All GPU tests passed!" \
    "Some GPU tests failed. See output above for details."

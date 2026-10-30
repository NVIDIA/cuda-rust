#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

# With no arguments, build and run locally. Archive mode needs only the bundled
# cargo-nextest executable, the two archives, and this checkout for fixtures.
# After extracting CI's cutile-gpu-tests.tar, --archive-dir points to:
#   DIR/cargo-nextest       executable used to create the archives
#   DIR/tile-tests.tar.zst  Tile workspace, with experimental-tune enabled
#   DIR/host-tests.tar.zst  cuda-bindings, cuda-core, and cuda-async workspace tests
# CI also bundles revision.txt and nextest-version.txt; the workflow checks the
# revision against the checkout before invoking this script. See the archive,
# bundle, and extract steps in .github/workflows/cutile-rs.yml at the git root.
# Both archives use nextest's standard format, including target/nextest metadata.
usage() {
    echo "Usage: $0 [--archive-dir DIR]"
}

ARCHIVE_DIR=""
case "$#" in
    0) ;;
    1)
        if [[ "$1" == --help || "$1" == -h ]]; then
            usage
            exit 0
        fi
        usage >&2
        exit 2
        ;;
    2)
        if [[ "$1" != --archive-dir || ! -d "$2" ]]; then
            usage >&2
            exit 2
        fi
        ARCHIVE_DIR="$(cd "$2" && pwd)"
        for archive in tile-tests.tar.zst host-tests.tar.zst; do
            if [[ ! -f "$ARCHIVE_DIR/$archive" ]]; then
                echo "Missing test archive: $ARCHIVE_DIR/$archive" >&2
                exit 2
            fi
        done
        if [[ ! -x "$ARCHIVE_DIR/cargo-nextest" ]]; then
            echo "Missing executable: $ARCHIVE_DIR/cargo-nextest" >&2
            exit 2
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac

if [[ -n "$ARCHIVE_DIR" ]]; then
    # Extract each workspace once, rather than unpacking it for every test target.
    EXTRACT_DIR="$(mktemp -d)" || exit 1
    trap 'rm -rf "$EXTRACT_DIR"' EXIT
    for workspace in tile host; do
        mkdir -p "$EXTRACT_DIR/$workspace" || exit 1
        workspace_root="$REPO_ROOT"
        if [[ "$workspace" == host ]]; then
            workspace_root="$GIT_ROOT"
        fi
        "$ARCHIVE_DIR/cargo-nextest" nextest list --list-type binaries-only \
            --archive-file "$ARCHIVE_DIR/$workspace-tests.tar.zst" \
            --extract-to "$EXTRACT_DIR/$workspace" \
            --workspace-remap "$workspace_root" > /dev/null || exit 1
    done
fi

# Keep suite selection below independent of the runner. Target is an integration
# test name, "lib" for unit tests, or "all" for the whole package.
# shellcheck disable=SC2329 # Invoked through run_gpu_step/run_step.
gpu_test() (
    local workspace="$1" package="$2" target="$3"
    shift 3
    local workspace_root="$REPO_ROOT"
    if [[ "$workspace" == host ]]; then
        workspace_root="$GIT_ROOT"
    fi
    cd "$workspace_root" || return 1

    local expression="package(=$package)"
    local cargo_args=(test -p "$package")
    local libtest_args=()
    local nextest_args=(--test-threads 1 --no-fail-fast)
    case "$target" in
        lib) cargo_args+=(--lib); expression+=" & kind(lib)" ;;
        all) ;;
        *) cargo_args+=(--test "$target"); expression+=" & binary(=$target)" ;;
    esac
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --filter)
                cargo_args+=("$2")
                expression+=" & test($2)"
                shift 2 ;;
            --skip)
                libtest_args+=(--skip "$2")
                expression+=" & not test($2)"
                shift 2 ;;
            --ignored)
                libtest_args+=(--ignored)
                nextest_args+=(--run-ignored only)
                shift ;;
            --serial)
                libtest_args+=(--test-threads=1)
                shift ;;
            --features)
                cargo_args+=(--features "$2")
                shift 2 ;;
            *) echo "Unknown GPU test option: $1" >&2; return 2 ;;
        esac
    done

    if [[ -n "$ARCHIVE_DIR" ]]; then
        # Serialize GPU tests so unrelated processes cannot compete for resources.
        local target_dir="$EXTRACT_DIR/$workspace/target"
        "$ARCHIVE_DIR/cargo-nextest" nextest run \
            --cargo-metadata "$target_dir/nextest/cargo-metadata.json" \
            --binaries-metadata "$target_dir/nextest/binaries-metadata.json" \
            --target-dir-remap "$target_dir" \
            --workspace-remap "$workspace_root" "${nextest_args[@]}" -E "$expression"
    else
        if [[ ${#libtest_args[@]} -gt 0 ]]; then
            cargo_args+=(-- "${libtest_args[@]}")
        fi
        cargo "${cargo_args[@]}"
    fi
)

# shellcheck disable=SC2329 # Invoked through run_gpu_step/run_step.
submission_test() (
    if [[ "$1" == default ]]; then
        # Local runs honor developer overrides; CI exercises the actual default.
        if [[ -n "$ARCHIVE_DIR" ]]; then
            unset CUDA_ASYNC_SPIN_BUDGET_US
        fi
    else
        export CUDA_ASYNC_SPIN_BUDGET_US="$1"
    fi
    gpu_test tile cutile submission_lifetimes --serial
)

run_gpu_step() {
    if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
        echo "::group::$1"
    fi
    local status=0
    run_step "$@" || status=$?
    if [[ "${GITHUB_ACTIONS:-}" == true ]]; then
        echo "::endgroup::"
    fi
    return "$status"
}

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
    run_gpu_step \
        "cutile GPU integration test ${test_target}" \
        gpu_test tile cutile "$test_target" --features experimental-tune
done

run_gpu_step \
    "cutile GPU integration test control_flow_ops runtime cases" \
    gpu_test tile cutile control_flow_ops --skip compile_

run_gpu_step \
    "cutile GPU integration test tensor_and_matrix_ops runtime cases" \
    gpu_test tile cutile tensor_and_matrix_ops --filter execute_

run_gpu_step \
    "cutile GPU integration test type_conversion_ops runtime cases" \
    gpu_test tile cutile type_conversion_ops --filter execute_

run_gpu_step \
    "cutile GPU integration test specialization_bits runtime cases" \
    gpu_test tile cutile specialization_bits --filter raw_pointer_launch

run_gpu_step \
    "cutile GPU warmup and disk-cache tests" \
    gpu_test tile cutile warmup_suite

run_gpu_step \
    "cutile GPU aggregate tests" \
    gpu_test tile cutile gpu

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
run_gpu_step \
    "cutile GPU submission-lifetime tests" \
    submission_test default

run_gpu_step \
    "cutile GPU submission-lifetime tests (reactor path, CUDA_ASYNC_SPIN_BUDGET_US=0)" \
    submission_test 0

run_gpu_step \
    "cutile GPU submission-lifetime tests (spin path, CUDA_ASYNC_SPIN_BUDGET_US=200000)" \
    submission_test 200000

run_gpu_step \
    "cutile cross-file debug source kernel" \
    gpu_test tile cutile debug_info --filter kernel_executes_in_all_debug_modes --ignored

run_gpu_step \
    "cuda-core GPU integration test vmm" \
    gpu_test host cuda-core vmm

run_gpu_step \
    "cuda-core GPU integration test vmm_multicast" \
    gpu_test host cuda-core vmm_multicast

# The SIMT integration tests carried over from cuda-oxide. vmm_p2p
# exercises the real two-GPU P2P path (it self-skips on single-device
# machines or when peer access is unavailable); vmm_multicast
# self-skips without NVLink switch multicast support.
for test_target in \
    borrow_raw \
    context_limits \
    context_sync_policy \
    device_buffer \
    device_buffer_leaks \
    pinned_host_buffer \
    stream_priority \
    stream_query \
    nvlink_multicast \
    shared_allocation \
    vmm_p2p
do
    run_gpu_step \
        "cuda-core GPU integration test ${test_target}" \
        gpu_test host cuda-core "$test_target"
done

run_gpu_step \
    "cuda-async unit tests (with live driver)" \
    gpu_test host cuda-async lib

# Unfiltered: includes the GPU-requiring `cuda_tests` module that the CPU
# path skips.
run_gpu_step \
    "cuda-bindings tests (with live driver)" \
    gpu_test host cuda-bindings all

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
    run_gpu_step \
        "cuda-async GPU integration test ${test_target}" \
        gpu_test host cuda-async "$test_target"
done

run_gpu_step "shared Tile and SIMT host buffer interop" \
    gpu_test tile cutile shared_host_buffers --serial

run_gpu_step "shared async Tile and SIMT graph replay" \
    gpu_test tile cutile shared_async_runtime --serial

print_summary_and_exit \
    "All GPU tests passed!" \
    "Some GPU tests failed. See output above for details."

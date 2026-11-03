#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Driver-free tests of the example runner's pass/skip/failure accounting.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

example_files=("$REPO_ROOT"/cutile-examples/examples/*.rs)
example_count=${#example_files[@]}

output=$(
    cargo() {
        local example=""
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == --example ]]; then example="$2"; break; fi
            shift
        done
        case "$example" in
            pdl) echo 'SKIP pdl: requires Tile IR 13.4'; return 0 ;;
            nvfp4)
                echo 'SKIP nvfp4: must not override the failing exit status'
                echo 'simulated launch failure' >&2
                return 1 ;;
            mxfp8) echo 'simulated compiler failure' >&2; return 1 ;;
            *) return 0 ;;
        esac
    }
    run_examples "$REPO_ROOT/cutile-examples/examples"
    [[ "$OVERALL_SUCCESS" == false ]]
) || { echo "runner failed to record an error"; exit 1; }

for expected in \
    "2 examples failed, $((example_count - 3)) passed, 1 skipped" \
    'SKIP pdl: requires Tile IR 13.4' \
    'simulated compiler failure' \
    'simulated launch failure'
do
    if [[ "$output" != *"$expected"* ]]; then
        printf 'Missing %s in runner output:\n%s\n' "$expected" "$output"
        exit 1
    fi
done

output=$(
    cargo() {
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == --example ]]; then
                echo "SKIP $2: unsupported test target"
                return 0
            fi
            shift
        done
        return 1
    }
    run_examples "$REPO_ROOT/cutile-examples/examples"
    [[ "$OVERALL_SUCCESS" == true ]]
) || { echo "expected skips were treated as failures"; exit 1; }
if [[ "$output" != *"0 examples passed, ${example_count} skipped"* ]]; then
    printf 'Skipped examples were counted as passes:\n%s\n' "$output"
    exit 1
fi

echo 'Example runner accounting tests passed.'

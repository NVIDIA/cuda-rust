#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

EXTRA_FEATURES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --features) EXTRA_FEATURES="--features $2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -n "$EXTRA_FEATURES" ]]; then
    print_header "Running benchmarks ($EXTRA_FEATURES)"
else
    print_header "Running benchmarks"
fi

echo -e "${YELLOW}>>> Running cutile-benchmarks (GPU)${NC}"
cd "$REPO_ROOT/cutile-benchmarks" || exit 1
run_benches "$REPO_ROOT/cutile-benchmarks/benches" $EXTRA_FEATURES

print_summary_and_exit \
    "All benchmarks passed!" \
    "Some benchmarks failed. See output above for details."

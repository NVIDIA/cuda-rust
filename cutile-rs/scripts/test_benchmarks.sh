#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke-test all benchmarks: verify each code path executes without
# measuring performance. Results are discarded (no baseline pollution).

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Smoke-testing benchmarks"

echo -e "${YELLOW}>>> Running cutile-benchmarks with smoke-test (GPU)${NC}"
cd "$REPO_ROOT/cutile-benchmarks" || exit 1
run_benches "$REPO_ROOT/cutile-benchmarks/benches" "smoke-test"

print_summary_and_exit \
    "All benchmarks passed!" \
    "Some benchmarks failed. See output above for details."

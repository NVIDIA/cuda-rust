#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

# Enable the features required by the CPU-reference and autotune examples.
EXTRA_FEATURES="--features reference-cpu,experimental-tune"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --features) EXTRA_FEATURES="--features $2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -n "$EXTRA_FEATURES" ]]; then
    print_header "Running examples ($EXTRA_FEATURES)"
else
    print_header "Running examples"
fi

run_step \
    "cutile-ir build_basic example" \
    cargo run -p cutile-ir --example build_basic --quiet

echo -e "${YELLOW}>>> Running cutile-examples (GPU)${NC}"
cd "$REPO_ROOT/cutile-examples" || exit 1
run_examples "$REPO_ROOT/cutile-examples/examples" $EXTRA_FEATURES

print_summary_and_exit \
    "All supported examples passed (see skipped examples above)." \
    "Some examples failed. See output above for details."

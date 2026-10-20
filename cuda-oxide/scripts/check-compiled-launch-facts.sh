#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arch="${1:-sm_120}"
oxide=(cargo oxide)
if [[ -n "${CARGO_OXIDE_BIN:-}" ]]; then
    oxide=("$CARGO_OXIDE_BIN")
fi
log=$(mktemp)
trap 'rm -f "$log"' EXIT

"${oxide[@]}" run compiled_launch_facts --arch "$arch"
"${oxide[@]}" run compiled_launch_facts --arch "$arch" --features generic
"${oxide[@]}" run compiled_launch_facts --arch "$arch" --materialize-cubin

if "${oxide[@]}" run compiled_launch_facts --arch "$arch" --features invalid-raw >"$log" 2>&1; then
    echo "error: inferred scratch accepted an unconstrained raw dynamic view" >&2
    exit 1
fi
grep -F 'inferred dynamic shared memory cannot include unconstrained' "$log"

"${oxide[@]}" run compiled_launch_facts --arch "$arch" --features generic --materialize-cubin
"${oxide[@]}" run cross_crate_kernel --arch "$arch" --features native-ambiguity
if "${oxide[@]}" run cross_crate_kernel --arch "$arch" --materialize-cubin --features native-ambiguity >"$log" 2>&1; then
    echo "error: generic loading accepted separate native artifacts" >&2
    exit 1
fi
grep -F 'IncompatibleGenericArtifacts' "$log"
grep -F 'kernel-lib' "$log"
echo 'PASS: compiled launch facts, incomplete-plan and native-ambiguity rejection'

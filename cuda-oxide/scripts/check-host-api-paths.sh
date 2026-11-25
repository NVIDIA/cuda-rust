#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reject paths into the removed cuda-core SIMT namespace. Shared handles and
# typed buffers now live at the crate root, as does the shared async runtime.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Files whose host-API spellings a reader copies. `git ls-files` paths are
# repository-relative; grep them from the git root.
mapfile -t files < <(git -C "${GIT_ROOT}" ls-files \
    README.md cuda-oxide/README.md CONTRIBUTING.md \
    'cuda-oxide/cuda-oxide-book/*.md' 'cuda-oxide/cuda-oxide-book/**/*.md' \
    'cuda-oxide/crates/*/README.md' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/README.md' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/README.md' \
    'cuda-oxide/examples/*/README.md' \
    'cuda-oxide/examples/*/*/README.md' \
    'cuda-oxide/models/*/README.md' \
    'cuda-oxide/models/*/*/README.md' \
    'cuda-oxide/crates/cargo-oxide/src/**/*.rs' 'cuda-oxide/crates/cargo-oxide/src/*.rs' \
    'cuda-oxide/crates/cuda-host/src/**/*.rs' 'cuda-oxide/crates/cuda-host/src/*.rs' \
    'cuda-oxide/crates/cuda-macros/src/**/*.rs' 'cuda-oxide/crates/cuda-macros/src/*.rs' \
    'cuda-oxide/crates/cuda-device/src/**/*.rs' 'cuda-oxide/crates/cuda-device/src/*.rs' \
    | sort -u)

# Each pattern is an ERE; `cuda_core::...` never matches because the
# module name has to follow `cuda_core::` directly.
patterns=(
    'cuda_core::simt'
    'cuda_core::error::'
    'cuda_async::simt'
)

status=0
for pattern in "${patterns[@]}"; do
    if hits="$(grep -n -E -- "${pattern}" "${files[@]/#/${GIT_ROOT}/}" 2>/dev/null)"; then
        status=1
        echo "error: host-API path that does not resolve on the shared crates (/${pattern}/):" >&2
        echo "${hits}" | sed "s|^${GIT_ROOT}/||; s/^/    /" >&2
    fi
done

if [ "${status}" -ne 0 ]; then
    echo "hint: cuda-core handles and buffers use the shared crate-root API." >&2
    exit 1
fi
echo "OK: no stale host-API root paths in $((${#files[@]})) doc and source files."

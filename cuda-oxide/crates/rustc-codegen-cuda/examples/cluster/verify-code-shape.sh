#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# map_shared_rank / map_shared_rank_mut must keep their results in the
# cluster-shared address space end to end. If the mapped pointer decays to a
# generic pointer, an ordinary Rust deref compiles to a CTA-local access and
# the remote read/write silently targets the wrong shared memory. Assert the
# LLVM IR loads and stores through `ptr addrspace(7)` and that the two
# plain-deref kernels emit `ld.shared::cluster` / `st.shared::cluster`, so a
# regression to generic accesses cannot pass compile-only CI.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${root}/cluster.ll"
ptx="${root}/cluster.ptx"

test -s "${ll}"
test -s "${ptx}"

require_ll_shape() {
    local description="$1"
    local pattern="$2"
    if ! grep -E "${pattern}" "${ll}" >/dev/null; then
        echo "error: missing ${description} in ${ll}" >&2
        exit 1
    fi
}

# Print the body of one PTX entry. Waits for the header's `{` so a forward
# declaration (which ends in `;`) is skipped.
entry_body() {
    local symbol="$1"
    awk -v marker="${symbol}(" '
        !emit && !candidate && index($0, marker) && index($0, ".entry") {
            candidate = 1
        }
        candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ {
            candidate = 0
            next
        }
        candidate && $0 ~ /^[[:space:]]*\{[[:space:]]*$/ {
            emit = 1
            candidate = 0
        }
        emit { print }
        emit && index($0, "End function") != 0 { exit }
    ' "${ptx}"
}

require_entry_shape() {
    local symbol="$1"
    local description="$2"
    local pattern="$3"
    if ! entry_body "${symbol}" | grep -E "${pattern}" >/dev/null; then
        echo "error: missing ${description} in ${ptx}:${symbol}" >&2
        exit 1
    fi
}

# The mapped remote read and write must stay in addrspace(7) at the IR level.
require_ll_shape "cluster-shared mapped load" \
    'load i32, ptr addrspace\(7\)'
require_ll_shape "cluster-shared mapped store" \
    'store i32 [^,]+, ptr addrspace\(7\)'

# The plain-deref kernels must select the cluster-shared PTX access forms.
require_entry_shape test_dsmem_ring_exchange \
    "cluster-shared remote load" 'ld\.shared::cluster\.'
require_entry_shape test_dsmem_mapped_store \
    "cluster-shared remote store" 'st\.shared::cluster\.'

echo "cluster code shape: PASS"

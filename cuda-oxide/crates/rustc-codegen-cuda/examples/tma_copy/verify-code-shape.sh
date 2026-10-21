#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cp_async_bulk_tensor_2d_g2s_cta must copy into the issuing CTA's shared
# memory: its kernel emits the `.shared::cta` destination form and, since its
# `SharedPtr` destination is shared by type, converts no address to the shared
# or cluster window. The cluster API must keep its
# `.shared::cluster` form, since its destination may belong to another CTA.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${root}/tma_copy.ll"
ptx="${root}/tma_copy.ptx"

test -s "${ll}"
test -s "${ptx}"

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

forbid_entry_shape() {
    local symbol="$1"
    local description="$2"
    local pattern="$3"
    if entry_body "${symbol}" | grep -E "${pattern}" >/dev/null; then
        echo "error: unexpected ${description} in ${ptx}:${symbol}" >&2
        exit 1
    fi
}

if ! grep -E 'call void @llvm\.nvvm\.cp\.async\.bulk\.tensor\.g2s\.cta\.tile\.2d\(ptr addrspace\(3\)' \
    "${ll}" >/dev/null; then
    echo "error: missing typed CTA-local G2S call in ${ll}" >&2
    exit 1
fi

require_entry_shape tma_copy_2d_cta_test "CTA-local TMA load" \
    'cp\.async\.bulk\.tensor\.2d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes'
forbid_entry_shape tma_copy_2d_cta_test "cluster-window destination" 'shared::cluster'
forbid_entry_shape tma_copy_2d_cta_test "generic-to-shared address conversion" 'cvta\.to\.shared'

require_entry_shape tma_copy_2d_test "cluster TMA load" \
    'cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes'

echo "tma_copy code shape: PASS"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime SUCCESS gate only proves the kernels are bit-correct, which holds
# with or without the optimization this example exists to demonstrate. The perf
# property lives in the PTX shape: an #[repr(C, align(8))] pair must fuse its
# two adjacent f32 field writes into one st.global.v2.b32, while the
# natural-align-4 control must keep two scalar stores. Assert both directions so
# a silent regression of either the fusion or the control cannot pass.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/aligned_field_stores.ptx"

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

reject_entry_shape() {
    local symbol="$1"
    local description="$2"
    local pattern="$3"
    if entry_body "${symbol}" | grep -E "${pattern}" >/dev/null; then
        echo "error: found ${description} in ${ptx}:${symbol}" >&2
        exit 1
    fi
}

vector_store='st\.global\.v2\.b32'
scalar_store='st\.global\.(b32|f32)'

# The align(8) kernels must fuse the pair into a single wide store. `lanes_store`
# reaches the same two f32 through an array index rather than named fields, which
# is the element path rather than the field path; both carry the aggregate's
# alignment on the address, and both are consumed by the store.
for kernel in aligned_store lanes_store hot_aligned; do
    require_entry_shape "${kernel}" \
        "vectorized field-pair store" "${vector_store}"
    reject_entry_shape "${kernel}" \
        "unfused scalar field store" "${scalar_store}"
done

# The natural-align-4 controls prove nothing wider than the f32 itself, so
# fusing them would be claiming alignment the source never guaranteed. Widening
# a store is worse than widening a load: an over-aligned store instruction on a
# under-aligned address is undefined, not merely slow.
for kernel in packed_store hot_packed; do
    require_entry_shape "${kernel}" \
        "scalar field store" "${scalar_store}"
    reject_entry_shape "${kernel}" \
        "over-claimed vectorized store" "${vector_store}"
done

echo "aligned_field_stores code shape: PASS"

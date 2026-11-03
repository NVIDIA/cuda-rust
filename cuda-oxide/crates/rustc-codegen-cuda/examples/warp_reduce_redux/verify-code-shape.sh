#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime PASS gate only proves the reductions are correct, which the
# shuffle butterfly also achieves. Which form was emitted lives in the PTX:
# at a floor of sm_80 or newer each integer reduction must be one `redux.sync`
# and the butterfly must be gone; the f32 and sub-warp controls must never use
# `redux`; and below sm_80 no `redux` may appear at all.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/warp_reduce_redux.ptx"

test -s "${ptx}"

target="$(grep -m1 -oE '^\.target sm_[0-9]+' "${ptx}" | grep -oE '[0-9]+$')" || true
if [[ -z "${target}" ]]; then
    echo "error: no .target sm_NN line in ${ptx}" >&2
    exit 1
fi

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

# count <kernel> <opcode-regex>: instructions in the kernel body whose opcode
# matches, anchored at instruction position so comments never count.
count() {
    local symbol="$1" opcode="$2" body
    body="$(entry_body "${symbol}")"
    if [[ -z "${body}" ]]; then
        echo "error: no .entry body found for ${symbol} in ${ptx}" >&2
        exit 1
    fi
    grep -cE "^[[:space:]]*(@%p[0-9]+[[:space:]]+)?${opcode}" <<<"${body}" || true
}

# require <kernel> <opcode-regex> <expected-count>
require() {
    local symbol="$1" opcode="$2" want="$3" got
    got="$(count "${symbol}" "${opcode}")"
    if [[ "${got}" -ne "${want}" ]]; then
        echo "error: sm_${target}: ${symbol} has ${got}x ${opcode}, expected ${want}x in ${ptx}" >&2
        exit 1
    fi
}

if [[ "${target}" -ge 80 ]]; then
    # Six u32 ops plus the unsigned min/max over values above 2^31.
    require reduce_u32_all_ops 'redux\.sync\.' 8
    require reduce_i32_signed_ops 'redux\.sync\.' 3
    require reduce_u32_all_ops 'shfl\.sync\.' 0
    require reduce_i32_signed_ops 'shfl\.sync\.' 0
else
    require reduce_u32_all_ops 'redux\.' 0
    require reduce_i32_signed_ops 'redux\.' 0
fi
require reduce_f32_control 'redux\.' 0
require reduce_subwarp_control 'redux\.' 0

echo "warp_reduce_redux code shape OK at sm_${target}"

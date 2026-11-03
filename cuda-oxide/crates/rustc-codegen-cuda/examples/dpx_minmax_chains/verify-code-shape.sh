#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The runtime PASS gate only proves the chains are bit-correct, which holds
# even if min/max degrades to compare-and-branch. The DPX property lives in
# the PTX shape: ptxas fuses dataflow-adjacent native min.*/max.* chains into
# DPX (VIMNMX) SASS on sm_90+, so each kernel must keep exactly its chain's
# min/max instructions — no more (no duplication into the edge-lane branch),
# no fewer (no lowering through setp/selp), and none of the other signedness.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/dpx_minmax_chains.ptx"

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

# require_shape <kernel> <min-count> <max-count> <type> <other-type>
#
# The kernel body must contain exactly <min-count> `min.<type>` and
# <max-count> `max.<type>` instructions, and no integer min/max of
# <other-type> at all.
require_shape() {
    local symbol="$1" min_count="$2" max_count="$3" ty="$4" other="$5"
    local body got_min got_max got_other
    body="$(entry_body "${symbol}")"
    if [[ -z "${body}" ]]; then
        echo "error: no .entry body found for ${symbol} in ${ptx}" >&2
        exit 1
    fi
    # Anchor to instruction position (start of line, then the opcode) so
    # comments or symbol names containing "min.s32" are never counted.
    got_min="$(grep -cE "^[[:space:]]*min\\.${ty}[[:space:]]" <<<"${body}")" || true
    got_max="$(grep -cE "^[[:space:]]*max\\.${ty}[[:space:]]" <<<"${body}")" || true
    got_other="$(grep -cE "^[[:space:]]*(min|max)\\.${other}[[:space:]]" <<<"${body}")" || true
    if [[ "${got_min}" -ne "${min_count}" || "${got_max}" -ne "${max_count}" ]]; then
        echo "error: ${symbol} expected ${min_count}x min.${ty} + ${max_count}x max.${ty}," \
            "got ${got_min}x + ${got_max}x in ${ptx}" >&2
        exit 1
    fi
    if [[ "${got_other}" -ne 0 ]]; then
        echo "error: ${symbol} contains ${got_other}x min/max.${other} in ${ptx}" >&2
        exit 1
    fi
}

#             kernel        min max type  other
require_shape vimax3_s32    0   2   s32   u32
require_shape vimin3_s32    2   0   s32   u32
require_shape vimnmx_s32    1   1   s32   u32
require_shape viaddmax_s32  0   1   s32   u32
require_shape viaddmin_s32  1   0   s32   u32
require_shape vimax3_u32    0   2   u32   s32
require_shape vimin3_u32    2   0   u32   s32
require_shape vimnmx_u32    1   1   u32   s32
require_shape viaddmax_u32  0   1   u32   s32
require_shape viaddmin_u32  1   0   u32   s32

echo "dpx_minmax_chains code shape: PASS"

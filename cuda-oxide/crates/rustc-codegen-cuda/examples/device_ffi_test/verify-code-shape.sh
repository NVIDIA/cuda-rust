#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A runtime q -> Q check cannot distinguish a plain i32 ABI from the plausible
# but wrong `zeroext i32` mapping: both reach the same PTX and hardware result.
# Pin the declaration and calls in the emitted LLVM IR before LTO.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${root}/device_ffi_test.ll"

test -s "${ll}"

require_count() {
    local description="$1"
    local expected="$2"
    local pattern="$3"
    local actual
    actual="$(grep -Ec "${pattern}" "${ll}" || true)"
    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "error: expected ${expected} ${description} in ${ll}, found ${actual}" >&2
        exit 1
    fi
}

require_count "plain char_to_upper declaration" 1 \
    '^declare i32 @char_to_upper\(i32\)( #[0-9]+)?$'
require_count "'q' char_to_upper call" 1 \
    'call i32 @char_to_upper\(i32 113\)'
require_count "'Z' char_to_upper call" 1 \
    'call i32 @char_to_upper\(i32 90\)'

# Legacy NVVM IR retains the i32 pointee while modern IR uses opaque `ptr`.
require_count "plain char_store declaration" 1 \
    '^declare void @char_store\((i32\*|ptr), i32\)( #[0-9]+)?$'
require_count "char_store pointer call" 1 \
    'call void @char_store\((i32\*|ptr) [^,]+, i32 955\)'

char_lines="$(grep -E '(declare|call).*@(char_to_upper|char_store)\(' "${ll}" || true)"

if grep -Eq '\b(signext|zeroext)\b' <<<"${char_lines}"; then
    echo "error: 32-bit char ABI must not carry an extension attribute" >&2
    echo "${char_lines}" >&2
    exit 1
fi

echo "device_ffi_test char ABI shape: PASS"

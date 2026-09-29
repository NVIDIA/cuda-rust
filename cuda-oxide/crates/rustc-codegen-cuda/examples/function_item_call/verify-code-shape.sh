#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm="${root}/function_item_call.ll"

test -s "${llvm}"

extract_llvm_function() {
    local needle="$1"
    awk -v needle="${needle}" '
        /^define / && index($0, needle) != 0 { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${llvm}"
}

kernel="$(extract_llvm_function '@diverging_callable(')"
test -n "${kernel}"

call_count="$(
    grep -Ec 'call void @[^ (]*write_then_diverge\(' <<<"${kernel}" || true
)"
if [[ "${call_count}" -ne 1 ]]; then
    echo "diverging_callable must contain exactly one direct call to write_then_diverge" >&2
    exit 1
fi

call_line="$(
    grep -nE 'call void @[^ (]*write_then_diverge\(' <<<"${kernel}" |
        head -n1 |
        cut -d: -f1
)"
next_line="$(sed -n "$((call_line + 1))p" <<<"${kernel}")"
if ! grep -Eq '^[[:space:]]*unreachable[[:space:]]*$' <<<"${next_line}"; then
    echo "direct diverging call must be immediately followed by LLVM unreachable" >&2
    exit 1
fi

helper="$(extract_llvm_function 'write_then_diverge')"
test -n "${helper}"
if ! grep -Eq 'store i32 286331153, ptr ' <<<"${helper}"; then
    echo "write_then_diverge lost its observable sentinel store" >&2
    exit 1
fi

echo "function_item_call code shape: PASS"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each inline request must reach LLVM as its function attribute and leave the
# matching mark in the PTX. The pre-opt inline_intent.ll shows the attribute
# on each definition. The PTX shows what opt made of it: the `noinline`
# helpers stay real functions that the entry reaches with `call.uni`, the
# `alwaysinline` helper leaves no function behind, and the const-generic
# `#[device]` helper (whose macro-added `#[inline(never)]` is a host-side
# boundary only) folds its factor into an immediate. A regression in either
# direction fails compile-only CI.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${root}/inline_intent.ll"
ptx="${root}/inline_intent.ptx"

test -s "${ll}"
test -s "${ptx}"

# The one `define` line whose function name matches the pattern.
define_line() {
    local pattern="$1"
    local lines
    lines="$(grep -E "^define .*@${pattern}\(" "${ll}" || true)"
    if [[ -z "${lines}" ]]; then
        echo "error: no definition matching ${pattern} in ${ll}" >&2
        exit 1
    fi
    if [[ "$(wc -l <<<"${lines}")" -ne 1 ]]; then
        echo "error: more than one definition matches ${pattern} in ${ll}:" >&2
        echo "${lines}" >&2
        exit 1
    fi
    printf '%s\n' "${lines}"
}

require_define_keyword() {
    local description="$1" pattern="$2" keyword="$3"
    local line
    line="$(define_line "${pattern}")"
    if ! grep -E "\) ${keyword} " <<<"${line}" >/dev/null; then
        echo "error: ${description}: ${keyword} missing on: ${line}" >&2
        exit 1
    fi
}

forbid_define_keywords() {
    local description="$1" pattern="$2"
    local line
    line="$(define_line "${pattern}")"
    if grep -E 'noinline|inlinehint|alwaysinline' <<<"${line}" >/dev/null; then
        echo "error: ${description}: unexpected inline keyword on: ${line}" >&2
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
    local symbol="$1" description="$2" pattern="$3"
    if ! entry_body "${symbol}" | grep -E "${pattern}" >/dev/null; then
        echo "error: missing ${description} in ${ptx}:${symbol}" >&2
        exit 1
    fi
}

require_ptx_function() {
    local description="$1" pattern="$2"
    if ! grep -E "^[[:space:]]*(\.visible |\.weak )?\.func .*${pattern}\(" "${ptx}" >/dev/null; then
        echo "error: missing ${description} in ${ptx}" >&2
        exit 1
    fi
}

forbid_ptx_function() {
    local description="$1" pattern="$2"
    if grep -E "^[[:space:]]*(\.visible |\.weak )?\.func .*${pattern}\(" "${ptx}" >/dev/null; then
        echo "error: ${description} is still a PTX function in ${ptx}" >&2
        exit 1
    fi
}

# Pre-opt LLVM IR: one keyword per request, none on the generic #[device]
# instance whose #[inline(never)] belongs to the macro.
require_define_keyword "plain #[inline(never)] helper" 'inline_intent__never_add' noinline
require_define_keyword "non-generic #[inline(never)] #[device] helper" 'never_dev' noinline
require_define_keyword "plain #[inline] helper" 'inline_intent__hint_mul' inlinehint
require_define_keyword "plain #[inline(always)] helper" 'inline_intent__always_sub' alwaysinline
forbid_define_keywords "const-generic #[device] helper" 'scale_by[A-Za-z0-9_.$]*'

# PTX: the noinline helpers survive opt as functions the entry calls.
require_ptx_function "plain noinline helper" 'inline_intent__never_add'
require_ptx_function "noinline #[device] helper" 'never_dev'
require_entry_shape intent "call to the plain noinline helper" 'inline_intent__never_add'
require_entry_shape intent "call to the noinline #[device] helper" 'never_dev'
if [[ "$(entry_body intent | grep -c 'call\.uni')" -lt 2 ]]; then
    echo "error: the intent entry must call both noinline helpers with call.uni" >&2
    exit 1
fi

# PTX: alwaysinline leaves no function. The const-generic #[device] helper
# keeps a `.func` because every #[device] function stays exported through
# `@llvm.used`, but the entry must not call it: its factor 4.0 (0f40800000)
# is an immediate in the entry instead.
forbid_ptx_function "alwaysinline helper" 'inline_intent__always_sub'
if entry_body intent | grep -E 'scale_by' >/dev/null; then
    echo "error: the intent entry still calls the const-generic #[device] helper in ${ptx}" >&2
    exit 1
fi
require_entry_shape intent "folded const-generic factor" '0f40800000'

echo "inline_intent code shape: PASS"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm="${root}/function_item_call.ll"
ptx="${root}/function_item_call.ptx"

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

# A kept diverging call must be followed by LLVM `unreachable` on the next
# line. A `--device-debug` build appends `, !dbg !N` to that line, so the
# match stops at the keyword. Monomorphized callees carry a mangled suffix
# after the Rust name (`apply_neverNvB2_12spin_foreverEB2_`), so the pattern
# allows symbol text on both sides of the name.
expect_kept_diverging_calls() {
    local callee="$1"
    local expected="$2"
    local pattern="call void @[^ (]*${callee}[^ (]*\\("
    local found
    found="$(grep -Ec "${pattern}" <<<"${kernel}" || true)"
    if [[ "${found}" -ne "${expected}" ]]; then
        echo "diverging_callable must keep ${expected} direct call(s) to ${callee}, found ${found}" >&2
        exit 1
    fi
    local line next
    while IFS= read -r line; do
        next="$(sed -n "$((line + 1))p" <<<"${kernel}")"
        if ! grep -Eq '^[[:space:]]*unreachable(,|[[:space:]]*$)' <<<"${next}"; then
            echo "the direct diverging call to ${callee} at kernel line ${line} is not followed by LLVM unreachable" >&2
            exit 1
        fi
    done < <(grep -nE "${pattern}" <<<"${kernel}" | cut -d: -f1)
}

# The three call sites the importer used to replace with a trap: two through
# the generic `apply_never` (a function item and a closure) and the ordinary
# direct call to the store-then-loop helper.
expect_kept_diverging_calls 'apply_never' 2
expect_kept_diverging_calls 'write_then_diverge' 1

# Retained-trap control. `panic_only` has a body that is nothing but a panic,
# which the collector does not emit, so its call site must still lower to the
# device trap and the callee must not be called.
trap_count="$(grep -Ec 'asm sideeffect "trap;"' <<<"${kernel}" || true)"
if [[ "${trap_count}" -ne 1 ]]; then
    echo "diverging_callable must keep one trap for the panic-only callee, found ${trap_count}" >&2
    exit 1
fi
if grep -Eq 'call void @[^ (]*panic_only[^ (]*\(' <<<"${kernel}"; then
    echo "the panic-only callee must not be called: the collector does not emit its body" >&2
    exit 1
fi

helper="$(extract_llvm_function 'write_then_diverge')"
test -n "${helper}"
if ! grep -Eq 'store i32 286331153, ptr ' <<<"${helper}"; then
    echo "write_then_diverge lost its observable sentinel store" >&2
    exit 1
fi

# The direct llc route also writes PTX; the NVVM route writes only `.ll` and
# `.target`. Check the PTX only when it exists and is at least as new as the
# IR, so a leftover from an older build is never checked. In the kernel entry
# the sentinel store must survive `opt` and `llc` (the helper is normally
# inlined) or the helper must still be called, and the panic-only control
# must keep its trap.
if [[ -s "${ptx}" && ! "${ptx}" -ot "${llvm}" ]]; then
    extract_ptx_function() {
        local pattern="$1"
        awk -v pattern="${pattern}" '
            /^\.(visible )?\.(entry|func) / && $0 ~ pattern && $0 ~ /\($/ { in_function = 1 }
            in_function { print }
            in_function && /^}/ { exit }
        ' "${ptx}"
    }
    entry="$(extract_ptx_function '[.]entry diverging_callable[(]')"
    test -n "${entry}"
    if ! grep -Eq 'st\.global\.b32[[:space:]]+\[[^]]+\], 286331153|call\.uni[[:space:]].*write_then_diverge' <<<"${entry}"; then
        echo "the diverging_callable PTX entry neither stores the sentinel nor calls write_then_diverge" >&2
        exit 1
    fi
    ptx_trap_count="$(grep -Ec '^[[:space:]]*trap;' <<<"${entry}" || true)"
    if [[ "${ptx_trap_count}" -ne 1 ]]; then
        echo "the diverging_callable PTX entry must keep one trap for the panic-only callee, found ${ptx_trap_count}" >&2
        exit 1
    fi
fi

echo "function_item_call code shape: PASS"

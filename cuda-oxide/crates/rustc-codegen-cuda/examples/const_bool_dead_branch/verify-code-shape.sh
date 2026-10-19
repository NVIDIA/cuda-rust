#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifacts=(
    "${root}/const_bool_dead_branch.ll"
    "${root}/const_bool_dead_branch.ptx"
)
if [[ -s "${root}/const_bool_dead_branch.opt.ll" ]]; then
    artifacts+=("${root}/const_bool_dead_branch.opt.ll")
fi

for artifact in "${artifacts[@]}"; do
    test -s "${artifact}"
    grep -q 'ExplicitOn.*ExplicitMode.*hook' "${artifact}"
    grep -q 'DefaultOn.*DefaultMode.*hook' "${artifact}"
    grep -q 'Dynamic.*DynamicMode.*hook' "${artifact}"
    if grep -q 'ExplicitOff.*ExplicitMode.*hook' "${artifact}"; then
        echo "dead explicit hook found in ${artifact}" >&2
        exit 1
    fi
    if grep -q 'DefaultOff.*DefaultMode.*hook' "${artifact}"; then
        echo "dead default hook found in ${artifact}" >&2
        exit 1
    fi
done

extract_llvm_function() {
    local pattern="$1"
    awk -v pattern="${pattern}" '
        /^define / && $0 ~ pattern { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${root}/const_bool_dead_branch.ll"
}

for pattern in \
    'select_default.*DefaultOn' \
    'select_default.*DefaultOff' \
    'select_explicit.*ExplicitOn' \
    'select_explicit.*ExplicitOff'; do
    body="$(extract_llvm_function "${pattern}")"
    test -n "${body}"
    if grep -q 'br i1' <<<"${body}"; then
        echo "const-selected function still contains a conditional branch: ${pattern}" >&2
        exit 1
    fi
done

dynamic_body="$(extract_llvm_function 'select_dynamic.*Dynamic')"
test -n "${dynamic_body}"
if [[ "$(grep -c 'br i1' <<<"${dynamic_body}")" -ne 1 ]]; then
    echo "dynamic control must retain exactly one conditional branch" >&2
    exit 1
fi

pointer_body="$(extract_llvm_function 'write_pointer.*GlobalPointerOnly')"
test -n "${pointer_body}"
if grep -q 'addrspace(3)' <<<"${pointer_body}"; then
    echo "const-dead shared-pointer arm contaminated the live global-pointer slot" >&2
    exit 1
fi
if ! grep -Eq 'store i32 7, ptr %' <<<"${pointer_body}"; then
    echo "global-pointer write is missing from the const-selected function" >&2
    exit 1
fi

dynamic_pointer_body="$(extract_llvm_function 'write_pointer_dynamic')"
test -n "${dynamic_pointer_body}"
if ! grep -Eq 'store i32 11, ptr %' <<<"${dynamic_pointer_body}"; then
    echo "runtime-selected pointer write did not remain generic" >&2
    exit 1
fi
if grep -Eq 'store i32 11, ptr addrspace\(3\)|to ptr addrspace\(3\)' <<<"${dynamic_pointer_body}"; then
    echo "runtime-selected global pointer was unsafely narrowed to shared" >&2
    exit 1
fi

extract_ptx_function() {
    local pattern="$1"
    awk -v pattern="${pattern}" '
        /^\.(visible )?\.(entry|func) / && $0 ~ pattern && $0 ~ /\($/ { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${root}/const_bool_dead_branch.ptx"
}

pointer_kernel="$(extract_ptx_function 'dead_shared_pointer_TID_')"
test -n "${pointer_kernel}"
if ! grep -Eq 'st\.global\.b32[[:space:]]+\[[^]]+\], 7|call\.uni .*dead_shared_pointer.*GlobalPointerOnly' <<<"${pointer_kernel}"; then
    echo "const-selected kernel neither performs nor calls its global write" >&2
    exit 1
fi
if grep -Eq 'cvta\.to\.shared|st\.shared' <<<"${pointer_kernel}"; then
    echo "const-dead shared-pointer code leaked into its kernel" >&2
    exit 1
fi

pointer_helper="$(extract_ptx_function 'write_pointer.*GlobalPointerOnly')"
test -n "${pointer_helper}"
if ! grep -Eq 'st\.b32[[:space:]]+\[[^]]+\], 7' <<<"${pointer_helper}"; then
    echo "const-selected pointer helper is missing its generic store" >&2
    exit 1
fi
if grep -Eq 'cvta\.to\.shared|st\.shared' <<<"${pointer_helper}"; then
    echo "const-dead shared-pointer code leaked into its helper" >&2
    exit 1
fi

dynamic_pointer_ptx="$(extract_ptx_function 'write_pointer_dynamic')"
test -n "${dynamic_pointer_ptx}"
if ! grep -Eq 'st\.b32[[:space:]]+\[[^]]+\], 11' <<<"${dynamic_pointer_ptx}"; then
    echo "runtime-selected pointer helper is missing its generic store" >&2
    exit 1
fi
if grep -Eq 'cvta\.to\.shared|st\.shared' <<<"${dynamic_pointer_ptx}"; then
    echo "runtime-selected global pointer was narrowed to shared in PTX" >&2
    exit 1
fi

echo "const_bool_dead_branch code shape: PASS"

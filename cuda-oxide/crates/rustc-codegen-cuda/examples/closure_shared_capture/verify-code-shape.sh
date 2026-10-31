#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${root}/closure_shared_capture.ll"
ptx="${root}/closure_shared_capture.ptx"
test -s "${ll}"
test -s "${ptx}"

extract_llvm_function() {
    local pattern="$1"
    awk -v pattern="${pattern}" '
        /^define / && $0 ~ pattern { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${ll}"
}

extract_ptx_function() {
    local pattern="$1"
    awk -v pattern="${pattern}" '
        /^\.(visible )?\.(entry|func) / && $0 ~ pattern && $0 ~ /\($/ { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${ptx}"
}

# an optional guard predicate before an instruction
guard='^[[:space:]]*(@!?%p[0-9]+[[:space:]]+)?'
# ordering and scope qualifiers that can sit between `ld`/`st` and the type
qualifiers='(\.(volatile|relaxed|weak|acquire|release|cta|cluster|gpu|sys|nc))*'

fail() {
    echo "$1" >&2
    exit 1
}

for kernel in dynamic_smem_closure static_smem_closure cp_async_closure; do
    # the pre-opt IR must still borrow the pointer local for an out-of-line
    # closure call; if rustc inlined the closure, this example would no longer
    # cover the by-reference capture
    body="$(extract_llvm_function "@${kernel}\\(")"
    test -n "${body}" || fail "${kernel}: kernel missing from ${ll}"
    grep -Eq "call void @[A-Za-z0-9_]*${kernel}0B[A-Za-z0-9_]*\\(" <<<"${body}" \
        || fail "${kernel}: closure call was inlined before codegen"
    # the borrowed slot holds the generic form of the shared pointer
    grep -Eq 'addrspacecast ptr addrspace\(3\) @__(dynamic_smem|shared_mem)_[A-Za-z0-9_]+ to ptr' \
        <<<"${body}" || fail "${kernel}: shared pointer not cast to generic before capture"

    # after optimization the accesses through the capture are shared again
    entry="$(extract_ptx_function "^\\.visible \\.entry ${kernel}\\(")"
    test -n "${entry}" || fail "${kernel}: entry missing from ${ptx}"
    # a generic access has no state space after its optional qualifiers, with
    # or without a guard predicate
    if grep -Eq "${guard}(ld|st)${qualifiers}\\.(v[248]\\.)?[bsuf](8|16|32|64)[[:space:]]" \
        <<<"${entry}"; then
        fail "${kernel}: generic shared-memory access in optimized PTX"
    fi
done

for kernel in dynamic_smem_closure static_smem_closure; do
    entry="$(extract_ptx_function "^\\.visible \\.entry ${kernel}\\(")"
    [[ "$(grep -Ec "${guard}st\\.shared\\.b16[[:space:]]" <<<"${entry}")" -eq 2 ]] \
        || fail "${kernel}: expected two st.shared.b16 stores from the closure"
    # the kernel's own neighbour reads go through the same pointer local
    [[ "$(grep -Ec "${guard}ld\\.shared\\.b16[[:space:]]" <<<"${entry}")" -eq 2 ]] \
        || fail "${kernel}: expected two ld.shared.b16 neighbour reads"
done

entry="$(extract_ptx_function '^\.visible \.entry cp_async_closure\(')"
[[ "$(grep -Ec "${guard}cp\\.async\\.cg\\.shared\\.global[[:space:]]" <<<"${entry}")" -eq 2 ]] \
    || fail "cp_async_closure: expected two cp.async.cg.shared.global copies from the closure"
[[ "$(grep -Ec "${guard}ld\\.shared\\.b32[[:space:]]" <<<"${entry}")" -eq 2 ]] \
    || fail "cp_async_closure: expected two ld.shared.b32 reads of the copied data"

echo "closure_shared_capture code shape: PASS"

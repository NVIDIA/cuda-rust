#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# `SharedPtr` keeps a CTA-shared pointer typed `addrspace(3)` across a call.
#
# LLVM IR: the non-inlined `shared_ptr_probe` helper takes one as its
# parameter and its call site passes one, the two fields of a by-value `Pair`
# reach `write_pair` as two such parameters, and `select_slot` returns one.
#
# PTX: `shared_ptr_probe` survives as a `.func` that the kernel calls once,
# and that `.func` initializes the barrier with `mbarrier.init.shared` on its
# parameter. No instruction converts a generic address to a shared one
# (`cvta.to.shared`) or asks which window an address is in (`isspacep`). The
# call into the `.func` is where a generic pointer would have needed one; a
# module-wide search alone proves little once the helper is inlined, because
# the generic-pointer control compiles to the same instructions.
#
# The PTX half reads the unoptimized build. scripts/smoketest.sh lists this
# example in NO_OPT_SHAPE_EXAMPLES and rebuilds it with CUDA_OXIDE_NO_OPT=1
# before running this script; by hand, build the same way first.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${dir}/shared_memory_helper.ll"
ptx="${dir}/shared_memory_helper.ptx"

for file in "${ll}" "${ptx}"; do
    [[ -f "${file}" ]] || { echo "FAIL: ${file} not found; build the example first" >&2; exit 1; }
done

# --- LLVM IR -----------------------------------------------------------------

if ! grep -Eq '^define .*shared_ptr_probe.*\(ptr addrspace\(3\) ' "${ll}"; then
    echo "FAIL: shared_ptr_probe does not take a ptr addrspace(3) parameter" >&2
    grep -E '^define .*shared_ptr_probe' "${ll}" >&2 || true
    exit 1
fi

if ! grep -Eq 'call i64 @[^(]*shared_ptr_probe[^(]*\(ptr addrspace\(3\) ' "${ll}"; then
    echo "FAIL: the call to shared_ptr_probe does not pass a ptr addrspace(3)" >&2
    grep -E 'call i64 @[^(]*shared_ptr_probe' "${ll}" >&2 || true
    exit 1
fi

if ! grep -Eq '^define .*write_pair.*\(ptr addrspace\(3\) %[A-Za-z0-9_.]+, ptr addrspace\(3\) %[A-Za-z0-9_.]+, ' "${ll}"; then
    echo "FAIL: write_pair does not take both Pair fields as ptr addrspace(3) parameters" >&2
    grep -E '^define .*write_pair' "${ll}" >&2 || true
    exit 1
fi

if ! grep -Eq '^define ptr addrspace\(3\) @[^(]*select_slot' "${ll}"; then
    echo "FAIL: select_slot does not return ptr addrspace(3)" >&2
    grep -E '^define .*select_slot' "${ll}" >&2 || true
    exit 1
fi

# --- PTX ---------------------------------------------------------------------

# The body of the PTX `.func` whose name contains the symbol: from the header
# line, past the parameter list, to the closing brace. A forward declaration
# ends in `;` before any `{` and is skipped.
func_body() {
    awk -v needle="$1" '
        !emit && !candidate && index($0, ".func") && index($0, needle) { candidate = 1 }
        candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ { candidate = 0; next }
        candidate && $0 ~ /^[[:space:]]*\{/ { emit = 1; candidate = 0 }
        emit { print }
        emit && $0 ~ /^\}/ { exit }
    ' "${ptx}"
}

probe_body="$(func_body shared_ptr_probe)"
if [[ -z "${probe_body}" ]]; then
    echo "FAIL: no PTX .func for shared_ptr_probe in ${ptx}" >&2
    echo "hint: this half reads the unoptimized build; rebuild with" >&2
    echo "      CUDA_OXIDE_NO_OPT=1 cargo oxide build shared_memory_helper" >&2
    exit 1
fi

if ! grep -Eq 'mbarrier\.init\.shared(::cta)?\.b64' <<<"${probe_body}"; then
    echo "FAIL: the shared_ptr_probe .func does not initialize the barrier with mbarrier.init.shared" >&2
    exit 1
fi

# One call into the helper: the symbol on a `call` line, outside `.func`
# headers and comments.
calls="$(grep -E 'shared_ptr_probe' "${ptx}" | grep -vE '\.func|^[[:space:]]*//' | grep -cE '(^|[[:space:]])call' || true)"
if [[ "${calls}" -ne 1 ]]; then
    echo "FAIL: expected one PTX call into shared_ptr_probe, found ${calls}" >&2
    grep -nE 'call.*shared_ptr_probe' "${ptx}" >&2 || true
    exit 1
fi

for forbidden in 'cvta\.to\.shared' 'isspacep'; do
    if grep -Eq "${forbidden}" "${ptx}"; then
        echo "FAIL: PTX contains ${forbidden}: a generic address was converted or classified" >&2
        grep -nE "${forbidden}" "${ptx}" >&2
        exit 1
    fi
done

echo "PASS: SharedPtr stays addrspace(3) across shared_ptr_probe, write_pair and select_slot; the shared_ptr_probe .func is called once and initializes its barrier in shared memory; no cvta.to.shared or isspacep"

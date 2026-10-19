#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# `SharedPtr` keeps a CTA-shared pointer typed `addrspace(3)` across a call:
# the non-inlined `shared_ptr_probe` helper takes one as its parameter, and
# no PTX converts a generic address to a shared one (`cvta.to.shared`).
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${dir}/shared_memory_helper.ll"
ptx="${dir}/shared_memory_helper.ptx"

for file in "${ll}" "${ptx}"; do
    [[ -f "${file}" ]] || { echo "FAIL: ${file} not found; build the example first" >&2; exit 1; }
done

if ! grep -Eq '^define .*shared_ptr_probe.*\(ptr addrspace\(3\) ' "${ll}"; then
    echo "FAIL: shared_ptr_probe does not take a ptr addrspace(3) parameter" >&2
    grep -E '^define .*shared_ptr_probe' "${ll}" >&2 || true
    exit 1
fi

if grep -q 'cvta\.to\.shared' "${ptx}"; then
    echo "FAIL: PTX converts a generic address to a shared one" >&2
    grep -n 'cvta\.to\.shared' "${ptx}" >&2
    exit 1
fi

echo "PASS: SharedPtr stays addrspace(3) across shared_ptr_probe, no cvta.to.shared"

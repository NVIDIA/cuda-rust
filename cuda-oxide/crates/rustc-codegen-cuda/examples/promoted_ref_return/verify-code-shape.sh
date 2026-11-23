#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Every helper returns a `&'static` reference across an `#[inline(never)]`
# call. Its pointee must live in a module-level read-only global, never in the
# helper's own frame (issue #1724).

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_ir="${root}/promoted_ref_return.ll"
ptx="${root}/promoted_ref_return.ptx"

test -s "${llvm_ir}"
test -s "${ptx}"

# Print one function body from the LLVM IR or the PTX
symbol_body() {
    local artifact="$1"
    local format="$2"
    local symbol="$3"

    if [[ "${format}" == "llvm" ]]; then
        awk -v marker="@${symbol}(" '
            !emit && index($0, marker) && $1 == "define" { emit = 1 }
            emit { print }
            emit && $0 == "}" { exit }
        ' "${artifact}"
    else
        # PTX lists prototypes first; a prototype ends at `;`, a body at `{`
        awk -v marker=") ${symbol}(" '
            !emit && !candidate && index($0, marker) && index($0, ".func") {
                candidate = 1
            }
            candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ { candidate = 0; next }
            candidate && index($0, "{") { emit = 1; candidate = 0 }
            emit { print }
            emit && index($0, "End function") != 0 { exit }
        ' "${artifact}"
    fi
}

require_shape_count() {
    local description="$1"
    local expected="$2"
    local pattern="$3"
    local actual

    actual="$(grep -Ec "${pattern}" "${llvm_ir}" || true)"
    if [[ "${actual}" -ne "${expected}" ]]; then
        echo "error: expected ${expected} ${description} in ${llvm_ir}, found ${actual}" >&2
        exit 1
    fi
}

helpers=(pair span literal answer nested second maybe wrapped some_ref via_static static_ref static_slice both raw raw_second raw_field)

for helper in "${helpers[@]}"; do
    llvm_body="$(symbol_body "${llvm_ir}" llvm "${helper}")"
    if [[ -z "${llvm_body}" ]]; then
        echo "error: ${helper} is missing from ${llvm_ir}" >&2
        exit 1
    fi
    if grep -q 'alloca' <<<"${llvm_body}"; then
        echo "error: ${helper} builds its constant in a stack slot:" >&2
        echo "${llvm_body}" >&2
        exit 1
    fi
    if ! grep -Eq 'ptr addrspace\(1\) @__device_global' <<<"${llvm_body}"; then
        echo "error: ${helper} does not address a device global:" >&2
        echo "${llvm_body}" >&2
        exit 1
    fi

    ptx_body="$(symbol_body "${ptx}" ptx "${helper}")"
    if [[ -z "${ptx_body}" ]]; then
        echo "error: ${helper} is missing from ${ptx}" >&2
        exit 1
    fi
    if grep -q '__local_depot' <<<"${ptx_body}"; then
        echo "error: ${helper} returns a pointer into its own .local frame:" >&2
        echo "${ptx_body}" >&2
        exit 1
    fi
done

# `PAIR` and `&PAIR.1` are one rustc allocation, so they share one read-only
# global, and `second` points 4 bytes into it
require_shape_count "read-only PAIR global" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \[8 x i8\] c"\\03\\00\\00\\00\\04\\00\\00\\00"'
if ! symbol_body "${llvm_ir}" llvm second | grep -Eq 'getelementptr inbounds i8, ptr addrspace\(1\) @__device_global[A-Za-z0-9_]*, i64 4'; then
    echo "error: second does not point 4 bytes into PAIR's global" >&2
    exit 1
fi

# `&&7` and `&&TABLE[2]`: the outer constant is a relocation to the inner
# allocation's own global, or to the device static with a byte addend
require_shape_count "read-only promoted pointer relocated to a promoted global" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \{ i64 \} \{ i64 ptrtoint \(ptr addrspacecast \(ptr addrspace\(1\) @__device_global[A-Za-z0-9_]* to ptr\) to i64\) \}'
require_shape_count "read-only promoted pointer relocated into the TABLE static" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \{ i64 \} \{ i64 ptrtoint \(ptr addrspacecast \(ptr addrspace\(1\) getelementptr \(i8, ptr addrspace\(1\) @__device_global[A-Za-z0-9_]*, i64 8\) to ptr\) to i64\) \}'

# `STATIC_SLICE`'s data word relocates to the promoted array's own read-only
# global, and its length stays literal initializer data
require_shape_count "read-only promoted array behind STATIC_SLICE" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \[12 x i8\] c"\\1F\\00\\00\\00\\25\\00\\00\\00\\29\\00\\00\\00"'
require_shape_count "STATIC_SLICE relocated to a promoted global" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) global \{ i64, \[8 x i8\] \} \{ i64 ptrtoint \(ptr addrspacecast \(ptr addrspace\(1\) @__device_global[A-Za-z0-9_]* to ptr\) to i64\), \[8 x i8\] c"\\03\\00\\00\\00\\00\\00\\00\\00" \}'

# `BOTH` views one allocation as `Option<u32>` and as the `u32` in its payload;
# both views must share the allocation's single global
require_shape_count "global holding BOTH's allocation" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \[8 x i8\] c"\\01\\00\\00\\00\\35\\00\\00\\00"'

# `RAW` is a `*mut` pointer, but its storage keeps rustc's read-only
# allocation: the global stays an LLVM `constant`
require_shape_count "read-only global behind RAW" 1 \
    '^@__device_global[A-Za-z0-9_]* = addrspace\(1\) constant \[4 x i8\] c"\\3B\\00\\00\\00"'

# A promoted global's key is the identity lowering names it by, so it must not
# depend on unrelated code. rustc numbers allocations in the order they are
# interned, which follows the hash-based mono-item order, so keying by that
# number renamed a helper's global whenever a kernel with its own constants
# was added. Build two copies of this example, one with such kernels spliced
# in at the `ADDED KERNEL SLOT` comment, and require every helper's promoted
# global keys, read from the dialect-mir dump, to match. The copies keep the
# package name and live next to this example so its path dependencies
# resolve; each run creates its own and removes only those on exit
examples="$(dirname "${root}")"
oxide_root="$(cd "${examples}/../../.." && pwd)"
work=""
copies=()
cleanup() {
    rm -rf "${work}" "${copies[@]}"
}
trap cleanup EXIT
work="$(mktemp -d)"
copies+=("$(mktemp -d "${examples}/.promoted_ref_return_base.XXXXXX")")
copies+=("$(mktemp -d "${examples}/.promoted_ref_return_added.XXXXXX")")

added_kernels=(added_a added_b added_c added_d added_e added_f)
added_source=""
value=100
for kernel in "${added_kernels[@]}"; do
    # the constants cross an `#[inline(never)]` call, as the helpers' do, so
    # rustc cannot fold them away before they reach the importer
    added_source+="    #[device] #[inline(never)] pub fn ${kernel}_ref() -> (&'static (u32, u32), &'static u32) { const C: &(u32, u32) = &(${value}, $((value + 1))); (C, &$((value + 2))) }
    #[kernel] pub unsafe fn ${kernel}(output: *mut u32) { let (pair, single) = ${kernel}_ref(); unsafe { output.write(pair.0 + pair.1 + *single) }; }
"
    value=$((value + 3))
done

for copy in "${copies[@]}"; do
    cp -R "${root}/Cargo.toml" "${root}/Cargo.lock" "${root}/src" "${copy}/"
done
ADDED_SOURCE="${added_source}" python3 - "${copies[1]}/src/main.rs" <<'PY'
import os
import sys

path = sys.argv[1]
marker = "    // ADDED KERNEL SLOT"
source = open(path).read()
if source.count(marker) != 1:
    sys.exit(f"error: {path} must contain exactly one `{marker.strip()}` comment")
open(path, "w").write(source.replace(marker, os.environ["ADDED_SOURCE"] + marker, 1))
PY

dumps=()
for copy in "${copies[@]}"; do
    name="$(basename "${copy}")"
    dumps+=("${work}/${name}.mir")
    # keys do not depend on the target, so pin one that needs no GPU to detect
    if ! (cd "${oxide_root}" && CUDA_OXIDE_DUMP_MIR=1 cargo oxide build "${name}" --arch sm_80) \
        >"${dumps[-1]}" 2>&1; then
        cat "${dumps[-1]}" >&2
        echo "error: cargo oxide build ${name} failed" >&2
        exit 1
    fi
done

python3 - "${dumps[0]}" "${dumps[1]}" \
    "${added_kernels[*]}" "${helpers[@]}" <<'PY'
import re
import sys

base_path, added_path, added, *helpers = sys.argv[1:]
FUNC = re.compile(r"^mir\.func @(\S+):")
KEY = re.compile(r'global_key: builtin\.string "(cuda-oxide:promoted:[^"]*)"')


def promoted_keys(path):
    keys = {}
    current = None
    for line in open(path):
        match = FUNC.match(line)
        if match:
            current = match.group(1)
            keys.setdefault(current, [])
        elif current is not None:
            keys[current].extend(KEY.findall(line))
    return keys


def function_keys(keys, name):
    found = [k for symbol, k in keys.items() if re.search(rf"(^|_device_[0-9a-f]+_){name}$", symbol)]
    if len(found) != 1:
        sys.exit(f"error: expected one dialect-mir function for {name}, found {len(found)}")
    return found[0]


base = promoted_keys(base_path)
added_build = promoted_keys(added_path)
for kernel in added.split():
    if not function_keys(added_build, f"{kernel}_ref"):
        sys.exit(f"error: added kernel {kernel} declares no promoted global; the check is vacuous")

changed = []
for helper in helpers:
    before = function_keys(base, helper)
    after = function_keys(added_build, helper)
    if not before:
        sys.exit(f"error: helper {helper} declares no promoted global in the dump")
    if before != after:
        changed.append(f"  {helper}: {before} -> {after}")
if changed:
    print("error: adding kernels changed these helpers' promoted global keys:", file=sys.stderr)
    print("\n".join(changed), file=sys.stderr)
    sys.exit(1)
PY

echo "promoted_ref_return: code shape OK"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The device-side drop no-op proof may erase a destructor call only when the
# glue is effect-free and returns. Assert the importer's decision in the
# pre-opt drop_glue.ll: the two diverging probes keep a call to their drop
# glue in the kernel body, and the reconverging control has none. Then check
# what LLVM made of it in the optimized PTX, so a regression in either
# direction cannot pass compile-only CI.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${1:-${root}/drop_glue.ll}"
ptx="${2:-${root}/drop_glue.ptx}"

for artifact in "${ll}" "${ptx}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "error: missing ${artifact}; run cargo oxide build drop_glue without CUDA_OXIDE_NO_OPT" >&2
        exit 1
    fi
done

python3 - "${ll}" "${ptx}" <<'VERIFY_PY'
import re
import sys

ll_path, ptx_path = sys.argv[1], sys.argv[2]
ll = open(ll_path, encoding="utf-8").read()
ptx = open(ptx_path, encoding="utf-8").read()
# Strip comments so a mention in a comment cannot satisfy or hide a check.
ll = re.sub(r";[^\n]*", "", ll)
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)

# A call to the monomorphized `core::ptr::drop_glue::<T>` shim, in the v0
# mangling the device build uses.
GLUE_CALL = re.compile(r"call void @\S*4core3ptr9drop_glue\S*\(")

def kernel_body(name):
    header = re.search(
        r"^define ptx_kernel void @" + re.escape(name) + r"\(.*$",
        ll,
        flags=re.M,
    )
    if header is None:
        raise SystemExit(f"error: missing LLVM kernel {name} in {ll_path}")

    end = re.compile(r"^\}$", flags=re.M).search(ll, header.end())
    if end is None:
        raise SystemExit(f"error: unterminated LLVM kernel {name} in {ll_path}")

    return ll[header.end():end.start()]

def require_glue_call(body, type_name, name):
    pattern = rf"call void @\S*drop_glue\d+{type_name}\S*\("
    if re.search(pattern, body) is None:
        raise SystemExit(
            f"error: {name} lost the drop_glue::<{type_name}> call in {ll_path}"
        )

def reject_glue_call(body, name):
    call = GLUE_CALL.search(body)
    if call is not None:
        raise SystemExit(
            f"error: {name} kept a drop glue call the no-op proof should have "
            f"erased: {call.group(0)}"
        )

def entry_body(name):
    header = re.search(
        r"\.entry\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{",
        ptx,
    )
    if header is None:
        raise SystemExit(f"error: missing PTX entry {name}")

    depth = 1
    end = header.end()

    while depth and end < len(ptx):
        if ptx[end] == "{":
            depth += 1
        elif ptx[end] == "}":
            depth -= 1
        end += 1

    if depth:
        raise SystemExit(f"error: unterminated PTX entry {name}")

    return ptx[header.end():end - 1]

def require_store(body, value, name, label="pre-drop"):
    pattern = rf"^\s*st\.global\.b32\s+\[[^\]]+\],\s*{value}\s*;"
    store = re.search(pattern, body, flags=re.M)
    if store is None:
        raise SystemExit(
            f"error: {name} lost required {label} store {value}"
        )
    return store.end()

# The PTX adjacency below holds because `opt -O2` inlines the retained glue
# and the destructor into the kernel. The destructors carry no inline
# attribute: `#[inline(never)]` reaches LLVM as `noinline` and would keep them
# real PTX functions, so the kernel would call them instead. MIR inlining is
# off for this crate (Cargo.toml) so the importer still sees each destructor's
# own MIR. The drop_glue.ll assertion is the one that tracks the importer's
# own decision; this check confirms what LLVM made of the call it kept.
def require_nonreturning_loop(body, after_store, name):
    # Both optimized probes lower to a store followed by an unconditional
    # self-loop. Tie the loop to that store's continuation: an unrelated
    # loop elsewhere in the entry would not prove the drop was preserved.
    pattern = r"\s*([\w$.]+):\s*bra(?:\.uni)?\s+\1\s*;"
    if not re.match(pattern, body[after_store:]):
        raise SystemExit(
            f"error: {name} pre-drop store is not followed by an unconditional self-loop"
        )

def reject_store(body, value, name):
    pattern = rf"\bst\.global\.b32\s+\[[^\]]+\],\s*{value}\s*;"
    if re.search(pattern, body):
        raise SystemExit(
            f"error: {name} retained unreachable post-drop store {value}"
        )

diverging = (
    ("diverging_cfg_drop_probe", "LoopingDrop", 0x11111111, 0x22222222),
    ("diverging_recursive_drop_probe", "RecursiveDrop", 0x33333333, 0x44444444),
)

for name, type_name, before, after in diverging:
    require_glue_call(kernel_body(name), type_name, name)
    body = entry_body(name)
    after_store = require_store(body, before, name)
    reject_store(body, after, name)
    require_nonreturning_loop(body, after_store, name)
    print(
        f"{name}: drop_glue::<{type_name}> call kept in drop_glue.ll, "
        "pre-drop store followed by non-returning self-loop in drop_glue.ptx PASS"
    )

# Precision control: an effect-free destructor whose branches join back
# together must still be proven a no-op, so no glue call may appear in the
# kernel and both of its stores survive.
name, before, after = ("reconverging_drop_probe", 0x55555555, 0x66666666)
reject_glue_call(kernel_body(name), name)
body = entry_body(name)
require_store(body, before, name)
require_store(body, after, name, label="post-drop")
print(
    f"{name}: no drop glue call in drop_glue.ll, "
    "pre-drop and post-drop stores kept in drop_glue.ptx PASS"
)

print("drop_glue code shape: PASS")
VERIFY_PY

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The frontend .ll precedes LLVM optimization; .opt.ll may infer extra facts.
python3 - "${1:-${root}/kernel_reference_validity.ll}" \
    "${2:-${root}/kernel_reference_validity.ptx}" <<'PY'
from pathlib import Path
import re
import sys

ir = Path(sys.argv[1]).read_text()
attribute = re.compile(r"\b(?:nonnull|noalias|readonly)\b|\balign \d+|\bdereferenceable(?:_or_null)?\(\d+\)")

def check(name, expected):
    header = next((line for line in ir.splitlines()
                   if line.startswith("define ") and f"@{name}(" in line), None)
    assert header is not None, f"missing kernel {name}"
    # Match balanced parentheses: dereferenceable(N) is inside the arguments.
    start = header.index(f"@{name}(") + len(name) + 2
    depth = 1
    end = start
    while depth:
        depth += (header[end] == "(") - (header[end] == ")")
        end += 1
    params = header[start:end - 1].split(",")
    actual = [set(attribute.findall(param)) for param in params]
    assert actual == expected, f"{name}: expected {expected}, got {actual}\n{header}"

reference = {"nonnull", "align 4", "noalias"}
bare = set()
check("shared_ref", [reference | {"dereferenceable(4)"}])
check("unique_ref", [reference | {"dereferenceable(4)"}])
check("shared_slice", [reference, bare])
check("unique_slice", [reference, bare])
check("nvvm_slice", [reference, bare, bare, bare])
check("align_one", [{"nonnull", "noalias", "dereferenceable(1)"}])
check("aligned_zst", [{"nonnull", "align 16", "noalias"}])
check("atomic_ref", [{"nonnull", "align 4"}])
check("pinned_ref", [{"nonnull", "align 4"}])
check("grid_constant", [{"nonnull", "align 16", "noalias", "dereferenceable(16)"}, bare])
check("read_across_write", [reference | {"dereferenceable(4)"}, bare, bare])
for name in ("volatile_direct", "volatile_helper"):
    check(name, [reference | {"dereferenceable(4)"}, bare])
check("raw_pointer", [bare])
check("disjoint_slice", [bare, bare])
check("by_value", [bare])
check("c_reference", [bare])
print("PASS: exact frontend reference attributes and bare ABI controls")

ptx = Path(sys.argv[2]).read_text()

def global_loads(name):
    entry = re.search(r"\.entry\s+" + re.escape(name) + r"\([^;]*?\n\{(.*?)^\}",
                      ptx, re.MULTILINE | re.DOTALL)
    assert entry is not None, f"missing PTX entry {name}"
    loads = re.findall(r"^\s*(?:@!?%[\w.$]+\s+)?(ld(?:\.[\w:]+)+)\s",
                       entry[1], re.MULTILINE)
    return [load for load in loads if "global" in load.split(".")]

# Runtime values alone cannot detect two volatile reads being merged into one.
# Check both the direct route and the route through an inlined helper.
for name in ("volatile_direct", "volatile_helper"):
    loads = global_loads(name)
    # LLVM and libNVVM may spell the same 32-bit integer load b32, u32, or s32.
    assert len(loads) == 2 and all(re.fullmatch(r"ld\.volatile\.global\.[bus]32", load)
                                  for load in loads), f"{name}: {loads}"

# Keeping noalias still lets LLVM reuse an ordinary read across a disjoint write.
loads = global_loads("read_across_write")
assert len(loads) == 1 and re.fullmatch(r"ld\.global(?:\.nc)?\.[bus]32", loads[0]), \
    f"read_across_write: {loads}"
print("PASS: volatile reads stay separate; ordinary reads still combine")
PY

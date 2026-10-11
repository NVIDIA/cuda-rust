#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The frontend .ll precedes LLVM optimization; .opt.ll may infer extra facts.
python3 - "${root}/kernel_reference_validity.ll" <<'PY'
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

shared = {"nonnull", "align 4", "noalias", "readonly"}
unique = {"nonnull", "align 4", "noalias"}
bare = set()
check("shared_ref", [shared | {"dereferenceable(4)"}])
check("unique_ref", [unique | {"dereferenceable(4)"}])
check("shared_slice", [shared, bare])
check("unique_slice", [unique, bare])
check("nvvm_slice", [shared, bare, bare, bare])
check("align_one", [{"nonnull", "noalias", "readonly", "dereferenceable(1)"}])
check("aligned_zst", [{"nonnull", "align 16", "noalias", "readonly"}])
check("atomic_ref", [{"nonnull", "align 4"}])
check("pinned_ref", [{"nonnull", "align 4"}])
check("grid_constant", [{"nonnull", "align 16", "noalias", "readonly", "dereferenceable(16)"}, bare])
check("read_across_write", [shared | {"dereferenceable(4)"}, bare, bare])
check("raw_pointer", [bare])
check("disjoint_slice", [bare, bare])
check("by_value", [bare])
check("c_reference", [bare])
print("PASS: exact frontend reference attributes and bare ABI controls")
PY

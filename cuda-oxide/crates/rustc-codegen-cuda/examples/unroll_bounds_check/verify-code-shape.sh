#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/unroll_bounds_check.ptx"
test -s "${ptx}"

python3 - "${ptx}" <<'PY'
import re
import sys

ptx_path = sys.argv[1]
ptx = open(ptx_path, encoding="utf-8").read()
# Strip comments so instructions mentioned in comments cannot satisfy a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
names = (
    "control", "full_before", "full_inside", "full_after",
    "partial_before", "partial_inside", "partial_after", "full_division",
)
for name in names:
    header = re.search(r"\.entry\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", ptx)
    if header is None:
        sys.exit(f"error: missing PTX entry definition {name} in {ptx_path}")
    depth = 1
    end = header.end()
    while depth and end < len(ptx):
        depth += (ptx[end] == "{") - (ptx[end] == "}")
        end += 1
    if depth:
        sys.exit(f"error: unterminated PTX entry {name}")
    body = ptx[header.end():end - 1]
    checks = {
        "comparison": r"\bsetp\.",
        "conditional branch": r"@!?%p\w*\s+bra(?:\.uni)?\s+",
        "trap": r"\btrap\s*;",
        "store": r"\bst(?:\.[A-Za-z0-9_]+)+\s+",
    }
    for description, pattern in checks.items():
        if not re.search(pattern, body):
            sys.exit(f"error: {name} lost its {description} in {ptx_path}")
    print(f"{name}: checked PTX PASS")
PY

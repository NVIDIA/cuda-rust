#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/unroll_smoke.ptx}"

if [[ ! -f "$ptx" ]]; then
  echo "PTX file not found: $ptx" >&2
  exit 1
fi

python3 - "$ptx" <<'PY'
import re
import sys

ptx_path = sys.argv[1]
ptx = open(ptx_path, encoding="utf-8").read()
# Strip comments so a mention in a comment cannot satisfy or hide a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
# Kernels whose accumulator must stay in registers. A `.local` depot here
# means the annotated loops were left for LLVM, which does not unroll
# `nested_lane_range` or `const_generic_bound`. A generic kernel's entry name
# carries a `_TID_<hash>` suffix.
names = ("nested_lane_range", "isize_range_for", "const_generic_bound")
local_memory = re.compile(r"(?m)(?:^|\s)\.local\b|ld\.local|st\.local")
for name in names:
    header = re.search(r"\.entry\s+" + re.escape(name) + r"(?:_TID_\w+)?\s*\(", ptx)
    if header is None:
        sys.exit(f"error: missing PTX entry {name} in {ptx_path}")
    brace = ptx.find("{", header.end())
    if brace < 0:
        sys.exit(f"error: entry {name} has no body")
    depth = 1
    end = brace + 1
    while depth and end < len(ptx):
        depth += (ptx[end] == "{") - (ptx[end] == "}")
        end += 1
    if depth:
        sys.exit(f"error: unterminated PTX entry {name}")
    body = ptx[brace + 1 : end - 1]
    if local_memory.search(body):
        sys.exit(f"error: {name} uses local memory in {ptx_path}")
    print(f"{name}: no local memory PASS")
print("unroll_smoke PTX shape: PASS")
PY

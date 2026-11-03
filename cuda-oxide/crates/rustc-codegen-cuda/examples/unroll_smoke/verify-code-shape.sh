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


def entry_body(name):
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
    return ptx[brace + 1 : end - 1]


for name in names:
    if local_memory.search(entry_body(name)):
        sys.exit(f"error: {name} uses local memory in {ptx_path}")
    print(f"{name}: no local memory PASS")

# A partial `#[unroll(N)]` marks both its main and its remainder loop
# `nounroll`. `max_ops = 8` is too small for four copies, so the unroller
# must not split `partial_low_budget`; LLVM may still split it once on its
# own and mark only its remainder loop
pragmas = entry_body("partial_low_budget").count('.pragma "nounroll";')
if pragmas >= 2:
    sys.exit(f"error: partial_low_budget has {pragmas} nounroll loops; max_ops = 8 was ignored")
print("partial_low_budget: budget respected PASS")
print("unroll_smoke PTX shape: PASS")
PY

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/unroll_smoke.ptx}"
# Output of the `cargo oxide build` that produced the PTX. The warning check
# needs it; `scripts/smoketest.sh` passes its log as `CUDA_OXIDE_BUILD_LOG`.
build_log="${2:-${CUDA_OXIDE_BUILD_LOG:-}}"

if [[ ! -f "$ptx" ]]; then
  echo "PTX file not found: $ptx" >&2
  exit 1
fi
if [[ -n "$build_log" && ! -f "$build_log" ]]; then
  echo "build log not found: $build_log" >&2
  exit 1
fi

python3 - "$ptx" "$root/src/main.rs" "$build_log" <<'PY'
import re
import sys

ptx_path, src_path, log_path = sys.argv[1:4]
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
# `nounroll`. LLVM runtime-unrolls this loop by itself when the request is
# declined, and then marks only its own remainder loop.
name = "partial_row_view"
pragmas = entry_body(name).count('.pragma "nounroll";')
if pragmas != 2:
    sys.exit(f"error: {name} has {pragmas} nounroll loops, expected the unroller's 2")
print(f"{name}: main and remainder loop PASS")

# The unroller prints each declined request as a `warning:` line, then a
# `-->` line at the annotation. None may point into `partial_row_view`.
if log_path:
    log = open(log_path, encoding="utf-8", errors="replace").read()
    warned = re.findall(
        r"warning: #\[unroll(?:\(\d+\))?\] requested but the loop was not unrolled:"
        r"[^\n]*\n\s*--> ([^\n]*?):(\d+):\d+",
        log,
    )
    # the example declines some loops on purpose, so a log without any of
    # their warnings is not the output of a device build of this example
    if not warned:
        sys.exit(f"error: no unroll warnings in {log_path}; is it this build's output?")
    src = open(src_path, encoding="utf-8").read().splitlines()
    start = next(
        (i for i, line in enumerate(src, 1) if f"fn {name}(" in line), None
    )
    if start is None:
        sys.exit(f"error: missing kernel {name} in {src_path}")
    end = next(i for i, line in enumerate(src[start:], start + 1) if line == "    }")
    for path, line in warned:
        if path.endswith("main.rs") and start <= int(line) <= end:
            sys.exit(f"error: the build warned that {name}'s loop was not unrolled ({path}:{line})")
    print(f"{name}: no not-unrolled warning PASS")
else:
    print(f"{name}: no build log given, warning check skipped")
print("unroll_smoke PTX shape: PASS")
PY

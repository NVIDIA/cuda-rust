#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/nounroll.ptx}"
ll="${2:-$root/nounroll.ll}"

for file in "$ptx" "$ll"; do
  if [[ ! -f "$file" ]]; then
    echo "artifact not found: $file" >&2
    exit 1
  fi
done

python3 - "$ptx" "$ll" <<'PY'
import re
import sys

ptx_path, ll_path = sys.argv[1], sys.argv[2]
ptx = open(ptx_path, encoding="utf-8").read()
ll = open(ll_path, encoding="utf-8").read()
# Strip comments so a mention in a comment cannot satisfy or hide a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
ll = re.sub(r";[^\n]*", "", ll)

# Marked loops per function. The full unroll leaves four copies of its inner
# loop; the partial unroll leaves four in its main loop and one in its
# remainder loop, plus the main and remainder latches of `#[unroll(4)]` itself.
# `taps_default` is the control.
expected = {
    "taps_rolled": 1,
    "taps_default": 0,
    "rolled_k_unrolled_inner": 1,
    "rolled_inside_full_unroll": 4,
    "rolled_inside_partial_unroll": 7,
    "rolled_sum": 1,
}


def body(text, header_pattern, what):
    # PTX forward-declares a callee defined after its caller; a declaration
    # reaches `;` before any `{`, so take the first header that opens a body
    brace = -1
    for header in re.finditer(header_pattern, text):
        candidate = text.find("{", header.end())
        semicolon = text.find(";", header.end())
        if candidate >= 0 and (semicolon < 0 or candidate < semicolon):
            brace = candidate
            break
    if brace < 0:
        sys.exit(f"error: missing {what}")
    depth = 1
    end = brace + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        sys.exit(f"error: unterminated {what}")
    return text[brace + 1 : end - 1]


for name, want in expected.items():
    # a device function's symbol is mangled or path-qualified around its name
    symbol = r"[\w$.]*" + re.escape(name) + r"[\w$.]*"
    ll_body = body(ll, r"define\b[^\n]*@\"?" + symbol + r"\"?\(", f"LLVM function {name}")
    ptx_body = body(
        ptx, r"\.(?:entry|func)\s+(?:\([^)]*\)\s*)?" + symbol + r"\s*\(", f"PTX function {name}"
    )
    # The exporter marks exactly one latch per annotated loop
    marked = len(re.findall(r"^\s*br\b[^\n]*!llvm\.loop\s+!\d+", ll_body, flags=re.M))
    if marked != want:
        sys.exit(f"error: {name} has {marked} latches with !llvm.loop in {ll_path}, expected {want}")
    # LLVM may copy a rolled loop (for example when it unrolls the remainder
    # loop around it), and each copy keeps its pragma, so PTX can hold more
    pragmas = len(re.findall(r'\.pragma\s+"nounroll"\s*;', ptx_body))
    if pragmas < want or (want == 0 and pragmas):
        sys.exit(f'error: {name} has {pragmas} `.pragma "nounroll";` in {ptx_path}, expected {want}')
    print(f"{name}: {marked} marked latch(es), {pragmas} nounroll pragma(s) PASS")
print("nounroll code shape: PASS")
PY

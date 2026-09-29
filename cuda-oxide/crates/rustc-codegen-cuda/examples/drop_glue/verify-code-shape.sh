#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ptx="${root}/drop_glue.ptx"

test -s "${ptx}"

python3 - "${ptx}" <<'VERIFY_PY'
import re
import sys

ptx_path = sys.argv[1]
ptx = open(ptx_path, encoding="utf-8").read()
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)

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

def require_store(body, value, name):
    pattern = rf"^\s*st\.global\.b32\s+\[[^\]]+\],\s*{value}\s*;"
    store = re.search(pattern, body, flags=re.M)
    if store is None:
        raise SystemExit(
            f"error: {name} lost required pre-drop store {value}"
        )
    return store.end()

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

cases = (
    ("diverging_cfg_drop_probe", 0x11111111, 0x22222222),
    ("diverging_recursive_drop_probe", 0x33333333, 0x44444444),
)

for name, before, after in cases:
    body = entry_body(name)
    after_store = require_store(body, before, name)
    reject_store(body, after, name)
    require_nonreturning_loop(body, after_store, name)
    print(f"{name}: pre-drop store followed by non-returning self-loop PASS")

print("drop_glue PTX shape: PASS")
VERIFY_PY

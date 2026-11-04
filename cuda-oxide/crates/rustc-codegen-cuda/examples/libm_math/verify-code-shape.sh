#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/libm_math.ptx}"
ll="${2:-$root/libm_math.ll}"

if [[ ! -s "$ll" ]]; then
  echo "LLVM IR not found: $ll" >&2
  exit 1
fi

# The direct llc route also writes PTX; the NVVM route (no matching llvm-link)
# writes only `.ll` and `.target`. Check the PTX only when it exists and is at
# least as new as the IR, so a leftover from an older build is never checked.
if [[ ! -s "$ptx" || "$ptx" -ot "$ll" ]]; then
  echo "no current PTX next to $ll; skipping the PTX checks"
  ptx=""
fi

python3 - "$ll" "$ptx" <<'PY'
import re
import sys

ll_path, ptx_path = sys.argv[1:3]
ll = open(ll_path, encoding="utf-8").read()
ptx = open(ptx_path, encoding="utf-8").read() if ptx_path else ""
# Strip comments so a mention in a comment cannot satisfy or hide a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)


def entry_body(name):
    header = re.search(r"\.entry\s+" + re.escape(name) + r"\s*\(", ptx)
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


def ll_function(name):
    body = re.search(r"(?ms)^define [^\n]*@" + re.escape(name) + r"\(.*?^\}", ll)
    if body is None:
        sys.exit(f"error: missing LLVM function {name} in {ll_path}")
    return body.group(0)


def calls(body, callee):
    return re.search(r"\bcall [^\n]*@" + re.escape(callee) + r"\(", body) is not None


# The arch-dispatched libm functions, and the `Libm::<f32>` helpers that
# forward to them, reach the importer as calls and become libdevice calls. The
# pre-link IR names the call each one became; the PTX is full of libdevice's
# own `fma`s and cannot. `libm_arch_f32` calls nothing else, so a `__nv_sinf`
# there can only come from `Libm::<f32>::sin`, and an f64 callee would mean a
# helper was routed by its suffix-less name.
for entry, callees, forbidden in (
    ("libm_arch_f32", ("__nv_sqrtf", "__nv_fmaf", "__nv_sinf"), ("__nv_sqrt", "__nv_sin")),
    ("libm_f64", ("__nv_sqrt", "__nv_fma"), ()),
):
    body = ll_function(entry)
    for callee in callees:
        if not calls(body, callee):
            sys.exit(f"error: {entry} does not call {callee} in {ll_path}")
    for callee in forbidden:
        if calls(body, callee):
            sys.exit(f"error: {entry} calls {callee} in {ll_path}")
    print(f"{entry}: calls {', '.join(callees)} PASS")


# libm's arch-dispatched sqrt (x86 `sqrtss` asm on the host) reaches
# libdevice's correctly rounded sqrt instead of being MIR-inlined as host asm.
if ptx_path:
    for entry, instruction in (("libm_arch_f32", "sqrt.rn.f32"), ("libm_f64", "sqrt.rn.f64")):
        body = entry_body(entry)
        if instruction not in body:
            sys.exit(f"error: no {instruction} in {entry} of {ptx_path}")
        if "sqrt.approx" in body:
            sys.exit(f"error: {entry} uses sqrt.approx in {ptx_path}")
        print(f"{entry}: {instruction} PASS")
print("libm_math code shape: PASS")
PY

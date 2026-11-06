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


# Each kernel reaches libdevice through a call in the pre-link IR: `erff` in
# the counted loops, whose inlined body is what `opt` must price correctly to
# unroll them, and the inherent `f32::sqrt`/`f64::sqrt` in the sqrt kernels.
for entry, callee in (
    ("erf_fill_loop", "__nv_erff"),
    ("erf_pair_loop", "__nv_erff"),
    ("prec_sqrt_f32", "__nv_sqrtf"),
    ("prec_sqrt_f64", "__nv_sqrt"),
):
    if not calls(ll_function(entry), callee):
        sys.exit(f"error: {entry} does not call {callee} in {ll_path}")
    print(f"{entry}: calls {callee} PASS")


# Whether `opt` unrolled the loops and folded the precision reflect values only
# shows after `opt` and `llc`, so these checks need the PTX of the direct llc
# route, the one compile-only CI builds.
if ptx_path:
    local_memory = re.compile(r"(?m)(?:^|\s)\.local\b|ld\.local|st\.local")
    # Counted loops that index a per-thread array and call libdevice's erff.
    # A `.local` depot means LLVM left the loop rolled: for `erf_fill_loop`,
    # `opt` priced the libdevice body before `__nvvm_reflect` folded (no
    # `-mcpu`); for `erf_pair_loop`, it ran O2's full-unroll threshold instead
    # of O3's.
    for entry in ("erf_fill_loop", "erf_pair_loop"):
        if local_memory.search(entry_body(entry)):
            sys.exit(f"error: {entry} uses local memory in {ptx_path}")
        print(f"{entry}: no local memory PASS")

    # libdevice's sqrt picks sqrt.rn over sqrt.approx only when `opt`, which
    # folds `__nvvm_reflect` once it has `-mcpu`, gets `__CUDA_PREC_SQRT=1`.
    for entry, instruction in (("prec_sqrt_f32", "sqrt.rn.f32"), ("prec_sqrt_f64", "sqrt.rn.f64")):
        body = entry_body(entry)
        if instruction not in body:
            sys.exit(f"error: no {instruction} in {entry} of {ptx_path}")
        if "sqrt.approx" in body:
            sys.exit(f"error: {entry} uses sqrt.approx in {ptx_path}")
        print(f"{entry}: {instruction} PASS")
print("libm_math code shape: PASS")
PY

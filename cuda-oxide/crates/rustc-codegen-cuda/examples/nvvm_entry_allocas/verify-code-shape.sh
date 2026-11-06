#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check the entry-normalization contract separately from the numerical oracle.
# Correct GPU results alone do not establish static placement of scratch storage.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ir="${root}/nvvm_entry_allocas.ll"
test -s "${ir}"

# This example has only fixed slots (across both `branch_scratch` and `core_zip`).
# Entry placement is also checkable in a headless compile-only lane, before its
# host runner can produce linked PTX.
python3 - "${ir}" <<'CHECK_IR'
import re
import sys
from pathlib import Path

function = None
block = None
slots = 0
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("define "):
        function = line
        block = None
    elif line == "}":
        function = None
    elif function and re.fullmatch(r"[A-Za-z0-9_.$]+:", line):
        block = line[:-1]
    elif function and " = alloca " in line:
        slots += 1
        if block != "entry":
            sys.exit(f"error: fixed scratch outside true entry block ({block}): {line.strip()}")
if slots == 0:
    sys.exit("error: example no longer exercises fixed scratch allocations")
CHECK_IR

# The runtime runner writes the split compiler's PTX. It includes a trailing
# NUL, hence grep's -a flag. Compile-only lanes have just the LLVM IR above.
ptx="${root}/nvvm_entry_allocas.ptx"
if [[ -s "${ptx}" ]]; then
    if grep -aEq '(^|[[:space:]])(alloca|stacksave|stackrestore)\.' "${ptx}"; then
        echo "error: fixed scratch introduced dynamic stack instructions in ${ptx}" >&2
        exit 1
    fi
    grep -aEq 'st\.param\.v4\.b32' "${ptx}"
    grep -aEq 'call[^;]*nvvm_entry_allocas__consume' "${ptx}"

    python3 - "${ptx}" <<'CHECK_CORE_ZIP'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_bytes().decode("utf-8", errors="ignore")
match = re.search(r"\.visible \.entry core_zip\b.*?\n\}", text, re.DOTALL)
if not match:
    sys.exit("error: missing core_zip entry in nvvm_entry_allocas.ptx")
body = match.group(0)
if re.search(r"\bcall(\.uni)?\b", body):
    sys.exit("error: core_zip still contains an uninlined call (#1687)")
if ".local" in body:
    sys.exit("error: core_zip still allocates a .local stack frame (#1687)")
CHECK_CORE_ZIP
fi

echo "nvvm_entry_allocas code shape: PASS"

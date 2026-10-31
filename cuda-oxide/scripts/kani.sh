#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the cuda-device Kani proofs and fail on an unreachable cover.
#
# Usage: scripts/kani.sh [cargo kani args...]
#   scripts/kani.sh
#   scripts/kani.sh --harness kani_proofs::p1_index_1d_d3_bx32 --exact
#   scripts/kani.sh --solver kissat
#
# Every harness in crates/cuda-device/src/kani_proofs.rs carries a
# `kani::cover!` on the branch it exists to exercise. Kani 0.68 reports a
# cover that no input reaches as UNSATISFIABLE (or UNREACHABLE when the
# statement itself is dead) but still prints VERIFICATION:- SUCCESSFUL and
# exits 0, and it has no flag to turn that into a failure. A harness whose
# assumptions contradict each other is vacuously true, so an unsatisfied cover
# must fail CI here. The gate reads Kani's regular output, so the flags that
# change it (`--output-format`, `--quiet`, `-j`/`--jobs`) are refused.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

for arg in "$@"; do
    case "${arg}" in
        -j|-j*|--jobs|--jobs=*|--output-format|--output-format=*|--quiet)
            echo "kani.sh: ${arg} changes the output this script reads; run cargo kani directly" >&2
            exit 2
            ;;
    esac
done

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

cargo kani -p cuda-device -Z stubbing --output-format regular "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ]; then
    echo "kani.sh: cargo kani exited $status" >&2
    exit "$status"
fi
if grep -q '^VERIFICATION:- FAILED' "$log"; then
    echo "kani.sh: a harness failed" >&2
    exit 1
fi

# Each harness prints `Checking harness <name>...` and, when it has covers,
# ` ** N of M cover properties satisfied`. Every harness must print the
# second line and every N must equal M.
harnesses="$(grep -c '^Checking harness ' "$log")"
cover_lines="$(grep -c ' cover properties satisfied' "$log")"
if [ "$harnesses" -eq 0 ]; then
    echo "kani.sh: no harness ran" >&2
    exit 1
fi
if [ "$harnesses" -ne "$cover_lines" ]; then
    echo "kani.sh: $harnesses harnesses ran but $cover_lines reported covers; every harness must carry one" >&2
    exit 1
fi
unsatisfied="$(awk '
    /^Checking harness / { harness = $3 }
    / cover properties satisfied/ && $2 != $4 { print harness ": " $0 }
    /^Check [0-9]+: .*\.cover\.[0-9]+$/ { check = $0; next }
    check != "" && /Status:/ && !/Status: SATISFIED/ { print harness ": " check; print; next }
    { check = "" }
' "$log")"
if [ -n "$unsatisfied" ]; then
    echo "kani.sh: cover properties not reached:" >&2
    echo "$unsatisfied" >&2
    exit 1
fi
echo "kani.sh: verification and covers OK ($harnesses harnesses)"

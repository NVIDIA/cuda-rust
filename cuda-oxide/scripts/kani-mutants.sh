#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prove the cuda-device Kani harnesses can fail: apply each seeded fault in
# kani-mutants/, run the harnesses that must catch it, and require FAILED.
#
# Usage: scripts/kani-mutants.sh [mutant...]
#   scripts/kani-mutants.sh           # every kani-mutants/*.patch
#   scripts/kani-mutants.sh m3 m4
#
# Each patch names its killers on a `Must be killed by:` line in its header.
set -euo pipefail
export LC_ALL=C

cd "$(dirname "${BASH_SOURCE[0]}")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
MUTANT_DIR="${PWD}/kani-mutants"

if [ "$#" -gt 0 ]; then
    mutants=("$@")
else
    mutants=()
    for patch in "${MUTANT_DIR}"/*.patch; do
        mutants+=("$(basename "${patch}" .patch)")
    done
fi

logs="$(mktemp -d)"
applied=""
revert() {
    if [ -n "${applied}" ]; then
        git -C "${GIT_ROOT}" apply -R "${applied}"
        applied=""
    fi
}
trap 'revert; rm -rf "${logs}"' EXIT
trap 'exit 130' INT TERM

# git apply ignores everything before the first `diff --git`, which leaves
# the header free for the fault description and its killers.
killers() {
    sed -n 's/^Must be killed by: //p' "$1" | tr ',' '\n' | sed 's/^ *//; s/ *$//; /^$/d'
}

touched_files() {
    git -C "${GIT_ROOT}" apply --numstat "$1" | cut -f3
}

# Kani lists each failed check as `Failed Checks: <message>` followed by a
# ` File: "<path>", line N, in <fn>` line.
failed_checks() {
    grep -A1 '^Failed Checks:' "$1"
}

# A kill counts only when an assertion in kani_proofs.rs fails. A fault that
# trips a Kani-inserted check (overflow, unwrap, out of bounds) in the crate
# itself still shows the mutant is reachable, but says nothing about the
# property the harness states.
failed_in_proofs() {
    failed_checks "$1" | grep -q 'File: ".*kani_proofs\.rs"'
}

declare -a rows=()
status=0
for m in "${mutants[@]}"; do
    patch="${MUTANT_DIR}/${m}.patch"
    if [ ! -f "${patch}" ]; then
        rows+=("${m}	ERROR	no such patch ${patch}")
        status=1
        continue
    fi
    mapfile -t harnesses < <(killers "${patch}")
    if [ "${#harnesses[@]}" -eq 0 ]; then
        rows+=("${m}	ERROR	no 'Must be killed by:' line")
        status=1
        continue
    fi
    mapfile -t files < <(touched_files "${patch}")
    if ! git -C "${GIT_ROOT}" diff --quiet HEAD -- "${files[@]}"; then
        echo "kani-mutants.sh: ${files[*]} differ from HEAD; refusing to mutate a dirty tree" >&2
        exit 1
    fi
    if ! git -C "${GIT_ROOT}" apply --check "${patch}" 2>"${logs}/${m}.apply"; then
        rows+=("${m}	ERROR	patch no longer applies: $(head -1 "${logs}/${m}.apply")")
        status=1
        continue
    fi
    git -C "${GIT_ROOT}" apply "${patch}"
    applied="${patch}"

    verdict=KILLED
    detail=""
    for h in "${harnesses[@]}"; do
        log="${logs}/${m}.${h##*::}.log"
        echo "kani-mutants.sh: ${m}: cargo kani --harness ${h} --exact"
        cargo kani -p cuda-device -Z stubbing --harness "${h}" --exact >"${log}" 2>&1 || true
        if grep -q '^VERIFICATION:- FAILED' "${log}"; then
            if failed_in_proofs "${log}"; then
                detail+="${h##*::} failed; "
            else
                [ "${verdict}" = SURVIVED ] || verdict=ERROR
                detail+="${h##*::} failed outside the proofs; "
                failed_checks "${log}" >&2
            fi
        elif grep -q '^VERIFICATION:- SUCCESSFUL' "${log}"; then
            verdict=SURVIVED
            detail+="${h##*::} verified; "
        else
            [ "${verdict}" = SURVIVED ] || verdict=ERROR
            detail+="${h##*::} no verdict; "
            tail -20 "${log}" >&2
        fi
    done

    revert
    if ! git -C "${GIT_ROOT}" diff --quiet HEAD -- "${files[@]}"; then
        echo "kani-mutants.sh: ${m} did not revert cleanly; ${files[*]} still differ from HEAD" >&2
        exit 1
    fi
    [ "${verdict}" = KILLED ] || status=1
    rows+=("${m}	${verdict}	${detail%; }")
done

printf '\n%-6s %-9s %s\n' mutant verdict detail
for row in "${rows[@]}"; do
    IFS=$'\t' read -r m verdict detail <<<"${row}"
    printf '%-6s %-9s %s\n' "${m}" "${verdict}" "${detail}"
done
exit "${status}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the dialect-nvvm book page's catalog stamp still matches
# intrinsics/catalog.json.
#
# The page states its own freshness rule: it prints the catalog SHA-256 and
# says that if the stamp no longer matches, every count on the page predates
# the catalog being read.  Nothing checked it.  The page failed that test for
# weeks -- 560 operations printed against 570 in the tree -- and only a hand
# audit caught it, twice (#1052 and the batch before it).
#
# So the counts rot silently while the rule that would expose them is itself
# unenforced.  This makes the rule executable: the stamp is machine-checked,
# which turns "the counts are stale" into a red check on the op-adding pull
# request rather than a discovery months later.  The counts stay hand-written;
# only the stamp needs a gate, because the page derives staleness from it.
#
# The hash is the one `cuda-intrinsics-gen` stamps into every generated file:
# `sha256_bytes(catalog_json.as_bytes())` over the file as committed, so a
# plain sha256sum of catalog.json reproduces it.  The page prints the first
# eight hex digits, which is what this compares.
#
# Run this after any change to intrinsics/catalog.json.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

CATALOG=intrinsics/catalog.json
PAGE=cuda-oxide-book/compiler/mlir-dialects.md
GENERATED_DIR=crates/dialect-nvvm/src/ops/generated

test -f "${CATALOG}"
test -f "${PAGE}"

actual="$(sha256sum "${CATALOG}" | cut -c1-8)"

# Self-test: a truncated or reworded hash pipeline must not read as agreement.
case "${actual}" in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
    *)
        echo "error: could not hash ${CATALOG} (got '${actual}')" >&2
        exit 1
        ;;
esac

# Cross-check against the generated sources, which carry the full hash in
# their headers.  If the generator's notion of the catalog hash ever stops
# being a plain sha256sum of the file, this fails here rather than silently
# comparing the page against the wrong number.
stamped="$(grep -ohE 'catalog SHA-256: [0-9a-f]{64}' "${GENERATED_DIR}"/*.rs |
    sed 's/.*: //' | sort -u)"
if [ "$(printf '%s\n' "${stamped}" | grep -c .)" -ne 1 ]; then
    echo "error: ${GENERATED_DIR} carries more than one catalog SHA-256:" >&2
    printf '  %s\n' ${stamped} >&2
    echo "       regenerate with 'cargo run -p cuda-intrinsics-gen -- generate'" >&2
    exit 1
fi
if [ "${stamped#"${actual}"}" = "${stamped}" ]; then
    echo "error: ${CATALOG} hashes to ${actual}, but the generated sources stamp" >&2
    echo "       ${stamped}. The generated outputs are stale; run" >&2
    echo "       'cargo run -p cuda-intrinsics-gen -- generate'." >&2
    exit 1
fi

# The page names the stamp twice: once stating it, once in the rule that tells
# a reader what a mismatch means. Both have to move together, or the rule
# contradicts the statement above it.
# `|| true` on the capture, not on a later pipeline: with pipefail a grep that
# matches nothing would abort the script before the diagnostics below, turning
# "the page was reworded" into a silent non-zero exit.
stamps="$(grep -oE '`[0-9a-f]{8}`' "${PAGE}" | tr -d '`' || true)"
count="$(printf '%s' "${stamps}" | grep -c . || true)"
found="$(printf '%s\n' "${stamps}" | sort -u | grep -c . >/dev/null 2>&1 &&
    printf '%s\n' "${stamps}" | sort -u || true)"

if [ "${count}" -lt 2 ]; then
    echo "error: found ${count} catalog stamps in ${PAGE}, expected at least 2" >&2
    echo "       (the statement and the staleness rule). The page was reworded;" >&2
    echo "       refusing to report success from a check that found nothing." >&2
    exit 1
fi

if [ "$(printf '%s\n' "${found}" | grep -c .)" -ne 1 ]; then
    echo "error: ${PAGE} prints more than one catalog stamp:" >&2
    printf '  %s\n' ${found} >&2
    echo "       the statement and the staleness rule must name the same hash." >&2
    exit 1
fi

if [ "${found}" != "${actual}" ]; then
    cat >&2 <<EOF
error: ${PAGE} is stale.

  page stamp:      ${found}
  ${CATALOG}: ${actual}

By the page's own rule, every count on it now predates the catalog. Update the
stamp in both places and re-derive the counts it guards:

  total operations   grep -c '^pub struct .*Op;' ${GENERATED_DIR}/*.rs \\
                       crates/dialect-nvvm/src/ops/*.rs
  catalog entries    jq '.intrinsics | length' ${CATALOG}

then check that both tables still sum to their stated totals.
EOF
    exit 1
fi

echo "OK: ${PAGE} stamps ${actual}, matching ${CATALOG} and ${GENERATED_DIR}."

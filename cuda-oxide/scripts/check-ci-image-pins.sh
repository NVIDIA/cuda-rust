#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the `rust:<version>` container images in the workflows still match the
# stable pin in the git-root rust-toolchain.toml.
#
# The NVIDIA runner VM image has no rustup, so the Rust jobs run in
# `rust:<version>-bookworm`. That tag is a copy of the root pin, repeated in
# many jobs, and nothing else keeps the copies together. A stale tag does not
# fail a job (rustup installs the pinned toolchain on first use) but it spends
# a toolchain download in every job and leaves the CodeQL Rust extractor on a
# different compiler than the one it analyzes.
#
# Run this after bumping the root pin.
set -euo pipefail

cd "$(dirname "$0")/../.."

PIN_FILE=rust-toolchain.toml
test -s "${PIN_FILE}"

pin="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' "${PIN_FILE}" | head -n 1)"
if [ -z "${pin}" ]; then
    echo "parse self-test failed: no channel found in ${PIN_FILE}" >&2
    exit 1
fi

# Every `rust:<major.minor.patch>` image tag in the workflows and local actions.
refs="$(grep -rnE 'rust:[0-9]+\.[0-9]+\.[0-9]+' .github || true)"
if [ -z "${refs}" ]; then
    echo "parse self-test failed: no rust:<version> image found under .github;" >&2
    echo "either the jobs stopped using it (delete this check) or the pattern is stale" >&2
    exit 1
fi

failed=0
count=0
while IFS= read -r line; do
    [ -n "${line}" ] || continue
    count=$((count + 1))
    tag="$(printf '%s\n' "${line}" | sed -n 's/.*rust:\([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p')"
    if [ "${tag}" != "${pin}" ]; then
        echo "  ${line%%:*}:$(printf '%s\n' "${line}" | cut -d: -f2) uses rust:${tag}, ${PIN_FILE} pins ${pin}" >&2
        failed=1
    fi
done <<< "${refs}"

if [ "${failed}" -ne 0 ]; then
    echo "error: a workflow container image does not match the root toolchain pin" >&2
    exit 1
fi

echo "OK: all ${count} rust:<version> image reference(s) match ${PIN_FILE} (${pin})."

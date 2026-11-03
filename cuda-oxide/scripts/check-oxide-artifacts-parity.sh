#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the in-tree oxide-artifacts crate and the crates.io release everyone
# consumes are the same version.
#
# crates/oxide-artifacts is the source the release is published from, but no
# workspace consumes it by path any more: the backend writes artifact bundles
# and the shared cuda-core (cutile-rs) reads them, both through crates.io
# oxide-artifacts, so writer and reader are one crate. The in-tree copy can
# still drift ahead of the release; if it does, an edit to the format lands in
# git without reaching any consumer, and the next publish silently changes
# what every reader accepts. This guard fails as soon as the in-tree version,
# the workspace dependency requirements, and the versions resolved in every
# lock file disagree.
#
# Format changes therefore go: bump crates/oxide-artifacts, publish, then move
# the requirement in the SIMT and backend manifests and relock.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

in_tree="$(sed -n -E 's/^version = "([^"]+)"/\1/p' crates/oxide-artifacts/Cargo.toml | head -1)"
[ -n "${in_tree}" ] || { echo "error: crates/oxide-artifacts/Cargo.toml has no version" >&2; exit 1; }
echo "in-tree oxide-artifacts: ${in_tree}"

status=0
req_of() {  # req_of FILE -> the version requirement on the oxide-artifacts line
    local line
    line="$(grep -E '^oxide-artifacts[[:space:]]*=' "$1" | head -1 || true)"
    if [[ "${line}" =~ version[[:space:]]*=[[:space:]]*\"([^\"]+)\" ]] || [[ "${line}" =~ ^oxide-artifacts[[:space:]]*=[[:space:]]*\"([^\"]+)\" ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "${line}" =~ path[[:space:]]*= ]]; then
        echo "path"
    else
        echo "missing"
    fi
}
for manifest in Cargo.toml crates/rustc-codegen-cuda/Cargo.toml; do
    req="$(req_of "${manifest}")"
    if [ "${req}" != "${in_tree}" ]; then
        echo "error: ${manifest}: oxide-artifacts requirement is '${req}', in-tree version is ${in_tree}" >&2; status=1
    fi
done

# Every lock that resolves oxide-artifacts must resolve exactly the in-tree
# version, from the registry (the SIMT lock also lists the in-tree member).
while IFS= read -r lock; do
    versions="$(awk '/^name = "oxide-artifacts"$/{getline; print $3}' "${lock}" | tr -d '"' | sort -u)"
    [ -n "${versions}" ] || continue
    if [ "${versions}" != "${in_tree}" ]; then
        echo "error: ${lock}: oxide-artifacts resolves to ${versions//$'\n'/, }, in-tree version is ${in_tree}" >&2; status=1
    fi
    if ! grep -A3 '^name = "oxide-artifacts"$' "${lock}" | grep -q 'source = "registry+'; then
        echo "error: ${lock}: oxide-artifacts is not consumed from crates.io" >&2; status=1
    fi
done < <(git -C "${GIT_ROOT}" ls-files 'Cargo.lock' '*/Cargo.lock' '**/Cargo.lock' | while read -r lock; do echo "${GIT_ROOT}/${lock}"; done)

[ "${status}" -eq 0 ] && echo "OK: oxide-artifacts is ${in_tree} in the tree, in both manifests, and in every lock (from crates.io)."
exit "${status}"

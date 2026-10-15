#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify that CUDA Oxide consumes the shared host crates at tip of tree.
#
# cuda-bindings, cuda-core, cuda-core-derive, and cuda-async live at the git
# root and are consumed by path. In-tree manifests must not carry a version
# requirement for them: a path dependency with no version always resolves to
# the sibling directory, so a release of the host crates needs no edit in this
# tree (#1459). Three things are checked:
#
#   1. The four root crates agree on one version.
#   2. CUDA Oxide's [workspace.dependencies] and every example workspace name
#      cuda-bindings, cuda-core, and cuda-async as path dependencies with no
#      `version` field. Examples are their own [workspace], so they cannot
#      inherit the SIMT entry, and a crates.io copy there would resolve a
#      second cuda_core and break type unification with path cuda-host.
#   3. `cargo oxide new` scaffolds out-of-tree projects, which cannot use the
#      git-root paths, so its SHARED_HOST_CRATES_VERSION is the one place a
#      crates.io version of the host crates lives. It must equal the root
#      crates' version until it is derived from the root manifests directly.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"
OXIDE="${GIT_ROOT}/cuda-oxide/Cargo.toml"
SCAFFOLD=crates/cargo-oxide/src/commands/scaffold.rs
status=0

# 1. Root crate versions agree.
root_version=""
for crate in cuda-bindings cuda-core cuda-core-derive cuda-async; do
    v="$(sed -n -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "${GIT_ROOT}/${crate}/Cargo.toml" | head -1)"
    [ -n "${v}" ] || { echo "error: ${crate}/Cargo.toml has no version" >&2; exit 1; }
    if [ -z "${root_version}" ]; then
        root_version="${v}"
    elif [ "${v}" != "${root_version}" ]; then
        echo "error: ${crate} is ${v}, cuda-bindings is ${root_version}" >&2; status=1
    fi
done
echo "root host crates: ${root_version}"

# `check_manifest FILE` flags a host-crate dependency line that is not a path
# dependency or that carries a version requirement. Renamed dependencies
# (`alias = { package = "cuda-core", ... }`) are covered too.
check_manifest() {
    local file="$1" line
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        if [[ ! "${line}" =~ path[[:space:]]*= ]]; then
            echo "error: ${file}: '${line}' (want a path dependency)" >&2; status=1
        elif [[ "${line}" =~ version[[:space:]]*= ]]; then
            echo "error: ${file}: '${line}' (drop the version; path dependencies track tip of tree)" >&2; status=1
        fi
    done < <(grep -E '^(cuda-bindings|cuda-core|cuda-async)[[:space:]]*=|^[A-Za-z0-9_-]+[[:space:]]*=[[:space:]]*\{[^}]*package[[:space:]]*=[[:space:]]*"cuda-(bindings|core|async)"' "${file}" || true)
}

# 2. CUDA Oxide and the example workspaces (nested member crates included).
check_manifest "${OXIDE}"
while IFS= read -r manifest; do
    check_manifest "${GIT_ROOT}/${manifest}"
done < <(git -C "${GIT_ROOT}" ls-files \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/*/Cargo.toml')

# 3. The scaffold's crates.io version.
scaffold_version="$(sed -n -E 's/^pub\(super\) const SHARED_HOST_CRATES_VERSION: &str = "([^"]+)";/\1/p' "${SCAFFOLD}")"
if [ -z "${scaffold_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION not found" >&2; status=1
elif [ "${scaffold_version}" != "${root_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION is ${scaffold_version}, root host crates are ${root_version}" >&2; status=1
fi

[ "${status}" -eq 0 ] && echo "OK: host crates are path dependencies at tip of tree; scaffold pins crates.io ${root_version}."
exit "${status}"

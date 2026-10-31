#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify that cuda-oxide consumes the shared host crates at tip of tree.
#
# cuda-bindings, cuda-core, cuda-core-derive, and cuda-async live at the git
# root and are consumed by path. In-tree manifests must not carry a version
# requirement for them: a path dependency with no version always resolves to
# the sibling directory, so a release of the host crates needs no edit in this
# tree (#1459). Three things are checked:
#
#   1. The four root crates agree on one version.
#   2. cuda-oxide's [workspace.dependencies] and every example or model workspace
#      name cuda-bindings, cuda-core, and cuda-async as path dependencies with no
#      `version` field. Examples are their own [workspace], so they cannot
#      inherit the SIMT entry, and a crates.io copy there would resolve a
#      second cuda_core and break type unification with path cuda-host.
#   3. `cargo oxide new` takes cuda-core and cuda-async from the same git
#      source as cuda-host. cuda-host depends on them by path, so in a project
#      that pulls cuda-host from git they resolve from that checkout; a
#      crates.io requirement beside it is a second cuda_core, and the
#      scaffolded project fails to type-check.
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

# 2. cuda-oxide and the example and model workspaces (nested member crates included).
check_manifest "${OXIDE}"
while IFS= read -r manifest; do
    check_manifest "${GIT_ROOT}/${manifest}"
done < <(git -C "${GIT_ROOT}" ls-files \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/*/Cargo.toml' \
    'cuda-oxide/models/*/Cargo.toml' \
    'cuda-oxide/models/*/*/Cargo.toml' \
    'cuda-oxide/models/*/*/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/examples/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/examples/*/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/examples/*/*/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/models/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/models/*/*/Cargo.toml' \
    'cuda-oxide/cute-rs-preview/models/*/*/*/Cargo.toml')

# 3. The scaffold names every host crate from GIT_REPO.
scaffold_lines="$(grep -E '^(cuda-bindings|cuda-core|cuda-async)[[:space:]]*=' "${SCAFFOLD}" || true)"
if [ -z "${scaffold_lines}" ]; then
    echo "error: ${SCAFFOLD}: no host-crate dependency lines in the templates" >&2; status=1
fi
while IFS= read -r line; do
    [ -n "${line}" ] || continue
    if [[ ! "${line}" =~ git[[:space:]]*=[[:space:]]*\"\{GIT_REPO\}\" ]]; then
        echo "error: ${SCAFFOLD}: '${line}' (want git = \"{GIT_REPO}\", the source cuda-host comes from)" >&2; status=1
    fi
done <<< "${scaffold_lines}"

[ "${status}" -eq 0 ] && echo "OK: host crates are path dependencies at tip of tree; the scaffold takes them from git beside cuda-host."
exit "${status}"

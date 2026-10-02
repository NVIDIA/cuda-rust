#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every copy of the shared host-crate pin agrees with the SIMT workspace.
#
# cuda-bindings, cuda-core, and cuda-async live at the git root. CUDA Oxide's
# `[workspace.dependencies]` path-depends on them once, and that pin is copied:
#
#   1. Into every example workspace's Cargo.toml as a path dependency (each
#      example is its own [workspace], so it cannot inherit the SIMT entry).
#      A crates.io copy here resolves a second cuda_core and breaks type
#      unification with path cuda-host.
#
#   2. Into the `cargo oxide new` templates (SHARED_HOST_CRATES_VERSION in
#      crates/cargo-oxide/src/commands/scaffold.rs) as the crates.io version
#      of the same pin. Out-of-tree projects cannot use the git-root paths.
#
# In-tree manifests must use the same form as CUDA Oxide (path). The scaffold
# stays on the registry form; only its version string must match.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

ROOT="${GIT_ROOT}/cuda-oxide/Cargo.toml"
SCAFFOLD=crates/cargo-oxide/src/commands/scaffold.rs

# `spec_of FILE CRATE` prints "<form> <version>" for the crate's dependency
# line in FILE, or nothing if the file does not name the crate.
spec_of() {
    local file="$1" crate="$2" line version
    line="$(grep -E "^${crate}[[:space:]]*=" "${file}" | head -1 || true)"
    [ -n "${line}" ] || return 0
    version="?"
    if [[ "${line}" =~ version[[:space:]]*=[[:space:]]*\"=?([0-9][^\"]*)\" ]]; then
        version="${BASH_REMATCH[1]}"
    elif [[ "${line}" =~ ^${crate}[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
        version="${BASH_REMATCH[1]}"
    fi
    if [[ "${line}" =~ path[[:space:]]*= ]]; then
        echo "path ${version}"
    elif [[ "${line}" =~ tag[[:space:]]*=[[:space:]]*\"v([0-9][^\"]*)\" ]]; then
        echo "git ${BASH_REMATCH[1]}"
    else
        echo "registry ${version}"
    fi
}

root_spec="$(spec_of "${ROOT}" cuda-core)"
[ -n "${root_spec}" ] || { echo "error: ${ROOT} has no cuda-core workspace dependency" >&2; exit 1; }
root_form="${root_spec% *}"; root_version="${root_spec#* }"
echo "CUDA Oxide pin: cuda-core ${root_form} ${root_version}"

status=0
for crate in cuda-bindings cuda-async; do
    spec="$(spec_of "${ROOT}" "${crate}")"
    if [ "${spec}" != "${root_spec}" ]; then
        echo "error: ${ROOT}: ${crate} is '${spec}', cuda-core is '${root_spec}'" >&2; status=1
    fi
done

# 1. Example manifests (nested member crates included). Must match the
# in-tree path pin, not the scaffold's crates.io form.
while IFS= read -r manifest; do
    for crate in cuda-bindings cuda-core cuda-async; do
        while IFS= read -r line; do
            [ -n "${line}" ] || continue
            version="?"
            if [[ "${line}" =~ version[[:space:]]*=[[:space:]]*\"=?([0-9][^\"]*)\" ]]; then
                version="${BASH_REMATCH[1]}"
            elif [[ "${line}" =~ =[[:space:]]*\"([^\"]*)\"[[:space:]]*$ ]]; then
                version="${BASH_REMATCH[1]}"
            fi
            if [[ "${line}" =~ path[[:space:]]*= ]]; then
                form=path
            elif [[ "${line}" =~ tag[[:space:]]*=[[:space:]]*\"v([0-9][^\"]*)\" ]]; then
                form=git; version="${BASH_REMATCH[1]}"
            else
                form=registry
            fi
            if [ "${form} ${version}" != "${root_spec}" ]; then
                echo "error: ${manifest}: '${line}' (want ${root_spec})" >&2; status=1
            fi
        done < <(grep -E "^([A-Za-z0-9_-]+[[:space:]]*=[[:space:]]*\{[^}]*package[[:space:]]*=[[:space:]]*\"${crate}\"|${crate}[[:space:]]*=)" "${manifest}" || true)
    done
done < <(git -C "${GIT_ROOT}" ls-files \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/Cargo.toml' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/*/Cargo.toml' \
    'cuda-oxide/cuteir/examples/*/Cargo.toml' \
    'cuda-oxide/cuteir/examples/*/*/Cargo.toml' \
    'cuda-oxide/cuteir/examples/*/*/*/Cargo.toml' \
    | while read -r manifest; do echo "${GIT_ROOT}/${manifest}"; done)

# 2. The scaffold constant (crates.io form of the same version).
scaffold_version="$(sed -n -E 's/^pub\(super\) const SHARED_HOST_CRATES_VERSION: &str = "([^"]+)";/\1/p' "${SCAFFOLD}")"
if [ -z "${scaffold_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION not found" >&2; status=1
elif [ "${scaffold_version}" != "${root_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION is ${scaffold_version}, CUDA Oxide pin is ${root_version}" >&2; status=1
fi

[ "${status}" -eq 0 ] && echo "OK: every shared host-crate pin agrees with CUDA Oxide (${root_form} ${root_version})."
exit "${status}"

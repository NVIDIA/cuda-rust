#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every copy of the shared host-crate pin agrees with the root workspace.
#
# cuda-bindings, cuda-core, and cuda-async come from cutile-rs. The root
# `[workspace.dependencies]` names the release once, but that line is copied:
#
#   1. Into every example workspace's Cargo.toml (each example is its own
#      [workspace], so it cannot inherit the root entry). A drifted copy
#      resolves a second version of the runtime and breaks the shared
#      example build cache; sync-example-locks.sh catches the lock, this
#      catches the manifest that produced it.
#
#   2. Into the `cargo oxide new` templates (SHARED_HOST_CRATES_VERSION in
#      crates/cargo-oxide/src/commands/scaffold.rs). That copy never breaks
#      this repository's CI: a stale one hands each new project a runtime
#      the generated `#[cuda_module]` code was not written against, and the
#      failure surfaces on the user's machine.
#
# The pin may be a crates.io version (`"0.3.1"`, `{ version = "0.3.1", ... }`)
# or, between a cutile-rs tag and its crates.io release, a git tag
# (`{ git = ".../cutile-rs", tag = "v0.3.1" }`). Both spell the same version,
# so the comparison is on the version string, and every manifest must also
# use the same *form* as the root (all git, or all registry).
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

ROOT="${GIT_ROOT}/Cargo.toml"
SCAFFOLD=crates/cargo-oxide/src/commands/scaffold.rs
CRATES='cuda-(bindings|core|async)'

# `spec_of FILE CRATE` prints "<form> <version>" for the crate's dependency
# line in FILE, or nothing if the file does not name the crate.
spec_of() {
    local file="$1" crate="$2" line
    line="$(grep -E "^${crate}[[:space:]]*=" "${file}" | head -1 || true)"
    [ -n "${line}" ] || return 0
    if [[ "${line}" =~ tag[[:space:]]*=[[:space:]]*\"v([0-9][^\"]*)\" ]]; then
        echo "git ${BASH_REMATCH[1]}"
    elif [[ "${line}" =~ version[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
        echo "registry ${BASH_REMATCH[1]}"
    elif [[ "${line}" =~ ^${crate}[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
        echo "registry ${BASH_REMATCH[1]}"
    else
        echo "unknown ?"
    fi
}

root_spec="$(spec_of "${ROOT}" cuda-core)"
[ -n "${root_spec}" ] || { echo "error: ${ROOT} has no cuda-core workspace dependency" >&2; exit 1; }
root_form="${root_spec% *}"; root_version="${root_spec#* }"
echo "root pin: cuda-core ${root_form} ${root_version}"

status=0
for crate in cuda-bindings cuda-async; do
    spec="$(spec_of "${ROOT}" "${crate}")"
    if [ "${spec}" != "${root_spec}" ]; then
        echo "error: ${ROOT}: ${crate} is '${spec}', cuda-core is '${root_spec}'" >&2; status=1
    fi
done

# 1. Example manifests (nested member crates included).
while IFS= read -r manifest; do
    for crate in cuda-bindings cuda-core cuda-async; do
        # `cutile-cuda-core = { ..., package = "cuda-core" }` is a renamed
        # dependency on the same crate; match it through its package key.
        while IFS= read -r line; do
            [ -n "${line}" ] || continue
            if [[ "${line}" =~ tag[[:space:]]*=[[:space:]]*\"v([0-9][^\"]*)\" ]]; then
                form=git; version="${BASH_REMATCH[1]}"
            elif [[ "${line}" =~ version[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
                form=registry; version="${BASH_REMATCH[1]}"
            elif [[ "${line}" =~ =[[:space:]]*\"([^\"]*)\"[[:space:]]*$ ]]; then
                form=registry; version="${BASH_REMATCH[1]}"
            elif [[ "${line}" =~ path[[:space:]]*= ]]; then
                form=path; version="?"
            else
                form=unknown; version="?"
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
    | while read -r manifest; do echo "${GIT_ROOT}/${manifest}"; done)

# 2. The scaffold constant.
scaffold_version="$(sed -n -E 's/^pub\(super\) const SHARED_HOST_CRATES_VERSION: &str = "([^"]+)";/\1/p' "${SCAFFOLD}")"
if [ -z "${scaffold_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION not found" >&2; status=1
elif [ "${scaffold_version}" != "${root_version}" ]; then
    echo "error: ${SCAFFOLD}: SHARED_HOST_CRATES_VERSION is ${scaffold_version}, root pin is ${root_version}" >&2; status=1
fi

[ "${status}" -eq 0 ] && echo "OK: every shared host-crate pin agrees with the root (${root_form} ${root_version})."
exit "${status}"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build both `cargo oxide new` templates against this checkout.
#
# The scaffold tests in cargo-oxide read the generated files as text; nothing
# compiles them, so an API change in cuda-host or cuda-core can break the first
# project a new user creates while every check stays green. This generates the
# sync and the async project, redirects the template's git source to this
# checkout with a [patch] table (the dependency lines stay as users get them),
# and runs `cargo oxide build` on each.
#
# Needs the SIMT toolchain, LLVM tools and a CUDA toolkit, as any example
# build does. Run from anywhere in the repository.

set -euo pipefail

root="$(git rev-parse --show-toplevel)"
oxide="${root}/cuda-oxide"
work="$(mktemp -d "${TMPDIR:-/tmp}/scaffold-builds.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

(cd "${oxide}" && cargo build --quiet -p cargo-oxide)
target_dir="$(cd "${oxide}" && cargo metadata --format-version 1 --no-deps |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
cargo_oxide="${target_dir}/debug/cargo-oxide"

ptxas_bin=""
for candidate in "${CUDA_OXIDE_PTXAS:-}" \
    "$(command -v ptxas 2>/dev/null || true)" \
    "${CUDA_TOOLKIT_PATH:+${CUDA_TOOLKIT_PATH}/bin/ptxas}" \
    "${CUDA_HOME:+${CUDA_HOME}/bin/ptxas}" \
    /usr/local/cuda/bin/ptxas; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        ptxas_bin="${candidate}"
        break
    fi
done

failed=0
for mode in sync async; do
    name="scaffold_${mode}"
    flags=()
    [[ "${mode}" == "async" ]] && flags+=(--async)
    (cd "${work}" && "${cargo_oxide}" oxide new "${name}" "${flags[@]}" >/dev/null)
    project="${work}/${name}"

    # Patch exactly the git source the template names.
    source_url="$(sed -nE 's/^cuda-device = \{ git = "([^"]+)".*/\1/p' "${project}/Cargo.toml")"
    if [[ -z "${source_url}" ]]; then
        echo "error: the ${mode} template no longer takes cuda-device from git" >&2
        failed=1
        continue
    fi
    {
        echo
        echo "[patch.\"${source_url}\"]"
        for crate in cuda-device cuda-host cuda-core cuda-async; do
            grep -q "^${crate} = " "${project}/Cargo.toml" || continue
            case "${crate}" in
                cuda-device | cuda-host) path="${oxide}/crates/${crate}" ;;
                *) path="${root}/${crate}" ;;
            esac
            echo "${crate} = { path = \"${path}\" }"
        done
    } >>"${project}/Cargo.toml"

    if ! (cd "${project}" && "${cargo_oxide}" oxide build) >"${work}/${name}.log" 2>&1; then
        echo "error: the ${mode} \`cargo oxide new\` template does not build:" >&2
        tail -n 80 "${work}/${name}.log" >&2
        failed=1
        continue
    fi
    # A patch that silently did not apply would have built GitHub's main.
    if grep -qE '^source = "git\+' "${project}/Cargo.lock"; then
        echo "error: the ${mode} project resolved a CUDA crate from git, not this checkout" >&2
        failed=1
        continue
    fi
    ptx="${project}/${name}.ptx"
    if [[ ! -s "${ptx}" ]]; then
        echo "error: the ${mode} project build produced no device PTX at ${ptx}" >&2
        failed=1
        continue
    fi
    if [[ -n "${ptxas_bin}" ]]; then
        ptx_arch="$(sed -nE 's/^\.target[[:space:]]+(sm_[0-9]+[af]?).*/\1/p' "${ptx}" | head -n 1)"
        if [[ -z "${ptx_arch}" ]]; then
            echo "error: ${ptx} has no .target directive" >&2
            failed=1
            continue
        fi
        if ! "${ptxas_bin}" -arch="${ptx_arch}" "${ptx}" -o "${work}/${name}.cubin" \
            >"${work}/${name}.ptxas.log" 2>&1; then
            echo "error: ptxas rejected ${mode} template PTX (${ptx_arch}):" >&2
            cat "${work}/${name}.ptxas.log" >&2
            failed=1
            continue
        fi
    fi
    echo "${mode} template: builds against this checkout"
done

exit "${failed}"

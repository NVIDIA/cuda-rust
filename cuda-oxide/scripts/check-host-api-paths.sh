#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reject host-API paths that no longer resolve on the shared host crates.
#
# cuda-bindings, cuda-core, and cuda-async are shared with cutile-rs. Their
# crate roots carry cutile's own Tile API; the cuda-oxide (SIMT) surface lives
# under `cuda_core::simt` and `cuda_async::simt`. cuda-core re-exports most of
# the SIMT items at its root, with four deliberate exceptions that collide with
# Tile-side names or were never re-exported:
#
#   cuda_core::LaunchConfig        -> cuda_core::simt::LaunchConfig
#                                     (root LaunchConfig is cutile's struct;
#                                      LaunchConfig1D/2D/3D stay at the root)
#   cuda_core::{memory,peer,vmm}   -> cuda_core::simt::{memory,peer,vmm}
#                                     (root vmm is cutile's fork)
#   cuda_core::error::IntoResult   -> cuda_core::IntoResult
#   cuda_async::<module>::<item>   -> cuda_async::simt::<module>::<item>
#                                     (root device_operation, device_context,
#                                      device_box, launch, error ... are Tile)
#
# A doc snippet, README, or `cargo oxide new` template that spells the old
# root path either fails to compile or, worse, binds the Tile-side type and
# fails later with a type mismatch. Nothing compiles those snippets, which is
# how some forty stale lines survived the crate switch, so this guard greps
# them. Rust sources under crates/ are covered too, for their doc comments;
# they are compiled, but a `//!` example is not.
#
# Root `cuda_async::zip!` / `unzip!` are fine: they expand to `.zip()` and
# dispatch on whichever `Zippable` is in scope, and they are the one shared
# entry point (the simt copies cannot be re-exported without a root collision).
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Files whose host-API spellings a reader copies. `git ls-files` paths are
# repository-relative; grep them from the git root.
mapfile -t files < <(git -C "${GIT_ROOT}" ls-files \
    README.md cuda-oxide/README.md CONTRIBUTING.md \
    'cuda-oxide/cuda-oxide-book/*.md' 'cuda-oxide/cuda-oxide-book/**/*.md' \
    'cuda-oxide/crates/*/README.md' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/README.md' \
    'cuda-oxide/crates/rustc-codegen-cuda/examples/*/*/README.md' \
    'cuda-oxide/crates/cargo-oxide/src/**/*.rs' 'cuda-oxide/crates/cargo-oxide/src/*.rs' \
    'cuda-oxide/crates/cuda-host/src/**/*.rs' 'cuda-oxide/crates/cuda-host/src/*.rs' \
    'cuda-oxide/crates/cuda-macros/src/**/*.rs' 'cuda-oxide/crates/cuda-macros/src/*.rs' \
    'cuda-oxide/crates/cuda-device/src/**/*.rs' 'cuda-oxide/crates/cuda-device/src/*.rs' \
    | sort -u)

# Each pattern is an ERE; `cuda_core::simt::...` never matches because the
# module name has to follow `cuda_core::` directly.
patterns=(
    'cuda_core::LaunchConfig([^0-9A-Za-z_]|$)'
    'cuda_core::(memory|peer|vmm|error|launch|stream|module|context|event|pinned_host_buffer|device_buffer)::'
    'use cuda_core::\{[^}]*(^|[^A-Za-z0-9_])(LaunchConfig|peer|vmm|memory)([^0-9A-Za-z_]|\})'
    'cuda_async::(device_operation|device_context|device_box|device_future|scheduling_policies|reclaim|launch|error)([^0-9A-Za-z_]|$)'
    'cuda_core::simt::simt|cuda_async::simt::simt'
)

status=0
for pattern in "${patterns[@]}"; do
    if hits="$(grep -n -E -- "${pattern}" "${files[@]/#/${GIT_ROOT}/}" 2>/dev/null)"; then
        status=1
        echo "error: host-API path that does not resolve on the shared crates (/${pattern}/):" >&2
        echo "${hits}" | sed "s|^${GIT_ROOT}/||; s/^/    /" >&2
    fi
done

if [ "${status}" -ne 0 ]; then
    echo "hint: SIMT items live under cuda_core::simt / cuda_async::simt; see the header of $0." >&2
    exit 1
fi
echo "OK: no stale host-API root paths in $((${#files[@]})) doc and source files."

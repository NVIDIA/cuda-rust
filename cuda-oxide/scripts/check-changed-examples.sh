#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prek passes paths relative to its project working directory.
# Format or lint each affected example package once, including nested packages.
set -euo pipefail

case "${1:-}" in
    fmt) args=(fmt) ;;
    clippy) args=(clippy --locked --all-targets -- -D warnings) ;;
    *) printf 'usage: %s {fmt|clippy} [files...]\n' "$0" >&2; exit 2 ;;
esac
shift

root=$(git rev-parse --show-toplevel)
examples="$root/cuda-oxide/crates/rustc-codegen-cuda/examples"
directories=()

for file in "$@"; do
    dir=$(dirname "$PWD/$file")
    while [[ "$dir" == "$examples/"* && ! -f "$dir/Cargo.toml" ]]; do
        dir=$(dirname "$dir")
    done
    if [[ "$dir" != "$examples/"* || ! -f "$dir/Cargo.toml" ]]; then
        printf 'error: no example Cargo.toml found for %s\n' "$file" >&2
        exit 1
    fi

    for seen in "${directories[@]}"; do
        if [[ "$dir" == "$seen" ]]; then
            continue 2
        fi
    done
    directories+=("$dir")
done

# Let rustup select each package's directory-scoped toolchain.
unset RUSTUP_TOOLCHAIN
for dir in "${directories[@]}"; do
    # Formatting avoids --all: it repeatedly formats shared path dependencies.
    (cd "$dir" && cargo "${args[@]}")
done

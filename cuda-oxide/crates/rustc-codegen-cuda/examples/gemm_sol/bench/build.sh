#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compiles bench/cublaslt_bench from cublaslt_bench.c.
#
# `gemm_sol/src/main.rs` invokes this binary at startup to measure the live
# cublasLtMatmul SoL on the host GPU (replacing the previously hardcoded B200
# constants). Run this once after entering your dev shell, or whenever
# CUDA_HOME / cublasLt changes.
#
# Picks the CUDA toolkit in this priority order:
#   1. $CUDA_HOME            (set by `nix develop` for cuda-oxide)
#   2. $CUDA_PATH            (alternative env var some toolchains use)
#   3. /usr/local/cuda       (system install)
#
# Usage:
#   cd bench
#   bash build.sh

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

cuda_root="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
if [[ ! -d "$cuda_root" ]]; then
    echo "error: CUDA toolkit not found." >&2
    echo "  tried CUDA_HOME=${CUDA_HOME:-<unset>}, CUDA_PATH=${CUDA_PATH:-<unset>}, /usr/local/cuda" >&2
    echo "  set CUDA_HOME to your CTK install (in cuda-oxide's nix devshell this is automatic)." >&2
    exit 1
fi

# nixpkgs ships .so files under $CUDA_HOME/lib; classic CTK installs use lib64.
# Pass both -L paths so this works in either layout, then pin them via -rpath
# so the binary can be invoked outside the dev shell.
lib_args=()
rpath_args=()
for d in "$cuda_root/lib" "$cuda_root/lib64"; do
    if [[ -d "$d" ]]; then
        lib_args+=("-L$d")
        rpath_args+=("-Wl,-rpath,$d")
    fi
done
if [[ ${#lib_args[@]} -eq 0 ]]; then
    echo "error: no lib/ or lib64/ found under $cuda_root" >&2
    exit 1
fi

# Sanity-check: the compile-time include must resolve cublasLt.h. If it
# doesn't, the user is missing libcublas.include / cuda_cccl from the dev
# shell and gcc would otherwise produce a confusing error mid-flight.
if [[ ! -f "$cuda_root/include/cublasLt.h" ]]; then
    echo "error: $cuda_root/include/cublasLt.h not found." >&2
    echo "  on nixpkgs, libcublas.include must be in cudaSymlinked." >&2
    echo "  on a system CTK install, install the cuBLAS dev package." >&2
    exit 1
fi

cc="${CC:-gcc}"
echo "Building cublaslt_bench:"
echo "  CC       = $cc"
echo "  CUDA     = $cuda_root"
echo "  -L paths = ${lib_args[*]}"

set -x
"$cc" -O2 -o cublaslt_bench cublaslt_bench.c \
    -I"$cuda_root/include" \
    "${lib_args[@]}" \
    -lcublasLt -lcudart -lm \
    "${rpath_args[@]}"
{ set +x; } 2>/dev/null

echo "✓ Built: $here/cublaslt_bench"
echo "  gemm_sol's main.rs will pick this up automatically on the next run."

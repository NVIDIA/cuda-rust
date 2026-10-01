#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=$(mktemp -d)
trap 'rm -rf -- "$build"' EXIT
nvcc_bin=${NVCC:-${CUDA_TOOLKIT_PATH:+$CUDA_TOOLKIT_PATH/bin/}nvcc}
"$nvcc_bin" -O3 -std=c++17 -arch=sm_90a \
  "$root/crates/rustc-codegen-cuda/examples/wgmma_mma_fp8/bench/fp8_vs_bf16.cu" \
  -lcublasLt -o "$build/fp8_vs_bf16"
if (($#)); then
  "$build/fp8_vs_bf16" "$@"
else
  for fast in 0 1; do
    for size in 64 256 1024 4096 8192; do
      "$build/fp8_vs_bf16" "$size" "$size" "$size" "$fast"
    done
    "$build/fp8_vs_bf16" 64 8192 8192 "$fast"
  done
fi

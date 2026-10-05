#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$HERE/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$EXP1/diagnostics/results/cublaslt_cpp}"
CUDA_HOME="${CUDA_HOME:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
OUT="$HERE/cublaslt_gemm_sanity"
PIN="${PIN:-}"
GPU_INDEX="${GPU_INDEX:-0}"
CUDA_ARCH="${CUDA_ARCH:-}"

if [[ -z "$CUDA_ARCH" ]]; then
  compute_cap="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -n "$compute_cap" ]]; then
    CUDA_ARCH="sm_${compute_cap//./}"
  else
    CUDA_ARCH="native"
  fi
fi

mkdir -p "$RESULTS_DIR"

echo "Building cuBLASLt C++ sanity check for CUDA_ARCH=$CUDA_ARCH"
"$NVCC" -std=c++17 -O3 -arch="$CUDA_ARCH" \
  "$HERE/cublaslt_gemm_sanity.cu" \
  -lcublasLt -lcublas -o "$OUT"

if [[ -n "$PIN" ]]; then
  RESULTS_DIR="$RESULTS_DIR" $PIN "$OUT"
else
  RESULTS_DIR="$RESULTS_DIR" "$OUT"
fi

#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TUNING="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$TUNING/.." && pwd)"

OUT="${GEMM_OUT:-$EXP1/tuning/results/gemm/gemm_rust_static_mlir_fix_all_sizes_tune_results.csv}"

GEMM_VARIANTS="static" \
GEMM_SIZES="1024 2048 4096 8192 16384 32768" \
GEMM_TILES="128,128,64 128,256,64 128,256,128 256,128,64 256,256,32 256,256,64 256,256,128" \
GEMM_TILE_HINTS="2,default,default 1,1,default" \
GEMM_HINT_SWEEP=0 \
GEMM_OUT="$OUT" \
"$EXP1/scripts/with_gpu_clock_2400.sh" "$TUNING/run_rust_tuning_sweep.sh" gemm

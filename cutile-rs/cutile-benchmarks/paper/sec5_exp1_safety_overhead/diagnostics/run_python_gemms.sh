#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Python-side GEMM benchmarks: cuTile Python, cuBLASLt, and the
# matching-size cuTile/cuBLAS fraction summary. Swizzle is an explicit
# diagnostic target because it has not been the fastest path on RTX 5090.
#
# Usage:
#   ./run_python_gemms.sh              # cuTile Python + cuBLAS + summary
#   ./run_python_gemms.sh swizzle      # only gemm_python_swizzle.py
#   ./run_python_gemms.sh sweep        # native/swizzle/persistent candidate sweep
#   SIZES="2048 4096" ./run_python_gemms.sh swizzle
#   SWIZZLE_IMPLEMENTATION=cutile-sample ./run_python_gemms.sh swizzle
#
# Set PIN_CPU= to disable CPU pinning. This script intentionally does not
# lock GPU clocks; use ../paper/run_gemm.sh for the full paper-final path.
set -euo pipefail

DIAGNOSTICS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$DIAGNOSTICS/.." && pwd)"
PAPER="$EXP1/paper"
PAPER_RESULTS="$PAPER/results/rtx5090"
SWIZZLE_RESULTS="$EXP1/diagnostics/results/swizzle"
PY="${PY:-python3}"
TARGET="${1:-all}"
PIN_CPU="${PIN_CPU:-0}"

if [[ -n "$PIN_CPU" ]]; then
  PIN_CMD=(taskset -c "$PIN_CPU")
else
  PIN_CMD=()
fi

run_py() {
  local results_dir=$1
  shift
  ( cd "$PAPER" && RESULTS_DIR="$results_dir" "${PIN_CMD[@]}" "$PY" "$@" )
}

run_unpinned_py() {
  ( cd "$PAPER" && RESULTS_DIR="$PAPER_RESULTS" "$PY" "$@" )
}

size_args=()
if [[ -n "${SIZES:-}" ]]; then
  read -r -a sizes <<< "$SIZES"
  size_args+=(--sizes "${sizes[@]}")
fi

swizzle_args=()
swizzle_args+=(--implementation "${SWIZZLE_IMPLEMENTATION:-slide}")
if [[ "${SWIZZLE_NUM_CTAS:-2}" != "default" ]]; then
  swizzle_args+=(--num-ctas "${SWIZZLE_NUM_CTAS:-2}")
fi
if [[ -n "${SWIZZLE_OCCUPANCY:-}" && "${SWIZZLE_OCCUPANCY:-}" != "default" ]]; then
  swizzle_args+=(--occupancy "$SWIZZLE_OCCUPANCY")
fi
swizzle_args+=("${size_args[@]}")
if [[ -n "${SWIZZLE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<< "$SWIZZLE_EXTRA_ARGS"
  swizzle_args+=("${extra[@]}")
fi

case "$TARGET" in
  all|cutile|swizzle|sweep|cublas) ;;
  -h|--help|help)
    sed -n '1,13p' "$0"
    exit 0
    ;;
  *)
    echo "error: target must be one of: all, cutile, swizzle, sweep, cublas" >&2
    exit 2
    ;;
esac

if [[ "$TARGET" == "all" || "$TARGET" == "cutile" ]]; then
  echo "=== GEMM Python: cuTile paper baseline ==="
  run_py "$PAPER_RESULTS" gemm_python.py
fi

if [[ "$TARGET" == "swizzle" ]]; then
  echo "=== GEMM Python: cuTile slide-style swizzle ==="
  ( cd "$DIAGNOSTICS" && RESULTS_DIR="$SWIZZLE_RESULTS" "${PIN_CMD[@]}" "$PY" gemm_python_swizzle.py "${swizzle_args[@]}" )
fi

if [[ "$TARGET" == "sweep" ]]; then
  echo "=== GEMM Python: cuTile candidate sweep ==="
  ( cd "$DIAGNOSTICS" && RESULTS_DIR="$SWIZZLE_RESULTS" "${PIN_CMD[@]}" "$PY" gemm_python_swizzle.py --sweep "${size_args[@]}" )
fi

if [[ "$TARGET" == "all" || "$TARGET" == "cublas" ]]; then
  echo "=== GEMM Python: cuBLASLt baseline ==="
  run_py "$PAPER_RESULTS" gemm_cublas.py
fi

if [[ -s "$PAPER_RESULTS/gemm_cublas_results.csv" ]]; then
  echo "=== Fraction of cuBLAS ==="
  run_unpinned_py summarize_gemm_cublas_fraction.py --markdown
else
  echo "Skipping fraction summary: gemm_cublas_results.csv is missing." >&2
fi

echo ""
echo "Python GEMM CSVs:"
ls -1 "$PAPER_RESULTS"/gemm_python_results.csv \
      "$PAPER_RESULTS"/gemm_cublas_results.csv \
      "$PAPER_RESULTS"/gemm_cublas_fraction.csv 2>/dev/null || true
if [[ "$TARGET" == "swizzle" || "$TARGET" == "sweep" ]]; then
  ls -1 "$SWIZZLE_RESULTS"/gemm_python_swizzle_results.csv \
        "$SWIZZLE_RESULTS"/gemm_python_swizzle_sweep_results.csv 2>/dev/null || true
fi

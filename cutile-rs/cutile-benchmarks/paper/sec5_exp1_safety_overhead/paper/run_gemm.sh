#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run §5.1 GEMM benchmarks at locked 2.4 GHz clocks, regenerate the
# GEMM plot. Matches run_elemwise.sh methodology so the two workloads
# are measured under identical conditions.
#
# Each bench is pinned to CPU 0 via taskset to minimise host-side
# scheduling noise.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$PAPER/.." && pwd)"
BENCHMARKS="$(cd "$EXP1/.." && pwd)"
REPO_ROOT="$(cd "$BENCHMARKS/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PAPER/results/rtx5090}"
GEMM_DIR="$EXP1/rust/gemm"
PY="${PY:-python3}"
PIN="taskset -c 0"
SM_CLOCK_GHZ="${SM_CLOCK_GHZ:-2.4}"
GPU_INDEX="${GPU_INDEX:-0}"

echo "=== Lock GPU clocks @ 2.4 GHz ==="
sudo nvidia-smi -i "$GPU_INDEX" -lgc 2400,2400
# Release clocks on any exit path (success, failure, Ctrl-C).
trap 'echo "=== Release GPU clocks ==="; sudo nvidia-smi -i "$GPU_INDEX" -rgc || true' EXIT

echo "=== Query theoretical peak device-memory bandwidth ==="
MAX_MEM_GB_S="$("$PY" "$BENCHMARKS/tools/query_nominal_memory_bandwidth.py" \
  --device "$GPU_INDEX" --value-only)"
MAX_MEM_TB_S="$("$PY" -c 'import sys; print(f"{float(sys.argv[1]) / 1000.0:.3f}")' "$MAX_MEM_GB_S")"
export RESULTS_DIR
echo "Theoretical peak device-memory bandwidth: ${MAX_MEM_TB_S} TB/s (${MAX_MEM_GB_S} GB/s); SM clock locked at ${SM_CLOCK_GHZ} GHz."

echo "=== Rust (build) ==="
( cd "$GEMM_DIR" && cargo build --release )

echo "=== GEMM Rust: static ==="
( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --static )

echo "=== GEMM Rust: optimized ==="
( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust )

echo "=== GEMM Python: cuTile ==="
( cd "$PAPER" && $PIN "$PY" gemm_python.py )

echo "=== Python: cuBLAS (nvmath) ==="
( cd "$PAPER" && $PIN "$PY" gemm_cublas.py )

echo "=== GEMM Rust: dynamic ==="
( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --safe )

echo "=== Regenerate GEMM plot ==="
( cd "$PAPER" && "$PY" plot_exp1.py )

echo "=== Summarize cuTile fraction of cuBLAS ==="
summary_args=(--markdown)
( cd "$PAPER" && "$PY" summarize_gemm_cublas_fraction.py "${summary_args[@]}" )

echo ""
echo "GEMM CSVs:"
ls -1 "$RESULTS_DIR"/gemm_python_results.csv \
      "$RESULTS_DIR"/gemm_cublas_results.csv \
      "$RESULTS_DIR"/gemm_cublas_fraction.csv \
      "$RESULTS_DIR"/gemm_rust_optimized_results.csv \
      "$RESULTS_DIR"/gemm_rust_safe_results.csv \
      "$RESULTS_DIR"/gemm_rust_static_results.csv 2>/dev/null
echo ""
echo "Plot:"
ls -1 "$REPO_ROOT"/figures/generated/exp1_safety_overhead.pdf
echo ""
echo "Memory roof: ${MAX_MEM_TB_S} TB/s (${MAX_MEM_GB_S} GB/s), theoretical peak device-memory bandwidth; SM clock was locked at ${SM_CLOCK_GHZ} GHz."
echo ""
echo "Note: GEMM static uses num_tiles for a dynamic MLIR loop bound while"
echo "preserving static bounds for check elimination."

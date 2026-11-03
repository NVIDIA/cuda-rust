#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run §5.1 element-wise benchmarks at locked 2.4 GHz clocks.
# Rust uses the isolated / CUDA-event / fixed-ITERS methodology. Each Rust
# variant is run as its own single-variant process so optimized/safe match
# the Python single-kernel measurement structure.
# Python is a single-variant cross-frontend baseline with the same
# CUDA-event timing and cache-rotation methodology.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$PAPER/.." && pwd)"
BENCHMARKS="$(cd "$EXP1/.." && pwd)"
REPO_ROOT="$(cd "$BENCHMARKS/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PAPER/results/rtx5090}"
ELEM_DIR="$EXP1/rust/elemwise"
PY="${PY:-python3}"
PIN="taskset -c 0"
SM_CLOCK_GHZ="${SM_CLOCK_GHZ:-2.4}"
GPU_INDEX="${GPU_INDEX:-0}"
CACHE_SWEEP_MIB="${ELEM_CACHE_SWEEP_MIB:-1024}"
ELEM_ITERS="${ELEM_ITERS:-10}"
# Shared Rust/Python elementwise tile schedule. Keep these aligned so
# cross-frontend results differ only by frontend lowering.
ELEM_B="${ELEM_B:-16384}"
ELEM_NUM_CTAS="${ELEM_NUM_CTAS:-default}"
RUST_ELEMWISE_VARIANTS="${RUST_ELEMWISE_VARIANTS:-optimized safe}"
RUST_ELEMWISE_ARGS=(--b "$ELEM_B" --cache-sweep-mib "$CACHE_SWEEP_MIB" --iters "$ELEM_ITERS")
PYTHON_ELEMWISE_ARGS=(--b "$ELEM_B" --iters "$ELEM_ITERS")
if [[ "$ELEM_NUM_CTAS" != "default" && -n "$ELEM_NUM_CTAS" ]]; then
  RUST_ELEMWISE_ARGS+=(--cta "$ELEM_NUM_CTAS")
  PYTHON_ELEMWISE_ARGS+=(--num-ctas "$ELEM_NUM_CTAS")
fi

echo "=== Lock GPU clocks @ 2.4 GHz ==="
sudo nvidia-smi -i "$GPU_INDEX" -lgc 2400,2400
trap 'echo "=== Release GPU clocks ==="; sudo nvidia-smi -i "$GPU_INDEX" -rgc || true' EXIT

echo "=== Query theoretical peak device-memory bandwidth ==="
MAX_MEM_GB_S="$("$PY" "$BENCHMARKS/tools/query_nominal_memory_bandwidth.py" \
  --device "$GPU_INDEX" --value-only)"
MAX_MEM_TB_S="$("$PY" -c 'import sys; print(f"{float(sys.argv[1]) / 1000.0:.3f}")' "$MAX_MEM_GB_S")"
export ELEM_MEM_ROOFLINE_GB_S="$MAX_MEM_GB_S"
export ELEM_CACHE_SWEEP_MIB="$CACHE_SWEEP_MIB"
export RESULTS_DIR
echo "Theoretical peak device-memory bandwidth: ${MAX_MEM_TB_S} TB/s (${MAX_MEM_GB_S} GB/s); SM clock locked at ${SM_CLOCK_GHZ} GHz."

echo "=== Rust (build) ==="
( cd "$ELEM_DIR" && cargo build --release )

echo "=== Elementwise Rust: isolated single-variant runtime benches ==="
rm -f "$RESULTS_DIR/elemwise_rust_runtime_results.csv"
for variant in $RUST_ELEMWISE_VARIANTS; do
  echo "Rust args: ${RUST_ELEMWISE_ARGS[*]} --variant $variant --append-csv"
  ( cd "$ELEM_DIR" && $PIN ./target/release/elemwise_rust "${RUST_ELEMWISE_ARGS[@]}" --variant "$variant" --append-csv )
done

echo "=== Elementwise Python: cuTile ==="
echo "Python args: ${PYTHON_ELEMWISE_ARGS[*]}"
( cd "$PAPER" && $PIN "$PY" elemwise_python.py "${PYTHON_ELEMWISE_ARGS[@]}" )

echo "=== Regenerate element-wise plot ==="
( cd "$PAPER" && "$PY" plot_exp1_elemwise.py )

echo ""
echo "Elementwise CSVs:"
ls -1 "$RESULTS_DIR"/elemwise_python_results.csv \
      "$RESULTS_DIR"/elemwise_rust_runtime_results.csv 2>/dev/null
echo ""
echo "Plot:"
ls -1 "$REPO_ROOT"/figures/generated/exp1_elemwise.pdf
echo ""
echo "Memory roof: ${MAX_MEM_TB_S} TB/s (${MAX_MEM_GB_S} GB/s), theoretical peak device-memory bandwidth; SM clock was locked at ${SM_CLOCK_GHZ} GHz."

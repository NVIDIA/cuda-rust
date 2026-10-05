#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect Rust JIT timing artifacts separately from the paper throughput
# runs, then regenerate the combined JIT breakdown plot.
set -euo pipefail

DIAGNOSTICS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$DIAGNOSTICS/.." && pwd)"
GEMM_DIR="$EXP1/rust/gemm"
ELEM_DIR="$EXP1/rust/elemwise"
RESULTS_DIR="$EXP1/diagnostics/results/jit"
PY="${PY:-python3}"
PIN="${PIN:-taskset -c 0}"
export RESULTS_DIR
mkdir -p "$RESULTS_DIR"

# Best configuration from tuning/results/elemwise/elemwise_rust_tune_results.csv:
# B=2048, num_cta_in_cga=2, max_divisibility=8.
RUST_ELEMWISE_ARGS=(--b 2048 --cta 2 --max-divisibility 8)

echo "=== Rust (build) ==="
( cd "$GEMM_DIR" && cargo build --release )
( cd "$ELEM_DIR" && cargo build --release )

echo "=== GEMM Rust: JIT cost (per-stage via CUTILE_JIT_TIMING) ==="
( cd "$GEMM_DIR" && \
  CUTILE_JIT_TIMING=1 $PIN ./target/release/gemm_rust --jit \
    2> "$RESULTS_DIR/gemm_rust_jit_timing.log" )

echo "=== Elementwise Rust: JIT cost (per-stage via CUTILE_JIT_TIMING) ==="
echo "Rust args: ${RUST_ELEMWISE_ARGS[*]}"
( cd "$ELEM_DIR" && \
  CUTILE_JIT_TIMING=1 $PIN ./target/release/elemwise_rust --jit "${RUST_ELEMWISE_ARGS[@]}" \
    2> "$RESULTS_DIR/elemwise_rust_jit_timing.log" )

echo "=== Regenerate combined JIT breakdown plot ==="
( cd "$DIAGNOSTICS" && "$PY" plot_jit_breakdown.py --quiet )

echo ""
echo "JIT CSVs:"
ls -1 "$RESULTS_DIR"/gemm_rust_jit_results.csv \
      "$RESULTS_DIR"/elemwise_rust_jit_results.csv 2>/dev/null
echo ""
echo "JIT timing logs:"
ls -1 "$RESULTS_DIR"/gemm_rust_jit_timing.log \
      "$RESULTS_DIR"/elemwise_rust_jit_timing.log 2>/dev/null

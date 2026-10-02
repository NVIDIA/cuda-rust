#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sweeps the bimodal-GEMMs GPU work-distribution experiment:
# three modes × a range of stream counts, appending to
# results_bimodal.csv. Renders the figure afterward.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/bimodal_gemms_rust}"

# Stream counts to sweep for the multi-stream modes.
S_VALUES="${S_VALUES:-1 2 4 8 16 32}"

# Tokio-thread values to sweep for async. T=1 is current_thread runtime
# (single OS thread for all tokio tasks); T>1 uses multi_thread with
# that many worker threads. We sweep up to 16 (physical core count) to
# match the thread-per-stream footprint for an apples-to-apples
# comparison.
T_VALUES="${T_VALUES:-1 2 4 8 16}"

# Workload knobs. n-work large enough to amortize JIT warmup cleanly,
# small enough that one sample finishes in <~10 s.
N_WORK="${N_WORK:-2000}"
SMALL_RATIO="${SMALL_RATIO:-0.8}"
SAMPLES="${SAMPLES:-5}"

# GEMM size knobs. Each work unit is M=N=K = (small_m or large_m).
# Both must be multiples of BM=128 (enforced by the binary).
SMALL_M="${SMALL_M:-512}"
LARGE_M="${LARGE_M:-2048}"

echo "=== building bimodal_gemms_rust (release) ==="
( cd "$HERE/bimodal_gemms_rust" && cargo build --release --offline ) || exit 1

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/results_bimodal.csv"
rm -f "$CSV"

# Mode-specific pinning: each mode gets exactly the CPU resources its
# threading model uses, which keeps the comparison fair.
#   serial  → 1 core (single host thread)
#   async   → 1 core (single host thread, tokio current_thread)
#   threaded → range of cores (S worker threads each block on sync;
#              they must be on separate cores to run concurrently)
# A wide range for async causes the OS to migrate the single host
# thread across cores, costing L1/L2 warmth and adding sample-to-sample
# variance; pinning to one core removes that.
PIN_SINGLE="${PIN_SINGLE:-8}"
PIN_RANGE="${PIN_RANGE:-1-16}"

echo
echo "=== sweeping (serial/async on CPU $PIN_SINGLE, threaded on CPUs $PIN_RANGE, small-m=$SMALL_M, large-m=$LARGE_M, n-work=$N_WORK, small-ratio=$SMALL_RATIO, samples=$SAMPLES) ==="

# Mode A (serial): single configuration; S is meaningless.
echo "--- mode=serial ---"
( cd "$HERE/bimodal_gemms_rust" && \
  taskset -c "$PIN_SINGLE" cargo run --release --offline --bin bimodal_gemms -- \
    --mode serial \
    --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
    --small-m "$SMALL_M" --large-m "$LARGE_M" \
    --samples "$SAMPLES" --warmup \
    --csv "$CSV" ) || exit 1

# Mode B (threaded): sweep S only (always T=1 threads-of-its-own-kind).
for s in $S_VALUES; do
  echo "--- mode=threaded streams=$s ---"
  ( cd "$HERE/bimodal_gemms_rust" && \
    taskset -c "$PIN_RANGE" cargo run --release --offline --bin bimodal_gemms -- \
      --mode threaded --streams "$s" \
      --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
      --small-m "$SMALL_M" --large-m "$LARGE_M" \
      --samples "$SAMPLES" --warmup \
      --csv "$CSV" ) || exit 1
done

# Mode C (async): sweep both S and T so we log every
# (stream-count × tokio-thread-count) cell.
for t in $T_VALUES; do
  for s in $S_VALUES; do
    echo "--- mode=async streams=$s tokio_threads=$t ---"
    ( cd "$HERE/bimodal_gemms_rust" && \
      taskset -c "$PIN_RANGE" cargo run --release --offline --bin bimodal_gemms -- \
        --mode async --streams "$s" --tokio-threads "$t" \
        --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
        --small-m "$SMALL_M" --large-m "$LARGE_M" \
        --samples "$SAMPLES" --warmup \
        --csv "$CSV" ) || exit 1
  done
done

echo
echo "=== results ==="
cat "$CSV"

echo
echo "=== rendering plot ==="
RESULTS_DIR="$RESULTS_DIR" "$PY" "$HERE/plot_bimodal.py"

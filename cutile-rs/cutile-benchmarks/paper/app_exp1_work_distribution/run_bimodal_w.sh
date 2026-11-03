#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sweeps the bimodal-GEMMs work-distribution experiment with
# per-task host work W (Option A merge of 6b + 6c). For each
# W value, runs serial, threaded (S=PIN_S), async-T=1 (S=PIN_S),
# and async-T=4 (S=PIN_S). Appends to results_bimodal_w.csv and
# renders the figure.
#
# Logic of the experiment: each work unit is GEMM + spin(W).
# At W=0 we recover the existing 6c story. As W grows, the
# tradeoff between async (overlap host work with other streams'
# GPU work, single host thread) and threaded (S threads spinning
# in parallel) becomes visible.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/bimodal_gemms_rust}"

# Per-task host work (microseconds). Sweep across the regime
# where the merged story shifts: W=0 is the GPU-only baseline;
# small W is dominated by GEMM; large W is dominated by host spin.
W_VALUES="${W_VALUES:-0 10 30 100 300 1000 3000}"

# Fixed stream count for this experiment. 16 is the GPU's
# multi-stream ceiling for this workload (per the existing 6c
# sweep) — past S=16 we get diminishing returns.
PIN_S="${PIN_S:-16}"

# Tokio thread counts to compare. T=1 is the headline async
# configuration; T=4 is shown for context (matches the second
# async line in 6c).
T_VALUES="${T_VALUES:-1 4}"

# Workload knobs (consistent with the main bimodal sweep).
N_WORK="${N_WORK:-2000}"
SMALL_RATIO="${SMALL_RATIO:-0.8}"
SAMPLES="${SAMPLES:-5}"
SMALL_M="${SMALL_M:-512}"
LARGE_M="${LARGE_M:-2048}"

echo "=== building bimodal_gemms_rust (release) ==="
( cd "$HERE/bimodal_gemms_rust" && cargo build --release --offline ) || exit 1

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/results_bimodal_w.csv"
rm -f "$CSV"

# Mode-specific pinning matches run_bimodal.sh.
PIN_SINGLE="${PIN_SINGLE:-8}"
PIN_RANGE="${PIN_RANGE:-1-16}"

echo
echo "=== W-sweep (S=$PIN_S, T in $T_VALUES, W in $W_VALUES, small-m=$SMALL_M, large-m=$LARGE_M, samples=$SAMPLES) ==="

for w in $W_VALUES; do
  echo "--- mode=serial w=$w ---"
  ( cd "$HERE/bimodal_gemms_rust" && \
    taskset -c "$PIN_SINGLE" cargo run --release --offline --bin bimodal_gemms -- \
      --mode serial \
      --w-host-us "$w" \
      --small-m "$SMALL_M" --large-m "$LARGE_M" \
      --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
      --samples "$SAMPLES" --warmup \
      --csv "$CSV" ) || exit 1
  echo "--- mode=threaded streams=$PIN_S w=$w ---"
  ( cd "$HERE/bimodal_gemms_rust" && \
    taskset -c "$PIN_RANGE" cargo run --release --offline --bin bimodal_gemms -- \
      --mode threaded --streams "$PIN_S" \
      --w-host-us "$w" \
      --small-m "$SMALL_M" --large-m "$LARGE_M" \
      --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
      --samples "$SAMPLES" --warmup \
      --csv "$CSV" ) || exit 1
  for t in $T_VALUES; do
    echo "--- mode=async streams=$PIN_S tokio=$t w=$w ---"
    ( cd "$HERE/bimodal_gemms_rust" && \
      taskset -c "$PIN_RANGE" cargo run --release --offline --bin bimodal_gemms -- \
        --mode async --streams "$PIN_S" --tokio-threads "$t" \
        --w-host-us "$w" \
        --small-m "$SMALL_M" --large-m "$LARGE_M" \
        --n-work "$N_WORK" --small-ratio "$SMALL_RATIO" \
        --samples "$SAMPLES" --warmup \
        --csv "$CSV" ) || exit 1
  done
done

echo
echo "=== results ==="
cat "$CSV"

echo
echo "=== rendering plot ==="
RESULTS_DIR="$RESULTS_DIR" "$PY" "$HERE/plot_bimodal_w.py"

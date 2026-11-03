#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Drives the §5.2 execution-mode scaling sweep: measured schedules × a sweep
# of pipeline lengths N, one cargo run per (schedule, N) cell, appending
# to results_part_a.csv. Then renders the line chart at
# figures/generated/exp2_execmode_latency.pdf.
#
# Prereqs:
#   - launch_overhead_rust/ builds (path-deps on local cutile-rs)
#   - Python environment with the plotting dependencies installed.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/launch_overhead_rust}"

# Pipeline lengths to sweep. The plot clips at N=1000 (plot_part_a.py),
# which is the regime shown in Figure 6.
N_VALUES="${N_VALUES:-1 2 3 5 10 30 100 300 1000}"

# Fixed measurement budget for every (mode, N) cell. Holding these
# constant across N keeps the statistics comparable along the x-axis:
# each cell draws SAMPLES samples (the population behind min/p25/p75),
# and each sample is the mean per-pipeline time over ITERS pipelines.
# Earlier revisions scaled these down with N to cap wall-clock, which
# made the large-N jitter bands artificially wide (fewer samples, less
# inner averaging) rather than reflecting real variance.
WARMUP="${WARMUP:-200}"
ITERS="${ITERS:-50}"
SAMPLES="${SAMPLES:-50}"

echo "=== building launch_overhead_rust (release) ==="
( cd "$HERE/launch_overhead_rust" && cargo build --release ) || exit 1

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/results_part_a.csv"
rm -f "$CSV"

# Auto-pick the idlest CPU core for pinning. We deliberately avoid CPU
# 0, which tends to service interrupts on most Linux boxes. Override
# with PIN_CPU=<id>.
PIN_CPU="${PIN_CPU:-}"
if [ -z "$PIN_CPU" ]; then
  if command -v mpstat >/dev/null 2>&1; then
    PIN_CPU=$(mpstat -P ALL 1 1 2>/dev/null \
      | awk 'NR>3 && $2 ~ /^[0-9]+$/ && $2 != "0" {print $NF, $2}' \
      | sort -rn | head -1 | awk '{print $2}')
  fi
  PIN_CPU="${PIN_CPU:-8}"
fi

echo
echo "=== sweeping (taskset -c $PIN_CPU), N in: $N_VALUES ==="
for mode in sync-individual sync-chained async graph; do
  for n in $N_VALUES; do
    echo "--- mode=$mode n=$n warmup=$WARMUP iters=$ITERS samples=$SAMPLES ---"
    ( cd "$HERE/launch_overhead_rust" && \
      taskset -c "$PIN_CPU" cargo run --release --bin launch_overhead -- \
        --mode "$mode" --n "$n" \
        --warmup "$WARMUP" --iters "$ITERS" --samples "$SAMPLES" \
        --csv "$CSV" ) || exit 1
  done
done

echo
echo "=== results ==="
cat "$CSV"

echo
echo "=== rendering plot ==="
RESULTS_DIR="$RESULTS_DIR" "$PY" "$HERE/plot_part_a.py"

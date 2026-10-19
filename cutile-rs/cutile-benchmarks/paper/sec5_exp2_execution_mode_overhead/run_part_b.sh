#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Drives §5.2 Part B — async vs sync throughput under host work W.
# For each (mode, W), one cargo run appending to results_part_b.csv.
# Then renders figures/generated/exp2_async_throughput.pdf.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
RESULTS_DIR="${RESULTS_DIR:-$HERE/launch_overhead_rust}"

# Host-work sweep. Span well below the async callback tax (so sync
# wins at small W) through the peak-overlap regime (~GPU time) to
# well above it (where the two asymptotically converge).
W_VALUES="${W_VALUES:-0 1 3 10 30 100 300 1000 3000 10000}"

# Pipeline length captured into the graph. N=300 puts the per-replay
# GPU time around a few hundred microseconds, so the 2x peak is
# reachable with W values we can measure accurately.
N_OPS="${N_OPS:-300}"
SAMPLES="${SAMPLES:-25}"

echo "=== building launch_overhead_rust (release) ==="
( cd "$HERE/launch_overhead_rust" && cargo build --release ) || exit 1

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/results_part_b.csv"
rm -f "$CSV"

# Auto-pick the idlest CPU core for pinning. Avoids CPU 0 which tends
# to service interrupts. Override with PIN_CPU=<id>.
PIN_CPU="${PIN_CPU:-}"
if [ -z "$PIN_CPU" ]; then
  if command -v mpstat >/dev/null 2>&1; then
    PIN_CPU=$(mpstat -P ALL 1 1 2>/dev/null \
      | awk 'NR>3 && $2 ~ /^[0-9]+$/ && $2 != "0" {print $NF, $2}' \
      | sort -rn | head -1 | awk '{print $2}')
  fi
  PIN_CPU="${PIN_CPU:-8}"
fi

# Adaptive iters/warmup so wall-clock per (mode, W) stays bounded.
# Per-iter wall time is roughly max(GPU_time, W + eps) for async and
# GPU_time + W for sync; pick iters so each sample times ~100 ms.
iters_for() {
  local w=$1
  if   (( w <=    30 )); then echo "100 300"
  elif (( w <=   300 )); then echo "80  150"
  elif (( w <=  1000 )); then echo "50  80"
  elif (( w <=  3000 )); then echo "40  40"
  else                        echo "30  20"
  fi
}

# Optional GPU clock lock to remove clock-variance noise from the sweep.
# Off by default (setting clocks is privileged and affects the whole GPU).
# Enable with LOCK_CLOCKS=1 (needs permission, e.g. run under sudo, or
# `nvidia-smi -acp UNRESTRICTED`); pin frequency with LOCK_CLOCKS_MHZ.
if [ "${LOCK_CLOCKS:-0}" = "1" ]; then
  LOCK_CLOCKS_MHZ="${LOCK_CLOCKS_MHZ:-$(nvidia-smi -q -d SUPPORTED_CLOCKS 2>/dev/null \
    | awk '/Graphics/{print $3; exit}')}"
  echo "=== locking GPU graphics clock to ${LOCK_CLOCKS_MHZ} MHz ==="
  if nvidia-smi --lock-gpu-clocks="${LOCK_CLOCKS_MHZ}" 2>/dev/null; then
    trap 'echo "=== resetting GPU clocks ==="; nvidia-smi --reset-gpu-clocks >/dev/null 2>&1' EXIT
  else
    echo "!! could not lock clocks (need permission); continuing unlocked" >&2
  fi
fi

echo
echo "=== sweeping (taskset -c $PIN_CPU), W in: $W_VALUES, N=$N_OPS, samples=$SAMPLES ==="
for mode in sync async; do
  for w in $W_VALUES; do
    read warmup iters <<< "$(iters_for "$w")"
    echo "--- mode=$mode w=$w warmup=$warmup iters=$iters ---"
    ( cd "$HERE/launch_overhead_rust" && \
      taskset -c "$PIN_CPU" cargo run --release --bin async_throughput -- \
        --mode "$mode" --w "$w" --n "$N_OPS" \
        --warmup "$warmup" --iters "$iters" --samples "$SAMPLES" \
        --csv "$CSV" ) || exit 1
  done
done

echo
echo "=== results ==="
cat "$CSV"

echo
echo "=== rendering plot ==="
RESULTS_DIR="$RESULTS_DIR" "$PY" "$HERE/plot_part_b.py"

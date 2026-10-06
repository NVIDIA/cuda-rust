#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect B200 locked-clock results for persistent GEMM and elementwise.
# This intentionally writes under paper/results/b200 instead of clobbering the
# 5090-oriented paper CSVs in paper/results/rtx5090.
set -euo pipefail

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP1="$(cd "$PAPER/.." && pwd)"
BENCHMARKS="$(cd "$EXP1/.." && pwd)"
PY="${PY:-python3}"
GEMM_DIR="$EXP1/rust/gemm"
ELEM_DIR="$EXP1/rust/elemwise"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_CLOCK_MHZ="${GPU_CLOCK_MHZ:-1830}"
GEMM_SOL_DENSE_TFLOPS="${GEMM_SOL_DENSE_TFLOPS:-2250}"
GEMM_SOL_SPARSE_TFLOPS="${GEMM_SOL_SPARSE_TFLOPS:-4500}"
GEMM_SOL_SOURCE="${GEMM_SOL_SOURCE:-https://www.nvidia.com/en-us/data-center/hgx/}"
GEMM_SOL_BASIS="${GEMM_SOL_BASIS:-HGX B200 FP16/BF16 Tensor Core 36 PFLOP/s sparse for 8 GPUs; dense is half sparse; dense per GPU is 2250 TFLOP/s}"
if [[ -z "${PIN+x}" ]]; then
  PIN=""
  if command -v taskset >/dev/null 2>&1; then
    affinity="$(taskset -pc $$ 2>/dev/null | awk -F: '{gsub(/^ +| +$/, "", $2); print $2}' || true)"
    first_cpu="$(printf '%s\n' "$affinity" | awk -F, '{split($1, range, "-"); print range[1]}')"
    if [[ -n "$first_cpu" ]] && taskset -c "$first_cpu" true >/dev/null 2>&1; then
      PIN="taskset -c $first_cpu"
    fi
  fi
fi
ELEM_B="${ELEM_B:-16384}"
ELEM_NUM_CTAS="${ELEM_NUM_CTAS:-default}"
ELEM_CACHE_SWEEP_MIB="${ELEM_CACHE_SWEEP_MIB:-1024}"
ELEM_ITERS="${ELEM_ITERS:-10}"
RUST_ELEMWISE_VARIANTS="${RUST_ELEMWISE_VARIANTS:-optimized safe}"
RESULTS_DIR="${RESULTS_DIR:-$PAPER/results/b200}"
FIGURES_DIR="${FIGURES_DIR:-}"
CUBLAS_DIAG_DIR="${CUBLAS_DIAG_DIR:-$EXP1/diagnostics/results/cublaslt_cpp/b200}"
GEMM_CHECK="${GEMM_CHECK:-1}"
RUN_CUBLAS="${RUN_CUBLAS:-1}"
REQUIRE_CLOCK_LOCK="${REQUIRE_CLOCK_LOCK:-0}"
CLOCK_LOCKED=0
CLOCK_LOCK_STATUS="not_attempted"

SUDO=()
if command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
fi

export RESULTS_DIR
export ELEM_CACHE_SWEEP_MIB

mkdir -p "$RESULTS_DIR"

cleanup() {
  if [[ "$CLOCK_LOCKED" == "1" ]]; then
    echo "=== Release GPU ${GPU_INDEX} clocks ==="
    "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -rgc || true
  fi
}
trap cleanup EXIT

echo "=== B200 persistent GEMM + elementwise run ==="
echo "Results: $RESULTS_DIR"
echo "Python: $PY"

echo "=== Lock GPU ${GPU_INDEX} clocks @ ${GPU_CLOCK_MHZ} MHz ==="
if "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -lgc "$GPU_CLOCK_MHZ,$GPU_CLOCK_MHZ"; then
  CLOCK_LOCKED=1
  CLOCK_LOCK_STATUS="locked"
else
  CLOCK_LOCK_STATUS="failed"
  if [[ "$REQUIRE_CLOCK_LOCK" != "0" ]]; then
    echo "error: failed to lock GPU clocks and REQUIRE_CLOCK_LOCK=${REQUIRE_CLOCK_LOCK}" >&2
    exit 1
  fi
  echo "warning: failed to lock GPU clocks; continuing with unlocked clocks" >&2
fi

echo "=== Device metadata ==="
{
  echo "key,value"
  echo "gpu_clock_mhz,$GPU_CLOCK_MHZ"
  echo "clock_lock_status,$CLOCK_LOCK_STATUS"
  echo "gpu_index,$GPU_INDEX"
  echo "gemm_sol_dense_tflops,$GEMM_SOL_DENSE_TFLOPS"
  echo "gemm_sol_sparse_tflops,$GEMM_SOL_SPARSE_TFLOPS"
  echo "gemm_sol_source,$GEMM_SOL_SOURCE"
  echo "gemm_sol_basis,$GEMM_SOL_BASIS"
  echo "elem_b,$ELEM_B"
  echo "elem_num_ctas,$ELEM_NUM_CTAS"
  echo "elem_cache_sweep_mib,$ELEM_CACHE_SWEEP_MIB"
  echo "elem_iters,$ELEM_ITERS"
  echo "elem_methodology,isolated_cuda_events_cache_rotation_fixed_iters_single_variant_processes"
  echo "pin_command,${PIN:-none}"
  echo "gemm_check,$GEMM_CHECK"
  echo "run_cublas,$RUN_CUBLAS"
  echo "cublas_baseline,direct_cublaslt_cpp"
  echo "cublas_timing,warmup5_samples10_iters3_no_cuda_graphs"
  nvidia-smi | sed -n 's/.*CUDA Version: *\([^ |]*\).*/cuda_version,\1/p' | head -n 1
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit \
    --format=csv,noheader,nounits |
    awk -F, '{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3);
              gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $6);
              print "gpu_name," $1;
              print "driver_version," $2;
              print "current_sm_clock_mhz," $3;
              print "max_sm_clock_mhz," $4;
              print "memory_clock_mhz," $5;
              print "power_limit_w," $6;}'
  "$PY" "$BENCHMARKS/tools/query_nominal_memory_bandwidth.py" \
    --device "$GPU_INDEX" --value-only |
    awk '{print "nominal_memory_bandwidth_gbps," $1}'
} > "$RESULTS_DIR/metadata.csv"
cat "$RESULTS_DIR/metadata.csv"

echo "=== Rust GEMM build ==="
( cd "$GEMM_DIR" && cargo build --release )

echo "=== Rust elementwise build ==="
( cd "$ELEM_DIR" && cargo build --release )

GEMM_CHECK_ARGS=()
if [[ "$GEMM_CHECK" != "0" ]]; then
  GEMM_CHECK_ARGS+=(--check)
fi

echo "=== Persistent GEMM Rust: dynamic safe ==="
( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --persistent --safe "${GEMM_CHECK_ARGS[@]}" )

echo "=== Persistent GEMM Rust: raw-pointer baseline ==="
( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --persistent --raw-pointer "${GEMM_CHECK_ARGS[@]}" )

if [[ "${RUN_EXTRA_GEMM_VARIANTS:-0}" != "0" ]]; then
  echo "=== Persistent GEMM Rust: optimized diagnostic ==="
  ( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --persistent "${GEMM_CHECK_ARGS[@]}" )

  echo "=== Persistent GEMM Rust: static diagnostic ==="
  ( cd "$GEMM_DIR" && $PIN ./target/release/gemm_rust --persistent --static "${GEMM_CHECK_ARGS[@]}" )
fi

echo "=== Persistent GEMM Python ==="
( cd "$PAPER" && $PIN "$PY" gemm_python_persistent.py "${GEMM_CHECK_ARGS[@]}" )

if [[ "$RUN_CUBLAS" != "0" ]]; then
  echo "=== GEMM cuBLASLt baseline (direct C++ heuristic sweep) ==="
  ( cd "$EXP1" && PIN="$PIN" RESULTS_DIR="$CUBLAS_DIAG_DIR" diagnostics/run_cublaslt_cpp_gemm.sh )
  cp "$CUBLAS_DIAG_DIR/gemm_cublaslt_cpp_best.csv" "$RESULTS_DIR/gemm_cublas_results.csv"
fi

RUST_ELEMWISE_ARGS=(--b "$ELEM_B" --cache-sweep-mib "$ELEM_CACHE_SWEEP_MIB" --iters "$ELEM_ITERS")
PYTHON_ELEMWISE_ARGS=(--b "$ELEM_B" --iters "$ELEM_ITERS")
if [[ "$ELEM_NUM_CTAS" != "default" && -n "$ELEM_NUM_CTAS" ]]; then
  RUST_ELEMWISE_ARGS+=(--cta "$ELEM_NUM_CTAS")
  PYTHON_ELEMWISE_ARGS+=(--num-ctas "$ELEM_NUM_CTAS")
fi

echo "=== Elementwise Rust ==="
rm -f "$RESULTS_DIR/elemwise_rust_runtime_results.csv"
for variant in $RUST_ELEMWISE_VARIANTS; do
  echo "Rust args: ${RUST_ELEMWISE_ARGS[*]} --variant $variant --append-csv"
  ( cd "$ELEM_DIR" && $PIN ./target/release/elemwise_rust "${RUST_ELEMWISE_ARGS[@]}" --variant "$variant" --append-csv )
done

echo "=== Elementwise Python ==="
echo "Python args: ${PYTHON_ELEMWISE_ARGS[*]}"
( cd "$PAPER" && $PIN "$PY" elemwise_python.py "${PYTHON_ELEMWISE_ARGS[@]}" )

if [[ -f "$RESULTS_DIR/gemm_cublas_results.csv" ]]; then
  echo "=== Summarize B200 cuTile fraction of cuBLAS ==="
  ( cd "$PAPER" && RESULTS_DIR="$RESULTS_DIR" "$PY" summarize_gemm_cublas_fraction.py --target b200 --markdown )
fi

echo "=== Regenerate B200 plots ==="
if "$PY" -c 'import matplotlib' >/dev/null 2>&1; then
  plot_args=(--target b200 --results-dir "$RESULTS_DIR")
  elem_plot_args=(--target b200 --results-dir "$RESULTS_DIR")
  if [[ -n "$FIGURES_DIR" ]]; then
    mkdir -p "$FIGURES_DIR"
    plot_args+=(--out "$FIGURES_DIR/exp1_safety_overhead.pdf")
    elem_plot_args+=(--out "$FIGURES_DIR/exp1_elemwise.pdf")
  fi
  ( cd "$PAPER" && RESULTS_DIR="$RESULTS_DIR" "$PY" plot_exp1.py "${plot_args[@]}" )
  ( cd "$PAPER" && RESULTS_DIR="$RESULTS_DIR" "$PY" plot_exp1_elemwise.py "${elem_plot_args[@]}" )
else
  echo "warning: matplotlib is not installed in $PY; skipping plot regeneration" >&2
fi

echo ""
echo "CSV results:"
ls -1 "$RESULTS_DIR"/*.csv

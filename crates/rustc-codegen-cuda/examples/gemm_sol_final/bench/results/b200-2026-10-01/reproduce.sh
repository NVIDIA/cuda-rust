#!/usr/bin/env bash
set -euo pipefail
# Run sequentially on an otherwise idle B200; no clock or power changes.
benchmark_repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$benchmark_repo"
bash crates/rustc-codegen-cuda/examples/gemm_sol_final/bench/build.sh
GEMM_SOL_MODE=both GEMM_CUBLAS_SKIP=1 \
  GEMM_SOL_WARMUP=2000 GEMM_SOL_ITERS=301 GEMM_SOL_REPEATS=5 \
  cargo oxide run gemm_sol_final --arch sm_100a
crates/rustc-codegen-cuda/examples/gemm_sol_final/bench/cublaslt_bench \
  --mode both --warmup 2000 --iters 301 --repeats 5

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect PR #1306 runtime evidence. See the example's HOPPER_VALIDATION.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_TOOLKIT_PATH:?Set CUDA_TOOLKIT_PATH to your CUDA 13+ installation}"
export CUDA_HOME="$CUDA_TOOLKIT_PATH"
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_TERM_COLOR=never
example=crates/rustc-codegen-cuda/examples/wgmma_mma_fp8
results=$(mktemp -d "${TMPDIR:-/tmp}/fp8-wgmma.XXXXXX")
marker='SUCCESS: FP8 WGMMA numeric check and BF16 comparison passed'

finish() {
    status=$?
    if ((status != 0)); then
        printf 'FAIL: exit %s; inspect the last log. Partial evidence is preserved.\n' "$status" > "$results/result.txt"
    fi
    printf 'exit_code=%s\n' "$status" > "$results/status.txt"
    tar -czf "$results.tar.gz" -C "$(dirname "$results")" "$(basename "$results")"
    printf '\nResults (exit %s): %s.tar.gz\n' "$status" "$results"
}
trap finish EXIT
printf 'Collecting logs in %s\n' "$results"

run() {
    local label=$1
    shift
    printf '\nRunning %s\n' "$label"
    printf '%q ' "$@" > "$results/$label.log"
    printf '\n' >> "$results/$label.log"
    "$@" 2>&1 | tee -a "$results/$label.log"
}

{
    date -u
    git rev-parse HEAD
    git status --short
    uname -a
    for name in CUDA_VISIBLE_DEVICES CUDA_TOOLKIT_PATH CARGO_BUILD_JOBS CUDA_OXIDE_DEBUG CUDA_OXIDE_LLC CUDA_OXIDE_NO_FMA CUDA_OXIDE_UNCHECKED_INDEXING RUSTFLAGS; do
        printf '%s=%s\n' "$name" "${!name-}"
    done
    nvidia-smi
    nvidia-smi -q
    rustc -Vv
    nvcc --version
    compute-sanitizer --version
} > "$results/environment.log" 2>&1
git diff --binary HEAD > "$results/changes.patch"
tar -czf "$results/source.tar.gz" "$example/src" "$example/bench" "$example/Cargo.toml" \
    "$example/Cargo.lock" "$example/README.md" "$example/HOPPER_VALIDATION.md" \
    scripts/validate-fp8-wgmma.sh scripts/bench-fp8-vs-bf16.sh scripts/test-fp8-wgmma-runner.py
run doctor cargo oxide doctor
run host-tests cargo test --locked --manifest-path "$example/Cargo.toml"
run correctness cargo oxide run wgmma_mma_fp8 --arch sm_90a -- --check-only
grep -Fx "$marker" "$results/correctness.log"

for tool in memcheck racecheck initcheck synccheck; do
    extra=()
    case "$tool" in
        memcheck|synccheck) extra=(--check-warpgroup-mma yes) ;;
        racecheck) extra=(--racecheck-report all --print-level info) ;;
    esac
    run "$tool" cargo oxide sanitize wgmma_mma_fp8 --arch sm_90a --lineinfo \
        --tool "$tool" -- --error-exitcode 86 --print-limit 0 "${extra[@]}" -- --check-only
    grep -Fx "$marker" "$results/$tool.log"
    if [[ "$tool" == racecheck ]]; then
        grep -E '^========= RACECHECK SUMMARY: 0 hazards' "$results/$tool.log"
    else
        grep -F '========= ERROR SUMMARY: 0 errors' "$results/$tool.log"
    fi
done

# Rebuild without lineinfo and time without instrumentation. Preserve every run.
for repeat in 1 2 3; do
    run "benchmark-$repeat" env -u CUDA_OXIDE_DEBUG cargo oxide run wgmma_mma_fp8 --arch sm_90a
    grep -Fx "$marker" "$results/benchmark-$repeat.log"
    grep -E '^BF16 m64n64k16 x4: .* TFLOPS$' "$results/benchmark-$repeat.log"
    grep -E '^FP8  m64n64k32 x2: .* TFLOPS$' "$results/benchmark-$repeat.log"
done
cp "$example/wgmma_mma_fp8.ptx" "$results/benchmark.ptx"
run ptxas "$CUDA_TOOLKIT_PATH/bin/ptxas" -arch=sm_90a --compile-only -v \
    "$results/benchmark.ptx" -o "$results/benchmark.cubin"
printf 'PASS: Rust correctness, four sanitizers, three tile benchmarks and ptxas\n' > "$results/pr-validation.txt"

# Independent cuBLASLt experiment: a slowdown is a valid result, not a failure.
# Keep it after the PR checks so a library/toolkit failure preserves their evidence.
for repeat in 1 2 3; do
    run "gemm-$repeat" env -u CUDA_OXIDE_DEBUG NVCC="$CUDA_TOOLKIT_PATH/bin/nvcc" \
        bash scripts/bench-fp8-vs-bf16.sh
    [[ $(grep -c '^RESULT ' "$results/gemm-$repeat.log") == 12 ]]
    [[ $(grep -c '^SUCCESS: both formats passed before and after timing' "$results/gemm-$repeat.log") == 12 ]]
done
{
    printf 'Rust tile benchmark (PR evidence):\n'
    grep -H -E '^(BF16 m64n64k16 x4|FP8  m64n64k32 x2):' "$results"/benchmark-*.log
    printf '\ncuBLASLt GEMM (independent experiment; speedup >1 means FP8 faster):\n'
    grep -H '^RESULT ' "$results"/gemm-*.log
} > "$results/summary.txt"
run gpu-after nvidia-smi -q
printf 'PASS: PR validation and three cuBLASLt sweeps; see summary.txt (FP8 wins are not required)\n' | tee "$results/result.txt"

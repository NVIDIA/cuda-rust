# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cuBLASLt f16 GEMM baseline for paper Exp 1.

Calls `nvmath.linalg.advanced.Matmul` with autotune enabled — this is
cuBLASLt directly, bypassing torch's matmul dispatch.

Rationale: torch.mm on Blackwell (sm_120) hits a suboptimal cuBLAS
algorithm and plateaus ~2x below cuBLASLt's autotuned ceiling
(~224 TFlops vs ~440 TFlops at M=N=K=16384). nvmath drives cuBLASLt
with heuristic search, so this script reports the real cuBLAS baseline.

Compute type forced to COMPUTE_16F to match the cuTile kernels' f16
accumulator.

Same measurement methodology as gemm_python.py: fixed N_WARMUP,
N_SAMPLES x ITERS between two syncs, wall-clock timed with
perf_counter, median per-launch reported.

Usage:
    python3 gemm_cublas.py
"""

import csv
import os
import time
from pathlib import Path

import torch
import nvmath
from nvmath.linalg.advanced import Matmul, MatmulOptions

HERE = Path(__file__).parent
EXP1 = HERE.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "results" / "rtx5090"))

N_WARMUP = 5
ITERS = 3
N_SAMPLES = 10


def bench(shapes, dtype):
    results = []
    opts = MatmulOptions(
        compute_type=nvmath.linalg.advanced.MatmulComputeType.COMPUTE_16F
    )

    for (m, n, k) in shapes:
        A = torch.ones(m, k, dtype=dtype, device="cuda")
        B = torch.ones(k, n, dtype=dtype, device="cuda")

        mm = Matmul(A, B, options=opts)
        mm.plan()
        mm.autotune(iterations=10)

        # Warmup.
        for _ in range(N_WARMUP):
            mm.execute()
        torch.cuda.synchronize()

        per_launch_s = []
        for _ in range(N_SAMPLES):
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(ITERS):
                mm.execute()
            torch.cuda.synchronize()
            per_launch_s.append((time.perf_counter() - s) / ITERS)

        mm.free()

        per_launch_s.sort()
        import statistics
        median_s = per_launch_s[len(per_launch_s) // 2]
        min_s = per_launch_s[0]
        max_s = per_launch_s[-1]
        stdev_s = statistics.stdev(per_launch_s) if len(per_launch_s) > 1 else 0.0
        flops = 2.0 * m * n * k
        tflops = flops / median_s / 1e12

        results.append({
            "M": m, "N": n, "K": k,
            "median_s": median_s,
            "min_s": min_s,
            "max_s": max_s,
            "stdev_s": stdev_s,
            "tflops": tflops,
        })
        print(f"  cublas M=N=K={m}: {tflops:.1f} TFlops "
              f"({median_s*1e6:.0f} us, min {min_s*1e6:.0f} / max {max_s*1e6:.0f} / "
              f"\u03c3 {stdev_s*1e6:.1f})")

    return results


def main():
    dtype = torch.float16
    shapes = [(2**i, 2**i, 2**i) for i in range(10, 16)]

    print("=== GEMM Benchmark: cuBLASLt baseline (sm_120, NVIDIA GeForce RTX 5090) ===")
    print("--- cublas (nvmath, f16 acc, autotuned) ---")
    results = bench(shapes, dtype)

    print(f"\n{'='*60}")
    print(f"  SUMMARY (cuBLAS, f16, NVIDIA GeForce RTX 5090)")
    print(f"{'='*60}")
    print(f"  {'M=N=K':>8}  {'cublas':>12}")
    for r in results:
        print(f"  {r['M']:>8}  {r['tflops']:>10.1f} TF")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "gemm_cublas_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "M", "N", "K", "tm", "tn", "tk",
                         "median_s", "min_s", "max_s", "stdev_s", "tflops"])
        for r in results:
            writer.writerow(["cublas", r["M"], r["N"], r["K"],
                             "", "", "",
                             f"{r['median_s']:.9f}",
                             f"{r['min_s']:.9f}",
                             f"{r['max_s']:.9f}",
                             f"{r['stdev_s']:.9f}",
                             f"{r['tflops']:.2f}"])
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()

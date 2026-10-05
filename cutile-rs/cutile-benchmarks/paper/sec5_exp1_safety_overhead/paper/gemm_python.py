# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM benchmark for cuTile Python — paper Exp 1.

Mirrors the Rust benchmark (`rust/gemm/`) layout:
  - source-level parity with the Rust `gemm` kernel (f16 accumulator, no
    swizzle, native 2D grid, default padding)
  - same measurement methodology: fixed N_WARMUP + N_SAMPLES x ITERS
    launches between two stream syncs, wall-clock timed with perf_counter

CLI:
  (no args)  : bench the kernel at all sizes
  --dump-ir  : compile at the largest M=N=K and dump CuTile IR text to stderr
               (no bench). Uses cuda.tile's built-in
               CUDA_TILE_LOGS=CUTILEIR hook; the env var must be set BEFORE
               `import cuda.tile`, which is why --dump-ir is parsed first.
"""

import os
import sys
from pathlib import Path

# --- CLI parsing must happen before importing cuda.tile so the log env var
# --- takes effect on first init.

_dump_ir = "--dump-ir" in sys.argv[1:]
if _dump_ir and "CUDA_TILE_LOGS" not in os.environ:
    os.environ["CUDA_TILE_LOGS"] = "CUTILEIR"


import cuda.tile as ct
import torch
import time
from math import ceil

HERE = Path(__file__).parent
EXP1 = HERE.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "results" / "rtx5090"))

ConstInt = ct.Constant[int]


@ct.kernel(num_ctas=2)
def gemm_optimized(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """GEMM with the fixed paper-run num_ctas=2 hint.

    Native 2D grid and A.dtype accumulator match the Rust gemm kernel source
    shape. Loads use `padding_mode=ZERO` so the partition-view emitted in Tile
    IR carries `padding_value = zero`, matching what Rust's `partition()` DSL
    emits.
    """
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=A.dtype)
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bid_m, k), shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        b = ct.load(B, index=(k, bid_n), shape=(tk, tn),
                    padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bid_m, bid_n), tile=acc.astype(C.dtype))


N_WARMUP = 5
ITERS = 3
N_SAMPLES = 10


def _alloc(m, n, k, dtype):
    A = torch.ones(m, k, dtype=dtype, device='cuda')
    B = torch.ones(k, n, dtype=dtype, device='cuda')
    C = torch.zeros(m, n, dtype=dtype, device='cuda')
    return A, B, C


def bench(kernel, label, shapes, hyper_params, dtype):
    stream = torch.cuda.current_stream()
    results = []

    for (m, n, k), (tm, tn, tk) in zip(shapes, hyper_params):
        A, B, C = _alloc(m, n, k, dtype)
        grid = (ceil(m / tm), ceil(n / tn), 1)

        # Warmup: fixed count, not wall-clock (async launches would queue up).
        for _ in range(N_WARMUP):
            ct.launch(stream, grid, kernel, (A, B, C, tm, tn, tk))
        torch.cuda.synchronize()

        per_launch_s = []
        for _ in range(N_SAMPLES):
            torch.cuda.synchronize()
            s = time.perf_counter()
            for _ in range(ITERS):
                ct.launch(stream, grid, kernel, (A, B, C, tm, tn, tk))
            torch.cuda.synchronize()
            per_launch_s.append((time.perf_counter() - s) / ITERS)

        per_launch_s.sort()
        import statistics
        median_s = per_launch_s[len(per_launch_s) // 2]
        min_s = per_launch_s[0]
        max_s = per_launch_s[-1]
        stdev_s = statistics.stdev(per_launch_s) if len(per_launch_s) > 1 else 0.0
        flops = 2.0 * m * n * k
        tflops = flops / median_s / 1e12

        results.append({
            'M': m, 'N': n, 'K': k,
            'tm': tm, 'tn': tn, 'tk': tk,
            'median_s': median_s,
            'min_s': min_s,
            'max_s': max_s,
            'stdev_s': stdev_s,
            'tflops': tflops,
        })
        print(f"  {label} M=N=K={m}: {tflops:.1f} TFlops "
              f"({median_s*1e6:.0f} us, min {min_s*1e6:.0f} / max {max_s*1e6:.0f} / "
              f"\u03c3 {stdev_s*1e6:.1f})")

    return results


def dump_ir_pass(kernel, m, n, k, tm, tn, tk, dtype):
    """Single launch at (m, n, k) to trigger JIT + CUTILEIR text dump to stderr."""
    stream = torch.cuda.current_stream()
    sys.stderr.write(f"-- M=N=K={m} --\n")
    A, B, C = _alloc(m, n, k, dtype)
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(stream, grid, kernel, (A, B, C, tm, tn, tk))
    torch.cuda.synchronize()


def main():
    dtype = torch.float16
    shapes = [(2**i, 2**i, 2**i) for i in range(10, 16)]
    hyper_params = [
        (128, 128, 64),   # 1024
        (256, 256, 128),  # 2048
        (128, 256, 128),  # 4096
        (128, 256, 128),  # 8192
        (256, 256, 128),  # 16384
        (128, 256, 128),  # 32768
    ]

    kernel = gemm_optimized
    label = "optimized"

    if _dump_ir:
        m, n, k = shapes[-1]
        tm, tn, tk = hyper_params[-1]
        sys.stderr.write(f"=== IR dump: {label} at largest size ===\n")
        dump_ir_pass(kernel, m, n, k, tm, tn, tk, dtype)
        return

    print(f"=== GEMM Benchmark: cuTile Python (sm_120, NVIDIA GeForce RTX 5090) ===")
    print(f"--- {label} ---")
    results = bench(kernel, label, shapes, hyper_params, dtype)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({label}, f16, NVIDIA GeForce RTX 5090)")
    print(f"{'='*60}")
    print(f"  {'M=N=K':>8}  {label:>12}")
    for r in results:
        print(f"  {r['M']:>8}  {r['tflops']:>10.1f} TF")

    import csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "gemm_python_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'M', 'N', 'K', 'tm', 'tn', 'tk',
                         'median_s', 'min_s', 'max_s', 'stdev_s', 'tflops'])
        for r in results:
            writer.writerow([label, r['M'], r['N'], r['K'],
                             r['tm'], r['tn'], r['tk'],
                             f"{r['median_s']:.9f}",
                             f"{r['min_s']:.9f}",
                             f"{r['max_s']:.9f}",
                             f"{r['stdev_s']:.9f}",
                             f"{r['tflops']:.2f}"])
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS CuTe DSL FP16 throughput diagnostic; not the live baseline.

Uses cute.compile() to pre-compile the kernel, then cute.testing.benchmark()
to measure GPU-only execution time (no Python host overhead).

Run from bench/:  PYTHONPATH=. ../bench/venv/bin/python3 cutlass_sol_bench.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# Import the tutorial's host_function and config
from tutorial_fp16_gemm import host_function, mma_tiler_mnk, io_dtype

WARMUP = 10
ITERS = 100


def bench_cutlass(m, n, k):
    """Benchmark CUTLASS GEMM using cute.compile + cute.testing.benchmark."""
    torch.manual_seed(42)
    current_stream = cutlass_torch.default_stream()

    # Create tensors (K-major, same as tutorial)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    # Pre-compile: separates JIT compilation from execution
    compiled = cute.compile(host_function, a_tensor, b_tensor, c_tensor)

    # Build workspace generator for benchmark
    def generate_tensors():
        args = cute.testing.JitArguments(a_tensor, b_tensor, c_tensor)
        args.add_to_scope([a, b, c])
        return args

    # Benchmark the COMPILED kernel (no Python host overhead)
    time_us = cute.testing.benchmark(
        compiled,
        workspace_generator=generate_tensors,
        workspace_count=1,
        stream=current_stream,
        warmup_iterations=WARMUP,
        iterations=ITERS,
    )

    avg_ms = time_us / 1000.0
    flops = 2.0 * m * n * k
    tflops = (flops / (avg_ms / 1000.0)) / 1e12

    return tflops, avg_ms


def main():
    from cuda.bindings import driver as cu_driver
    cu_driver.cuInit(0)

    print("=" * 70)
    print("  CUTLASS CuTe DSL SoL Benchmark (FP16 tcgen05 MMA)")
    print(f"  Tile: {mma_tiler_mnk[0]}x{mma_tiler_mnk[1]}x{mma_tiler_mnk[2]}")
    print(f"  Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 70)

    sizes = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    results = []
    for m, n, k in sizes:
        assert m % mma_tiler_mnk[0] == 0, f"M={m} not divisible by {mma_tiler_mnk[0]}"
        assert n % mma_tiler_mnk[1] == 0, f"N={n} not divisible by {mma_tiler_mnk[1]}"

        tflops, avg_ms = bench_cutlass(m, n, k)
        print(f"  {m:5d}x{n:5d}x{k:5d}  {avg_ms:8.4f} ms  {tflops:8.1f} TFLOPS")
        results.append((m, n, k, avg_ms, tflops))

    print(f"\n{'=' * 70}")
    print("  SUMMARY — CUTLASS measured results")
    print("  Compare only with a live same-size reference from cublaslt_bench.")
    print(f"{'=' * 70}")
    for m, n, k, avg_ms, tflops in results:
        print(f"  {m:5d}x{n:5d}x{k:5d}  {avg_ms:8.4f} ms  {tflops:8.1f} TFLOPS")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

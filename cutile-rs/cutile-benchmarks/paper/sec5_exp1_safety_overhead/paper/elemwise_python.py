# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Element-wise add benchmark for cuTile Python — paper Exp 1.

Cross-frontend baseline for the Rust element-wise bench. Methodology
mirrors rust/elemwise/src/main.rs:
  - CUDA events (`torch.cuda.Event(enable_timing=True)`) for GPU-side
    timing, removing host launch/sync jitter that dominated at small N
    under wall-clock `perf_counter`.
  - Timed windows rotate continuously across multiple independent x/y/z
    chunks when N is small, matching the Rust harness and avoiding
    repeated measurement of cache-hot buffers.
  - N_WARMUP = 20 launches, N_SAMPLES = 25 timed windows, fixed ITERS
    launches per timed window.
  - Median reported, p25/p75 for robust error bars.

cuTile Python has a single kernel variant (no dynamic/safe analog); we
compare its numbers against the Rust `unchecked` variant, both produced
via zero-padded partition-style tile-block loads. The tile size and CTA
hint are configurable with `--b` and `--num-ctas` so architecture-specific
baselines can match the chosen Rust tuning point. Rust also has a
max_divisibility knob, which is not exposed as a Python kernel decorator
option.

CLI:
  (no args)  bench the kernel across the N sweep
  --dump-ir  compile at the largest N and dump CuTile IR text to stderr
             (no bench). Env var must be set BEFORE `import cuda.tile`.
"""

import argparse
import os
import sys
from pathlib import Path

_dump_ir = "--dump-ir" in sys.argv[1:]
if _dump_ir and "CUDA_TILE_LOGS" not in os.environ:
    os.environ["CUDA_TILE_LOGS"] = "CUTILEIR"


import cuda.tile as ct
import torch
import statistics
from math import ceil

HERE = Path(__file__).parent
EXP1 = HERE.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "results" / "rtx5090"))

ConstInt = ct.Constant[int]


@ct.kernel()
def add_optimized(X, Y, Z, tb: ConstInt):
    """z = x + y, 1D tile-block partitioning over N.
    Matches Rust `add` kernel's partition+load pattern at the Tile IR
    level via zero-padded loads.
    """
    bid = ct.bid(0)
    x = ct.load(X, index=(bid,), shape=(tb,),
                padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(Y, index=(bid,), shape=(tb,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(Z, index=(bid,), tile=x + y)


N_WARMUP = 20
N_SAMPLES = 25
DEFAULT_ITERS = int(os.environ.get("ELEM_ITERS", "10"))
TILE_B = int(os.environ.get("ELEM_PYTHON_B", "16384"))
NUM_CTAS_ENV = os.environ.get("ELEM_PYTHON_NUM_CTAS", "default")
NUM_CTAS = None if NUM_CTAS_ENV in (None, "", "default") else int(NUM_CTAS_ENV)
CACHE_SWEEP_MIB = int(os.environ.get("ELEM_CACHE_SWEEP_MIB", "1024"))
MAX_CACHE_CHUNKS = 128


def _cache_chunks_for_n(n):
    if CACHE_SWEEP_MIB == 0:
        return 1
    target_bytes = CACHE_SWEEP_MIB * 1024 * 1024
    logical_bytes_per_launch = 3.0 * n * 2.0
    return max(1, min(MAX_CACHE_CHUNKS, ceil(target_bytes / logical_bytes_per_launch)))


def _alloc_chunks(n, dtype, cache_chunks):
    # Match the Rust harness: each cache-rotation chunk is an
    # independent 1D allocation. A single 2D backing allocation can
    # cross the Python frontend's large-array index threshold at the
    # largest paper sizes even though each launched slice is only 1D.
    return [_alloc(n, dtype) for _ in range(cache_chunks)]


def _alloc(n, dtype):
    X = torch.ones((n,), dtype=dtype, device='cuda')
    Y = torch.ones((n,), dtype=dtype, device='cuda')
    Z = torch.zeros((n,), dtype=dtype, device='cuda')
    return X, Y, Z


def _event_time_us(stream, fn):
    """Returns elapsed microseconds over `fn` using CUDA events on the
    given stream. Matches Rust's `event_time_us`.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    fn()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0  # ms -> us


def bench(kernel, label, ns, b, dtype, iters):
    stream = torch.cuda.current_stream()
    results = []

    for n in ns:
        cache_chunks = _cache_chunks_for_n(n)
        chunks = _alloc_chunks(n, dtype, cache_chunks)
        grid = (ceil(n / b), 1, 1)
        logical_mib = cache_chunks * 3.0 * n * 2.0 / (1024.0 * 1024.0)

        print(f"--- N={n} cache_chunks={cache_chunks} "
              f"logical_window_footprint={logical_mib:.1f} MiB ---")

        launch_counter = 0

        def _launch_next():
            nonlocal launch_counter
            X, Y, Z = chunks[launch_counter % cache_chunks]
            launch_counter += 1
            ct.launch(stream, grid, kernel, (X, Y, Z, b))

        for _ in range(N_WARMUP):
            _launch_next()
        torch.cuda.synchronize()

        per_launch_us = []
        for _ in range(N_SAMPLES):
            torch.cuda.synchronize()

            def _timed():
                for _ in range(iters):
                    _launch_next()

            us = _event_time_us(stream, _timed)
            per_launch_us.append(us / iters)

        per_launch_us.sort()
        median_us = per_launch_us[len(per_launch_us) // 2]
        min_us = per_launch_us[0]
        max_us = per_launch_us[-1]
        def _pct(q):
            idx = round((len(per_launch_us) - 1) * q)
            return per_launch_us[min(idx, len(per_launch_us) - 1)]
        p25_us = _pct(0.25)
        p75_us = _pct(0.75)
        stdev_us = statistics.stdev(per_launch_us) if len(per_launch_us) > 1 else 0.0
        bytes_transferred = 3.0 * n * 2.0  # 3 tensors * N * sizeof(f16)
        gb_per_s = bytes_transferred / (median_us / 1e6) / 1e9

        results.append({
            'N': n, 'B': b, 'cache_chunks': cache_chunks, 'iters': iters,
            'median_us': median_us,
            'min_us': min_us,
            'max_us': max_us,
            'p25_us': p25_us,
            'p75_us': p75_us,
            'stdev_us': stdev_us,
            'gb_per_s': gb_per_s,
        })
        print(f"  {label} N={n:>10}: iters={iters:>4} {gb_per_s:>7.2f} GB/s "
              f"(median {median_us:>7.2f} us, min {min_us:>7.2f} / max {max_us:>7.2f} / "
              f"\u03c3 {stdev_us:>6.2f})")

    return results


def specialize_kernel(kernel, num_ctas):
    if num_ctas is None:
        return kernel
    return ct.kernel(kernel._pyfunc, num_ctas=num_ctas)


def dump_ir_pass(kernel, n, b, dtype):
    stream = torch.cuda.current_stream()
    sys.stderr.write(f"-- N={n} --\n")
    X, Y, Z = _alloc(n, dtype)
    grid = (ceil(n / b), 1, 1)
    ct.launch(stream, grid, kernel, (X, Y, Z, b))
    torch.cuda.synchronize()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-ir", action="store_true")
    ap.add_argument("--b", type=int, default=TILE_B)
    ap.add_argument(
        "--n",
        type=int,
        action="append",
        help="Run one N value. May be repeated. Defaults to 2^20..2^28.",
    )
    ap.add_argument(
        "--num-ctas",
        default=NUM_CTAS,
        help="Compile with this num_ctas hint, or 'default' to leave it unset.",
    )
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    return ap.parse_args()


def parse_num_ctas(value):
    if value in (None, "", "default"):
        return None
    return int(value)


def validate_config(ns, b):
    if b <= 0:
        raise ValueError(f"B must be positive, got {b}")
    bad = [n for n in ns if n % b != 0]
    if bad:
        raise ValueError(f"all N must be divisible by B={b}; failing N values: {bad}")


def main():
    args = parse_args()
    dtype = torch.float16
    ns = args.n if args.n else [1 << i for i in range(20, 29)]
    b = args.b
    validate_config(ns, b)
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    num_ctas = parse_num_ctas(args.num_ctas)

    kernel = specialize_kernel(add_optimized, num_ctas)
    label = "optimized" if num_ctas is None else f"optimized_cta{num_ctas}"

    if args.dump_ir:
        sys.stderr.write(f"=== IR dump: {label} at largest size ===\n")
        dump_ir_pass(kernel, ns[-1], b, dtype)
        return

    device_name = torch.cuda.get_device_name()
    print(f"=== Element-wise Add Benchmark: cuTile Python ({device_name}) ===")
    print(f"--- {label} (B={b}, num_ctas={num_ctas if num_ctas is not None else 'default'}, "
          f"cache_sweep_mib={CACHE_SWEEP_MIB}, max_cache_chunks={MAX_CACHE_CHUNKS}, "
          f"warmup={N_WARMUP}, samples={N_SAMPLES}, iters={args.iters}) ---")
    results = bench(kernel, label, ns, b, dtype, args.iters)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({label}, f16, {device_name})")
    print(f"{'='*60}")
    print(f"  {'N':>10}  {label:>12}")
    for r in results:
        print(f"  {r['N']:>10}  {r['gb_per_s']:>8.1f} GB/s")

    import csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "elemwise_python_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'N', 'B', 'cache_chunks', 'iters',
                    'median_us', 'min_us', 'max_us',
                    'p25_us', 'p75_us', 'stdev_us', 'gb_per_s'])
        for r in results:
            w.writerow([label, r['N'], r['B'], r['cache_chunks'], r['iters'],
                        f"{r['median_us']:.3f}",
                        f"{r['min_us']:.3f}",
                        f"{r['max_us']:.3f}",
                        f"{r['p25_us']:.3f}",
                        f"{r['p75_us']:.3f}",
                        f"{r['stdev_us']:.3f}",
                        f"{r['gb_per_s']:.2f}"])
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent GEMM benchmark for cuTile Python.

This is the Python companion to `rust/gemm --persistent`:
  - same persistent work scheduling and optional swizzled tile order
  - same per-size tile shapes and num_ctas choices as the Rust persistent path
  - scheduler parameters are passed as compile-time constants
  - same fixed GEMM timing methodology used by the paper GEMM harnesses:
    N_WARMUP launches, then N_SAMPLES windows of fixed ITERS launches

The accumulator intentionally uses the output dtype for f16 inputs, matching
the Rust persistent kernels used for the paper results.
"""

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cuda.tile as ct
import torch


ConstInt = ct.Constant[int]
HERE = Path(__file__).parent
EXP1 = HERE.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "results" / "b200"))

N_WARMUP = 5
ITERS = 3
N_SAMPLES = 10
GROUP_SIZE_M = 8
OCCUPANCY = 1


@ct.kernel
def gemm_persistent(
    A,
    B,
    C,
    tm: ConstInt,
    tn: ConstInt,
    tk: ConstInt,
    group_size_m: ConstInt,
    swizzle: ConstInt,
):
    start_bid = ct.bid(0)

    m = A.shape[0]
    n = B.shape[1]
    num_bid_m = ct.cdiv(m, tm)
    num_bid_n = ct.cdiv(n, tn)
    total_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)
    k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for tile_id in range(start_bid, total_tiles, num_programs):
        if swizzle != 0:
            num_bid_in_group = group_size_m * num_bid_n
            group_id = tile_id // num_bid_in_group
            first_bid_m = group_id * group_size_m
            this_group_size_m = min(num_bid_m - first_bid_m, group_size_m)
            bid_m = first_bid_m + (tile_id % this_group_size_m)
            bid_n = (tile_id % num_bid_in_group) // this_group_size_m
        else:
            bid_m = tile_id // num_bid_n
            bid_n = tile_id % num_bid_n

        acc = ct.full((tm, tn), 0, dtype=C.dtype)
        for k in range(k_tiles):
            a = ct.load(A, (bid_m, k), shape=(tm, tk)).astype(dtype)
            b = ct.load(B, (k, bid_n), shape=(tk, tn)).astype(dtype)
            acc = ct.mma(a, b, acc)

        ct.store(C, (bid_m, bid_n), acc)


@dataclass(frozen=True)
class PersistentConfig:
    tm: int
    tn: int
    tk: int
    num_ctas: int
    occupancy: int
    group_size_m: int
    swizzle: bool
    num_programs: int | None = None


def fallback_config(size: int) -> PersistentConfig:
    if size <= 1024:
        return PersistentConfig(128, 512, 64, 4, 1, GROUP_SIZE_M, True)
    return PersistentConfig(256, 256, 64, 2, 1, GROUP_SIZE_M, True)


def default_config(size: int, sm_arch: int, num_sms: int) -> PersistentConfig:
    return fallback_config(size)


def persistent_num_programs(
    size: int,
    tm: int,
    tn: int,
    num_sms: int,
    num_ctas: int,
    occupancy: int,
    override: int | None = None,
) -> int:
    if override is not None:
        return max(1, override)
    tiles_m = ceil(size / tm)
    tiles_n = ceil(size / tn)
    total_tiles = max(1, tiles_m * tiles_n)
    sm_programs = max(1, num_sms // max(1, num_ctas))
    return max(1, min(sm_programs, total_tiles) * max(1, occupancy))


def alloc(size: int):
    a = torch.ones(size, size, dtype=torch.float16, device="cuda")
    b = torch.ones(size, size, dtype=torch.float16, device="cuda")
    c = torch.zeros(size, size, dtype=torch.float16, device="cuda")
    return a, b, c


def sample_check(c, size: int) -> tuple[float, bool]:
    torch.cuda.synchronize()
    positions = [
        (0, 0),
        (size // 2, size // 2),
        (size - 1, size - 1),
        (0, size - 1),
        (size - 1, 0),
    ]
    expected = float(size)
    max_err = 0.0
    for i, j in positions:
        got = float(c[i, j].item())
        max_err = max(max_err, abs(got - expected))
    return max_err, max_err == 0.0


def specialize_kernel(num_ctas: int, occupancy: int):
    return ct.kernel(gemm_persistent._pyfunc, num_ctas=num_ctas, occupancy=occupancy)


def bench_one(
    size: int,
    config: PersistentConfig,
    warmup: int,
    samples: int,
    iters: int,
    check: bool,
):
    tm, tn, tk = config.tm, config.tn, config.tk
    if size % tm != 0 or size % tn != 0 or size % tk != 0:
        raise ValueError(f"size={size} must be divisible by tile=({tm},{tn},{tk})")

    a, b, c = alloc(size)
    stream = torch.cuda.current_stream()
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    grid = (
        persistent_num_programs(
            size,
            tm,
            tn,
            num_sms,
            config.num_ctas,
            config.occupancy,
            config.num_programs,
        ),
        1,
        1,
    )
    kernel = specialize_kernel(config.num_ctas, config.occupancy)
    kernel_args = (
        a,
        b,
        c,
        tm,
        tn,
        tk,
        config.group_size_m,
        int(config.swizzle),
    )

    for _ in range(warmup):
        ct.launch(stream, grid, kernel, kernel_args)
    torch.cuda.synchronize()

    sample_max_err = 0.0
    ok = True
    if check:
        sample_max_err, ok = sample_check(c, size)

    per_launch_s = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            ct.launch(stream, grid, kernel, kernel_args)
        torch.cuda.synchronize()
        per_launch_s.append((time.perf_counter() - start) / iters)

    per_launch_s.sort()
    median_s = per_launch_s[len(per_launch_s) // 2]
    min_s = per_launch_s[0]
    max_s = per_launch_s[-1]
    stdev_s = statistics.stdev(per_launch_s) if len(per_launch_s) > 1 else 0.0
    tflops = (2.0 * size * size * size) / median_s / 1e12

    return {
        "config": "persistent",
        "M": size,
        "N": size,
        "K": size,
        "tm": tm,
        "tn": tn,
        "tk": tk,
        "group_size_m": config.group_size_m,
        "num_ctas": config.num_ctas,
        "occupancy": config.occupancy,
        "swizzle": "on" if config.swizzle else "off",
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
        "iters": iters,
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "stdev_s": stdev_s,
        "tflops": tflops,
        "sample_max_err": sample_max_err,
        "ok": ok,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "gemm_python_persistent_results.csv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sm-arch", type=int, default=None, help="Override detected SM architecture, e.g. 120.")
    ap.add_argument("--tm", type=int, default=None, help="Override persistent GEMM M tile size.")
    ap.add_argument("--tn", type=int, default=None, help="Override persistent GEMM N tile size.")
    ap.add_argument("--tk", type=int, default=None, help="Override persistent GEMM K tile size.")
    ap.add_argument("--num-ctas", type=int, default=None, help="Override CTAs per CGA.")
    ap.add_argument("--occupancy", type=int, default=None, help="Override occupancy hint.")
    ap.add_argument("--group-size-m", type=int, default=None, help="Override persistent swizzle group size.")
    ap.add_argument("--num-programs", type=int, default=None, help="Override persistent grid x dimension.")
    ap.add_argument(
        "--swizzle",
        action="store_true",
        help="Force swizzled persistent tile order for every size.",
    )
    ap.add_argument(
        "--no-swizzle",
        action="store_true",
        help="Use linear persistent tile order instead of the swizzled tile order.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.swizzle and args.no_swizzle:
        raise ValueError("--swizzle and --no-swizzle are mutually exclusive")

    rows = []
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    sm_arch = args.sm_arch if args.sm_arch is not None else major * 10 + minor

    print("=== Persistent GEMM Benchmark: cuTile Python ===")
    print(
        f"fixed timing: warmup={args.warmup} samples={args.samples} iters={args.iters}; "
        f"sm_arch=sm_{sm_arch} num_sms={num_sms}"
    )
    for size in args.sizes:
        config = default_config(size, sm_arch, num_sms)
        config = PersistentConfig(
            args.tm if args.tm is not None else config.tm,
            args.tn if args.tn is not None else config.tn,
            args.tk if args.tk is not None else config.tk,
            args.num_ctas if args.num_ctas is not None else config.num_ctas,
            args.occupancy if args.occupancy is not None else config.occupancy,
            args.group_size_m if args.group_size_m is not None else config.group_size_m,
            config.swizzle,
            args.num_programs if args.num_programs is not None else config.num_programs,
        )
        if args.swizzle:
            config = PersistentConfig(
                config.tm,
                config.tn,
                config.tk,
                config.num_ctas,
                config.occupancy,
                config.group_size_m,
                True,
                config.num_programs,
            )
        elif args.no_swizzle:
            config = PersistentConfig(
                config.tm,
                config.tn,
                config.tk,
                config.num_ctas,
                config.occupancy,
                config.group_size_m,
                False,
                config.num_programs,
            )
        row = bench_one(
            size,
            config,
            args.warmup,
            args.samples,
            args.iters,
            args.check,
        )
        rows.append(row)
        print(
            f"  persistent M=N=K={size}: {row['tflops']:.1f} TFlops "
            f"({row['median_s'] * 1e6:.0f} us, min {row['min_s'] * 1e6:.0f} / "
            f"max {row['max_s'] * 1e6:.0f} / sigma {row['stdev_s'] * 1e6:.1f} us, "
            f"tile=({config.tm},{config.tn},{config.tk}), cta={config.num_ctas}, "
            f"occupancy={config.occupancy}, grid={row['grid_x']}, "
            f"group={config.group_size_m}, swizzle={row['swizzle']}, iters={args.iters})"
        )

    print("\n============================================================")
    print("  SUMMARY (persistent, f16)")
    print("============================================================")
    print(f"  {'M=N=K':>8}  {'persistent':>11}")
    for row in rows:
        print(f"  {row['M']:>8}  {row['tflops']:>9.1f} TF")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config",
        "M",
        "N",
        "K",
        "tm",
        "tn",
        "tk",
        "group_size_m",
        "num_ctas",
        "occupancy",
        "swizzle",
        "grid_x",
        "grid_y",
        "grid_z",
        "iters",
        "median_s",
        "min_s",
        "max_s",
        "stdev_s",
        "tflops",
        "sample_max_err",
        "ok",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "median_s": f"{row['median_s']:.9f}",
                    "min_s": f"{row['min_s']:.9f}",
                    "max_s": f"{row['max_s']:.9f}",
                    "stdev_s": f"{row['stdev_s']:.9f}",
                    "tflops": f"{row['tflops']:.2f}",
                    "sample_max_err": f"{row['sample_max_err']:.6g}",
                }
            )
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

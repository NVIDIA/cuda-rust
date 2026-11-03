# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cuTile Python GEMM benchmark for slide-style and tuned candidate implementations.

This intentionally differs from `gemm_python.py`:
  - one-dimensional launch grid
  - swizzled CTA-to-output-tile mapping
  - direct `ct.load` / `ct.mma` / `ct.store` loop
  - f16 accumulator when run with f16 inputs/output

Default sizes match the slide: M=N=K in {2048, 4096, 8192, 16384, 32768}.
Use `--implementation cutile-sample` to run the swizzled implementation used
by the cuTile Python sample: fp32 accumulation, zero padding, and an output
cast. Use `--implementation tilegym-persistent` for TileGym-style static
persistent scheduling over the same swizzled tile order. Use `--sweep` to
compare these against the native 2D cuTile Python launch used by the paper
baseline.
"""

import argparse
import csv
import os
import statistics
import time
from math import ceil
from pathlib import Path
from typing import NamedTuple

import cuda.tile as ct
import torch


ConstInt = ct.Constant[int]
HERE = Path(__file__).parent
EXP1 = HERE.parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", EXP1 / "diagnostics" / "results" / "swizzle"))


class Config(NamedTuple):
    implementation: str
    tm: int
    tn: int
    tk: int
    num_ctas: int | None = None
    occupancy: int | None = None


def swizzle2d_from_bid(m, n, mt, nt, group_size_m, bid):
    num_bid_m = ct.cdiv(m, mt)
    num_bid_n = ct.cdiv(n, nt)
    num_bid_in_group = group_size_m * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * group_size_m
    this_group_size_m = min(num_bid_m - first_bid_m, group_size_m)
    i = first_bid_m + (bid % this_group_size_m)
    j = (bid % num_bid_in_group) // this_group_size_m
    return i, j


def swizzle2d(m, n, mt, nt):
    return swizzle2d_from_bid(m, n, mt, nt, 8, ct.bid(0))


@ct.kernel
def matmul_native(X, Y, Z, mt: ConstInt, nt: ConstInt, kt: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)

    nk = ct.num_tiles(X, axis=1, shape=(mt, kt))
    Zt = ct.full((mt, nt), 0, dtype=Z.dtype)
    dtype = ct.tfloat32 if X.dtype == ct.float32 else X.dtype

    for k in range(nk):
        Xt = ct.load(X, (i, k), shape=(mt, kt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        Yt = ct.load(Y, (k, j), shape=(kt, nt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        Zt = ct.mma(Xt, Yt, Zt)

    ct.store(Z, (i, j), ct.astype(Zt, Z.dtype))


@ct.kernel
def matmul_slide(X, Y, Z, mt: ConstInt, nt: ConstInt, kt: ConstInt):
    Zt = ct.full((mt, nt), 0, dtype=Z.dtype)

    m = X.shape[0]
    n = Y.shape[1]
    i, j = swizzle2d(m, n, mt, nt)

    nk = ct.num_tiles(X, axis=1, shape=(mt, kt))
    for k in range(nk):
        Xt = ct.load(X, (i, k), shape=(mt, kt))
        Yt = ct.load(Y, (k, j), shape=(kt, nt))
        Zt = ct.mma(Xt, Yt, Zt)

    ct.store(Z, (i, j), Zt)


@ct.kernel
def matmul_cutile_sample(X, Y, Z, mt: ConstInt, nt: ConstInt, kt: ConstInt):
    m = X.shape[0]
    n = Y.shape[1]
    i, j = swizzle2d(m, n, mt, nt)

    nk = ct.num_tiles(X, axis=1, shape=(mt, kt))
    Zt = ct.full((mt, nt), 0, dtype=ct.float32)
    dtype = ct.tfloat32 if X.dtype == ct.float32 else X.dtype

    for k in range(nk):
        Xt = ct.load(X, (i, k), shape=(mt, kt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        Yt = ct.load(Y, (k, j), shape=(kt, nt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        Zt = ct.mma(Xt, Yt, Zt)

    ct.store(Z, (i, j), ct.astype(Zt, Z.dtype))


@ct.kernel
def matmul_tilegym_persistent(X, Y, Z, mt: ConstInt, nt: ConstInt, kt: ConstInt):
    group_size_m = 8
    start_bid = ct.bid(0)

    m = X.shape[0]
    n = Y.shape[1]
    num_bid_m = ct.cdiv(m, mt)
    num_bid_n = ct.cdiv(n, nt)
    num_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)

    nk = ct.num_tiles(X, axis=1, shape=(mt, kt))
    dtype = ct.tfloat32 if X.dtype == ct.float32 else X.dtype

    for tile_id in range(start_bid, num_tiles, num_programs):
        i, j = swizzle2d_from_bid(m, n, mt, nt, group_size_m, tile_id)
        Zt = ct.full((mt, nt), 0, dtype=Z.dtype)

        for k in range(nk):
            Xt = ct.load(X, (i, k), shape=(mt, kt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
            Yt = ct.load(Y, (k, j), shape=(kt, nt), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
            Zt = ct.mma(Xt, Yt, Zt)

        ct.store(Z, (i, j), Zt)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--tile-m", type=int, default=256)
    ap.add_argument("--tile-n", type=int, default=256)
    ap.add_argument("--tile-k", type=int, default=64)
    ap.add_argument(
        "--implementation",
        choices=["native", "slide", "cutile-sample", "tilegym-persistent"],
        default="slide",
        help="Kernel body to benchmark.",
    )
    ap.add_argument("--num-ctas", type=int, default=None, help="Specialize the kernel with this num_ctas hint.")
    ap.add_argument("--occupancy", type=int, default=None, help="Specialize the kernel with this occupancy hint.")
    ap.add_argument("--sweep", action="store_true", help="Run an architecture-oriented candidate sweep.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true", help="Compare one output element against torch.mm.")
    return ap.parse_args()


def alloc(m: int, n: int, k: int):
    X = torch.ones(m, k, dtype=torch.float16, device="cuda")
    Y = torch.ones(k, n, dtype=torch.float16, device="cuda")
    Z = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    return X, Y, Z


def check_result(X, Y, Z) -> None:
    torch.cuda.synchronize()
    ref = torch.mm(X[:1, :], Y[:, :1])
    got = Z[:1, :1]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def specialize_kernel(kernel, num_ctas: int | None, occupancy: int | None):
    if num_ctas is None and occupancy is None:
        return kernel

    options = {}
    if num_ctas is not None:
        options["num_ctas"] = num_ctas
    if occupancy is not None:
        options["occupancy"] = occupancy
    return ct.kernel(kernel._pyfunc, **options)


def select_kernel(implementation: str, num_ctas: int | None, occupancy: int | None):
    kernels = {
        "native": matmul_native,
        "slide": matmul_slide,
        "cutile-sample": matmul_cutile_sample,
        "tilegym-persistent": matmul_tilegym_persistent,
    }
    return specialize_kernel(kernels[implementation], num_ctas, occupancy)


def config_label(implementation: str, num_ctas: int | None, occupancy: int | None) -> str:
    parts = [implementation.replace("-", "_")]
    if num_ctas is not None:
        parts.append(f"cta{num_ctas}")
    if occupancy is not None:
        parts.append(f"occ{occupancy}")
    return "_".join(parts)


def sweep_configs() -> list[Config]:
    configs = [
        # Native 2D launch, matching the main paper cuTile Python baseline.
        Config("native", 128, 128, 64, 1, 1),
        Config("native", 128, 128, 64, 2, 1),
        Config("native", 128, 256, 64, 1, 1),
        Config("native", 128, 256, 64, 2, 1),
        Config("native", 256, 128, 64, 1, 1),
        Config("native", 256, 128, 64, 2, 1),
        Config("native", 256, 256, 32, 2, 1),
        Config("native", 256, 256, 64, 1, 1),
        Config("native", 256, 256, 64, 2, 1),
        Config("native", 256, 256, 128, 2, 1),
        # Literal slide body, tuned as a candidate for this architecture.
        Config("slide", 128, 64, 32, 1, 1),
        Config("slide", 128, 64, 32, 1, 2),
        Config("slide", 128, 64, 64, 1, 1),
        Config("slide", 128, 128, 32, 1, 1),
        Config("slide", 128, 128, 64, 1, 1),
        Config("slide", 256, 128, 64, 1, 1),
        Config("slide", 256, 256, 64, 1, 1),
        Config("slide", 256, 256, 64, 2, 1),
        Config("slide", 512, 256, 64, 2, 1),
        # cuTile Python sample / TileGym non-persistent body.
        Config("cutile-sample", 128, 64, 32, 1, 2),
        Config("cutile-sample", 128, 64, 64, 1, 1),
        Config("cutile-sample", 128, 128, 32, 1, 1),
        Config("cutile-sample", 256, 128, 64, 1, 1),
        Config("cutile-sample", 256, 256, 64, 2, 1),
        # TileGym static-persistent scheduling. Keep this separate because it
        # changes grid sizing and usually only helps for selected shapes.
        Config("tilegym-persistent", 64, 64, 64, 1, 1),
        Config("tilegym-persistent", 64, 64, 64, 1, 2),
        Config("tilegym-persistent", 128, 64, 64, 1, 1),
        Config("tilegym-persistent", 128, 64, 64, 1, 2),
        Config("tilegym-persistent", 256, 256, 64, 1, 1),
        Config("tilegym-persistent", 256, 256, 64, 2, 1),
    ]
    return configs


def launch_grid(implementation: str, size: int, tm: int, tn: int, num_ctas: int | None, occupancy: int | None):
    if implementation == "native":
        return (ceil(size / tm), ceil(size / tn), 1)

    total_tiles = ceil(size / tm) * ceil(size / tn)
    if implementation != "tilegym-persistent":
        return (total_tiles, 1, 1)

    device = torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ctas = num_ctas if num_ctas is not None else 1
    occ = occupancy if occupancy is not None else 1
    programs = max(1, min(sms // ctas, total_tiles) * occ)
    return (programs, 1, 1)


def bench_one(
    kernel,
    config: str,
    implementation: str,
    num_ctas: int | None,
    occupancy: int | None,
    size: int,
    tm: int,
    tn: int,
    tk: int,
    warmup: int,
    samples: int,
    iters: int,
    check: bool,
):
    if size % tm != 0 or size % tn != 0 or size % tk != 0:
        raise ValueError(
            f"benchmark expects divisible sizes; got size={size}, "
            f"tile=({tm},{tn},{tk})"
        )

    X, Y, Z = alloc(size, size, size)
    stream = torch.cuda.current_stream()
    grid = launch_grid(implementation, size, tm, tn, num_ctas, occupancy)

    for _ in range(warmup):
        ct.launch(stream, grid, kernel, (X, Y, Z, tm, tn, tk))
    torch.cuda.synchronize()

    if check:
        check_result(X, Y, Z)

    per_launch_s = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            ct.launch(stream, grid, kernel, (X, Y, Z, tm, tn, tk))
        torch.cuda.synchronize()
        per_launch_s.append((time.perf_counter() - start) / iters)

    per_launch_s.sort()
    median_s = per_launch_s[len(per_launch_s) // 2]
    min_s = per_launch_s[0]
    max_s = per_launch_s[-1]
    stdev_s = statistics.stdev(per_launch_s) if len(per_launch_s) > 1 else 0.0
    tflops = (2.0 * size * size * size) / median_s / 1e12
    return {
        "config": config,
        "implementation": implementation,
        "num_ctas": num_ctas if num_ctas is not None else "default",
        "occupancy": occupancy if occupancy is not None else "default",
        "M": size,
        "N": size,
        "K": size,
        "tm": tm,
        "tn": tn,
        "tk": tk,
        "grid_x": grid[0],
        "median_s": median_s,
        "min_s": min_s,
        "max_s": max_s,
        "stdev_s": stdev_s,
        "tflops": tflops,
    }


def main() -> int:
    args = parse_args()
    rows = []
    out = args.out
    if out is None:
        name = "gemm_python_swizzle_sweep_results.csv" if args.sweep else "gemm_python_swizzle_results.csv"
        out = RESULTS_DIR / name
    sizes = args.sizes
    if sizes is None:
        sizes = [2048] if args.sweep else [2048, 4096, 8192, 16384, 32768]

    if args.sweep:
        configs = sweep_configs()
    else:
        configs = [Config(args.implementation, args.tile_m, args.tile_n, args.tile_k, args.num_ctas, args.occupancy)]

    print("=== Swizzled cuTile Python GEMM ===")
    if args.sweep:
        print(f"sweep configs={len(configs)}")

    for cfg in configs:
        kernel = select_kernel(cfg.implementation, cfg.num_ctas, cfg.occupancy)
        config = config_label(cfg.implementation, cfg.num_ctas, cfg.occupancy)
        print(f"implementation={cfg.implementation} config={config} tile=({cfg.tm},{cfg.tn},{cfg.tk})")

        for size in sizes:
            row = bench_one(
                kernel,
                config,
                cfg.implementation,
                cfg.num_ctas,
                cfg.occupancy,
                size,
                cfg.tm,
                cfg.tn,
                cfg.tk,
                args.warmup,
                args.samples,
                args.iters,
                args.check,
            )
            rows.append(row)
            print(
                f"  M=N=K={size}: {row['tflops']:.1f} TFlops "
                f"({row['median_s'] * 1e6:.0f} us, "
                f"min {row['min_s'] * 1e6:.0f} / max {row['max_s'] * 1e6:.0f} / "
                f"σ {row['stdev_s'] * 1e6:.1f})"
            )

    if args.sweep:
        print("\nBest by size:")
        for size in sizes:
            best = max((row for row in rows if row["M"] == size), key=lambda row: row["tflops"])
            print(
                f"  M=N=K={size}: {best['tflops']:.1f} TFlops "
                f"{best['config']} tile=({best['tm']},{best['tn']},{best['tk']}) "
                f"cta={best['num_ctas']} occ={best['occupancy']}"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = [
            "config",
            "implementation",
            "num_ctas",
            "occupancy",
            "M",
            "N",
            "K",
            "tm",
            "tn",
            "tk",
            "grid_x",
            "median_s",
            "min_s",
            "max_s",
            "stdev_s",
            "tflops",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({
                **row,
                "median_s": f"{row['median_s']:.9f}",
                "min_s": f"{row['min_s']:.9f}",
                "max_s": f"{row['max_s']:.9f}",
                "stdev_s": f"{row['stdev_s']:.9f}",
                "tflops": f"{row['tflops']:.2f}",
            })
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

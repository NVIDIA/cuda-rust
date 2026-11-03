#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize cuTile GEMM throughput as a fraction of cuBLASLt.

The denominator is the cuBLASLt result at the matching M=N=K size.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


HERE = Path(__file__).parent
EXP1 = HERE.parent
RTX5090_RESULTS_DIR = HERE / "results" / "rtx5090"
B200_RESULTS_DIR = HERE / "results" / "b200"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", RTX5090_RESULTS_DIR))
DIAG_RESULTS_DIR = EXP1 / "diagnostics" / "results"


def rtx5090_inputs(results_dir: Path) -> dict[str, Path]:
    return {
        "cutile_python": results_dir / "gemm_python_results.csv",
        "rust_unchecked": results_dir / "gemm_rust_optimized_results.csv",
        "rust_dynamic": results_dir / "gemm_rust_safe_results.csv",
        "rust_static": results_dir / "gemm_rust_static_results.csv",
    }


def b200_inputs(results_dir: Path) -> dict[str, Path]:
    return {
        "cutile_python_persistent": results_dir / "gemm_python_persistent_results.csv",
        "rust_persistent_dynamic": results_dir / "gemm_rust_persistent_safe_results.csv",
    }

SWIZZLE_INPUT = {
    "cutile_python_swizzle": DIAG_RESULTS_DIR / "swizzle" / "gemm_python_swizzle_results.csv",
}

FULL_OUTPUT_INPUT = {
    "rust_full_output": DIAG_RESULTS_DIR / "full_output" / "gemm_rust_full_output_results.csv",
}

DISPLAY = {
    "cutile_python": "cuTile Python",
    "cutile_python_persistent": "cuTile Python persistent",
    "cutile_python_swizzle": "cuTile Python swizzle",
    "rust_unchecked": "Rust unchecked",
    "rust_full_output": "Rust full-output diagnostic",
    "rust_dynamic": "Rust dynamic",
    "rust_static": "Rust static",
    "rust_persistent_dynamic": "Rust persistent dynamic",
}


def select_results_dir(target: str) -> Path:
    if "RESULTS_DIR" in os.environ:
        return RESULTS_DIR
    if target == "b200":
        return B200_RESULTS_DIR
    return RTX5090_RESULTS_DIR


def select_inputs(target: str, results_dir: Path) -> dict[str, Path]:
    if target == "rtx5090":
        return rtx5090_inputs(results_dir)
    if target == "b200":
        return b200_inputs(results_dir)

    # Auto mode preserves the two paper-facing paths:
    # - paper/results/rtx5090: original non-persistent RTX 5090 results
    # - paper/results/b200: persistent B200 results
    if (results_dir / "gemm_python_results.csv").exists():
        return rtx5090_inputs(results_dir)
    if (results_dir / "gemm_python_persistent_results.csv").exists():
        return b200_inputs(results_dir)
    return rtx5090_inputs(results_dir)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def read_by_m(path: Path) -> dict[int, dict[str, str]]:
    return {int(r["M"]): r for r in read_rows(path)}


def build_rows(results_dir: Path, inputs: dict[str, Path]) -> list[dict[str, object]]:
    cublas = read_by_m(results_dir / "gemm_cublas_results.csv")
    out: list[dict[str, object]] = []
    for impl, path in inputs.items():
        if not path.exists():
            continue
        rows = read_by_m(path)
        for m in sorted(rows):
            if m not in cublas:
                continue
            r = rows[m]
            c = cublas[m]
            tflops = float(r["tflops"])
            cublas_tflops = float(c["tflops"])
            frac = tflops / cublas_tflops
            out.append({
                "impl": impl,
                "config": r["config"],
                "M": m,
                "N": int(r["N"]),
                "K": int(r["K"]),
                "tflops": f"{tflops:.2f}",
                "cublas_tflops": f"{cublas_tflops:.2f}",
                "frac_cublas": f"{frac:.6f}",
                "pct_cublas": f"{100.0 * frac:.2f}",
            })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "impl",
        "config",
        "M",
        "N",
        "K",
        "tflops",
        "cublas_tflops",
        "frac_cublas",
        "pct_cublas",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict[str, object]], inputs: dict[str, Path]) -> None:
    print("| impl | frac @ 32768 | geometric mean | arithmetic mean |")
    print("|---|---:|---:|---:|")
    by_impl: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_impl.setdefault(str(row["impl"]), []).append(row)
    for impl in inputs:
        rs = by_impl.get(impl, [])
        if not rs:
            continue
        fracs = [float(r["frac_cublas"]) for r in rs]
        largest = next((float(r["frac_cublas"]) for r in rs if int(r["M"]) == 32768), fracs[-1])
        geomean = math.prod(fracs) ** (1.0 / len(fracs))
        arith = sum(fracs) / len(fracs)
        print(
            f"| {DISPLAY[impl]} | {largest:.3f} ({100 * largest:.1f}%) | "
            f"{geomean:.3f} ({100 * geomean:.1f}%) | "
            f"{arith:.3f} ({100 * arith:.1f}%) |"
        )


def print_per_size(rows: list[dict[str, object]], inputs: dict[str, Path]) -> None:
    by_m_impl: dict[int, dict[str, dict[str, object]]] = {}
    cublas_by_m: dict[int, float] = {}
    for row in rows:
        m = int(row["M"])
        impl = str(row["impl"])
        by_m_impl.setdefault(m, {})[impl] = row
        cublas_by_m[m] = float(row["cublas_tflops"])

    present_impls = [impl for impl in inputs if any(impl in by_m_impl[m] for m in by_m_impl)]
    headers = " | ".join(DISPLAY[impl] for impl in present_impls)
    aligns = "|".join(["---:"] * (2 + len(present_impls)))
    print(f"| M=N=K | cuBLAS TFLOP/s | {headers} |")
    print(f"|{aligns}|")
    for m in sorted(by_m_impl):
        impls = by_m_impl[m]
        cells = []
        for impl in present_impls:
            row = impls.get(impl)
            if row:
                cells.append(f"{float(row['pct_cublas']):.1f}%")
            else:
                cells.append("")
        print(
            f"| {m} | {cublas_by_m[m]:.2f} | "
            f"{' | '.join(cells)} |"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--target", choices=("auto", "rtx5090", "b200"), default="auto")
    ap.add_argument("--markdown", action="store_true", help="Print a markdown summary table.")
    ap.add_argument("--include-swizzle", action="store_true", help="Include diagnostic swizzle CSV rows if present.")
    ap.add_argument("--include-full-output", action="store_true", help="Include Rust full-output diagnostic CSV rows if present.")
    args = ap.parse_args()

    results_dir = select_results_dir(args.target)
    inputs = select_inputs(args.target, results_dir)
    if args.include_full_output:
        inputs.update(FULL_OUTPUT_INPUT)
    if args.include_swizzle:
        inputs.update(SWIZZLE_INPUT)

    rows = build_rows(results_dir, inputs)
    out = args.out or (results_dir / "gemm_cublas_fraction.csv")
    write_csv(out, rows)
    if args.markdown:
        print_per_size(rows, inputs)
        print()
        print_summary(rows, inputs)
    else:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

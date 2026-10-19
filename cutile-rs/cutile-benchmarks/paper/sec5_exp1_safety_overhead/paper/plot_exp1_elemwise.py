# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.1 element-wise safety-overhead results.

Reads elemwise_rust_runtime_results.csv (isolated, CUDA-event timed;
single file with a `config` column of "optimized"/"safe"/"static") and
writes figures/generated/exp1_elemwise.pdf by default.

Usage:
    python3 plot_exp1_elemwise.py
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EXP1 = HERE.parent
BENCHMARKS = EXP1.parent
REPO = BENCHMARKS.parent
B200_RESULTS_DIR = HERE / "results" / "b200"

TARGETS = {
    "rtx5090": {
        "results_dir": HERE / "results" / "rtx5090",
        "out": REPO / "figures" / "generated" / "exp1_elemwise_rtx5090.pdf",
        "roofline_gb_s": None,
    },
    "b200": {
        "results_dir": B200_RESULTS_DIR,
        "out": REPO / "figures" / "generated" / "exp1_elemwise.pdf",
        "roofline_gb_s": 7680.0,
    },
}


def read_roofline_tb_s(override_gb_s=None):
    """Theoretical peak device-memory bandwidth in TB/s.

    `run_elemwise.sh` exports this from `query_nominal_memory_bandwidth.py`.
    For direct plot runs, query NVML with the same helper. If no peak
    value is available, the plot omits the reference line.
    """
    if override_gb_s is not None:
        return float(override_gb_s) / 1000.0
    env = (
        os.environ.get("ELEM_MEM_ROOFLINE_GB_S")
        or os.environ.get("MEM_ROOFLINE_GB_S")
        or os.environ.get("ELEM_NOMINAL_MEM_BW_GB_S")
        or os.environ.get("NOMINAL_MEM_BW_GB_S")
    )
    if env:
        return float(env) / 1000.0
    try:
        sys.path.insert(0, str(BENCHMARKS / "tools"))
        import query_nominal_memory_bandwidth as nominal_bw

        info = nominal_bw.query(int(os.environ.get("GPU_INDEX", "0")))
        transfers = float(os.environ.get("MEM_TRANSFERS_PER_CLOCK", "2.0"))
        return (
            float(info["max_memory_clock_mhz"])
            * transfers
            * (float(info["memory_bus_width_bits"]) / 8.0)
            / 1000.0
            / 1000.0
        )
    except Exception:
        pass
    return None


CONFIG_LABELS = {
    "python":    "Python",
    "optimized": "unsafe Rust",
    "safe":      "Rust",
    "static":    "Rust static",
}
STYLE = {
    "python":    dict(color="#E69F00", marker="s"),
    "optimized": dict(color="#009E73", marker="^"),
    "safe":      dict(color="#0072B2", marker="o"),
    "static":    dict(color="#D55E00", marker="D"),
}


def read_runtime(results_dir):
    """Returns {config -> [(n, tbs, lo_tbs, hi_tbs)]} sorted by n.
    Bandwidth reported in TB/s. Rust CSV has per-variant rows keyed
    by `config`; Python CSV is a single-variant baseline merged under
    key "python".
    """
    runtime_csv = results_dir / "elemwise_rust_runtime_results.csv"
    python_csv = results_dir / "elemwise_python_results.csv"
    by_cfg = defaultdict(list)
    if runtime_csv.exists():
        with runtime_csv.open() as f:
            for r in csv.DictReader(f):
                cfg = r["config"]
                n = int(r["N"])
                median_us = float(r["median_us"])
                bytes_transferred = 6.0 * n
                median_tbs = bytes_transferred / (median_us / 1e6) / 1e12
                p25_us = float(r.get("p25_us", median_us))
                p75_us = float(r.get("p75_us", median_us))
                hi_tbs = bytes_transferred / (p25_us / 1e6) / 1e12
                lo_tbs = bytes_transferred / (p75_us / 1e6) / 1e12
                by_cfg[cfg].append((n, median_tbs, lo_tbs, hi_tbs))
    if python_csv.exists():
        with python_csv.open() as f:
            for r in csv.DictReader(f):
                n = int(r["N"])
                bytes_transferred = 6.0 * n
                # Python CSV uses seconds; Rust CSV uses us. Branch on
                # whichever column is present.
                if "median_s" in r:
                    median_s = float(r["median_s"])
                    min_s = float(r.get("min_s", median_s))
                    max_s = float(r.get("max_s", median_s))
                    median_tbs = bytes_transferred / median_s / 1e12
                    hi_tbs = bytes_transferred / min_s / 1e12
                    lo_tbs = bytes_transferred / max_s / 1e12
                else:
                    median_us = float(r["median_us"])
                    median_tbs = bytes_transferred / (median_us / 1e6) / 1e12
                    p25_us = float(r.get("p25_us", median_us))
                    p75_us = float(r.get("p75_us", median_us))
                    hi_tbs = bytes_transferred / (p25_us / 1e6) / 1e12
                    lo_tbs = bytes_transferred / (p75_us / 1e6) / 1e12
                by_cfg["python"].append((n, median_tbs, lo_tbs, hi_tbs))
    for cfg in by_cfg:
        by_cfg[cfg].sort()
    return by_cfg


def plot_runtime(ax, runtime, sol_tb_s):
    plotted = []
    if sol_tb_s is not None:
        plotted.append(sol_tb_s)
        ax.axhline(sol_tb_s, color="#888888", linestyle=":",
                   linewidth=0.9, zorder=0,
                   label="Peak")

    for cfg in ("python", "optimized", "safe"):
        rows = runtime.get(cfg)
        if not rows:
            continue
        xs = [n for n, *_ in rows]
        ys = [tbs for _, tbs, *_ in rows]
        los = [tbs - lo for _, tbs, lo, _ in rows]
        his = [hi - tbs for _, tbs, _, hi in rows]
        plotted.extend(ys)
        plotted.extend(lo for _, _, lo, _ in rows)
        plotted.extend(hi for _, _, _, hi in rows)
        ax.errorbar(xs, ys, yerr=[los, his], label=CONFIG_LABELS[cfg],
                    capsize=1.3, elinewidth=0.5, **STYLE[cfg])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$N$ (elements)")
    ax.set_ylabel("TB/s (f16, memory)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    ax.set_xticks([2**20, 2**24, 2**28])
    ax.set_xticklabels([r"$2^{20}$", r"$2^{24}$", r"$2^{28}$"])
    ax.set_xticks([2**i for i in range(20, 29)], minor=True)
    ax.set_xlim(2**20 * 0.92, 2**28 * 1.08)
    if plotted:
        hi = max(plotted)
        lo = min(plotted)
        pad = max((hi - lo) * 0.08, hi * 0.03)
        ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03),
              handlelength=1.8, borderpad=0.25,
              labelspacing=0.3, handletextpad=0.4,
              framealpha=0.85, frameon=True, edgecolor="none")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=os.environ.get("EXP1_PLOT_TARGET", "b200"),
    )
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--roofline-gb-s", type=float)
    return ap.parse_args()


def main():
    args = parse_args()
    target = TARGETS[args.target]
    results_dir = args.results_dir or Path(os.environ.get("RESULTS_DIR", target["results_dir"]))
    out = args.out or target["out"]
    roofline_gb_s = args.roofline_gb_s
    if roofline_gb_s is None:
        roofline_gb_s = target["roofline_gb_s"]
    runtime = read_runtime(results_dir)
    sol_tb_s = read_roofline_tb_s(roofline_gb_s)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.33, 1.8))
    plot_runtime(ax, runtime, sol_tb_s)

    fig.tight_layout(pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

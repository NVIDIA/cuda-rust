# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot the merged 6b+6c experiment (Option A): bimodal-GEMM queue
with per-task host work W, fixed S, three modes (serial / threaded /
async). X = W (us, log). Y = throughput (work units / ms).

Reads bimodal_gemms_rust/results_bimodal_w.csv and writes
figures/generated/exp3_bimodal_w.pdf.

Story this plot tells:
  - W=0 (left edge): same as 6c at S=PIN_S — threaded ≈ async at the
    plateau, both well above serial.
  - W ≈ GPU time per task: async pulls ahead because its single host
    thread can submit launches on other streams while one task spins.
  - W large: each thread is dominated by spin time; threaded scales
    with S threads, async-T=1 caps at single-thread throughput.
    async-T=4 sits between.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
REPO = HERE.parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "bimodal_gemms_rust"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO / "figures" / "generated"))
CSV = RESULTS_DIR / "results_bimodal_w.csv"
OUT = FIGURES_DIR / "exp3_bimodal_w.pdf"

# Wong palette consistent with the other plots.
COLORS = {
    "serial":          "#D55E00",
    "threaded":        "#E69F00",
    "async_t1":        "#009E73",
}
MARKERS = {
    "serial":          "o",
    "threaded":        "s",
    "async_t1":        "^",
}


def read_csv(path: Path):
    """Returns rows as list of dicts with throughput in WU/ms."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "mode":          r["mode"],
                "streams":       int(r["streams"]),
                "tokio_threads": int(r["tokio_threads"]),
                "w_host_us":     int(r["w_host_us"]),
                "tp_median":     float(r["tp_median"]) / 1000.0,
                "tp_min":        float(r["tp_min"])    / 1000.0,
                "tp_max":        float(r["tp_max"])    / 1000.0,
            })
    return rows


def _series(rows, predicate):
    """Returns sorted [(w, med, lo, hi)] for rows matching predicate."""
    out = [(r["w_host_us"], r["tp_median"], r["tp_min"], r["tp_max"])
           for r in rows if predicate(r)]
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run ./run_bimodal_w.sh first.")
    rows = read_csv(CSV)

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         13,
        "axes.labelsize":    13,
        "axes.titlesize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })
    fig, ax = plt.subplots(figsize=(3.33, 2.3))

    series = {
        "serial":      _series(rows, lambda r: r["mode"] == "serial"),
        "threaded":    _series(rows, lambda r: r["mode"] == "threaded"),
        "async_t1":    _series(rows, lambda r: r["mode"] == "async" and r["tokio_threads"] == 1),
    }

    pretty = {
        "serial":      "serial",
        "threaded":    "threaded",
        "async_t1":    "async",
    }
    order = ["serial", "threaded", "async_t1"]

    # X axis is W (microseconds). Substitute W=0 with a small number
    # so log-x doesn't drop the leftmost point.
    def xs_with_zero_floor(ws):
        return [max(w, 1) for w in ws]

    for k in order:
        s = series[k]
        if not s:
            continue
        ws  = [t[0] for t in s]
        med = [t[1] for t in s]
        lo  = [t[2] for t in s]
        hi  = [t[3] for t in s]
        xs = xs_with_zero_floor(ws)
        ax.plot(xs, med, marker=MARKERS[k], markersize=4,
                color=COLORS[k], linewidth=1.2, label=pretty[k])
        ax.fill_between(xs, lo, hi, color=COLORS[k], alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Host work $W$ ($\mu$s)")
    ax.set_ylabel("Throughput (WU/ms)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper right",
              handlelength=1.8, labelspacing=0.3)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

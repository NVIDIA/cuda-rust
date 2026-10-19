# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.2 Part B — async vs sync throughput under host-side work.

Reads launch_overhead_rust/results_part_b.csv (schema
`mode,n_ops,w_us,tp_median,tp_min,tp_p25,tp_p75,tp_max,n_samples`) and
writes figures/generated/exp2_async_throughput.pdf.

The figure shows iterations per second (y) against per-iteration host
work W (x), one line per execution mode. The line is the *max*
observed throughput over samples (intrinsic capacity unperturbed by
preemption on a shared machine); the shaded band spans p25 to max so
the jitter envelope is visible.

Usage:
    python3 plot_part_b.py
"""

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
REPO = HERE.parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "launch_overhead_rust"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO / "figures" / "generated"))
CSV = RESULTS_DIR / "results_part_b.csv"
OUT = FIGURES_DIR / "exp2_async_throughput.pdf"

ORDER = ["async", "sync"]

PRETTY = {
    "sync":  "sync",
    "async": "async",
}

COLORS = {
    "sync":  "#E69F00",  # gold — consistent with Fig 6's "sync (chained)"
    "async": "#009E73",  # green — consistent with Fig 6's "async (chained)"
}

MARKERS = {
    "sync":  "s",
    "async": "^",
}


def read_csv(path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    """Returns {mode: [(w_us, tp_line, tp_lo, tp_hi), ...]}, sorted by
    w_us, taking the *last* row per (mode, w_us) when the CSV holds
    multiple runs. Line = max (intrinsic throughput); band spans p25
    to max."""
    latest: dict[tuple[str, int], tuple[float, float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            w = int(row["w_us"])
            latest[(mode, w)] = (
                float(row["tp_max"]),
                float(row["tp_p25"]),
                float(row["tp_max"]),
            )
    out: dict[str, list[tuple[int, float, float, float]]] = {}
    for (mode, w), (line, lo, hi) in latest.items():
        out.setdefault(mode, []).append((w, line, lo, hi))
    for mode in out:
        out[mode].sort(key=lambda t: t[0])
    return out


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run ./run_part_b.sh first.")
    data = read_csv(CSV)

    missing = [m for m in ORDER if m not in data]
    if missing:
        raise SystemExit(f"results_part_b.csv missing modes: {missing}")

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         13,
        "axes.labelsize":    13,
        "axes.titlesize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(3.33, 2.3))

    for mode in ORDER:
        series = data[mode]
        # Use w+1 for the x-axis so W=0 is plottable on log-x.
        xs  = [max(t[0], 1) for t in series]
        ln  = [t[1] for t in series]
        lo  = [t[2] for t in series]
        hi  = [t[3] for t in series]
        ax.plot(
            xs, ln,
            marker=MARKERS[mode], markersize=4,
            color=COLORS[mode], linewidth=1.2,
            label=PRETTY[mode],
        )
        ax.fill_between(xs, lo, hi, color=COLORS[mode], alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Host work $W$ ($\mu$s)")
    ax.set_ylabel("Throughput (iters/s)")

    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.25)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

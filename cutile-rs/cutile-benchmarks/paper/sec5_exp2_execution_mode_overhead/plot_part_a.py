# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.2 — execution-mode scaling.

Reads launch_overhead_rust/results_part_a.csv (schema
`mode,d,n_ops,median_us,min_us,p25_us,p75_us,max_us,n_samples`) and writes
figures/generated/exp2_execmode_latency.pdf.

The plot shows host-visible latency per operation (y) vs pipeline length
in ops N (x), one line per measured schedule. Dividing by N exposes the
asymptotic per-op launch/scheduling cost, so each schedule should flatten
as the fixed per-pipeline component is amortized.

Usage:
    python3 plot_part_a.py
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
CSV = RESULTS_DIR / "results_part_a.csv"
OUT = FIGURES_DIR / "exp2_execmode_latency.pdf"

# Legend order matches the visual top-to-bottom ordering at the right
# edge of the chart, so the reader can match lines to labels by looking
# straight across.
ORDER = ["sync-individual", "async", "sync-chained", "graph"]

PRETTY = {
    "sync-individual": "sync (individual)",
    "sync-chained":    "sync (chained)",
    "async":           "async (chained)",
    "graph":           "graph replay",
}

# Wong-style colorblind-friendly palette, kept consistent with the
# old bar chart and with §5.1 figures.
COLORS = {
    "sync-individual": "#D55E00",  # warm orange — the "loss"
    "sync-chained":    "#E69F00",  # gold
    "async":           "#009E73",  # green
    "graph":           "#0072B2",  # blue  — the "win"
}

MARKERS = {
    "sync-individual": "o",
    "sync-chained":    "s",
    "async":           "^",
    "graph":           "D",
}


def read_csv(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Returns {mode: [(n_ops, min_us_per_op, p75_us_per_op), ...]}, sorted by n_ops,
    taking the *last* row per (mode, n_ops) if the CSV was appended to
    across multiple runs. We use `min_us` as the line value because the
    benchmark machine is shared: min represents the intrinsic API cost
    unperturbed by preemption. The min..p75 band anchored at the line
    shows the one-sided upper jitter envelope."""
    # Clip at N_MAX so the figure focuses on the small-to-medium regime
    # where the mode differences are pedagogically interesting. CSV may
    # still contain larger N from an extended sweep; those rows are
    # ignored by the plot.
    N_MAX = 1000
    latest: dict[tuple[str, int], tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            n = int(row["n_ops"])
            if n > N_MAX:
                continue
            latest[(mode, n)] = (
                float(row["min_us"]) / n,
                float(row["p75_us"]) / n,
            )
    out: dict[str, list[tuple[int, float, float]]] = {}
    for (mode, n), (mn, p75) in latest.items():
        out.setdefault(mode, []).append((n, mn, p75))
    for mode in out:
        out[mode].sort(key=lambda t: t[0])
    return out


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run the harness first (./run.sh).")
    data = read_csv(CSV)

    missing = [m for m in ORDER if m not in data]
    if missing:
        raise SystemExit(
            f"results_part_a.csv missing modes: {missing}; re-run the harness."
        )

    plt.rcParams.update({
        "font.family":       "serif",
        # The PDF is tightly cropped and scaled to 0.90\columnwidth in
        # LaTeX, so source text needs to be slightly smaller than the
        # nominal body size to render near body text in the paper.
        "font.size":         8,
        "axes.labelsize":    8,
        "axes.titlesize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(3.33, 1.8))

    for mode in ORDER:
        series = data[mode]
        ns  = [t[0] for t in series]
        mn  = [t[1] for t in series]
        p75 = [t[2] for t in series]
        ax.plot(
            ns, mn,
            marker=MARKERS[mode], markersize=4,
            color=COLORS[mode], linewidth=1.2,
            label=PRETTY[mode],
        )
        ax.fill_between(ns, mn, p75, color=COLORS[mode], alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Keep the log tick *positions* (the decades are the technically
    # accurate scale) but render them as plain numbers (1, 10, 100, 1000;
    # 1, 10) instead of 10^k, which is hard to read at this size. Drop the
    # trailing ".0" so whole decades print as integers.
    def plain(v, _pos):
        return f"{v:g}"
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(plain))
        axis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"Pipeline length (# chained ops, $N$)")
    ax.set_ylabel(r"Execution time per op ($\mu$s)")

    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=1.0, labelspacing=0.2)

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

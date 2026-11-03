# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.2 Part B2 — bimodal-GEMM work-distribution throughput.

Reads bimodal_gemms_rust/results_bimodal.csv (schema:
mode,streams,tokio_threads,n_work,small_ratio,tp_median,tp_min,tp_max,n_samples)
and writes two figures:

  figures/generated/exp3_bimodal_throughput.pdf
      Main scaling plot. X = stream count S. Three lines:
      serial (dashed reference), threaded, async (T=1).

  figures/generated/exp3_bimodal_tokio.pdf
      Tokio-thread sweep at each S. X = tokio thread count T.
      One line per S value (for async mode only). Shows that
      additional tokio threads do not materially improve
      throughput — the plateau is GPU-bound, not executor-bound.

Y axis: work units per millisecond.
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
CSV = RESULTS_DIR / "results_bimodal.csv"
OUT_MAIN  = FIGURES_DIR / "exp3_bimodal_throughput.pdf"
OUT_TOKIO = FIGURES_DIR / "exp3_bimodal_tokio.pdf"

# Wong palette — consistent with Figs 6/7.
COLORS = {
    "serial":          "#D55E00",
    "threaded":        "#E69F00",
    "async_t1":        "#009E73",  # green
}
MARKERS = {
    "serial":          "o",
    "threaded":        "s",
    "async_t1":        "^",
}


def read_csv(path: Path):
    """Returns rows as list of dicts with tp_* converted to WU/ms."""
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "mode":    row["mode"],
                "streams": int(row["streams"]),
                "tokio_threads": int(row["tokio_threads"]),
                "tp_median": float(row["tp_median"]) / 1000.0,
                "tp_min":    float(row["tp_min"])    / 1000.0,
                "tp_max":    float(row["tp_max"])    / 1000.0,
            })
    return rows


def _rcparams(font_size=13, axes_size=13, tick_size=11, legend_size=10):
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         font_size,
        "axes.labelsize":    axes_size,
        "axes.titlesize":    axes_size,
        "xtick.labelsize":   tick_size,
        "ytick.labelsize":   tick_size,
        "legend.fontsize":   legend_size,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })


def plot_main(rows) -> None:
    """Main scaling figure: S on x, three lines (serial, threaded, async T=1)."""
    _rcparams()
    fig, ax = plt.subplots(figsize=(3.33, 2.3))

    # Serial: flat baseline across the sweep range.
    serial_tp = next(
        (r["tp_median"] for r in rows if r["mode"] == "serial"), None
    )
    multi_s = sorted(
        {r["streams"] for r in rows if r["mode"] in ("threaded", "async")}
    )
    if serial_tp is not None and multi_s:
        ax.plot(multi_s, [serial_tp] * len(multi_s),
                color=COLORS["serial"], linestyle="--", linewidth=1.0,
                alpha=0.7, zorder=1,
                label="serial")

    # Threaded: sweep S, one line.
    thr = sorted(
        [(r["streams"], r["tp_median"], r["tp_min"], r["tp_max"])
         for r in rows if r["mode"] == "threaded"],
        key=lambda t: t[0],
    )
    if thr:
        xs, med, lo, hi = zip(*thr)
        ax.plot(xs, med, marker=MARKERS["threaded"], markersize=4,
                color=COLORS["threaded"], linewidth=1.2,
                label="threaded")
        ax.fill_between(xs, lo, hi, color=COLORS["threaded"],
                        alpha=0.15, linewidth=0)

    # Async T=1: single host thread, S streams.
    asy1 = sorted(
        [(r["streams"], r["tp_median"], r["tp_min"], r["tp_max"])
         for r in rows if r["mode"] == "async" and r["tokio_threads"] == 1],
        key=lambda t: t[0],
    )
    if asy1:
        xs, med, lo, hi = zip(*asy1)
        ax.plot(xs, med, marker=MARKERS["async_t1"], markersize=4,
                color=COLORS["async_t1"], linewidth=1.2,
                label="async")
        ax.fill_between(xs, lo, hi, color=COLORS["async_t1"],
                        alpha=0.15, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_xticks(multi_s if multi_s else [1, 2, 4, 8, 16, 32])
    ax.set_xticklabels([str(s) for s in (multi_s if multi_s else [1, 2, 4, 8, 16, 32])])
    ax.set_xlabel(r"Streams $S$")
    ax.set_ylabel("Throughput (WU/ms)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left",
              handlelength=1.8, labelspacing=0.3)

    fig.tight_layout(pad=0.2)
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN, bbox_inches="tight")
    print(f"wrote {OUT_MAIN}")


def plot_tokio(rows) -> None:
    """Tokio-thread sweep: X = T, one line per S, async only."""
    _rcparams(font_size=10, axes_size=10, tick_size=9, legend_size=7)
    fig, ax = plt.subplots(figsize=(3.33, 2.15))

    # Group async rows by stream count.
    by_s: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for r in rows:
        if r["mode"] != "async":
            continue
        by_s[r["streams"]].append(
            (r["tokio_threads"], r["tp_median"], r["tp_min"], r["tp_max"])
        )
    for s in by_s:
        by_s[s].sort(key=lambda t: t[0])

    # Reference: threaded at the same S = S (matches thread count; our
    # comparison baseline). Pull from rows.
    thr_by_s = {
        r["streams"]: r["tp_median"]
        for r in rows if r["mode"] == "threaded"
    }

    # Serial reference (flat).
    serial_tp = next(
        (r["tp_median"] for r in rows if r["mode"] == "serial"), None
    )

    # Distinct colorblind-friendly colors per S value, picked from
    # an extended Wong-style palette. Avoids clashing with the main
    # plot's threaded (gold) and async (green).
    s_list = sorted(by_s.keys())
    s_palette = {
        1:  "#CC79A7",  # pink
        2:  "#56B4E9",  # sky blue
        4:  "#009E73",  # green
        8:  "#F0E442",  # yellow
        16: "#0072B2",  # dark blue
        32: "#D55E00",  # vermilion
    }
    s_markers = {
        1: "o", 2: "s", 4: "^", 8: "D", 16: "v", 32: "X",
    }

    for s in s_list:
        series = by_s[s]
        ts, med, lo, hi = zip(*series)
        c = s_palette.get(s, "#888888")
        m = s_markers.get(s, "o")
        ax.plot(ts, med, marker=m, markersize=4,
                color=c, linewidth=1.2,
                label=f"$S{{=}}{s}$")
        ax.fill_between(ts, lo, hi, color=c, alpha=0.12, linewidth=0)

    # Dashed horizontal refs: serial baseline and best-threaded value.
    if serial_tp is not None:
        ax.axhline(serial_tp, color=COLORS["serial"], linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label="serial (reference)")
    if thr_by_s:
        best_thr = max(thr_by_s.values())
        ax.axhline(best_thr, color=COLORS["threaded"], linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label=f"threaded best ({best_thr:.0f} wu/ms)")

    ax.set_xscale("log", base=2)
    t_ticks = sorted({t for s in by_s for t, *_ in by_s[s]})
    ax.set_xticks(t_ticks if t_ticks else [1, 2, 4, 8, 16])
    ax.set_xticklabels([str(t) for t in (t_ticks if t_ticks else [1, 2, 4, 8, 16])])
    ax.set_xlabel(r"Runtime threads $T$")
    ax.set_ylabel("Throughput (WU/ms)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.23),
              handlelength=1.4, labelspacing=0.25,
              columnspacing=0.8, ncol=4)

    fig.tight_layout(pad=0.2)
    OUT_TOKIO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TOKIO, bbox_inches="tight")
    print(f"wrote {OUT_TOKIO}")


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}; run ./run_bimodal.sh first.")
    rows = read_csv(CSV)
    plot_main(rows)
    plot_tokio(rows)


if __name__ == "__main__":
    main()

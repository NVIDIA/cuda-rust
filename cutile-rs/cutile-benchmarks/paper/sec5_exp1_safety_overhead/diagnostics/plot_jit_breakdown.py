# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT overhead panel for Figure 5c.

Plots first-use GEMM JIT overhead for the checked Rust path, grouped by
BM and stacked by JIT stage. This keeps the phase breakdown while avoiding
the static-vs-dynamic comparison that is no longer part of the paper story.

Reads CUTILE_JIT_TIMING log files produced by running *_rust --jit with the
env var set.

Usage:
    python3 plot_jit_breakdown.py
    python3 plot_jit_breakdown.py --quiet
"""

import argparse
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
EXP1 = HERE.parent
REPO = EXP1.parents[1]
OUT = REPO / "figures" / "generated" / "exp1_jit_breakdown.pdf"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", EXP1 / "diagnostics" / "results" / "jit"))
GEMM_LOG = RESULTS_DIR / "gemm_rust_jit_timing.log"

# This is the path called "Rust" in Figure 5b.
PLOT_FUNCTION = "gemm_safe"

LINE_RE = re.compile(
    r"CUTILE_JIT_TIMING\s+module=(\S+)\s+function=(\S+)\s+key=\S+\s+"
    r"stage1_ms=([\d.]+)\s+stage2_ms=([\d.]+)\s+stage3_ms=([\d.]+)\s+generics=(\S+)"
)

STAGE_ORDER = ("back-end", "front-end", "module-load")
STAGE_COLORS = {
    "back-end": "#4C72B0",
    "front-end": "#DD8452",
    "module-load": "#55A868",
}


def parse_gemm_jit_by_bm(path: Path):
    buckets = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run benchmarks/exp1_safety_overhead/diagnostics/run_jit_breakdown.sh"
        )

    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        _mod, fn, stage1, stage2, stage3, generics = m.groups()
        if fn != PLOT_FUNCTION:
            continue

        parts = generics.split(",")
        if len(parts) < 2:
            continue
        bm = int(parts[1])

        # CUTILE_JIT_TIMING reports stage1=front-end, stage2=back-end,
        # stage3=module-load.
        buckets[bm]["front-end"].append(float(stage1))
        buckets[bm]["back-end"].append(float(stage2))
        buckets[bm]["module-load"].append(float(stage3))

    return {
        bm: {stage: vals for stage, vals in stages.items()}
        for bm, stages in sorted(buckets.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    by_bm = parse_gemm_jit_by_bm(GEMM_LOG)
    if not by_bm:
        raise SystemExit(f"no {PLOT_FUNCTION!r} rows found in {GEMM_LOG}")

    bms = list(by_bm)
    means = {
        stage: [statistics.mean(by_bm[bm].get(stage, [0.0])) for bm in bms]
        for stage in STAGE_ORDER
    }
    totals = [sum(means[stage][i] for stage in STAGE_ORDER) for i in range(len(bms))]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    # Figure 5 places this panel in a 0.17\textwidth subfigure next to
    # two 0.40\textwidth panels whose generated bboxes are about
    # 0.63 as tall as they are wide. 0.40 * 0.63 / 0.17 ~= 1.49, so
    # preserving this native aspect ratio makes the displayed panel
    # mathematically match the height of Figures 5a and 5b.
    fig, ax = plt.subplots(figsize=(1.40, 1.8))
    xs = range(len(bms))
    bottoms = [0.0] * len(bms)
    for stage in STAGE_ORDER:
        ax.bar(
            xs,
            means[stage],
            bottom=bottoms,
            width=0.55,
            color=STAGE_COLORS[stage],
            edgecolor="#222222",
            linewidth=0.35,
            label=stage,
        )
        bottoms = [b + v for b, v in zip(bottoms, means[stage])]

    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(bm) for bm in bms])
    ax.set_xlabel("BM")
    ax.set_ylabel("JIT overhead (ms)")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(totals) * 1.40)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.14),
        handlelength=0.9,
        borderpad=0.25,
        labelspacing=0.2,
        handletextpad=0.3,
        frameon=False,
    )

    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    if not args.quiet:
        print("\nGEMM checked Rust JIT overhead by BM:")
        for i, bm in enumerate(bms):
            parts = ", ".join(f"{stage}={means[stage][i]:.2f} ms" for stage in STAGE_ORDER)
            print(f"  BM={bm:>3}: total={totals[i]:7.2f} ms ({parts})")


if __name__ == "__main__":
    main()

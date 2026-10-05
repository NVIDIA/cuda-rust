# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.3 Grout inference-perf sweep, two-panel:

  Left  (prefill): latency (ms) vs pp at tg=36, from pp sweep
  Right (request): generated-token throughput (tok/s) vs tg at pp=18,
                   from tg sweep

Prefill latency comes from direct per-request measurements when the
engine reports them in run.jsonl. Engines without a direct prefill field
fall back to cross-sweep derivation:
  prefill_ms = e2e_ms − tg × decode_ms_per_tok
using each engine's `decode_fit_ms_per_tok` from the tg-sweep fit.

Writes figures/generated/exp3_grout_sweep.pdf (figure* full-width).
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
REPO = HERE.parents[1]

TARGETS = {
    "rtx5090": {
        "tg_csv": HERE / "data" / "sweep_tg_20260508_111703_plus_115728_tg8192" / "aggregate.csv",
        "pp_csv": HERE / "data" / "sweep_pp_20260508_114340" / "aggregate.csv",
        "pp_runjsonl": HERE / "data" / "sweep_pp_20260508_114340" / "run.jsonl",
        "out": REPO / "figures" / "generated" / "exp3_grout_sweep.pdf",
        "title": "NVIDIA GeForce RTX 5090 / Qwen3-4B",
    },
    "b200": {
        "tg_csv": HERE / "data" / "b200_qwen3_32b" / "tg_sweep_pp18_8k" / "aggregate.csv",
        "pp_csv": HERE / "data" / "b200_qwen3_32b" / "pp_sweep_tg36_8k" / "aggregate.csv",
        "pp_runjsonl": HERE / "data" / "b200_qwen3_32b" / "pp_sweep_tg36_8k" / "run.jsonl",
        "out": REPO / "figures" / "generated" / "exp3_grout_sweep_b200.pdf",
        "title": "NVIDIA B200 / Qwen3-32B",
    },
}

# (engine_key, display_label, color, marker, linewidth, zorder)
# Grout is drawn last and slightly heavier so it reads as the subject.
ENGINES = [
    ("vllm",      "vLLM",      "#009E73", "^", 1.3, 2),
    ("sglang",    "SGLang",    "#0072B2", "s", 1.3, 2),
    ("grout-v2",  "Grout",     "#D55E00", "o", 1.9, 3),
]


def read_csv(path: Path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def decode_rate_per_engine(tg_rows):
    """Return {engine: decode_ms_per_tok} from the tg-sweep linear fit."""
    out = {}
    for row in tg_rows:
        e = row["engine"]
        if e in out:
            continue
        v = row.get("decode_fit_ms_per_tok", "") or row.get("derived_decode_ms_per_tok", "")
        if v and v.lower() != "nan":
            out[e] = float(v)
    return out


def decode_series(tg_rows, engine):
    """(tgs, request_gen_tps_med, tps_lo, tps_hi) sorted by tg for engine."""
    cell = [r for r in tg_rows if r["engine"] == engine]
    cell.sort(key=lambda r: int(r["tg"]))
    tgs = [int(r["tg"]) for r in cell]
    gen = [int(r["gen_tokens"]) for r in cell]
    med = [g / (float(r["e2e_median_ms"]) / 1000.0) for g, r in zip(gen, cell)]
    lo  = [g / (float(r["e2e_p75_ms"])    / 1000.0) for g, r in zip(gen, cell)]
    hi  = [g / (float(r["e2e_p25_ms"])    / 1000.0) for g, r in zip(gen, cell)]
    return tgs, med, lo, hi


def request_roofline_series(tg_rows):
    """
    (tgs, nominal_tps, effective_tps) sorted by tg.

    The paper-facing roof sets prefill time to zero and uses only the
    HBM decode payload term: tg / T_decode_roof. The aggregate CSV also
    includes a nonzero ideal prefill term, so compute directly from the
    decode-time columns when they are present.
    """
    by_tg = {}
    for row in tg_rows:
        tg = int(row["tg"])
        if tg in by_tg:
            continue
        gen = int(row["gen_tokens"])
        nominal_decode_ms = row.get("request_nominal_roofline_decode_ms", "")
        effective_decode_ms = row.get("request_effective_roofline_decode_ms", "")
        if nominal_decode_ms and effective_decode_ms:
            nominal = gen / (float(nominal_decode_ms) / 1000.0)
            effective = gen / (float(effective_decode_ms) / 1000.0)
        elif row.get("request_nominal_roofline_tps") and row.get("request_effective_roofline_tps"):
            nominal = float(row["request_nominal_roofline_tps"])
            effective = float(row["request_effective_roofline_tps"])
        else:
            continue
        by_tg[tg] = (nominal, effective)
    tgs = sorted(by_tg)
    return tgs, [by_tg[t][0] for t in tgs], [by_tg[t][1] for t in tgs]


def prefill_series(pp_rows, pp_reps, engine, decode_ms_per_tok):
    """
    (pps, ms_med, ms_lo, ms_hi) sorted by pp for engine.

    Preferred source: per-rep `prefill_ms` from run.jsonl, with median +
    p25/p75 computed over the reps. Used for any engine that reports
    prefill_ms (grout-v2, llama.cpp, sglang in the current sweep).

    Fallback for engines without direct prefill (vLLM): derive from
    cross-sweep as e2e_ms − tg × decode_ms_per_tok, with residuals
    below 0.5 ms clamped to the log-axis floor.
    """
    cell = [r for r in pp_rows if r["engine"] == engine]
    cell.sort(key=lambda r: int(r["pp"]))
    pps = [int(r["pp"]) for r in cell]
    tgs = [int(r["tg"]) for r in cell]

    # Group per-rep prefill_ms by pp for this engine.
    by_pp: dict[int, list[float]] = {}
    for rec in pp_reps:
        if rec["engine"] != engine:
            continue
        pf = rec.get("prefill_ms")
        if pf is None:
            continue
        by_pp.setdefault(int(rec["pp"]), []).append(float(pf))

    ms_med, ms_lo, ms_hi = [], [], []
    for pp, tg, row in zip(pps, tgs, cell):
        reps = by_pp.get(pp)
        if reps:
            ms_med.append(percentile(reps, 50))
            ms_lo .append(percentile(reps, 25))
            ms_hi .append(percentile(reps, 75))
        else:
            rate = decode_ms_per_tok.get(engine)
            if rate is None:
                return pps, [], [], []
            FLOOR = 0.5
            ms_med.append(max(float(row["e2e_median_ms"]) - tg * rate, FLOOR))
            ms_lo .append(max(float(row["e2e_p25_ms"])    - tg * rate, FLOOR))
            ms_hi .append(max(float(row["e2e_p75_ms"])    - tg * rate, FLOOR))

    return pps, ms_med, ms_lo, ms_hi


def _rcparams():
    # Each generated PDF has two horizontal panels and is included as a
    # half-width subfigure, so source text is larger than final body text.
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         16,
        "axes.labelsize":    16,
        "axes.titlesize":    16,
        "xtick.labelsize":   13,
        "ytick.labelsize":   13,
        "legend.fontsize":   13,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "pdf.fonttype":      42,
    })


def plot(
    tg_rows,
    pp_rows,
    pp_runjsonl: Path,
    out: Path,
    roofline: str,
    title: str | None,
) -> None:
    _rcparams()
    rates = decode_rate_per_engine(tg_rows)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.85, 2.15),
        gridspec_kw={
            "wspace": 0.55,
            "top": 0.68 if title else 0.76,
            "bottom": 0.22,
            "left": 0.09,
            "right": 0.99,
        },
    )
    if title:
        fig.suptitle(title, y=1.10, fontsize=18, fontweight="semibold")

    pp_reps = read_jsonl(pp_runjsonl)

    # Left panel: prefill latency (TTFT) vs pp. Down is better, which
    # is the standard convention for prefill / time-to-first-token.
    pp_ticks = sorted({int(row["pp"]) for row in pp_rows})
    prefill_values = []
    for engine, label, color, marker, lw, zorder in ENGINES:
        pps, ms_med, ms_lo, ms_hi = prefill_series(pp_rows, pp_reps,
                                                    engine, rates)
        if not pps:
            continue
        prefill_values.extend(ms_lo)
        prefill_values.extend(ms_med)
        prefill_values.extend(ms_hi)
        axL.plot(pps, ms_med, marker=marker, markersize=7, color=color,
                 linewidth=lw, label=label, zorder=zorder)
        axL.fill_between(pps, ms_lo, ms_hi, color=color,
                         alpha=0.15, linewidth=0, zorder=zorder - 1)
    axL.set_xscale("log", base=2)
    axL.set_yscale("log")
    axL.set_xticks(pp_ticks)
    axL.set_xticklabels([str(t) for t in pp_ticks])
    if prefill_values and max(prefill_values) >= 250:
        prefill_ticks = [10, 30, 100, 300, 1000]
    else:
        prefill_ticks = [5, 10, 30, 100]
    axL.set_yticks(prefill_ticks)
    axL.set_yticklabels([str(t) for t in prefill_ticks])
    axL.minorticks_off()
    axL.set_xlabel(r"Prompt tokens $p_p$")
    axL.set_ylabel("ms")
    axL.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.4)

    # Right panel: request-level generated-token throughput vs tg.
    throughput_values = []
    tg_ticks = sorted({int(row["tg"]) for row in tg_rows})
    for engine, label, color, marker, lw, zorder in ENGINES:
        tgs, med, lo, hi = decode_series(tg_rows, engine)
        if not tgs:
            continue
        throughput_values.extend(med)
        axR.plot(tgs, med, marker=marker, markersize=7, color=color,
                 linewidth=lw, label=label, zorder=zorder)
        axR.fill_between(tgs, lo, hi, color=color,
                         alpha=0.15, linewidth=0, zorder=zorder - 1)
    roof_tgs, nominal_roof, effective_roof = request_roofline_series(tg_rows)
    if roof_tgs and roofline != "none":
        if roofline in ("both", "nominal"):
            throughput_values.extend(nominal_roof)
        if roofline in ("both", "effective"):
            throughput_values.extend(effective_roof)
        if roofline in ("both", "nominal"):
            axR.plot(roof_tgs, nominal_roof, color="#666666", linestyle="--",
                     linewidth=0.9, zorder=1)
            axR.text(roof_tgs[-1], nominal_roof[-1] + 1.0, "nominal roof",
                     ha="right", va="bottom", fontsize=10, color="#555555")
        if roofline in ("both", "effective"):
            axR.plot(roof_tgs, effective_roof, color="#888888", linestyle=":",
                     linewidth=0.9, zorder=1)
            axR.text(roof_tgs[-1], effective_roof[-1] + 1.0, "85% BW ref.",
                     ha="right", va="bottom", fontsize=10, color="#666666")
    axR.set_xscale("log", base=2)
    axR.set_xticks(tg_ticks)
    axR.set_xticklabels([str(t) for t in tg_ticks])
    axR.minorticks_off()
    axR.set_xlabel(r"Generated tokens $t_g$")
    axR.set_ylabel("tokens/s")
    if throughput_values:
        lo = min(throughput_values)
        hi = max(throughput_values)
        pad = max((hi - lo) * 0.12, hi * 0.04)
        axR.set_ylim(max(0, lo - pad), hi + pad)
    axR.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.4)

    # Shared legend above both panels, one row, spread across the
    # full figure width for readability. Grout is bolded to mark it
    # as our work and placed first.
    handles, labels = axR.get_legend_handles_labels()
    desired_order = ["Grout", "SGLang", "vLLM", "llama.cpp"]
    idx = [labels.index(l) for l in desired_order if l in labels]
    handles = [handles[i] for i in idx]
    labels  = [labels[i]  for i in idx]
    leg = fig.legend(handles, labels, loc="upper center",
                     ncol=len(labels), frameon=False,
                     bbox_to_anchor=(0.5, 0.96 if title else 1.0),
                     handlelength=1.6, handletextpad=0.4,
                     columnspacing=1.1)
    for text in leg.get_texts():
        if text.get_text() == "Grout":
            text.set_fontweight("bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def parse_args():
    target = os.environ.get("GROUT_PLOT_TARGET", "rtx5090")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), default=target)
    ap.add_argument("--tg-csv", type=Path)
    ap.add_argument("--pp-csv", type=Path)
    ap.add_argument("--pp-runjsonl", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--roofline",
        choices=("both", "nominal", "effective", "none"),
        default="none",
    )
    ap.add_argument(
        "--show-title",
        action="store_true",
        help="Add a figure title naming the target GPU/model combo.",
    )
    ap.add_argument(
        "--title",
        help="Override the default title text; implies --show-title.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    defaults = TARGETS[args.target]
    tg_csv = args.tg_csv or defaults["tg_csv"]
    pp_csv = args.pp_csv or defaults["pp_csv"]
    pp_runjsonl = args.pp_runjsonl or defaults["pp_runjsonl"]
    out = args.out or defaults["out"]
    title = args.title or (defaults["title"] if args.show_title else None)
    tg_rows = read_csv(tg_csv)
    pp_rows = read_csv(pp_csv)
    plot(tg_rows, pp_rows, pp_runjsonl, out, args.roofline, title)

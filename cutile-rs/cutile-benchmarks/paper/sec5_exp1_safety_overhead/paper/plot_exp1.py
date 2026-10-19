# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot §5.1 GEMM safety-overhead results.

Reads B200 CSVs from paper/results/b200 by default and writes
figures/generated/exp1_safety_overhead.pdf in the paper repo.

The B200 plot compares the safe mapped persistent Rust GEMM against
the raw-pointer persistent variant, cuTile Python, and cuBLAS.

Usage:
    python3 plot_exp1.py
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
EXP1 = HERE.parent
REPO = EXP1.parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "results" / "b200"))
OUT = REPO / "figures" / "generated" / "exp1_safety_overhead.pdf"

# NVIDIA GeForce RTX 5090 f16 tensor-core peak.
# Formula mirrors cutile-examples/src/lib.rs:93-125
# (blackwell_tensorcore_sol_tflops / rtx_5090_tensorcore_f16_sol_tflops_at):
#
#   peak = num_sms * tensor_cores_per_sm * issue_freq * flops_per_op * clock_hz * 1e-12
#
# Fastest f16 MMA op on Blackwell is HMMA.16816.F16:
#   - issue freq: 1 / 16 cycles
#   - flops per op: 2 * 16 * 8 * 16 = 4096
# NVIDIA GeForce RTX 5090 has 170 SMs, 4 tensor cores per SM.
NUM_SMS = 170
TENSOR_CORES_PER_SM = 4
HMMA_ISSUE_FREQ = 1.0 / 16.0
HMMA_FLOPS_PER_OP = 2 * 16 * 8 * 16


def _observe_sm_clock_under_load_mhz(fallback=2400.0):
    """Return the SM clock observed while the GPU is under sustained load.

    nvidia-smi `clocks.max.sm` always reports the device's physical
    maximum (3090 MHz on NVIDIA GeForce RTX 5090) regardless of `-lgc` locks, and
    `clocks.current.sm` reports the idle clock when queried with no
    workload. To get the value that reflects whatever `-lgc` the user
    has set (or the boost clock when unlocked), we run a sustained
    large GEMM and sample `clocks.current.sm` repeatedly, taking the
    max. If the observation is implausibly low (< 1800 MHz), we fall
    back to `clocks.max.sm` — signalling the sampling didn't catch
    the GPU under full load.
    """
    import subprocess
    try:
        import torch
        # Large enough that each matmul runs for tens of milliseconds,
        # keeping the GPU pinned at its sustained clock.
        n = 16384
        a = torch.ones(n, n, dtype=torch.float16, device="cuda")
        b = torch.ones(n, n, dtype=torch.float16, device="cuda")
        # Warm-up.
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        import threading, time
        samples = []
        done = threading.Event()

        def _sample():
            while not done.is_set():
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=clocks.current.sm",
                         "--format=csv,noheader,nounits"],
                        stderr=subprocess.DEVNULL, timeout=2,
                    ).decode().strip().splitlines()[0]
                    samples.append(float(out))
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        # Sustained: ~2-3 s of continuous GEMM work.
        for _ in range(50):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        done.set()
        t.join(timeout=1)

        if samples:
            observed = max(samples)
            if observed >= 1800.0:
                return observed

        # Fall back to the physical max.
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.max.sm",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip().splitlines()[0]
        return float(out)
    except Exception:
        return fallback


# Locked to 2.4 GHz across all benchmarks (see NVIDIA GeForce RTX 5090 setup in §5).
CLOCK_MHZ = 2400.0
SOL_TFLOPS = (
    NUM_SMS
    * TENSOR_CORES_PER_SM
    * HMMA_ISSUE_FREQ
    * HMMA_FLOPS_PER_OP
    * (CLOCK_MHZ * 1e6)
    * 1e-12
)

B200_RESULTS_DIR = HERE / "results" / "b200"
B200_SOL_TFLOPS = 2250.0

TARGETS = {
    "rtx5090": {
        "results_dir": HERE / "results" / "rtx5090",
        "out": REPO / "figures" / "generated" / "exp1_safety_overhead_rtx5090.pdf",
        "sol_tflops": SOL_TFLOPS,
        "sol_label": "Peak",
        "paths": {
            "Python": "gemm_python_results.csv",
            "Rust (unchecked)": "gemm_rust_optimized_results.csv",
            "Rust (dynamic)": "gemm_rust_safe_results.csv",
        },
    },
    "b200": {
        "results_dir": B200_RESULTS_DIR,
        "out": REPO / "figures" / "generated" / "exp1_safety_overhead.pdf",
        "sol_tflops": B200_SOL_TFLOPS,
        "sol_label": "Peak",
        "paths": {
            "Python": "gemm_python_persistent_results.csv",
            "Rust": "gemm_rust_persistent_safe_results.csv",
            "unsafe Rust": "gemm_rust_persistent_raw_results.csv",
            "cuBLAS": "gemm_cublas_results.csv",
        },
    },
}


def read_csv(path):
    """Returns (M, median_tflops, lo_tflops, hi_tflops) tuples, sorted by M.

    Error-bar bounds prefer stdev_s (median ± 1σ in time, converted to
    TFlops). Falls back to min_s/max_s if stdev_s is absent, then to
    zero-width if neither is present. Higher TFlops = lower wall-clock,
    so time bounds invert when mapped to TFlops.
    """
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            m = int(r["M"])
            n = int(r["N"])
            k = int(r["K"])
            median_s = float(r["median_s"])
            flops = 2.0 * m * n * k
            median_tf = flops / median_s / 1e12

            lo_tf = hi_tf = median_tf
            if r.get("stdev_s"):
                sigma = float(r["stdev_s"])
                lo_s = median_s + sigma
                hi_s = max(median_s - sigma, 1e-12)
                lo_tf = flops / lo_s / 1e12
                hi_tf = flops / hi_s / 1e12
            elif r.get("min_s") and r.get("max_s"):
                min_s = float(r["min_s"])
                max_s = float(r["max_s"])
                hi_tf = flops / min_s / 1e12
                lo_tf = flops / max_s / 1e12

            rows.append((m, median_tf, lo_tf, hi_tf))
    rows.sort()
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=os.environ.get("EXP1_PLOT_TARGET", "b200"),
    )
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sol-tflops", type=float)
    return ap.parse_args()


def main():
    args = parse_args()
    target = TARGETS[args.target]
    results_dir = args.results_dir or Path(os.environ.get("RESULTS_DIR", target["results_dir"]))
    out = args.out or target["out"]
    sol_tflops = args.sol_tflops if args.sol_tflops is not None else target["sol_tflops"]

    # Paper story: mapped persistent Rust keeps checked, disjoint output
    # stores while matching the unsafe raw-pointer persistent implementation.
    # cuTile Python and cuBLAS provide frontend and library baselines.
    paths = {
        label: results_dir / filename
        for label, filename in target["paths"].items()
    }

    # Single-column width in ACM sigplan is ~3.33 in. Shrunk fonts and
    # markers to keep 3 series + error bars + legend legible at that size.
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

    fig_size = (3.33, 1.8)
    fig, ax = plt.subplots(figsize=fig_size)

    # Colorblind-friendly palette (Wong).
    style = {
        "Python":                  dict(color="#E69F00", marker="s"),
        "Rust (unchecked)":        dict(color="#0072B2", marker="o"),
        "Rust (dynamic)":          dict(color="#D55E00", marker="D"),
        "Rust (static)":           dict(color="#009E73", marker="^"),
        "Rust":                    dict(color="#0072B2", marker="o"),
        "Rust static":             dict(color="#D55E00", marker="D"),
        "unsafe Rust":             dict(color="#009E73", marker="^"),
        "cuBLAS":                  dict(color="#000000", marker="x", linestyle="--"),
    }

    # ylim: don't tie to the peak line alone. If the queried peak is low
    # or data runs above it, use the max of peak and observed data.
    _all_y = []
    for _label, _path in paths.items():
        _rows = read_csv(_path)
        if _rows:
            _all_y.extend(hi for _, _, _, hi in _rows)
    _ymax = max([sol_tflops] + _all_y) if _all_y else sol_tflops
    ax.set_ylim(0, _ymax * 1.08)

    # Peak reference line (dotted, light) at the device's f16/bf16 peak.
    ax.axhline(sol_tflops, color="#888888", linestyle=":", linewidth=0.9,
               zorder=0,
               label=target["sol_label"])

    for label, path in paths.items():
        rows = read_csv(path)
        if rows is None:
            continue
        xs = [m for m, *_ in rows]
        ys = [t for _, t, *_ in rows]
        los = [t - lo for _, t, lo, _ in rows]
        his = [hi - t for _, t, _, hi in rows]
        ax.errorbar(xs, ys, yerr=[los, his], label=label,
                    capsize=1.3, elinewidth=0.5, **style[label])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$M = N = K$")
    ax.set_ylabel("TFlops (f16)")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.4)
    # Sparser x-ticks for single-column: every other power of 2 + the
    # endpoint, so labels don't collide but the data range is obvious.
    ax.set_xticks([1024, 4096, 16384])
    ax.set_xticklabels([r"$2^{10}$", r"$2^{12}$", r"$2^{14}$"])
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768], minor=True)
    # Ensure the 32768 endpoint is visible even if the data stops there.
    ax.set_xlim(1024 * 0.92, 32768 * 1.08)

    ax.legend(loc="upper left", handlelength=1.4, borderpad=0.25,
              labelspacing=0.25, handletextpad=0.35,
              framealpha=0.85, frameon=True, edgecolor="none")

    fig.tight_layout(pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

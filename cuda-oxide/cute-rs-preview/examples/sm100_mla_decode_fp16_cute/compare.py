# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate identical paged inputs before comparing complete decode invocations."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import hashlib
import os
import importlib.metadata
from pathlib import Path
import statistics
import subprocess
import tempfile
import sys
from datetime import datetime, timezone

import numpy as np

from contract import COMMIT, SOURCE_HASHES, Config, check_close, generate, oracle, write_inputs
from cutedsl import sm100_mla_decode_fp16_cutedsl


def run_case(rust_bin, cfg, *, seed=1111, pattern="random", warmup=10, iterations=100,
             rounds=3, clusters=None, validation_only=False, timeout=180, timing_order="cutedsl-first"):
    if timing_order not in ("cutedsl-first", "rust-first"):
        raise ValueError("timing_order must be cutedsl-first or rust-first")
    if warmup < 0 or iterations <= 0 or rounds <= 0 or timeout <= 0:
        raise ValueError("warmup must be nonnegative; iterations, rounds and timeout must be positive")
    if clusters is not None and clusters <= 0:
        raise ValueError("clusters must be positive")
    cfg.validate()
    tolerance = 0.001 if pattern == "zero-query" else 0.01
    data = generate(cfg, seed, pattern)
    expected, expected_lse = oracle(cfg, data)
    with tempfile.TemporaryDirectory(prefix="sm100-mla-") as temporary:
        directory = Path(temporary)
        digests = write_inputs(directory, cfg, data)
        reference = sm100_mla_decode_fp16_cutedsl(cfg, data, clusters)
        reference_validation = reference.validate(data, atol=tolerance)
        command = [str(Path(rust_bin).resolve()), "--input-dir", str(directory),
                   "--output-dir", str(directory), "--clusters", str(reference.clusters)]
        completed = subprocess.run(command + ["--validation-only"], check=True,
                                   capture_output=True, text=True, timeout=timeout)
        rust_report = json.loads(completed.stdout)
        rust_o = np.fromfile(directory / "output.f16", dtype="<f2").reshape(cfg.batch, 128, 512)
        rust_lse = np.fromfile(directory / "lse.f32", dtype="<f4").reshape(cfg.batch, 128)
        python_o, python_lse = reference.read()
        validation = {
            "rust_oracle": {"output": check_close(rust_o, expected, tolerance),
                            "lse": check_close(rust_lse, expected_lse, tolerance)},
            "cutedsl_oracle": reference_validation,
            "cross_implementation": {"output": check_close(rust_o, python_o, tolerance),
                                     "lse": check_close(rust_lse, python_lse, tolerance)},
        }
        gpu_state = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit", "--format=csv"],
            check=True, capture_output=True, text=True).stdout.strip()
        gpu_processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
            check=True, capture_output=True, text=True).stdout.strip()
        report = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "configuration": asdict(cfg), "seed": seed, "pattern": pattern,
                  "input_sha256": digests, "validation": validation,
                  "reference": {"commit": COMMIT, "sha256": SOURCE_HASHES},
                  "rust": rust_report, "clusters": reference.clusters,
                  "environment": {"gpu_state": gpu_state, "gpu_processes": gpu_processes,
                                  "python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                  "torch": reference.torch.__version__,
                                  "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
                                  "compiler": "CUTLASS 4.7.0", "architecture": "sm_100a",
                                  "reference_optimization_level": 2,
                                  "rust_optimization_level": rust_report.get("cutlass_optimization_level")},
                  "rust_binary_sha256": hashlib.sha256(Path(rust_bin).read_bytes()).hexdigest()}
        if not validation_only:
            if rust_report.get("cutlass_optimization_level") != "2":
                raise ValueError("timing requires Rust built with CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 to match the original reference")
            if timing_order == "cutedsl-first":
                python_samples = reference.benchmark(warmup, iterations, rounds)
            timed = subprocess.run(command + ["--warmup", str(warmup), "--iterations", str(iterations),
                                              "--rounds", str(rounds)], check=True, capture_output=True,
                                   text=True, timeout=timeout)
            report["rust"] = json.loads(timed.stdout)
            after_o = np.fromfile(directory / "output.f16", dtype="<f2").reshape(rust_o.shape)
            after_lse = np.fromfile(directory / "lse.f32", dtype="<f4").reshape(rust_lse.shape)
            if after_o.tobytes() != rust_o.tobytes() or after_lse.tobytes() != rust_lse.tobytes():
                raise AssertionError("Rust output changed across identical timed invocations")
            if timing_order == "rust-first":
                python_samples = reference.benchmark(warmup, iterations, rounds)
            after_python_o, after_python_lse = reference.read()
            if (after_python_o.tobytes() != python_o.tobytes()
                    or after_python_lse.tobytes() != python_lse.tobytes()):
                raise AssertionError("CuTeDSL output changed across identical timed invocations")
            report["environment"]["gpu_state_after_timing"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit", "--format=csv"],
                check=True, capture_output=True, text=True).stdout.strip()
            report["environment"]["gpu_processes_after_timing"] = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
                check=True, capture_output=True, text=True).stdout.strip()
            order = ["sm100_mla_decode_fp16_cutedsl", "sm100_mla_decode_fp16_cute_rs"]
            if timing_order == "rust-first":
                order.reverse()
            report["cutedsl"] = {"implementation": "sm100_mla_decode_fp16_cutedsl",
                                 "samples_us": python_samples, "median_us": statistics.median(python_samples)}
            report["timing"] = {"method": "cuda-events-repeated-complete-invocation", "graphs": False,
                                "cache": "hot", "warmup": warmup, "iterations": iterations,
                                "rounds": rounds, "includes_split_reduction": cfg.splits > 1,
                                "order": order}
        return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust-bin", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--kv-length", type=int, default=1024)
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--softmax-scale", type=float, default=0.0416)
    p.add_argument("--output-scale", type=float, default=1.0)
    p.add_argument("--clusters", type=int)
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--pattern", choices=("random", "zero-query"), default="random")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", "--iters", type=int, default=100)
    p.add_argument("--rounds", "--windows", type=int, default=3)
    p.add_argument("--validation-only", "--check-only", action="store_true")
    p.add_argument("--timing-order", choices=("cutedsl-first", "rust-first"), default="cutedsl-first")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--json", type=Path)
    return p


def main():
    args = parser().parse_args()
    cfg = Config(batch=args.batch, kv_length=args.kv_length, splits=args.splits,
                 softmax_scale=args.softmax_scale, output_scale=args.output_scale)
    with redirect_stdout(sys.stderr):
        report = run_case(args.rust_bin, cfg, seed=args.seed, pattern=args.pattern, warmup=args.warmup,
                          iterations=args.iterations, rounds=args.rounds, clusters=args.clusters,
                          validation_only=args.validation_only, timeout=args.timeout,
                          timing_order=args.timing_order)
    rendered = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    if args.validation_only:
        print("CuTeDSL: PASS")
        print("CuTe-RS: PASS")
    else:
        print(f"CuTeDSL: {report['cutedsl']['median_us']:.3f} us")
        print(f"CuTe-RS: {report['rust']['median_us']:.3f} us")


if __name__ == "__main__":
    main()

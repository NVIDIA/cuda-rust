#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and compare FMHA CuTeDSL and cute-rs using direct CUDA events.

Shared by the prefill and decode executables. Workload flags use the existing
reference/launch.py interface. Nsight profiling remains in reference/compare.py.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(workload):
    directory = HERE if workload == "prefill" else HERE.parent / "fmha_decode_sm120a_cute"
    launch = load_module("fmha_direct_launch", directory / "reference/launch.py")
    # The prefill evidence helpers import launch by name. Keep that import
    # local so callers/tests cannot accidentally supply the decode fixture.
    previous = sys.modules.get("launch")
    try:
        sys.modules["launch"] = launch
        checks = load_module("fmha_direct_checks", directory / "reference/compare.py")
    finally:
        if previous is None:
            sys.modules.pop("launch", None)
        else:
            sys.modules["launch"] = previous
    return directory / "reference/launch.py", launch, checks


def validate_timing(report, count):
    samples = report.get("samples_us")
    if (report.get("metric_unit") != "us" or not isinstance(samples, list)
            or len(samples) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in samples)):
        raise ValueError("expected one positive finite CUDA-event sample per iteration in us")
    metric = report.get("metric_value")
    if samples:
        if type(metric) not in (int, float) or metric != statistics.median(samples):
            raise ValueError("CUDA-event timing must equal the sample median")
    elif metric is not None:
        raise ValueError("verification-only runs must not report a timing")


def runner_arguments(args, workload):
    common = ["--seq-q", str(args.seq_q), "--seq-k", str(args.seq_k), "--heads", str(args.heads),
              "--causal" if args.causal else "--noncausal", "--input-scale", str(args.input_scale),
              "--device", str(args.device), "--warmup", str(args.warmup), "--iters", str(args.iters)]
    if workload == "prefill":
        common += ["--shape-mode", args.shape_mode]
    else:
        common += ["--specialization", args.specialization]
    if args.zero_query:
        common.append("--zero-query")
    return common


def compare(options, runner_cli):
    reference_path, launch, checks = load_reference(options.workload)
    args = launch.parse_args(runner_cli)
    if args.output is not None:
        raise ValueError("--output requires --rust-only; comparison checks both outputs internally")
    binary = options.rust_binary.resolve(strict=True)
    args.samples = args.iters
    args.warmup = args.warmup if args.iters else 0
    common = runner_arguments(args, options.workload)
    hashes = checks.expected_input_hashes(args)
    reports, outputs = {}, {}
    # Preserve CUDA_VISIBLE_DEVICES and the reference compiler environment.
    # Each process creates its own CUDA context and times only its own work.
    with tempfile.TemporaryDirectory(prefix=f"fmha-{options.workload}-compare-") as temporary:
        directory = Path(temporary)
        for key, label, prefix, implementation in (
            ("python_reference", "CuTeDSL", [sys.executable, str(reference_path)], "cutlass-cute-dsl"),
            ("cuda_oxide", "cute-rs", [str(binary), "--rust-only"], "cute-rs"),
        ):
            result_path, output_path = directory / f"{key}.json", directory / f"{key}.f16"
            command = [*prefix, *common, "--json", str(result_path), "--output", str(output_path)]
            started = time.time_ns()
            completed = subprocess.run(command, capture_output=True, text=True, timeout=options.timeout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode:
                if completed.stdout:
                    print(completed.stdout, file=sys.stderr, end="")
                raise RuntimeError(
                    f"{label} runner exited {completed.returncode}; select the reference environment "
                    "with --python or CUDA_OXIDE_CUTE_PYTHON; use --rust-only for standalone execution"
                )
            report, output = checks.validate_report(result_path, output_path, args, started, hashes)
            checks.require_fields(report, {"implementation": implementation, "workload": args.workload}, "report")
            validate_timing(report, args.iters)
            reports[key], outputs[key] = report, output
    checks.require_fields(reports["cuda_oxide"]["provenance"], {
        key: reports["python_reference"]["provenance"][key]
        for key in ("gpu", "compute_capability", "device")
    }, "matched device")
    parity = launch.verify(outputs["cuda_oxide"], outputs["python_reference"])
    return {
        "workload": args.workload, "verification": "both-pass", **reports,
        "output_comparison": parity,
        "timing": {
            "mode": "direct", "statistic": "median", "unit": "us",
            "scope": "partial-plus-merge sequence" if options.workload == "decode" else "single kernel",
            "order": ["python_reference", "cuda_oxide"],
            "note": "CUDA-event intervals include host submission gaps; these are not Nsight kernel-active times.",
        },
    }


def compare_all_cases(options, runner_cli):
    _, launch, _ = load_reference(options.workload)
    args = launch.parse_args(runner_cli)
    if options.workload == "prefill":
        if (args.seq_q, args.seq_k, args.heads) != launch.STATIC_SHAPE:
            raise ValueError("--all-cases requires --seq-q 1024 --seq-k 1024 --heads 8 for static cases")
        masks = (True, False)
    else:
        if (args.seq_k, args.heads) not in launch.STATIC_SHAPES:
            raise ValueError("--all-cases requires (seq-k, heads) in " + str(launch.STATIC_SHAPES))
        masks = (False,)
    if args.output is not None:
        raise ValueError("--output requires --rust-only; comparison checks both outputs internally")
    cases = []
    for shape_mode in ("runtime", "static"):
        if options.workload == "prefill":
            args.shape_mode = shape_mode
        else:
            args.specialization = shape_mode
        for causal in masks:
            args.causal = causal
            print(f"Comparing {options.workload}: {shape_mode} / {'causal' if causal else 'noncausal'}...",
                  file=sys.stderr, flush=True)
            report = compare(options, runner_arguments(args, options.workload))
            cases.append({"shape_mode": shape_mode, "causal": causal, **report})
    return {"workload": args.workload, "verification": "all-pass", "cases": cases,
            "timing": cases[0]["timing"]}


def print_summary(report):
    cases = report.get("cases")
    first = cases[0] if cases is not None else report
    rust = first["cuda_oxide"]
    provenance = rust["provenance"]
    batch, seq_q, seq_k, heads, dim = rust["shape"]
    print(f"{report['workload']}: B={batch}, Sq={seq_q}, Sk={seq_k}, H={heads}, D={dim}")
    if provenance["iters"]:
        print(f"Timing: direct CUDA-event median (us); warmup={provenance['warmup']}, "
              f"iters={provenance['iters']}")
        print("CUDA-event intervals include host submission gaps; not Nsight kernel-active times.")
    else:
        print("Verification only (timing disabled).")
    if cases is not None:
        print("| Shapes | Attention | cute-rs (us) | CuTeDSL (us) | Relative delta |")
        print("| --- | --- | ---: | ---: | ---: |")
        for case in cases:
            rust_time = case["cuda_oxide"]["metric_value"]
            dsl_time = case["python_reference"]["metric_value"]
            values = ("verified | verified | n/a" if rust_time is None else
                      f"{rust_time:.3f} | {dsl_time:.3f} | {(rust_time / dsl_time - 1) * 100:+.1f}%")
            print(f"| {case['shape_mode'].title()} | {'Causal' if case['causal'] else 'Noncausal'} | {values} |")
        if provenance["iters"]:
            print("Relative delta = (cute-rs / CuTeDSL - 1) * 100%; negative means a shorter cute-rs event interval.")
    else:
        print(f"Case: {provenance['shape_specialization']} / {'causal' if rust['causal'] else 'noncausal'}")
        for key, label in (("python_reference", "CuTeDSL"), ("cuda_oxide", "cute-rs")):
            metric = report[key]["metric_value"]
            print(f"{label}: {metric:.3f} us" if metric is not None
                  else f"{label}: verified (timing disabled)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workload", choices=("decode", "prefill"), required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--all-cases", action="store_true",
                        help="compare runtime/static shapes; prefill also sweeps causal/noncausal attention")
    parser.add_argument("--json", type=Path, help="save both checked reports and output comparison")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per implementation")
    options, runner_cli = parser.parse_known_args(argv)
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        parser.error("timeout must be positive and finite")
    try:
        report = compare_all_cases(options, runner_cli) if options.all_cases else compare(options, runner_cli)
        if options.json:
            options.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print_summary(report)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

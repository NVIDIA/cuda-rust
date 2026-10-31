#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate full split-KV decode outputs, tails, replay, hashes and head isolation.

Runs no timing iterations. --reference additionally checks the independently
authored cuda-oxide CuTeDSL decode example on identical inputs. Select the GPU
externally with CUDA_VISIBLE_DEVICES. Each run writes fresh evidence.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("split_kv_decode_comparison", HERE / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

CASES = tuple(
    (f"sk-{length}-heads-{heads}", length, heads, 1, False)
    for length in (1, 3, 31, 32, 33, 63, 64, 65, 127, 257, 1024, 4097)
    for heads in (1, 3, 8)
) + (
    ("unequal-splits-uniform-65", 65, 3, 1, True),
    ("unequal-splits-uniform-257", 257, 8, 1, True),
    ("large-logits-tail-65", 65, 3, 16, False),
    ("large-logits-tail-4097", 4097, 8, 16, False),
)


def validation_plan(cases, specialization):
    """Keep the original shape sweep, then force both paths on the AOT portfolio."""
    for _, seq_k, heads, _, _ in cases:
        compare.effective_specialization(seq_k, heads, specialization)
    plan = [(*case, specialization) for case in cases]
    paired = [(seq_k, heads, 1) for seq_k, heads in compare.STATIC_PORTFOLIO]
    paired.extend(((65, 3, 16), (4097, 8, 16)))
    plan.extend((f"specialization-sk-{seq_k}-heads-{heads}-scale-{scale}-{mode}",
                 seq_k, heads, scale, False, mode)
                for seq_k, heads, scale in paired for mode in ("runtime", "static"))
    return plan


def main(argv=None, *, cases=CASES):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-runner", type=Path, default=HERE / "launch.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--specialization", choices=("auto", "runtime", "static"), default="auto",
                        help="mode for the base shape sweep; runtime/static portfolio checks always follow")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be an existing executable")
    if args.reference and not args.reference_runner.is_file():
        parser.error("--reference-runner must be an existing file")
    if args.device < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("device must be nonnegative; timeout must be finite and positive")
    try:
        plan = validation_plan(cases, args.specialization)
    except ValueError as error:
        parser.error(str(error))
    args.binary = args.binary.resolve()
    args.reference_runner = args.reference_runner.resolve()
    parent = args.output_dir.resolve() if args.output_dir else None
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="decode-validation-", dir=parent))
    environment = os.environ.copy()
    summary = {
        "status": "running", "started_ns": time.time_ns(),
        "binary": compare.file_identity(args.binary), "checks": [], "cross_checks": [],
        "specialization_cross_checks": [], "specialization": args.specialization,
        "static_portfolio": [list(shape) for shape in compare.STATIC_PORTFOLIO],
        "reference_origin": "independently authored cuda-oxide CuTeDSL decode example",
        "environment": {key: environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "verification": "every FP16 output against independent CPU FP64 stable-softmax oracle",
        "launch_contract": {"correctness_sequences": 2, "warmup_sequences": 0,
                            "measured_sequences": 0, "kernel_sequence": compare.KERNEL_SEQUENCE},
    }
    if args.reference:
        summary["reference_runner"] = compare.file_identity(args.reference_runner)
    report_path = directory / "report.json"

    def save():
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    print(f"Evidence: {directory}", flush=True)
    entry, gpu_identity = None, None
    oracle_cache, paired_outputs = {}, {}
    save()
    try:
        for name, seq_k, heads, scale, zero_query, specialization in plan:
            options = SimpleNamespace(seq_q=1, seq_k=seq_k, heads=heads, causal=False,
                                      input_scale=scale, zero_query=zero_query, device=args.device,
                                      warmup=0, samples=0, specialization=specialization)
            shape_key = (seq_k, heads, scale, zero_query)
            if shape_key not in oracle_cache:
                oracle_cache[shape_key] = (compare.expected_input_hashes(options), compare.cpu_oracle(options))
            input_hashes, expected = oracle_cache[shape_key]
            cli = ["--seq-q", "1", "--seq-k", str(seq_k), "--heads", str(heads), "--noncausal",
                   "--input-scale", str(scale), "--device", str(args.device), "--warmup", "0", "--iters", "0",
                   "--specialization", specialization]
            if zero_query:
                cli.append("--zero-query")
            implementations = [("rust", [str(args.binary), "--rust-only"])]
            if args.reference:
                implementations.append(("authored-cute-dsl", [args.python, str(args.reference_runner)]))
            outputs = {}
            for label, prefix in implementations:
                result_path = directory / f"{name}-{label}.json"
                output_path = directory / f"{name}-{label}.f16"
                command = [*prefix, *cli, "--json", str(result_path), "--output", str(output_path)]
                entry = {"name": name, "implementation": label, "command": command, "status": "running",
                         "started_ns": time.time_ns(), "runner_report": str(result_path),
                         "output": str(output_path), "expected_input_hashes": input_hashes,
                         "specialization": specialization,
                         "shape_specialization": compare.effective_specialization(seq_k, heads, specialization)}
                summary["checks"].append(entry)
                save()
                if result_path.exists() or output_path.exists():
                    raise ValueError("refusing to reuse existing validation evidence")
                entry["exit_code"] = compare.run_command(command, args.timeout, environment,
                                                          directory / f"{name}-{label}.log")
                if entry["exit_code"]:
                    raise ValueError(f"{name}/{label} exited {entry['exit_code']}; see its log")
                checked, output = compare.validate_report(result_path, output_path, options, entry["started_ns"],
                                                          input_hashes, expected)
                compare.require_fields(checked, {
                    "implementation": "cute-rs" if label == "rust" else "cutlass-cute-dsl"}, "report")
                identity = {key: checked["provenance"][key] for key in ("gpu", "compute_capability", "device")}
                if gpu_identity is None:
                    gpu_identity = identity
                    summary["gpu"] = identity
                elif identity != gpu_identity:
                    raise ValueError("GPU identity changed between validation runners/cases")
                outputs[label] = output
                entry.update(checked_report=checked, independent_cpu_validation=compare.fixture.verify(output, expected))
                if name.startswith("specialization-"):
                    key = (*shape_key, label)
                    if specialization == "runtime":
                        paired_outputs[key] = output
                    else:
                        runtime = paired_outputs.pop(key)
                        summary["specialization_cross_checks"].append({
                            "shape": [1, 1, seq_k, heads, 64], "input_scale": scale,
                            "zero_query": zero_query, "implementation": label, "status": "passed",
                            "runtime_output_hash": compare.fixture.bits_hash(runtime),
                            "static_output_hash": compare.fixture.bits_hash(output),
                            "bitwise_equal": compare.fixture.encode_f16(runtime) == compare.fixture.encode_f16(output),
                            **compare.fixture.verify(output, runtime),
                        })
                entry.update(status="passed", finished_ns=time.time_ns())
                save()
            if args.reference:
                summary["cross_checks"].append({"name": name, "status": "passed",
                    **compare.fixture.verify(outputs["rust"], outputs["authored-cute-dsl"])})
            print(f"PASS {name}", flush=True)
            save()
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        summary.update(status="failed", error=str(error) or "interrupted")
        if entry and entry["status"] == "running":
            entry.update(status="failed", error=summary["error"], finished_ns=time.time_ns())
        print(f"FAIL: {summary['error']}", file=sys.stderr)
    finally:
        summary["finished_ns"] = time.time_ns()
        save()
    print(f"Validation report: {report_path}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

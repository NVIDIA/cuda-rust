#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate all outputs for multiple tiles, masks, heads, and logit ranges.

Select the intended GPU through CUDA_VISIBLE_DEVICES. No timing is performed.
--reference also runs the unchanged public CuTeDSL kernel on identical inputs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent / "reference"))
from launch import decode_f16, verify

CASES = (
    ("causal-one-tile", 64, 64, 2, True, 1, False),
    ("noncausal-one-tile", 64, 64, 2, False, 1, False),
    ("causal-two-tiles", 128, 128, 8, True, 1, False),
    ("noncausal-two-tiles", 128, 128, 8, False, 1, False),
    ("causal-three-tiles", 192, 192, 3, True, 1, False),
    ("noncausal-single-query", 1, 192, 3, False, 1, False),
    ("more-kv-tiles", 64, 128, 3, False, 1, False),
    ("more-query-tiles", 128, 64, 3, False, 1, False),
    ("causal-prefill", 1024, 1024, 8, True, 1, False),
    ("noncausal-fmha", 1024, 1024, 8, False, 1, False),
    ("causal-large-logits", 128, 128, 2, True, 16, False),
    ("noncausal-large-logits", 128, 128, 2, False, 16, False),
    ("moderate-logits", 64, 128, 2, False, 4, False),
    ("causal-uniform-softmax", 128, 128, 3, True, 1, True),
    ("noncausal-uniform-softmax", 128, 128, 3, False, 1, True),
)


def checked_report(path, seq_q, seq_k, heads, causal, scale, zero_query):
    report = json.loads(path.read_text())
    expected = {"shape": [1, seq_q, seq_k, heads, 64], "causal": causal,
                "input_scale": scale, "zero_query": zero_query,
                "verification": "cpu-pass", "replay_bitwise_equal": True}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"wrong report {key}: {report.get(key)!r} != {value!r}")
    validation = report["validation"]
    if validation["elements"] != seq_q * heads * 64:
        raise ValueError("report does not cover every output")
    for field in ("max_abs_error", "max_tolerance_ratio"):
        if not math.isfinite(validation[field]) or validation[field] < 0:
            raise ValueError(f"invalid {field}")
    if validation["max_tolerance_ratio"] > 1:
        raise ValueError("verification tolerance exceeded")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--shape-mode", choices=("runtime", "static"), default="runtime",
                        help="static checks the supported 1024-token, 8-head specialization")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    binary = args.binary.resolve()
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="fmha-prefill-validation-"))
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
               "status": "running", "checks": []}
    report_path = directory / "report.json"
    cases = CASES
    if args.shape_mode == "static":
        cases = (
            ("static-causal", 1024, 1024, 8, True, 1, False),
            ("static-noncausal", 1024, 1024, 8, False, 1, False),
            ("static-causal-large-logits", 1024, 1024, 8, True, 16, False),
            ("static-noncausal-uniform", 1024, 1024, 8, False, 1, True),
        )
    summary["shape_mode"] = args.shape_mode
    try:
        for name, sq, sk, heads, causal, scale, zero_query in cases:
            common = ["--seq-q", str(sq), "--seq-k", str(sk), "--heads", str(heads),
                      "--causal" if causal else "--noncausal", "--input-scale", str(scale),
                      "--warmup", "0", "--iters", "0"]
            common += ["--shape-mode", args.shape_mode]
            if zero_query:
                common.append("--zero-query")
            implementations = [("rust", [str(binary), "--rust-only"])]
            if args.reference:
                script = Path(__file__).parent / "reference" / "launch.py"
                implementations.append(("cutlass", [args.python, str(script)]))
            reports, outputs = {}, {}
            for label, prefix in implementations:
                result_path = directory / f"{name}-{label}.json"
                output_path = directory / f"{name}-{label}.f16"
                result_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                command = [*prefix, *common, "--json", str(result_path), "--output", str(output_path)]
                entry = {"name": name, "implementation": label, "command": command, "status": "running"}
                summary["checks"].append(entry)
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    (directory / f"{name}-{label}.log").write_text(completed.stdout + completed.stderr)
                except subprocess.TimeoutExpired as error:
                    entry["status"] = "timeout"
                    raise RuntimeError(f"{name}/{label} exceeded {args.timeout:g} seconds") from error
                entry["exit_code"] = completed.returncode
                if completed.returncode:
                    entry["status"] = "failed"
                    raise RuntimeError(f"{name}/{label} exited {completed.returncode}; see its log")
                reports[label] = checked_report(result_path, sq, sk, heads, causal, scale, zero_query)
                if reports[label].get("provenance", {}).get("shape_specialization") != args.shape_mode:
                    raise ValueError(f"wrong shape specialization for {name}/{label}")
                outputs[label] = decode_f16(output_path.read_bytes())
                if len(outputs[label]) != sq * heads * 64 or not all(map(math.isfinite, outputs[label])):
                    raise ValueError(f"invalid or incomplete output for {name}/{label}")
                entry["status"] = "passed"
            if args.reference:
                if reports["rust"]["input_hashes"] != reports["cutlass"]["input_hashes"]:
                    raise ValueError(f"Rust/CuTeDSL input mismatch for {name}")
                parity = verify(outputs["rust"], outputs["cutlass"])
                summary["checks"].append({"name": name, "implementation": "cross-check", "status": "passed", **parity})
            report_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(f"PASS {name}", flush=True)
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr)
    report_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Validation report: {report_path}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

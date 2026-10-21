#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the full bit-exact MXFP4 oracle, persistent grids and optional sanitizers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile


def run(command, timeout):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.returncode, output
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
            return None, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sanitizer", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("timeout must be finite and positive")
    binary = args.binary.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
    report = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "status": "running", "checks": []}
    cases = [("resident", [], []), ("single-cta", [], ["--ctas", "1"]),
             ("phase-wraps", [], ["--ctas", "17"])]
    if args.sanitizer:
        cases += [("memcheck", ["compute-sanitizer", "--tool", "memcheck", "--check-tensor-ops", "yes", "--error-exitcode", "1"], ["--ctas", "17"]),
                  ("synccheck", ["compute-sanitizer", "--tool", "synccheck", "--error-exitcode", "1"], ["--ctas", "17"])]
    for name, prefix, options in cases:
        result_file = directory / f"{name}.json"
        command = prefix + [str(binary), *options, "--warmup", "0", "--samples", "2", "--json", str(result_file)]
        try:
            status, output = run(command, args.timeout)
        except OSError as error:
            status, output = -1, str(error)
        (directory / f"{name}.log").write_text(output)
        entry = {"name": name, "command": command, "exit_code": status}
        report["checks"].append(entry)
        try:
            if status != 0:
                raise ValueError("timeout" if status is None else f"exit {status}")
            result = json.loads(result_file.read_text())
            if not (result["verification"] == "bit-exact-pass"
                    and result["checked_outputs"] == 2048*2048
                    and result["mnk"] == [2048,2048,1024]
                    and result["compute_capability"] == [12,0]
                    and result["replay_verified"] is True):
                raise ValueError("incomplete or invalid GPU verification")
            if options and result["ctas"] != int(options[1]):
                raise ValueError("persistent grid mismatch")
            entry.update(status="passed", result=result)
        except (ValueError, OSError, KeyError, TypeError) as error:
            entry.update(status="failed", error=str(error))
            report["status"] = "failed"
            (directory / "report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"FAIL {name}: {error}\n{output}\nEvidence: {directory}", flush=True)
            return 1
        (directory / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"PASS {name}: all 4,194,304 FP16 outputs exact", flush=True)
    report["status"] = "passed"
    (directory / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Evidence: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

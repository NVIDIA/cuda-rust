#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check FP16 CuTe GEMM tails, persistent tiles, and five-stage pipeline reuse.

Runs on Linux with an SM100 GPU. Every case uses the executable's exhaustive
CPU oracle and captured-graph replay check. Optional CUDA sanitizer runs add
memory/TMA and synchronization checks; these short runs are not benchmarks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


CASES = (
    ("full", "256,352,64", ()),
    ("small-tails-bias", "256,8,8", ("--has-bias",)),
    ("persistent-tails-bias", "512,712,72", ("--has-bias", "--clusters", "1")),
    ("five-stages", "512,704,320", ("--clusters", "1")),
    ("stage-wrap-tails-bias", "512,712,392", ("--has-bias", "--clusters", "1")),
    ("multicluster-phase-wraps", "768,1064,704", ("--has-bias", "--clusters", "2")),
)


def run_command(command, timeout):
    """Bound the entire Linux process group, including sanitizer children."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
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


def validate(binary, directory, timeout, sanitizer):
    report = {
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "timeout_seconds": timeout,
        "checks": [],
        "status": "running",
    }

    def save_report():
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(name, command):
        try:
            status, output = run_command(command, timeout)
        except OSError as error:
            status, output = -1, str(error)
        (directory / f"{name}.log").write_text(output)
        entry = {"name": name, "command": command, "exit_code": status}
        report["checks"].append(entry)
        if status != 0:
            entry["status"] = "timeout" if status is None else "failed"
            report["status"] = "failed"
            save_report()
            reason = f"timeout after {timeout:g}s" if status is None else f"exit {status}"
            print(f"FAIL {name}: {reason}\n{output}", file=sys.stderr, flush=True)
            return None
        return entry

    for name, shape, extra in CASES:
        result_path = directory / f"{name}.json"
        # A failed or malformed invocation must not reuse an older result.
        result_path.unlink(missing_ok=True)
        command = [
            str(binary), "--rust-only", "--mnk", shape, *extra, "--warmup", "1", "--iters", "2",
            "--graph-launches", "2", "--json", str(result_path),
        ]
        entry = check(name, command)
        if entry is None:
            return False
        try:
            result = json.loads(result_path.read_text())
            if result["verification"] != "cpu-pass":
                raise ValueError(f'expected cpu-pass, got {result["verification"]!r}')
            if result["mnk"] != [int(value) for value in shape.split(",")]:
                raise ValueError("result dimensions differ from the requested case")
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry["status"] = "failed"
            report["status"] = "failed"
            save_report()
            print(f"FAIL {name}: invalid verification report: {error}", file=sys.stderr)
            return False
        entry.update(status="passed", result=result)
        save_report()
        print(f"PASS {name}: {shape}", flush=True)

    if sanitizer:
        for tool in ("memcheck", "synccheck"):
            command = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "1"]
            if tool == "memcheck":
                command += ["--check-tensor-ops", "yes"]
            command += [
                str(binary), "--rust-only", "--mnk", "512,712,392", "--has-bias", "--clusters", "1",
                "--warmup", "0", "--iters", "1", "--graph-launches", "1",
            ]
            entry = check(tool, command)
            if entry is None:
                return False
            entry["status"] = "passed"
            save_report()
            print(f"PASS {tool}", flush=True)

    report["status"] = "passed"
    save_report()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path,
        default=Path(__file__).resolve().parent / "target/release/fp16_gemm_256x352_cute",
        help="compiled executable; override when Cargo uses a custom target directory",
    )
    parser.add_argument("--output-dir", type=Path, help="preserve per-case logs, JSON, and report.json")
    parser.add_argument("--timeout", type=float, default=60, help="seconds per check (default: 60)")
    parser.add_argument("--sanitizer", action="store_true", help="also run memcheck and synccheck")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("this CUDA runner requires Linux process-group handling")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be a finite positive number")
    try:
        binary = args.binary.resolve(strict=True)
    except OSError as error:
        parser.error(f"build the example first or pass --binary: {error}")
    if args.output_dir is not None:
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        passed = validate(binary, directory, args.timeout, args.sanitizer)
        print(f"Validation artifacts: {directory}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="fp16-cute-validate-") as temporary:
            passed = validate(binary, Path(temporary), args.timeout, args.sanitizer)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

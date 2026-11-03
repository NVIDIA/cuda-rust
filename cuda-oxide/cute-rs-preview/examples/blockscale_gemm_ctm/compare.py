#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare bit-exact GEMM binaries using serial Nsight Systems kernel-active times.

Each binary must accept --ctas, --warmup, --samples and --json, and launch only
its exact configured kernel name (default: blockscale_gemm): one correctness
launch, then warmups, then measured samples.
CUDA event times remain in the original executable reports and are not used here.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import tempfile
import time


def validate_report(path, ctas, warmup, samples, started_ns):
    """Reject stale evidence and require both initial and replay correctness."""
    try:
        if path.stat().st_mtime_ns < started_ns:
            raise ValueError(f"stale correctness report: {path}")
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read correctness report {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("correctness report must be a JSON object")
    expected = {
        "verification": "bit-exact-pass", "checked_outputs": 4194304,
        "mnk": [2048, 2048, 1024], "compute_capability": [12, 0],
        "ctas": ctas, "warmup": warmup, "samples": samples,
        "replay_verified": True,
    }
    for key, value in expected.items():
        actual = result.get(key)
        if (type(actual) is not type(value) or actual != value
                or isinstance(actual, list) and any(type(x) is not int for x in actual)):
            raise ValueError(f"invalid or missing {key}: expected {value!r}, got {actual!r}")
    return result


def kernel_samples(path, warmup, samples, expected_name="blockscale_gemm"):
    """Return the last N ordered kernel durations in microseconds, failing closed."""
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT k.start, k.end, s.value "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
                "LEFT JOIN StringIds AS s ON s.id = k.shortName "
                "ORDER BY k.start, k.end"
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"cannot read kernel trace {path}: {error}") from error
    expected = 1 + warmup + samples
    if len(rows) != expected:
        raise ValueError(f"expected {expected} kernel launches, found {len(rows)}")
    for start, end, name in rows:
        if name != expected_name:
            raise ValueError(f"unexpected kernel shortName: {name!r}")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError(f"invalid kernel timestamps: {start!r}, {end!r}")
    return [(end - start) / 1000 for start, end, _ in rows[-samples:]]


def summarize_samples(samples):
    """Match src/main.rs percentile: rounded index into sorted samples."""
    ordered = sorted(samples)
    if not ordered or any(not math.isfinite(x) or x <= 0 for x in ordered):
        raise ValueError("kernel durations must be finite and positive")

    def percentile(numerator, denominator):
        return ordered[((len(ordered) - 1) * numerator + denominator // 2) // denominator]

    return {"median_us": percentile(1, 2), "p10_us": percentile(1, 10),
            "p90_us": percentile(9, 10)}


def aggregate_rounds(entries):
    summary = {}
    for entry in entries:
        result = summary.setdefault(entry["label"], {"round_medians_us": [], "samples_us": []})
        result["round_medians_us"].append(entry["median_us"])
        result["samples_us"].extend(entry["samples_us"])
    for result in summary.values():
        result["median_of_round_medians_us"] = summarize_samples(result["round_medians_us"])["median_us"]
    return summary


def execution_order(labels, rounds):
    return [labels[r % len(labels):] + labels[:r % len(labels)] for r in range(rounds)]


def run_command(command, timeout, environment, log_path):
    """Wait for each process; kill the whole process group on timeout or interrupt."""
    with log_path.open("w") as log:
        with subprocess.Popen(command, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                return process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", required=True, metavar="LABEL=PATH",
                        help="repeat for at least two implementations with unique labels")
    parser.add_argument("--kernel-name", action="append", default=[], metavar="LABEL=NAME",
                        help="exact Nsight kernel shortName for a binary label (default: blockscale_gemm)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="parent for a fresh run directory containing raw traces and reports")
    parser.add_argument("--device", default="1", help="single physical CUDA device index or UUID (default: 1)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--ctas", type=int, default=170)
    parser.add_argument("--timeout", type=float, default=240,
                        help="maximum seconds for each tool invocation (default: 240)")
    args = parser.parse_args(argv)
    if args.rounds <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("rounds and samples must be positive; warmup must be nonnegative")
    if not 1 <= args.ctas <= 256:
        parser.error("ctas must be between 1 and 256")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not args.device or "," in args.device:
        parser.error("device must identify exactly one GPU")
    binaries = {}
    for specification in args.binary:
        label, separator, filename = specification.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            parser.error("binary must be LABEL=PATH; labels use letters, digits, underscores or hyphens")
        if label in binaries:
            parser.error(f"duplicate binary label: {label}")
        try:
            path = Path(filename).resolve(strict=True)
        except OSError as error:
            parser.error(f"invalid binary {filename!r}: {error}")
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"binary must be an executable file: {path}")
        binaries[label] = path
    if len(binaries) < 2:
        parser.error("provide at least two --binary LABEL=PATH arguments")
    kernel_names = {}
    for specification in args.kernel_name:
        label, separator, name = specification.partition("=")
        if not separator or not name.strip():
            parser.error("kernel-name must be LABEL=NAME with a nonempty name")
        if label not in binaries:
            parser.error(f"unknown kernel-name label: {label}")
        if label in kernel_names:
            parser.error(f"duplicate kernel-name label: {label}")
        kernel_names[label] = name
    args.kernel_names = {label: kernel_names.get(label, "blockscale_gemm") for label in binaries}
    return args, binaries


def main(argv=None):
    args, binaries = parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=args.device, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    report = {
        "status": "running", "metric": "Nsight Systems kernel-active duration",
        "units": "microseconds", "kernel": "blockscale_gemm", "started_ns": time.time_ns(),
        "kernel_names": args.kernel_names, "kernel_name_field": "shortName",
        "parameters": {"device": args.device, "rounds": args.rounds, "warmup": args.warmup,
                       "samples": args.samples, "ctas": args.ctas, "timeout_seconds": args.timeout},
        "environment": {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
        "percentile_method": "sorted[((count - 1) * numerator + denominator // 2) // denominator]",
        "binaries": {}, "execution_order": execution_order(list(binaries), args.rounds),
        "commands": [], "runs": [],
    }

    def save():
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(directory / "report.json")

    def execute(command, log_path):
        entry = {"command": command, "log": str(log_path), "status": "running",
                 "started_ns": time.time_ns()}
        report["commands"].append(entry)
        save()
        try:
            entry["exit_code"] = run_command(command, args.timeout, environment, log_path)
            if entry["exit_code"] != 0:
                raise ValueError(f"command exited {entry['exit_code']}; see {log_path}")
            entry["status"] = "passed"
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            entry.update(status="failed", error=str(error) or "interrupted")
            raise
        finally:
            entry["finished_ns"] = time.time_ns()
            save()

    save()
    print(f"Evidence: {directory}", flush=True)
    active = None
    try:
        for label, binary in binaries.items():
            report["binaries"][label] = {
                "path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
        version_log = directory / "nsys-version.log"
        execute(["nsys", "--version"], version_log)
        report["nsys_version"] = version_log.read_text().strip()
        snapshot = directory / "nvidia-smi.log"
        execute(["nvidia-smi", "--id", args.device, "-q"], snapshot)
        report["gpu_snapshot"] = {"path": str(snapshot), "text": snapshot.read_text()}
        for round_index, order in enumerate(report["execution_order"], start=1):
            for label in order:
                prefix = directory / f"round-{round_index:02d}-{label}"
                result_path = prefix.with_suffix(".json")
                trace_path = prefix.with_suffix(".nsys-rep")
                database_path = prefix.with_suffix(".sqlite")
                profile_log = prefix.with_suffix(".profile.log")
                export_log = prefix.with_suffix(".export.log")
                binary_command = [str(binaries[label]), "--ctas", str(args.ctas),
                                  "--warmup", str(args.warmup), "--samples", str(args.samples),
                                  "--json", str(result_path)]
                profile_command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                                   "--cpuctxsw=none", "--output", str(prefix), *binary_command]
                export_command = ["nsys", "export", "--type=sqlite", "--output",
                                  str(database_path), str(trace_path)]
                active = {"round": round_index, "label": label, "status": "running",
                          "correctness_report": str(result_path), "raw_trace": str(trace_path),
                          "sqlite": str(database_path), "profile_log": str(profile_log),
                          "export_log": str(export_log), "profile_command": profile_command,
                          "export_command": export_command, "started_ns": time.time_ns()}
                report["runs"].append(active)
                save()
                if any(path.exists() for path in (result_path, trace_path, database_path)):
                    raise ValueError("refusing to reuse existing benchmark evidence")
                print(f"Round {round_index}/{args.rounds}: {label}", flush=True)
                execute(profile_command, profile_log)
                active["verification"] = validate_report(
                    result_path, args.ctas, args.warmup, args.samples, active["started_ns"])
                if not trace_path.is_file():
                    raise ValueError(f"missing raw Nsight Systems trace: {trace_path}")
                execute(export_command, export_log)
                measured = kernel_samples(database_path, args.warmup, args.samples,
                                          expected_name=args.kernel_names[label])
                active.update(status="passed", samples_us=measured,
                              kernel_launches=1 + args.warmup + args.samples,
                              finished_ns=time.time_ns(), **summarize_samples(measured))
                save()
                print(f"  kernel-active median={active['median_us']:.3f} us, "
                      f"p10={active['p10_us']:.3f} us, p90={active['p90_us']:.3f} us", flush=True)
                active = None
        report.update(status="passed", summary=aggregate_rounds(report["runs"]))
    except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        message = str(error) or "interrupted"
        if active is not None:
            active.update(status="failed", error=message, finished_ns=time.time_ns())
        report.update(status="failed", error=message)
        print(f"FAIL: {message}", flush=True)
    report["finished_ns"] = time.time_ns()
    save()
    if report["status"] != "passed":
        print(f"Evidence: {directory}", flush=True)
        return 1
    for label, summary in report["summary"].items():
        print(f"{label}: median of round medians "
              f"{summary['median_of_round_medians_us']:.3f} us", flush=True)
    print(f"Evidence: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

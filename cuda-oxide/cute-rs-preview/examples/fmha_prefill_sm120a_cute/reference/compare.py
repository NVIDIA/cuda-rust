#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare checked attention runners using serial Nsight Systems kernel times.

Select the GPU externally with CUDA_VISIBLE_DEVICES; --device is the resulting
visible ordinal. Defaults: 100 warmups, 31 samples, three alternating rounds.
Both runners must use launch.py's CLI/report protocol and issue two correctness
launches, then warmups, then samples of exactly one selected attention kernel.
CUDA event measurements are retained as runner evidence, never aggregated here.
The historical single-query prefill baseline uses --seq-q 1 --noncausal.
Dedicated split-KV decode uses fmha_decode_sm120a_cute/reference/compare.py,
which measures its partial and merge kernels as one ordered sequence.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import statistics
import struct
import subprocess
import sys
import tempfile
import time

from launch import ATOL, RTOL, HEAD_DIM, STATIC_SHAPE, Q_SEED, K_SEED, V_SEED, decode_f16, mix32, verify


CORRECTNESS_LAUNCHES = 2


def reject_nonfinite_json(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def hash_f16_bytes(data):
    result = 0xCBF29CE484222325
    for byte in data:
        result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def expected_input_hashes(args):
    """Hash the shared deterministic fixture without allocating large tensors."""
    hashes = {}
    for name, length, seed, scale in (
        ("q", args.seq_q, Q_SEED, args.input_scale),
        ("k", args.seq_k, K_SEED, args.input_scale),
        ("v", args.seq_k, V_SEED, 1),
    ):
        result = 0xCBF29CE484222325
        for index in range(length * args.heads * HEAD_DIM):
            value = 0.0 if name == "q" and args.zero_query else ((mix32(index ^ seed) & 63) - 32) * scale / 128.0
            for byte in struct.pack("<e", value):
                result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        hashes[name] = f"{result:016x}"
    return hashes


def require_fields(actual, expected, context):
    if not isinstance(actual, dict):
        raise ValueError(f"{context} must be an object")
    for key, want in expected.items():
        got = actual.get(key)
        if (type(got) is not type(want) or got != want
                or isinstance(want, list) and any(type(x) is not type(y) for x, y in zip(got, want))):
            raise ValueError(f"{context}.{key}: expected {want!r}, got {got!r}")


def validate_report(path, output_path, args, started_ns, input_hashes):
    """Require fresh full-output CPU evidence, replay checks, and identical inputs."""
    try:
        for filename in (path, output_path):
            if filename.stat().st_mtime_ns < started_ns:
                raise ValueError(f"stale runner evidence: {filename}")
        result = json.loads(path.read_text(), parse_constant=reject_nonfinite_json)
        data = output_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read runner evidence {path}: {error}") from error
    require_fields(result, {
        "shape": [1, args.seq_q, args.seq_k, args.heads, HEAD_DIM],
        "causal": args.causal, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "verification": "cpu-pass", "replay_bitwise_equal": True,
        "input_hashes": input_hashes,
    }, "report")
    count = args.seq_q * args.heads * HEAD_DIM
    validation = result.get("validation")
    require_fields(validation, {"elements": count, "atol": ATOL, "rtol": RTOL,
                                "oracle": "cpu-fp64-stable-softmax"}, "validation")
    for field in ("max_abs_error", "max_tolerance_ratio"):
        value = validation.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid validation.{field}: {value!r}")
    if validation["max_tolerance_ratio"] > 1:
        raise ValueError("CPU verification tolerance exceeded")
    provenance = result.get("provenance")
    require_fields(provenance, {"device": args.device, "warmup": args.warmup,
                               "iters": args.samples, "dtype": "float16",
                               "causal_specialization": "static",
                               "shape_specialization": args.shape_mode,
                               "tool": "cuda-events-single-kernel", "timing_mode": "direct"}, "provenance")
    capability = provenance.get("compute_capability")
    if (not isinstance(provenance.get("gpu"), str) or not provenance["gpu"]
            or not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)):
        raise ValueError("missing GPU identity/compute capability in runner report")
    if len(data) != count * 2:
        raise ValueError(f"output contains {len(data)} bytes; expected {count * 2}")
    output = decode_f16(data)
    if not all(map(math.isfinite, output)):
        raise ValueError("nonfinite FP16 output")
    if result.get("output_hash") != hash_f16_bytes(data):
        raise ValueError("FP16 output does not match the checked report's output hash")
    return result, output


def kernel_samples(path, warmup, samples, *, expected_name=None, name_contains=None):
    """Resolve one full kernel identity, then remove checks/warmup by launch count.

    Nsight may truncate shortName. Prefer demangledName for identity; exact
    overrides can name either field, but a colliding shortName is ambiguous.
    Older schemas without demangledName retain exact shortName behavior.
    """
    if bool(expected_name) == bool(name_contains):
        raise ValueError("provide one exact kernel name or one name resolver")
    if warmup < 0 or samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
            device_column = "k.deviceId" if "deviceId" in columns else "NULL"
            full_names = "demangledName" in columns
            name_column = "d.value" if full_names else "s.value"
            name_join = "LEFT JOIN StringIds AS d ON d.id = k.demangledName " if full_names else ""
            rows = db.execute(
                "SELECT k.start, k.end, s.value, " + device_column + ", " + name_column + " "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
                "LEFT JOIN StringIds AS s ON s.id = k.shortName "
                + name_join +
                "ORDER BY k.start, k.end"
            ).fetchall()
            gpu_metadata = []
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='TARGET_INFO_GPU'").fetchone():
                cursor = db.execute("SELECT * FROM TARGET_INFO_GPU")
                fields = [column[0] for column in cursor.description]
                gpu_metadata = [dict(zip(fields, [value.hex() if isinstance(value, bytes) else value
                                                  for value in row])) for row in cursor]
    except sqlite3.Error as error:
        raise ValueError(f"cannot read kernel trace {path}: {error}") from error
    for start, end, short_name, _, full_name in rows:
        if any(not isinstance(name, str) or not name for name in (short_name, full_name)):
            raise ValueError("unresolved kernel shortName/demangledName in trace")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError(f"invalid kernel timestamps: {start!r}, {end!r}")
    if name_contains:
        matches = sorted({row[4] for row in rows if name_contains in row[4]})
    else:
        matches = sorted({row[4] for row in rows if expected_name in (row[2], row[4])})
    if len(matches) != 1:
        raise ValueError(f"expected one matching kernel name for {expected_name or name_contains!r}, found {matches!r}")
    kernel_name = matches[0]
    selected = [row for row in rows if row[4] == kernel_name]
    expected = CORRECTNESS_LAUNCHES + warmup + samples
    if len(selected) != expected:
        raise ValueError(f"expected {expected} launches of exact kernel {kernel_name!r}, found {len(selected)}")
    devices = {row[3] for row in selected}
    if len(devices) != 1:
        raise ValueError(f"selected kernel ran on multiple devices: {devices}")
    durations = [(row[1] - row[0]) / 1000 for row in selected]
    split = CORRECTNESS_LAUNCHES + warmup
    return {
        "kernel_name": kernel_name, "kernel_name_field": "demangledName" if full_names else "shortName",
        "kernel_short_names": sorted({row[2] for row in selected}), "kernel_launches": len(selected),
        "trace_device_id": selected[0][3], "trace_gpu_metadata": gpu_metadata,
        "correctness_samples_us": durations[:CORRECTNESS_LAUNCHES],
        "warmup_samples_us": durations[CORRECTNESS_LAUNCHES:split], "samples_us": durations[split:],
        "selected_kernel_timestamps_ns": [{"start": row[0], "end": row[1]} for row in selected],
        "auxiliary_kernel_counts": dict(sorted(Counter(row[4] for row in rows if row[4] != kernel_name).items())),
    }


def aggregate_rounds(entries):
    summary = {}
    for entry in entries:
        result = summary.setdefault(entry["label"], {"round_medians_us": [], "samples_us": []})
        result["round_medians_us"].append(entry["median_us"])
        result["samples_us"].extend(entry["samples_us"])
    for result in summary.values():
        result["median_of_round_medians_us"] = statistics.median(result["round_medians_us"])
    return summary


def run_command(command, timeout, environment, log_path):
    """Finish each process before starting another; terminate its children on timeout."""
    with log_path.open("w") as log:
        with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                return process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise


def file_identity(path):
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
        if digest is None:
            digest = hashlib.sha256(stream.read()).hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def source_snapshot(reference_runner):
    """Current source evidence is separate from proof of how a binary was built."""
    here = Path(__file__).resolve()
    oxide = next((parent for parent in here.parents if (parent / "rust-toolchain.toml").is_file()), None)
    paths = set()
    for example in (here.parent.parent, reference_runner.parent.parent):
        for child in (example / "src", example / "reference"):
            if child.is_dir():
                paths.update(path for path in child.rglob("*") if path.suffix in (".py", ".rs", ".txt", ".md"))
        paths.update(path for path in (example / "Cargo.toml", example / "Cargo.lock") if path.is_file())
    paths.add(reference_runner)
    pins = {}
    if oxide:
        for relative in ("rust-toolchain.toml", "Cargo.lock",
                         "crates/cargo-oxide/src/cutlass_toolchain.rs",
                         "crates/cuda-oxide-mlir-export/src/profile.rs",
                         "crates/cuda-oxide-mlir-export/src/cute_attention.rs",
                         "crates/cuda-oxide-mlir-export/src/nvvm_attention.rs",
                         "cute-rs-preview/cute-rs/src/mma/mod.rs"):
            path = oxide / relative
            if path.is_file():
                paths.add(path)
        pin_path = oxide / "crates/cargo-oxide/src/cutlass_toolchain.rs"
        match = re.search(r'const CUTLASS_VERSION: &str = "([^"]+)"', pin_path.read_text()) if pin_path.is_file() else None
        pins["rust_cutlass_compiler_source_pin"] = match[1] if match else None
        pins["rust_toolchain_file"] = (oxide / "rust-toolchain.toml").read_text()
    requirements = here.parent / "requirements.txt"
    if requirements.is_file():
        pins["reference_requirements"] = requirements.read_text()
    return {"scope": "current attention sources and selected compiler/API files; not binary build provenance",
            "files": [file_identity(path) for path in sorted(paths) if path.is_file()], "pins": pins}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--rust-build-manifest", type=Path,
                        help="build provenance JSON; must include matching top-level binary_sha256")
    parser.add_argument("--python", default=sys.executable, help="reference Python interpreter")
    parser.add_argument("--reference-runner", type=Path, default=Path(__file__).with_name("launch.py"))
    parser.add_argument("--rust-kernel-name",
                        help="exact Nsight shortName or demangledName; otherwise resolve one full fmha_prefill name")
    parser.add_argument("--reference-kernel-name", help="exact shortName or demangledName; otherwise resolve one full FlashAttentionForwardAmpere name")
    parser.add_argument("--nsys", default=shutil.which("nsys") or "nsys")
    parser.add_argument("--output-dir", type=Path, required=True, help="parent for a fresh evidence directory")
    parser.add_argument("--seq-q", type=int, default=64)
    parser.add_argument("--seq-k", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--shape-mode", choices=("runtime", "static"), default="runtime",
                        help="match both runners' shape and stride specialization; static requires Sq=Sk=1024, H=8")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--causal", dest="causal", action="store_true")
    masks.add_argument("--noncausal", dest="causal", action="store_false")
    parser.set_defaults(causal=True)
    parser.add_argument("--input-scale", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--zero-query", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="ordinal after external CUDA_VISIBLE_DEVICES remapping")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", "--iters", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600, help="seconds per tool invocation")
    args = parser.parse_args(argv)
    if args.shape_mode == "static" and (args.seq_q, args.seq_k, args.heads) != STATIC_SHAPE:
        parser.error("static shape mode requires --seq-q 1024 --seq-k 1024 --heads 8")
    if min(args.seq_q, args.seq_k, args.heads, args.samples, args.rounds) <= 0:
        parser.error("lengths, heads, samples, and rounds must be positive")
    if (args.seq_q != 1 and args.seq_q % 64) or args.seq_k % 64:
        parser.error("query length must be 1 or a multiple of 64; KV length must be a multiple of 64")
    if args.causal and args.seq_q != args.seq_k:
        parser.error("causal cases require equal query and KV lengths")
    if max(args.seq_q, args.seq_k) * args.heads * HEAD_DIM > 0xFFFFFFFF:
        parser.error("tensor element count must fit u32 indexing")
    if args.device < 0 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("device/warmup must be nonnegative and timeout finite and positive")
    if any(name is not None and not name.strip()
           for name in (args.rust_kernel_name, args.reference_kernel_name)):
        parser.error("kernel names must be nonempty")
    for field in ("rust_binary", "reference_runner", "rust_build_manifest"):
        path = getattr(args, field)
        if path is None:
            continue
        if not path.is_file() or field == "rust_binary" and not os.access(path, os.X_OK):
            parser.error(f"{field} must be an existing {'executable ' if field == 'rust_binary' else ''}file: {path}")
        setattr(args, field, path.resolve())
    return args


def main(argv=None):
    args = parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
    environment = os.environ.copy()
    report = {
        "status": "running", "metric": "Nsight Systems kernel-active duration", "metric_unit": "us",
        "duration_source": "CUPTI_ACTIVITY_KIND_KERNEL.end - start (nanoseconds / 1000)",
        "started_ns": time.time_ns(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {key: getattr(args, key) for key in ("seq_q", "seq_k", "heads", "causal", "shape_mode",
                       "input_scale", "zero_query", "device", "warmup", "samples", "rounds", "timeout")},
        "launch_contract": {"correctness": CORRECTNESS_LAUNCHES, "warmup": args.warmup, "measured": args.samples},
        "environment": {key: environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                        "CUTE_DSL_ARCH", "CUDA_HOME", "CUDA_PATH")},
        "aggregation": "median of each round's kernel samples, then median of round medians",
        "runner_event_times": "retained in runner_report; excluded from kernel-active summaries",
        "execution_order": [["rust", "cutlass"] if round_index % 2 == 0 else ["cutlass", "rust"]
                            for round_index in range(args.rounds)],
        "provenance": {}, "commands": [], "runs": [],
    }

    def save():
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "report.json")

    def execute(command, log_path):
        entry = {"command": command, "log": str(log_path), "status": "running", "started_ns": time.time_ns()}
        report["commands"].append(entry)
        save()
        try:
            entry["exit_code"] = run_command(command, args.timeout, environment, log_path)
            if entry["exit_code"]:
                raise ValueError(f"command exited {entry['exit_code']}; see {log_path}")
            entry["status"] = "passed"
        except (OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            entry.update(status="failed", error=str(error) or "interrupted")
            raise
        finally:
            entry["finished_ns"] = time.time_ns()
            save()

    print(f"Evidence: {directory}", flush=True)
    active = None
    save()
    try:
        provenance = report["provenance"]
        provenance["rust_binary"] = file_identity(args.rust_binary)
        provenance["reference_runner"] = file_identity(args.reference_runner)
        provenance["source_snapshot"] = source_snapshot(args.reference_runner)
        provenance["rust_build"] = {"status": "unavailable; current source pins do not prove the binary's build toolchain"}
        if args.rust_build_manifest:
            manifest = json.loads(args.rust_build_manifest.read_text(), parse_constant=reject_nonfinite_json)
            if not isinstance(manifest, dict) or manifest.get("binary_sha256") != provenance["rust_binary"]["sha256"]:
                raise ValueError("Rust build manifest must contain binary_sha256 matching --rust-binary")
            provenance["rust_build"] = {"file": file_identity(args.rust_build_manifest), "manifest": manifest}
        for key, command in (
            ("nsys_version", [args.nsys, "--version"]),
            ("python_version", [args.python, "--version"]),
            ("git_head", ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"]),
            ("git_status", ["git", "-C", str(Path(__file__).parent), "status", "--short"]),
        ):
            log = directory / f"{key}.log"
            execute(command, log)
            provenance[key] = log.read_text().strip()
        report["expected_input_hashes"] = expected_input_hashes(args)
        common = ["--seq-q", str(args.seq_q), "--seq-k", str(args.seq_k), "--heads", str(args.heads),
                  "--shape-mode", args.shape_mode,
                  "--causal" if args.causal else "--noncausal", "--input-scale", str(args.input_scale),
                  "--device", str(args.device), "--warmup", str(args.warmup), "--iters", str(args.samples)]
        if args.zero_query:
            common.append("--zero-query")
        implementations = {"rust": [str(args.rust_binary), "--rust-only"], "cutlass": [args.python, str(args.reference_runner)]}
        output_hashes, kernel_names, baseline = {}, {}, None
        for round_index, order in enumerate(report["execution_order"], start=1):
            outputs = {}
            for label in order:
                prefix = directory / f"round-{round_index:02d}-{label}"
                result_path, output_path = prefix.with_suffix(".json"), prefix.with_suffix(".f16")
                trace_path, database_path = prefix.with_suffix(".nsys-rep"), prefix.with_suffix(".sqlite")
                profile_log, export_log = prefix.with_suffix(".profile.log"), prefix.with_suffix(".export.log")
                profile = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                           "--output", str(prefix), *implementations[label], *common,
                           "--json", str(result_path), "--output", str(output_path)]
                export = [args.nsys, "export", "--type=sqlite", "--output", str(database_path), str(trace_path)]
                active = {"round": round_index, "label": label, "status": "running", "started_ns": time.time_ns(),
                          "correctness_report": str(result_path), "output": str(output_path),
                          "raw_trace": str(trace_path), "sqlite": str(database_path)}
                report["runs"].append(active)
                save()
                if any(path.exists() for path in (result_path, output_path, trace_path, database_path)):
                    raise ValueError("refusing to reuse existing profiling evidence")
                print(f"Round {round_index}/{args.rounds}: {label}", flush=True)
                execute(profile, profile_log)
                checked, output = validate_report(result_path, output_path, args, active["started_ns"],
                                                  report["expected_input_hashes"])
                active["runner_report"] = checked
                outputs[label] = output
                gpu = {key: checked["provenance"][key] for key in ("gpu", "compute_capability", "device")}
                if baseline is None:
                    baseline = gpu
                    provenance["gpu"] = gpu
                elif gpu != baseline:
                    raise ValueError(f"GPU identity changed between runners/rounds: {gpu!r} != {baseline!r}")
                if label in output_hashes and output_hashes[label] != checked["output_hash"]:
                    raise ValueError(f"{label} output changed across rounds")
                output_hashes[label] = checked["output_hash"]
                if not trace_path.is_file():
                    raise ValueError(f"missing raw Nsight Systems trace: {trace_path}")
                execute(export, export_log)
                name = args.rust_kernel_name if label == "rust" else args.reference_kernel_name
                resolver = "fmha_prefill" if label == "rust" else "FlashAttentionForwardAmpere"
                trace_data = kernel_samples(database_path, args.warmup, args.samples, expected_name=name,
                                            name_contains=None if name else resolver)
                if label in kernel_names and kernel_names[label] != trace_data["kernel_name"]:
                    raise ValueError(f"{label} kernel name changed across rounds")
                kernel_names[label] = trace_data["kernel_name"]
                if "trace_device_id" not in provenance:
                    provenance["trace_device_id"] = trace_data["trace_device_id"]
                elif trace_data["trace_device_id"] != provenance["trace_device_id"]:
                    raise ValueError("Nsight kernel device changed between runners/rounds")
                active.update(trace_data, median_us=statistics.median(trace_data["samples_us"]),
                              status="passed", finished_ns=time.time_ns())
                save()
            report.setdefault("cross_checks", []).append({"round": round_index, "status": "passed",
                                                          **verify(outputs["rust"], outputs["cutlass"])})
            save()
        report.update(status="passed", summary=aggregate_rounds(report["runs"]))
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error) or "interrupted")
        if active is not None and active["status"] == "running":
            active.update(status="failed", error=report["error"], finished_ns=time.time_ns())
        print(f"FAIL: {report['error']}", file=sys.stderr)
    finally:
        report.update(finished_ns=time.time_ns(), finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    if report["status"] == "passed":
        for label, result in report["summary"].items():
            print(f"{label}: {result['median_of_round_medians_us']:.3f} us kernel-active (median of round medians)")
    print(f"Comparison report: {directory / 'report.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

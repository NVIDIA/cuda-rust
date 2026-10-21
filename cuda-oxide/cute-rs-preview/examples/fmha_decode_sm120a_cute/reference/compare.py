#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare dedicated split-KV decode sequences with serial Nsight Systems runs.

The CuTeDSL comparator is an independently authored cuda-oxide example, not an
official upstream decode benchmark. Both implementations launch partial then
merge on one stream. Report active duration (sum of both kernels) and wall span
(first start through last end) separately. Select the GPU externally through
CUDA_VISIBLE_DEVICES. Defaults: 100 warmups, 31 samples, three alternating rounds.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
KERNEL_SEQUENCE = ["fmha_decode_partial", "fmha_decode_merge"]
CORRECTNESS_LAUNCHES = 2
KV_CHUNK = 64
STATIC_PORTFOLIO = ((64, 8), (65, 3), (1024, 8), (4097, 8))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse only fixture and process/provenance helpers. Isolate the prefill
# comparator's absolute launch import so importing this module cannot silently
# select the decode launch module or alter a caller's existing launch module.
_prefill = HERE.parent.parent / "fmha_prefill_sm120a_cute/reference"
fixture = load_module("decode_compare_fixture", _prefill / "launch.py")
_previous_launch = sys.modules.get("launch")
try:
    sys.modules["launch"] = fixture
    common = load_module("decode_compare_common", _prefill / "compare.py")
finally:
    if _previous_launch is None:
        sys.modules.pop("launch", None)
    else:
        sys.modules["launch"] = _previous_launch

expected_input_hashes = common.expected_input_hashes
file_identity = common.file_identity
hash_f16_bytes = common.hash_f16_bytes
reject_nonfinite_json = common.reject_nonfinite_json
require_fields = common.require_fields
run_command = common.run_command


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number: {value}")
    return number


def cpu_oracle(args):
    return fixture.reference_cpu(
        *fixture.make_inputs(1, args.seq_k, args.heads, args.input_scale, args.zero_query),
        1, args.seq_k, args.heads, False)


def effective_specialization(seq_k, heads, requested):
    if requested not in ("auto", "runtime", "static"):
        raise ValueError(f"unknown specialization mode: {requested!r}")
    supported = (seq_k, heads) in STATIC_PORTFOLIO
    if requested == "static" and not supported:
        raise ValueError(f"static specialization is unavailable for (seq_k, heads)=({seq_k}, {heads})")
    return "static" if supported and requested != "runtime" else "runtime"


def validate_report(path, output_path, args, started_ns, input_hashes, expected_output=None):
    """Require fresh, full CPU/replay evidence bound to exactly these FP16 bytes."""
    for filename in (path, output_path):
        if filename.stat().st_mtime_ns < started_ns:
            raise ValueError(f"stale runner evidence: {filename}")
    result = json.loads(path.read_text(), parse_constant=reject_nonfinite_json, parse_float=finite_json_float)
    data = output_path.read_bytes()
    require_fields(result, {
        "workload": "fmha-decode", "shape": [1, 1, args.seq_k, args.heads, 64],
        "causal": False, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "verification": "cpu-pass", "replay_bitwise_equal": True, "input_hashes": input_hashes,
        "metric_unit": "us",
    }, "report")
    events = result.get("samples_us")
    if (not isinstance(events, list) or len(events) != args.samples
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in events)):
        raise ValueError("runner must report one positive finite event sample per measured kernel sequence")
    metric = result.get("metric_value")
    if (events and (type(metric) not in (float, int) or metric != statistics.median(events))
            or not events and metric is not None):
        raise ValueError("runner event metric must equal the sample median, or null when no timing was requested")
    count = args.heads * 64
    validation = result.get("validation")
    require_fields(validation, {"elements": count, "atol": fixture.ATOL, "rtol": fixture.RTOL,
                                "oracle": "cpu-fp64-stable-softmax"}, "validation")
    for field in ("max_abs_error", "max_tolerance_ratio"):
        value = validation.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid validation.{field}: {value!r}")
    if validation["max_tolerance_ratio"] > 1:
        raise ValueError("CPU verification tolerance exceeded")
    provenance = result.get("provenance")
    require_fields(provenance, {
        "device": args.device, "warmup": args.warmup, "iters": args.samples, "dtype": "float16",
        "tool": "cuda-events-kernel-sequence", "timing_mode": "direct",
        "algorithm": "single-query-split-kv", "kv_chunk": KV_CHUNK,
        "split_count": (args.seq_k + KV_CHUNK - 1) // KV_CHUNK,
        "kernel_count": 2, "kernel_sequence": KERNEL_SEQUENCE,
        "shape_specialization": effective_specialization(args.seq_k, args.heads, args.specialization),
    }, "provenance")
    capability = provenance.get("compute_capability")
    if (not isinstance(provenance.get("gpu"), str) or not provenance["gpu"]
            or not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)):
        raise ValueError("missing GPU identity/compute capability in runner report")
    if len(data) != count * 2:
        raise ValueError(f"output contains {len(data)} bytes; expected {count * 2}")
    output = fixture.decode_f16(data)
    if not all(map(math.isfinite, output)):
        raise ValueError("nonfinite FP16 output")
    if result.get("output_hash") != hash_f16_bytes(data):
        raise ValueError("FP16 output does not match the checked report's output hash")
    fixture.verify(output, cpu_oracle(args) if expected_output is None else expected_output)
    return result, output


def sequence_samples(path, warmup, samples, *, expected_names=None, name_contains=None):
    """Resolve two kernel identities and accept only complete serial sequences.

The trace must expose stream/device identity. Every stage has exactly two
correctness launches plus warmups and samples. Auxiliary initialization work
may occur outside a pair, but never within a pair or the measured interval.
"""
    if bool(expected_names) == bool(name_contains):
        raise ValueError("provide two exact kernel names or two name resolvers")
    selectors = expected_names or name_contains
    if len(selectors) != 2 or any(not isinstance(name, str) or not name.strip() for name in selectors):
        raise ValueError("two nonempty kernel selectors are required")
    if warmup < 0 or samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
            if not {"start", "end", "shortName", "deviceId", "streamId"} <= columns:
                raise ValueError("kernel trace lacks timestamps, names, deviceId or streamId")
            full_names = "demangledName" in columns
            name_column = "d.value" if full_names else "s.value"
            name_join = "LEFT JOIN StringIds AS d ON d.id=k.demangledName " if full_names else ""
            context_column = "k.contextId" if "contextId" in columns else "0"
            rows = db.execute(
                "SELECT k.start, k.end, s.value, " + name_column + ", k.deviceId, k.streamId, "
                + context_column + " FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
                "LEFT JOIN StringIds AS s ON s.id=k.shortName " + name_join + "ORDER BY k.start, k.end"
            ).fetchall()
            gpu_metadata = []
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='TARGET_INFO_GPU'").fetchone():
                cursor = db.execute("SELECT * FROM TARGET_INFO_GPU")
                fields = [column[0] for column in cursor.description]
                gpu_metadata = [dict(zip(fields, [value.hex() if isinstance(value, bytes) else value
                                                  for value in row])) for row in cursor]
    except sqlite3.Error as error:
        raise ValueError(f"cannot read kernel trace {path}: {error}") from error
    for start, end, short, full, *_ in rows:
        if any(not isinstance(name, str) or not name for name in (short, full)):
            raise ValueError("unresolved kernel shortName/demangledName in trace")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError(f"invalid kernel timestamps: {start!r}, {end!r}")
    identities = []
    for selector in selectors:
        matches = sorted({row[3] for row in rows if
                          (selector in row[3] if name_contains else selector in (row[2], row[3]))})
        if len(matches) != 1:
            raise ValueError(f"expected one matching kernel name for {selector!r}, found {matches!r}")
        identities.append(matches[0])
    if len(set(identities)) != 2:
        raise ValueError("partial and merge must resolve to distinct kernels")
    selected = [row for row in rows if row[3] in identities]
    expected = CORRECTNESS_LAUNCHES + warmup + samples
    counts = Counter(row[3] for row in selected)
    if any(counts[name] != expected for name in identities):
        raise ValueError(f"expected {expected} launches of each kernel, found {dict(counts)!r}")
    for field, index in (("device", 4), ("stream", 5), ("context", 6)):
        values = {row[index] for row in selected}
        if len(values) != 1 or any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"selected sequence has invalid or mismatched {field} identities: {values!r}")
    if [row[3] for row in selected] != identities * expected:
        raise ValueError("kernel sequence must alternate partial then merge")
    if any(previous[1] > following[0] for previous, following in zip(selected, selected[1:])):
        raise ValueError("kernel sequence launches overlap; serial timing is required")
    pairs = list(zip(selected[::2], selected[1::2]))
    auxiliary = [row for row in rows if row[3] not in identities]
    split = CORRECTNESS_LAUNCHES + warmup
    intervals = [(partial[0], merge[1]) for partial, merge in pairs]
    intervals.append((pairs[split][0][0], pairs[-1][1][1]))
    for row in auxiliary:
        if any(row[0] < end and row[1] > start for start, end in intervals):
            raise ValueError(f"auxiliary kernel {row[3]!r} intersects a decode sequence or measured interval")
    active = [(partial[1] - partial[0] + merge[1] - merge[0]) / 1000 for partial, merge in pairs]
    wall = [(merge[1] - partial[0]) / 1000 for partial, merge in pairs]
    result = {
        "kernel_names": identities, "kernel_name_field": "demangledName" if full_names else "shortName",
        "kernel_launches": dict(counts), "sequence_count": expected,
        "trace_device_id": selected[0][4], "trace_stream_id": selected[0][5],
        "trace_context_id": selected[0][6] if "contextId" in columns else None,
        "trace_gpu_metadata": gpu_metadata,
        "selected_kernel_timestamps_ns": [{"name": row[3], "start": row[0], "end": row[1]} for row in selected],
        "auxiliary_kernel_counts": dict(sorted(Counter(row[3] for row in auxiliary).items())),
        "individual_kernel_samples_us": {name: [(row[1] - row[0]) / 1000 for row in selected
                                                   if row[3] == name][split:] for name in identities},
    }
    for metric, values in (("active", active), ("wall", wall)):
        result[f"correctness_{metric}_samples_us"] = values[:CORRECTNESS_LAUNCHES]
        result[f"warmup_{metric}_samples_us"] = values[CORRECTNESS_LAUNCHES:split]
        result[f"{metric}_samples_us"] = values[split:]
    return result


def aggregate_rounds(entries):
    return {label: {metric: common.aggregate_rounds([
        {"label": label, "samples_us": entry[f"{metric}_samples_us"],
         "median_us": statistics.median(entry[f"{metric}_samples_us"])}
        for entry in entries if entry["label"] == label])[label]
        for metric in ("active", "wall")} for label in dict.fromkeys(entry["label"] for entry in entries)}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--rust-build-manifest", type=Path,
                        help="optional build JSON containing matching binary_sha256")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reference-runner", type=Path, default=HERE / "launch.py")
    parser.add_argument("--rust-kernel-names", nargs=2, metavar=("PARTIAL", "MERGE"),
                        help="exact trace names; defaults resolve unique fmha_decode_partial/merge substrings")
    parser.add_argument("--reference-kernel-names", nargs=2, metavar=("PARTIAL", "MERGE"),
                        help="exact trace names; defaults resolve fmha_decode_partial/merge substrings")
    parser.add_argument("--nsys", default=shutil.which("nsys") or "nsys")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-q", type=int, default=1, choices=(1,))
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--specialization", choices=("auto", "runtime", "static"), default="auto",
                        help="match both implementations' shape specialization; auto uses the shared static portfolio")
    parser.add_argument("--noncausal", dest="causal", action="store_false", default=False)
    parser.add_argument("--input-scale", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--zero-query", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", "--iters", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if min(args.seq_k, args.heads, args.samples, args.rounds) <= 0:
        parser.error("KV length, heads, samples and rounds must be positive")
    if args.seq_k * args.heads * 64 > 0xFFFFFFFF:
        parser.error("tensor element count must fit u32 indexing")
    if args.device < 0 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("device/warmup must be nonnegative and timeout finite and positive")
    try:
        effective_specialization(args.seq_k, args.heads, args.specialization)
    except ValueError as error:
        parser.error(str(error))
    if any(not value.strip() for values in (args.rust_kernel_names or [], args.reference_kernel_names or [])
           for value in values):
        parser.error("kernel names must be nonempty")
    for field in ("rust_binary", "reference_runner", "rust_build_manifest"):
        path = getattr(args, field)
        if path is not None:
            if not path.is_file() or field == "rust_binary" and not os.access(path, os.X_OK):
                parser.error(f"{field} must be an existing {'executable ' if field == 'rust_binary' else ''}file")
            setattr(args, field, path.resolve())
    return args


def main(argv=None):
    args = parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="decode-", dir=args.output_dir.resolve()))
    environment = os.environ.copy()
    report = {
        "status": "running", "metric_unit": "us", "workload": "fmha-decode",
        "metrics": {"active": "sum of partial and merge CUPTI kernel durations",
                    "wall": "merge end minus partial start, including inter-kernel gap"},
        "duration_source": "CUPTI_ACTIVITY_KIND_KERNEL timestamps (nanoseconds / 1000)",
        "reference_origin": "independently authored cuda-oxide CuTeDSL decode example; not an official upstream decode baseline",
        "started_ns": time.time_ns(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {key: getattr(args, key) for key in ("seq_q", "seq_k", "heads", "causal",
                       "input_scale", "zero_query", "device", "specialization", "warmup", "samples", "rounds", "timeout")},
        "shape_specialization": effective_specialization(args.seq_k, args.heads, args.specialization),
        "static_portfolio": [list(shape) for shape in STATIC_PORTFOLIO],
        "launch_contract": {"correctness_sequences": 2, "warmup_sequences": args.warmup,
                            "measured_sequences": args.samples, "kernels_per_sequence": 2,
                            "kernel_sequence": KERNEL_SEQUENCE},
        "environment": {key: environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                        "CUTE_DSL_ARCH", "CUDA_HOME", "CUDA_PATH")},
        "aggregation": "median per round, then median of round medians, separately for active and wall",
        "runner_event_times": "retained in runner_report; excluded from Nsight timing summaries",
        "execution_order": [["rust", "authored-cute-dsl"] if i % 2 == 0 else ["authored-cute-dsl", "rust"]
                            for i in range(args.rounds)],
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
        provenance["source_snapshot"] = common.source_snapshot(args.reference_runner)
        provenance["rust_build"] = {"status": "unavailable; current source pins do not prove the binary's build toolchain"}
        if args.rust_build_manifest:
            manifest = json.loads(args.rust_build_manifest.read_text(), parse_constant=reject_nonfinite_json,
                                  parse_float=finite_json_float)
            if not isinstance(manifest, dict) or manifest.get("binary_sha256") != provenance["rust_binary"]["sha256"]:
                raise ValueError("Rust build manifest must contain binary_sha256 matching --rust-binary")
            provenance["rust_build"] = {"file": file_identity(args.rust_build_manifest), "manifest": manifest}
        for key, command in (
            ("nsys_version", [args.nsys, "--version"]), ("python_version", [args.python, "--version"]),
            ("git_head", ["git", "-C", str(HERE), "rev-parse", "HEAD"]),
            ("git_status", ["git", "-C", str(HERE), "status", "--short"]),
        ):
            log = directory / f"{key}.log"
            execute(command, log)
            provenance[key] = log.read_text().strip()
        report["expected_input_hashes"] = expected_input_hashes(args)
        expected_output = cpu_oracle(args)
        cli = ["--seq-q", "1", "--seq-k", str(args.seq_k), "--heads", str(args.heads), "--noncausal",
               "--input-scale", str(args.input_scale), "--device", str(args.device),
               "--specialization", args.specialization,
               "--warmup", str(args.warmup), "--iters", str(args.samples)]
        if args.zero_query:
            cli.append("--zero-query")
        implementations = {"rust": [str(args.rust_binary), "--rust-only"], "authored-cute-dsl": [args.python, str(args.reference_runner)]}
        output_hashes, kernel_names, baseline = {}, {}, None
        for round_index, order in enumerate(report["execution_order"], start=1):
            outputs = {}
            for label in order:
                prefix = directory / f"round-{round_index:02d}-{label}"
                result_path, output_path = prefix.with_suffix(".json"), prefix.with_suffix(".f16")
                trace_path, database_path = prefix.with_suffix(".nsys-rep"), prefix.with_suffix(".sqlite")
                profile = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                           "--output", str(prefix), *implementations[label], *cli,
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
                execute(profile, prefix.with_suffix(".profile.log"))
                checked, output = validate_report(result_path, output_path, args, active["started_ns"],
                                                  report["expected_input_hashes"], expected_output)
                require_fields(checked, {"implementation": "cute-rs" if label == "rust" else "cutlass-cute-dsl"},
                               "report")
                active["runner_report"] = checked
                active["independent_cpu_validation"] = fixture.verify(output, expected_output)
                outputs[label] = output
                gpu = {key: checked["provenance"][key] for key in ("gpu", "compute_capability", "device")}
                if baseline is None:
                    baseline = gpu
                    provenance["gpu"] = gpu
                elif gpu != baseline:
                    raise ValueError("GPU identity changed between runners/rounds")
                if label in output_hashes and output_hashes[label] != checked["output_hash"]:
                    raise ValueError(f"{label} output changed across rounds")
                output_hashes[label] = checked["output_hash"]
                if not trace_path.is_file():
                    raise ValueError(f"missing raw Nsight Systems trace: {trace_path}")
                execute(export, prefix.with_suffix(".export.log"))
                names = args.rust_kernel_names if label == "rust" else args.reference_kernel_names
                trace_data = sequence_samples(database_path, args.warmup, args.samples, expected_names=names,
                                              name_contains=None if names else KERNEL_SEQUENCE)
                if label in kernel_names and kernel_names[label] != trace_data["kernel_names"]:
                    raise ValueError(f"{label} kernel names changed across rounds")
                kernel_names[label] = trace_data["kernel_names"]
                if "trace_device_id" not in provenance:
                    provenance["trace_device_id"] = trace_data["trace_device_id"]
                elif trace_data["trace_device_id"] != provenance["trace_device_id"]:
                    raise ValueError("Nsight kernel device changed between runners/rounds")
                active.update(trace_data, status="passed", finished_ns=time.time_ns())
                save()
            report.setdefault("cross_checks", []).append({"round": round_index, "status": "passed",
                **fixture.verify(outputs["rust"], outputs["authored-cute-dsl"])})
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
        for label, metrics in report["summary"].items():
            print(f"{label}: " + ", ".join(f"{name} {value['median_of_round_medians_us']:.3f} us"
                  for name, value in metrics.items()) + " (median of round medians)")
    print(f"Comparison report: {directory / 'report.json'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

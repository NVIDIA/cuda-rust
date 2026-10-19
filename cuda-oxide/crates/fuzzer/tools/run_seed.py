#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generate a rustlantis seed, inject it into rustlantis-smoke, and run it.

This is the first automation layer over `rustlantis-smoke`. The smoke example
stays as the stable CPU/GPU execution harness; this script rewrites only its
`src/generated_case.rs` file for each seed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MIR_GENERATOR = ROOT / "crates" / "fuzzer" / "tools" / "mir_generator.py"
SMOKE_EXAMPLE = ROOT / "crates" / "rustc-codegen-cuda" / "examples" / "rustlantis-smoke"
GENERATED_CASE = SMOKE_EXAMPLE / "src" / "generated_case.rs"
ARTIFACTS = ROOT / "crates" / "fuzzer" / "artifacts"
SUMMARY = ARTIFACTS / "summary.jsonl"


def run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def display_path(path: Path) -> str:
    """Render an artifact path relative to the repo root.

    The absolute form ties the printed summary (and the transcripts quoted
    in the fuzzer README) to one particular checkout; the repo-relative form
    is what a reader can actually resolve.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def reason_from_output(output: str) -> str:
    prefixes = (
        "Unsupported construct:",
        "unsupported ",
        "expected ",
        "Symbol ",
        "Translation failed:",
        "Compilation error:",
    )
    for line in output.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            return stripped
    return output.splitlines()[0].strip() if output.splitlines() else "no output"


def generated_case_snapshot() -> str | None:
    if not GENERATED_CASE.exists():
        return None
    return GENERATED_CASE.read_text()


def write_log(
    *,
    seed: int,
    status: str,
    stage: str,
    reason: str,
    command: list[str],
    returncode: int,
    output: str,
    include_generated_case: bool,
) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"seed-{seed}-{status.lower()}.log"
    lines = [
        f"seed: {seed}",
        f"status: {status}",
        f"stage: {stage}",
        f"reason: {reason}",
        f"returncode: {returncode}",
        f"command: {shlex.join(command)}",
        "",
        "=== command output ===",
        output.rstrip(),
        "",
    ]

    if include_generated_case:
        case = generated_case_snapshot()
        if case is not None:
            lines.extend(
                [
                    "=== generated_case.rs ===",
                    case.rstrip(),
                    "",
                ]
            )

    path.write_text("\n".join(lines))
    return path


def append_summary(record: dict[str, object]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def make_record(
    *,
    seed: int,
    status: str,
    stage: str,
    reason: str,
    log_path: Path | None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "status": status,
        "stage": stage,
        "reason": reason,
        "log": display_path(log_path) if log_path else None,
    }


def clear_artifacts() -> None:
    """Start every fuzzer run with a clean artifacts directory."""
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def remove_stale_ptx() -> None:
    for name in ("rustlantis_smoke.ptx", "rustlantis_smoke.ll"):
        path = SMOKE_EXAMPLE / name
        if path.exists():
            path.unlink()


def classify_run(returncode: int, output: str) -> tuple[str, str, str]:
    if returncode == 0 and (
        "\nMATCH\n" in output or "PASS: CPU/GPU traces match" in output
    ):
        return ("PASS", "run", "CPU/GPU traces matched")
    if "MISMATCH" in output:
        return ("MISMATCH", "run", "CPU/GPU traces differed")
    return ("COMPILE_FAIL", "backend", reason_from_output(output))


def run_seed(seed: int, *, no_build: bool, keep_logs: bool) -> dict[str, object]:
    generator_cmd = [
        sys.executable,
        str(MIR_GENERATOR),
        "--seed",
        str(seed),
        "--output",
        str(GENERATED_CASE),
    ]
    if no_build:
        generator_cmd.append("--no-build")

    generated = run(generator_cmd, cwd=ROOT)
    if generated.returncode != 0:
        status = "UNSUPPORTED"
        stage = "adapter"
        reason = reason_from_output(generated.stdout)
        log_path = write_log(
            seed=seed,
            status=status,
            stage=stage,
            reason=reason,
            command=generator_cmd,
            returncode=generated.returncode,
            output=generated.stdout,
            include_generated_case=False,
        )
        record = make_record(
            seed=seed,
            status=status,
            stage=stage,
            reason=reason,
            log_path=log_path,
        )
        append_summary(record)
        print(f"seed {seed}: {status} [{stage}] {reason} ({display_path(log_path)})")
        return record

    remove_stale_ptx()
    # --no-fmad keeps the two backends comparable on float seeds. Device
    # codegen contracts an fmul feeding an fadd into a single fma.rn by
    # default, matching nvcc's --fmad=true, while the CPU oracle rounds the
    # multiply and the add separately. Both are correct, so the resulting
    # trace difference says nothing about a miscompile, and it would mask the
    # differences that do. Contraction is worth fuzzing on its own terms, as
    # a comparison against a contracted reference.
    run_cmd = ["cargo", "oxide", "run", "--no-fmad", "rustlantis-smoke"]
    result = run(run_cmd, cwd=ROOT)
    status, stage, reason = classify_run(result.returncode, result.stdout)

    if status == "PASS":
        print(f"seed {seed}: PASS")
        log_path = None
        if keep_logs:
            log_path = write_log(
                seed=seed,
                status=status,
                stage=stage,
                reason=reason,
                command=run_cmd,
                returncode=result.returncode,
                output=result.stdout,
                include_generated_case=True,
            )
            print(f"  log: {display_path(log_path)}")
    else:
        log_path = write_log(
            seed=seed,
            status=status,
            stage=stage,
            reason=reason,
            command=run_cmd,
            returncode=result.returncode,
            output=result.stdout,
            include_generated_case=True,
        )
        print(f"seed {seed}: {status} [{stage}] {reason} ({display_path(log_path)})")

    record = make_record(
        seed=seed,
        status=status,
        stage=stage,
        reason=reason,
        log_path=log_path,
    )
    append_summary(record)
    return record


def print_run_summary(records: list[dict[str, object]], statuses: dict[str, int]) -> None:
    print("\nresults:")
    for record in records:
        seed = record["seed"]
        status = record["status"]
        stage = record["stage"]
        reason = record["reason"]
        log = record["log"]
        suffix = f" ({log})" if log else ""
        print(f"  seed {seed}: {status} [{stage}] {reason}{suffix}")

    summary = ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
    print(f"summary: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="single seed to run")
    parser.add_argument("--start", type=int, default=0, help="first seed for a range")
    parser.add_argument("--count", type=int, default=1, help="number of seeds to run")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip building the vendored rustlantis generator",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="write logs for passing seeds as well as failures",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after the first non-PASS seed",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else range(args.start, args.start + args.count)
    statuses: dict[str, int] = {}
    records: list[dict[str, object]] = []

    clear_artifacts()

    for idx, seed in enumerate(seeds):
        # Build the vendored rustlantis generator once, then reuse it.
        skip_generator_build = args.no_build or idx > 0
        record = run_seed(seed, no_build=skip_generator_build, keep_logs=args.keep_logs)
        records.append(record)
        status = str(record["status"])
        statuses[status] = statuses.get(status, 0) + 1
        if status != "PASS" and not args.keep_going:
            break

    print_run_summary(records, statuses)
    return 0 if statuses.get("PASS", 0) == sum(statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

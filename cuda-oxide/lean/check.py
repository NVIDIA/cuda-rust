#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the mathematical access model; this does not extract or certify Rust."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ALLOWED_AXIOMS = {"propext", "Quot.sound"}
REJECTED = (
    ("RejectedInclusiveGuard.lean", "inclusive output guard"),
    ("RejectedLaneOnly.lean", "duplicate writers across blocks"),
    ("RejectedMissingLength.lean", "missing input-length precondition"),
    ("RejectedCoverage.lean", "insufficient launch coverage"),
)


def find_lake():
    executable = shutil.which("lake")
    if executable:
        # Preserve a proxy's filename: Elan dispatches using the name "lake".
        return str(Path(executable).absolute())
    elan_home = Path(os.environ.get("ELAN_HOME") or Path.home() / ".elan").expanduser()
    candidate = elan_home.resolve() / "bin" / "lake"
    executable = shutil.which(str(candidate))
    if executable:
        return str(candidate)
    raise RuntimeError(
        f"Could not find Lean's build tool 'lake' on PATH or at {candidate}. "
        "Install Lean through Elan, or add your Lean installation's bin directory to PATH."
    )


def run_lake(*args):
    result = subprocess.run(
        [find_lake(), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    output = result.stdout
    if result.returncode < 0:
        raise RuntimeError(f"lake terminated by a signal:\n{output}")
    if "sorryAx" in output or "declaration uses 'sorry'" in output:
        raise RuntimeError(f"An admitted proof was reported:\n{output}")
    for group in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", output):
        axioms = {name.strip() for name in group.split(",") if name.strip()}
        unexpected = axioms - ALLOWED_AXIOMS
        if unexpected:
            raise RuntimeError(f"Unexpected proof dependencies: {sorted(unexpected)}\n{output}")
    return result


def require_success(*args):
    result = run_lake(*args)
    if result.returncode != 0:
        raise RuntimeError(f"lake {' '.join(args)} failed:\n{result.stdout}")


def main():
    require_success("build")
    require_success("env", "lean", "tests/exporter/Scalar.lean")
    require_success("env", "lean", "tests/vecadd/Accepted.lean")
    require_success("env", "lean", "tests/vecadd/Counterexamples.lean")
    print("PASS: scalar semantics, vecadd access model, typed indices, and counterexamples.", flush=True)
    for filename, label in REJECTED:
        result = run_lake("env", "lean", f"tests/vecadd/{filename}")
        if result.returncode == 0:
            raise RuntimeError(f"The false obligation unexpectedly passed: {filename}")
        # Require a failed proof obligation, not an import or tool error.
        if "error: omega could not prove the goal:" not in result.stdout:
            raise RuntimeError(f"Unexpected failure in {filename}:\n{result.stdout}")
        print(f"REJECTED as expected: {label}.", flush=True)
    print("Model checks passed. Rust/MIR extraction is not implemented by this checker.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

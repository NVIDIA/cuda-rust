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
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
REJECTED = (
    ("RejectedInclusiveGuard.lean", "inclusive output guard"),
    ("RejectedLaneOnly.lean", "duplicate writers across blocks"),
    ("RejectedMissingLength.lean", "missing input-length precondition"),
    ("RejectedCoverage.lean", "insufficient launch coverage"),
    ("RejectedTmemColOverflow.lean", "TMEM column range overflow"),
    ("RejectedTmemWarpSlice.lean", "TMEM warpgroup lane-slice violation"),
    ("RejectedMutSliceRace.lean", "cross-thread &mut [T] read-write race"),
    ("RejectedTrappingWrite.lean", "trapping mir.assert before writes_all_elems store"),
    ("RejectedInsufficientRequiresScale.lean", "insufficient scaled requires precondition"),
    ("RejectedDivergentSharedRace.lean", "shared-memory race masked by divergent sync_threads"),
    ("tests/program/RejectedUnknownGuard.lean", "branch pruned on an untracked guard"),
    ("tests/program/RejectedUnknownWriteCoverage.lean", "write coverage claimed across an untracked guard"),
    ("tests/program/RejectedDivergentLoopUniformity.lean", "barrier loop bounded by a split-thread join"),
    ("tests/program/RejectedCrossWarpSyncMaskRace.lean", "cross-warp shared-memory race across sync_mask alone"),
    ("tests/program/Rejected2DNon32RowSyncMaskRace.lean", "2D non-multiple-of-32 row cross-warp sync_mask race"),
)
# Refused by the CTA state machine: `decide` evaluates `checkUniformBarriersWith` to `false`.
REJECTED_BARRIERS = (
    ("RejectedTmaReadAfterSync.lean", "TMA destination read after sync_threads without an mbarrier wait"),
    ("RejectedSubwarpSyncMask.lean", "sub-warp divergent warp::sync_mask"),
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
    require_success("env", "lean", "tests/program/UnknownGuard.lean")
    require_success("env", "lean", "tests/program/OrSplitAfterIteFuel.lean")
    print("PASS: scalar semantics, vecadd access model, typed indices, and counterexamples.", flush=True)
    for filename, label in REJECTED:
        rel_path = filename if "/" in filename else f"tests/vecadd/{filename}"
        result = run_lake("env", "lean", rel_path)
        if result.returncode == 0:
            raise RuntimeError(f"The false obligation unexpectedly passed: {filename}")
        # Require a failed proof obligation, not an import or tool error.
        if "error: omega could not prove the goal:" not in result.stdout:
            raise RuntimeError(f"Unexpected failure in {filename}:\n{result.stdout}")
        print(f"REJECTED as expected: {label}.", flush=True)
    for filename, label in REJECTED_BARRIERS:
        result = run_lake("env", "lean", f"tests/vecadd/{filename}")
        if result.returncode == 0:
            raise RuntimeError(f"The false obligation unexpectedly passed: {filename}")
        # Require the barrier check itself to evaluate to `false`, not an import or tool error.
        if ("error: Tactic `decide` proved that the proposition" not in result.stdout
                or "checkUniformBarriersWith" not in result.stdout):
            raise RuntimeError(f"Unexpected failure in {filename}:\n{result.stdout}")
        print(f"REJECTED as expected: {label}.", flush=True)
    print("Model checks passed. Rust/MIR extraction is not implemented by this checker.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the MIR exporter and Lean together; this is not a kernel proof gate."""

from pathlib import Path
import subprocess
import sys
import tempfile

from check import ROOT, require_success, run_lake

REPO = ROOT.parent
FIXTURE = REPO / "crates/lean-exporter/fixtures/arithmetic.mir"
PROOFS = """
namespace CudaOxide.Generated

theorem index_formula (inputs : Fin 3 → BitVec 64) :
    evaluate inputs =
      inputs ⟨0, by decide⟩ * inputs ⟨1, by decide⟩ + inputs ⟨2, by decide⟩ := by
  rfl

example : evaluate (fun i =>
    if i.val = 0 then 3 else if i.val = 1 then 256 else 255) =
    (1023 : BitVec 64) := by decide

-- The exported arithmetic must retain the source operation's wraparound.
example : evaluate (fun i =>
    if i.val = 0 then 18446744073709551615 else 1) =
    (0 : BitVec 64) := by decide

#print axioms index_formula
end CudaOxide.Generated
"""


def export(path):
    return subprocess.run(
        ["cargo", "run", "--quiet", "--package", "lean-exporter", "--",
         str(path), "global_index"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def require_export(path):
    result = export(path)
    if result.returncode != 0:
        raise RuntimeError(f"Exporter failed for {path}:\n{result.stderr}")
    return result.stdout


def main():
    require_success("build")
    source = FIXTURE.read_text()
    if source.count("mir.mul") != 1:
        raise RuntimeError("The arithmetic fixture must contain exactly one mir.mul")
    with tempfile.TemporaryDirectory(prefix="cuda-oxide-lean-export-") as directory:
        temporary = Path(directory)
        generated = require_export(FIXTURE)
        valid = temporary / "Valid.lean"
        valid.write_text(generated + PROOFS)
        require_success("env", "lean", str(valid))
        print("PASS: generated Lean proves the MIR arithmetic formula and wraparound.", flush=True)

        mutated_mir = temporary / "mutated.mir"
        mutated_mir.write_text(source.replace("mir.mul", "mir.add", 1))
        mutated = require_export(mutated_mir)
        if mutated == generated:
            raise RuntimeError("Changing a MIR operation did not change the generated Lean")
        invalid = temporary / "Mutated.lean"
        # Lean inserts an internal placeholder after a failed declaration.
        # Audit dependencies only on the successful fixture; this file must
        # instead fail with the specific formula-proof diagnostic below.
        invalid.write_text(mutated + PROOFS.replace("#print axioms index_formula\n", ""))
        result = run_lake("env", "lean", str(invalid))
        if result.returncode == 0 or "Tactic `rfl` failed" not in result.stdout:
            raise RuntimeError(f"Expected the arithmetic formula proof to fail:\n{result.stdout}")
        print("REJECTED as expected: changed MIR no longer proves the original formula.", flush=True)

        unsupported_mir = temporary / "unsupported.mir"
        unsupported_mir.write_text(source.replace("mir.mul", "mir.div", 1))
        rejected = export(unsupported_mir)
        if rejected.returncode <= 0 or rejected.stdout or "mir.div" not in rejected.stderr:
            raise RuntimeError(f"Expected explicit rejection without Lean output:\n{rejected.stderr}")
        print("REJECTED as expected: unsupported MIR operation emits no Lean.", flush=True)
    print("Exporter checks passed for scalar MIR fixtures; full vecadd extraction remains unsupported.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

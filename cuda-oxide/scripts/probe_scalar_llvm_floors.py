# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 ssoj13
# SPDX-License-Identifier: Apache-2.0
# Adapted from cuda-intrinsics-gen's licensed probe contracts; retains their attribution.
from pathlib import Path
import json
import hashlib
import subprocess
import re
import argparse
import tomllib

parser = argparse.ArgumentParser(
    description="Reproduce scalar arithmetic LLVM floor evidence"
)
parser.add_argument("--llc", type=Path, required=True)
parser.add_argument("--ptxas", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.out
out.mkdir(parents=True, exist_ok=True)
llc = args.llc
ptxas = args.ptxas


def run(args):
    r = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{args}: {r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


catalog = json.loads((root / "intrinsics/catalog.json").read_text(encoding="utf-8"))
records = [r for r in catalog["intrinsics"] if r["family"] == "scalar_arithmetic"]
assert len(records) == 64
capture_root = root / "intrinsics/evidence/scalar-arithmetic-sm20-20261013"
inputs = {
    r["id"]: r
    for r in map(
        json.loads,
        (capture_root / "inputs.jsonl").read_text(encoding="utf-8").splitlines(),
    )
}
saved = json.loads((capture_root / "captures.json").read_text(encoding="utf-8"))
expected_input = {r["id"]: r["input_sha256"] for r in saved["captures"]}
input_dir = out / "inputs"
input_dir.mkdir(exist_ok=True)
proof = {
    "llvm_source": tomllib.loads(
        (root / "intrinsics/upstream.lock").read_text(encoding="utf-8")
    )["llvm"]["revision"],
    "llc": {
        "path": str(llc),
        "version": run([llc, "--version"]),
        "sha256": digest(llc),
    },
    "ptxas": {
        "path": str(ptxas),
        "version": run([ptxas, "--version"]),
        "sha256": digest(ptxas),
    },
    "captures": [],
}
for r in records:
    source = input_dir / f"{r['id']}.ll"
    text = inputs[r["id"]]["probe_source"]
    candidates = [text.encode("utf-8"), text.replace("\n", "\r\n").encode("utf-8")]
    data = next(
        b
        for b in candidates
        if hashlib.sha256(b).hexdigest() == expected_input[r["id"]]
    )
    source.write_bytes(data)
    instruction = (
        r["expected_ptx"]["mnemonic"] + "." + ".".join(r["expected_ptx"]["modifiers"])
    )
    for arch, feature, terminal in [
        ("sm_20", "+ptx32", "sm_75"),
        ("sm_75", "+ptx63", "sm_75"),
        ("sm_80", "+ptx70", "sm_80"),
    ]:
        stem = out / f"{r['id']}-{arch}"
        ptx = stem.with_suffix(".ptx")
        cubin = stem.with_suffix(".cubin")
        run(
            [
                llc,
                source,
                "-march=nvptx64",
                f"-mcpu={arch}",
                f"-mattr={feature}",
                "-o",
                ptx,
            ]
        )
        text = ptx.read_text(encoding="utf-8")
        assert len(re.findall(r"\b" + re.escape(instruction) + r"\s", text)) == 1, (
            r["id"],
            arch,
            instruction,
        )
        assert ".extern .func" not in text, (r["id"], arch, "unresolved extern")
        run([ptxas, f"-arch={terminal}", ptx, "-o", cubin])
        proof["captures"].append(
            {
                "id": r["id"],
                "codegen_gpu": arch,
                "ptx_feature": feature,
                "terminal_gpu": terminal,
                "input_sha256": digest(source),
                "ptx_sha256": digest(ptx),
                "cubin_sha256": digest(cubin),
                "instruction": instruction,
            }
        )
    print(r["id"], "3 codegen/assembly targets passed", flush=True)
(out / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(f"All {len(proof['captures'])} captures passed", flush=True)

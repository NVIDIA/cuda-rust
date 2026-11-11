#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Keep the artifact format version consistent. Only the isolated backend uses
# the in-tree COFF writer; host runtimes retain the registry crate.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from pathlib import Path
import subprocess
import sys
import tomllib

def read_toml(path):
    with Path(path).open("rb") as file:
        return tomllib.load(file)

git_root = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True,
).strip())
simt_root = Path.cwd()
version = read_toml("crates/oxide-artifacts/Cargo.toml")["package"]["version"]
root = read_toml("Cargo.toml")
backend_path = simt_root / "crates/rustc-codegen-cuda"
backend = read_toml(backend_path / "Cargo.toml")
errors = []
for name, dependency in (
    ("cuda-oxide/Cargo.toml", root["workspace"]["dependencies"]["oxide-artifacts"]),
    ("cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml", backend["dependencies"]["oxide-artifacts"]),
):
    requirement = dependency if isinstance(dependency, str) else dependency.get("version")
    if requirement != version:
        errors.append(f"{name}: oxide-artifacts must require {version}")

patch = backend.get("patch", {}).get("crates-io", {}).get("oxide-artifacts", {})
writer_path = patch.get("path")
if not writer_path or (backend_path / writer_path).resolve() != (simt_root / "crates/oxide-artifacts").resolve():
    errors.append("backend must patch oxide-artifacts to the in-tree writer")

locks = subprocess.check_output(
    ["git", "-C", str(git_root), "ls-files", "Cargo.lock", "*/Cargo.lock", "**/Cargo.lock"], text=True,
).splitlines()
for path in locks:
    lock_path = git_root / path
    packages = [p for p in read_toml(lock_path).get("package", []) if p["name"] == "oxide-artifacts"]
    if not packages:
        continue
    if any(package["version"] != version for package in packages):
        errors.append(f"{path}: inconsistent oxide-artifacts format version")
    if lock_path == backend_path / "Cargo.lock":
        if len(packages) != 1 or "source" in packages[0]:
            errors.append(f"{path}: backend must resolve one in-tree oxide-artifacts crate")
    elif not any(package.get("source", "").startswith("registry+") for package in packages):
        errors.append(f"{path}: host runtime must consume oxide-artifacts from crates.io")

if errors:
    sys.exit("\n".join(f"error: {error}" for error in errors))
print(f"OK: oxide-artifacts {version}; registry host runtime, in-tree backend writer.")
PY

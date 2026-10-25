#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Keep artifact writers and readers on one in-tree implementation and version.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."
GIT_ROOT="$(git rev-parse --show-toplevel)"

python3 - "${GIT_ROOT}" <<'PYTHON'
import pathlib
import subprocess
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
artifact_dir = root / "cuda-oxide/crates/oxide-artifacts"

def read_toml(path):
    with path.open("rb") as source:
        return tomllib.load(source)

version = read_toml(artifact_dir / "Cargo.toml")["package"]["version"]
errors = []
for relative, workspace in [
    ("cuda-core/Cargo.toml", False),
    ("cuda-oxide/Cargo.toml", True),
    ("cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml", False),
]:
    manifest = root / relative
    data = read_toml(manifest)
    dependencies = data["workspace"]["dependencies"] if workspace else data["dependencies"]
    dependency = dependencies["oxide-artifacts"]
    if not isinstance(dependency, dict) or dependency.get("version") != version:
        errors.append(f"{relative}: oxide-artifacts must declare version {version}")
        continue
    path = dependency.get("path")
    if path is None or (manifest.parent / path).resolve() != artifact_dir.resolve():
        errors.append(f"{relative}: oxide-artifacts must use {artifact_dir}")

lockfiles = subprocess.check_output(
    ["git", "-C", str(root), "ls-files", "-z", "*Cargo.lock"]
).decode().split("\0")
checked = 0
for relative in filter(None, lockfiles):
    packages = [
        package for package in read_toml(root / relative).get("package", [])
        if package["name"] == "oxide-artifacts"
    ]
    if not packages:
        continue
    checked += 1
    if len(packages) != 1 or packages[0]["version"] != version or "source" in packages[0]:
        errors.append(f"{relative}: oxide-artifacts must resolve once to the in-tree version {version}")

for error in errors:
    print(f"error: {error}", file=sys.stderr)
if errors:
    sys.exit(1)
print(f"OK: oxide-artifacts {version} uses one in-tree source in all three manifests and {checked} locks.")
PYTHON

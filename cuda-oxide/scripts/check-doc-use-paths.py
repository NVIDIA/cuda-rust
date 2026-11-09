#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile every `use` of the cuda-oxide and shared host crates in markdown.

Readers copy imports from the book and READMEs, but no build reads a fenced
block in a markdown page. This collects each `use cuda_device|cuda_host|
cuda_core|cuda_async|cuda_bindings::...` statement from ```rust fences in
tracked markdown under cuda-oxide/, docs/ and the shared host crates, compiles
each path in its own module of a throwaway crate, and reports every path that
does not resolve with the page and line it came from.

Only import paths are checked. Call sites, items from other crates, and paths
mentioned in prose are not. Needs the SIMT toolchain and a CUDA toolkit
(CUDA_TOOLKIT_PATH or CUDA_HOME), because cuda-core builds cuda-bindings.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("cuda-oxide/", "docs/", "cuda-core/", "cuda-async/", "cuda-bindings/", "cuda-core-derive/")
CRATES = ("cuda_device", "cuda_host", "cuda_core", "cuda_async", "cuda_bindings")
FENCE = re.compile(r"^```rust[^\n]*\n(.*?)^```", re.S | re.M)
USE = re.compile(r"^[ \t]*(?:pub[ \t]+)?use[ \t]+((?:%s)\b[^;]*);" % "|".join(CRATES), re.M)


def markdown_pages():
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "*.md"], check=True, capture_output=True, text=True
    ).stdout.split()
    return [p for p in tracked if p.startswith(SCOPES) or p == "README.md"]


def expand(path):
    """`a::{b, c::{d, e}, self} as x` -> [a::b, a::c::d, a::c::e, a]."""
    path = re.sub(r"\s+as\s+\w+", "", path)
    path = re.sub(r"\s+", "", path)
    match = re.fullmatch(r"(.*?)::\{(.*)\}", path)
    if not match:
        return [path]
    head, body = match.groups()
    parts, depth, current = [], 0, ""
    for ch in body:
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        depth += ch == "{"
        depth -= ch == "}"
        current += ch
    parts.append(current)
    paths = []
    for part in filter(None, parts):
        paths += [head] if part == "self" else expand(f"{head}::{part}")
    return paths


def collect():
    """Map each distinct path to the first page:line that imports it."""
    found, pages = {}, set()
    for page in markdown_pages():
        text = (ROOT / page).read_text(errors="replace")
        for fence in FENCE.finditer(text):
            for use in USE.finditer(fence.group(1)):
                line = text.count("\n", 0, fence.start(1) + use.start()) + 1
                pages.add(page)
                for path in expand(use.group(1)):
                    found.setdefault(path, f"{page}:{line}")
    return found, pages


def write_probe(crate, paths):
    deps = {
        "cuda-device": "cuda-oxide/crates/cuda-device",
        "cuda-host": "cuda-oxide/crates/cuda-host",
        "cuda-core": "cuda-core",
        "cuda-async": "cuda-async",
        "cuda-bindings": "cuda-bindings",
    }
    manifest = ['[package]\nname = "doc-use-paths"\nversion = "0.0.0"\nedition = "2024"\n\n[workspace]\n\n[dependencies]']
    for name, rel in deps.items():
        features = ', features = ["async"]' if name == "cuda-host" else ""
        manifest.append(f'{name} = {{ path = "{(ROOT / rel).as_posix()}"{features} }}')
    (crate / "Cargo.toml").write_text("\n".join(manifest) + "\n")
    # The probe declares its own [workspace], so it must carry the SIMT pin.
    shutil.copy(ROOT / "cuda-oxide" / "rust-toolchain.toml", crate / "rust-toolchain.toml")
    (crate / "src").mkdir()
    lines = ["#![allow(unused_imports)]"]
    lines += [f"mod p{i} {{ use {path}; }}" for i, path in enumerate(paths)]
    (crate / "src" / "lib.rs").write_text("\n".join(lines) + "\n")


def main():
    found, pages = collect()
    paths = sorted(found)
    with tempfile.TemporaryDirectory(prefix="doc-use-paths-") as tmp:
        crate = Path(tmp)
        write_probe(crate, paths)
        # cuda-device, cuda-host and cuda-macros use #![feature] on the stable pin.
        env = dict(os.environ, RUSTC_BOOTSTRAP="1")
        env.setdefault("CARGO_TARGET_DIR", str(crate / "target"))
        result = subprocess.run(
            ["cargo", "check", "--quiet", "--message-format=json"],
            cwd=crate, env=env, capture_output=True, text=True,
        )
    failures = []
    for line in result.stdout.splitlines():
        message = json.loads(line) if line.startswith("{") else {}
        diagnostic = message.get("message") or {}
        if message.get("reason") != "compiler-message" or diagnostic.get("level") != "error":
            continue
        spans = [s for s in diagnostic.get("spans", []) if s.get("file_name") == "src/lib.rs"]
        if not spans:
            continue
        path = paths[spans[0]["line_start"] - 2]
        failures.append(f"{found[path]}: `use {path};` does not resolve: {diagnostic['message']}")
    if failures:
        print("error: imports in markdown that do not compile:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  {failure}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("error: the probe crate failed to build for a reason other than an import", file=sys.stderr)
        return 1
    print(f"OK: all {len(paths)} cuda_* import paths in markdown resolve ({len(pages)} pages import them).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the SM120a C++ primitives example and record reproducible evidence; no GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc",
                        help="CUDA compiler executable (path or command on PATH)")
    parser.add_argument("--cxx", default="c++",
                        help="host compiler for both CPU tests and nvcc -ccbin")
    args = parser.parse_args()
    for option in ("nvcc", "cxx"):
        executable = shutil.which(getattr(args, option))
        if executable is None:
            parser.error(f"--{option}: executable not found: {getattr(args, option)}")
        setattr(args, option, str(Path(executable).resolve(strict=True)))
    source = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
    keep = directory / "keep"
    keep.mkdir()
    manifest = {"status": "building", "directory": str(directory), "commands": [],
                "compilers": {"cuda": args.nvcc, "host": args.cxx},
                "sources": {name: sha256(source / name) for name in
                            ("blockscale_gemm.cu", "layout.cuh", "test_layout.cpp", "build.py")}}
    manifest_path = directory / "manifest.json"

    def record():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(command, log_name):
        manifest["commands"].append(command)
        record()
        result = subprocess.run(command, cwd=directory, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / log_name).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"exit {result.returncode}: {command}\n{result.stdout}")
        return result.stdout

    record()
    try:
        manifest["nvcc_version"] = run([args.nvcc, "--version"], "nvcc-version.log")
        manifest["cxx_version"] = run([args.cxx, "--version"], "cxx-version.log")
        host_test = directory / "test_layout"
        run([args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             str(source / "test_layout.cpp"), "-o", str(host_test)], "test-build.log")
        manifest["host_tests"] = run([str(host_test)], "test-layout.log").strip()
        binary = directory / "blockscale_gemm_cpp"
        # An explicit gencode avoids nvcc's additional generic compute_120
        # fallback, which cannot represent architecture-specific MXFP4 MMA.
        run([args.nvcc, "-ccbin", args.cxx, "-std=c++17", "-O3",
             "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo", "-Xptxas=-v",
             "--keep", "--keep-dir", str(keep), str(source / "blockscale_gemm.cu"),
             "-lcuda", "-o", str(binary)], "build.log")
        cuobjdump = str(Path(args.nvcc).resolve().parent / "cuobjdump")
        manifest["resources"] = run([cuobjdump, "--dump-resource-usage", str(binary)], "resources.log")
        run([cuobjdump, "--dump-sass", str(binary)], "kernel.sass")
        # These are the exact intermediate files consumed by this executable's build.
        manifest["artifacts"] = {str(path.relative_to(directory)): sha256(path)
                                 for path in sorted(keep.iterdir())
                                 if path.suffix in (".ptx", ".cubin")}
        manifest.update(status="compiled", binary=str(binary), binary_sha256=sha256(binary))
        record()
    except (OSError, RuntimeError) as error:
        manifest.update(status="failed", error=str(error))
        record()
        print(f"FAIL: {error}\nEvidence: {manifest_path}")
        return 1
    print(f"PASS CPU physical-layout checks and SM120a compilation\nBinary: {binary}\nEvidence: {manifest_path}")
    print(manifest["resources"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

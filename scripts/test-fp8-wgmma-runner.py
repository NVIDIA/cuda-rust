#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only check: runner preserves evidence and rejects skips/findings/failures."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

RUNNER = Path(__file__).with_name("validate-fp8-wgmma.sh")
FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
case = os.environ["TEST_CASE"]
if tool == "nvcc" and "-o" in args:
    binary = Path(args[args.index("-o") + 1])
    binary.write_text(Path(sys.argv[0]).read_text())
    binary.chmod(0o755)
if tool == "fp8_vs_bf16":
    if case == "gemm-failure":
        print("GEMM correctness failed")
        sys.exit(13)
    if case != "missing-gemm-timings":
        print(f"RESULT M={args[0]} N={args[1]} K={args[2]} fast_accum={args[3]} speedup=0.900x")
    print("SUCCESS: both formats passed before and after timing (speedup is measured, not required)")
    sys.exit(0)
if tool != "cargo":
    print("mock " + tool)
    sys.exit(0)
with open(os.environ["CALLS"], "a") as log:
    log.write(json.dumps(args) + "\n")
if "run" not in args and "sanitize" not in args:
    sys.exit(0)
if case == "skip":
    print("WARNING: WGMMA requires Hopper; PTX check only")
    sys.exit(0)
if case == "benchmark-failure" and "--check-only" not in args:
    print("timing failed")
    sys.exit(12)
print("SUCCESS: FP8 WGMMA numeric check and BF16 comparison passed")
if case == "missing-timings" and "--check-only" not in args:
    sys.exit(0)
if "sanitize" in args:
    if case == "sanitizer-failure":
        print("========= ERROR SUMMARY: 1 error")
        sys.exit(86)
    if "racecheck" in args:
        hazards = 1 if case == "hazard" else 0
        print(f"========= RACECHECK SUMMARY: {hazards} hazards displayed (0 errors, {hazards} warnings)")
    else:
        print("========= ERROR SUMMARY: 0 errors")
elif "--check-only" not in args:
    print("BF16 m64n64k16 x4: 0.02 ms, 1.00 TFLOPS")
    print("FP8  m64n64k32 x2: 0.01 ms, 2.00 TFLOPS")
'''


def main():
    for case, expected_code in (("pass", 0), ("skip", 1), ("sanitizer-failure", 86),
                                ("hazard", 1), ("benchmark-failure", 12), ("missing-timings", 1),
                                ("gemm-failure", 13), ("missing-gemm-timings", 1)):
        with tempfile.TemporaryDirectory(prefix="fp8-runner-test-") as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copy2(RUNNER, root / "scripts" / RUNNER.name)
            for name in ("bench-fp8-vs-bf16.sh", "test-fp8-wgmma-runner.py"):
                shutil.copy2(RUNNER.with_name(name), root / "scripts" / name)
            example = root / "crates/rustc-codegen-cuda/examples/wgmma_mma_fp8"
            (example / "src").mkdir(parents=True)
            (example / "bench").mkdir()
            for name in ("Cargo.toml", "Cargo.lock", "src/main.rs", "wgmma_mma_fp8.ptx",
                         "bench/fp8_vs_bf16.cu", "README.md", "HOPPER_VALIDATION.md"):
                (example / name).write_text("mock source\n")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            for name in ("cargo", "git", "nvidia-smi", "rustc", "nvcc", "compute-sanitizer", "ptxas"):
                path = fake_bin / name
                path.write_text(FAKE_TOOL)
                path.chmod(0o755)
            env = dict(os.environ, CUDA_TOOLKIT_PATH=tmp, TMPDIR=tmp,
                       TEST_CASE=case, CALLS=str(root / "calls"))
            result = subprocess.run(["bash", str(root / "scripts" / RUNNER.name)],
                                    env=env, text=True, capture_output=True)
            assert result.returncode == expected_code, result.stdout + result.stderr
            archive, = root.glob("fp8-wgmma.*.tar.gz")
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
                status, = [name for name in names if name.endswith("/status.txt")]
                assert bundle.extractfile(status).read() == f"exit_code={result.returncode}\n".encode()
                assert any(name.endswith("/environment.log") for name in names)
                result_name, = [name for name in names if name.endswith("/result.txt")]
                assert bundle.extractfile(result_name).read().startswith(b"PASS:" if case == "pass" else b"FAIL:")
                if case in ("pass", "gemm-failure", "missing-gemm-timings"):
                    assert any(name.endswith("/pr-validation.txt") for name in names)
                if case == "pass":
                    summary, = [name for name in names if name.endswith("/summary.txt")]
                    summary_text = bundle.extractfile(summary).read().decode()
                    assert summary_text.count("RESULT ") == 36
                    assert summary_text.count("fast_accum=0") == 18
                    assert summary_text.count("fast_accum=1") == 18
                    assert "speedup=0.900x" in summary_text  # A slowdown must not fail validation.
            calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
            if case == "pass":
                sanitizers = [args for args in calls if "sanitize" in args]
                assert len(sanitizers) == 4
                for args in sanitizers:
                    assert args[-2:] == ["--", "--check-only"]
                    assert args[args.index("--error-exitcode") + 1] == "86"
                    assert "sm_90a" in args and "--lineinfo" in args
                    if "memcheck" in args or "synccheck" in args:
                        assert args[args.index("--check-warpgroup-mma") + 1] == "yes"
                assert sum("run" in args and "--check-only" not in args for args in calls) == 3
            elif case not in ("gemm-failure", "missing-gemm-timings"):
                assert "ptxas" not in result.stdout
            print(f"PASS: {case}")


if __name__ == "__main__":
    main()

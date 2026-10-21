#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and time Rust and the public Python CTM FP16 reference on identical tensors."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile


def shape(value):
    """Parse M,N,K and enforce the device launch's dimension and index bounds."""
    try:
        dims = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected M,N,K integers") from exc
    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise argparse.ArgumentTypeError("expected three positive M,N,K integers")
    if dims[0] % 256 or dims[1] % 8 or dims[2] % 8:
        raise argparse.ArgumentTypeError("M must be divisible by 256; N and K by 8")
    i32_max = (1 << 31) - 1
    m, n, k = dims
    ntiles = (n + 351) // 352
    if max(dims) > i32_max or (ntiles - 1) * 352 + 320 > i32_max:
        raise argparse.ArgumentTypeError("dimensions and padded TMA coordinates must fit i32")
    if (m // 256) * ntiles > i32_max // 2:
        raise argparse.ArgumentTypeError("two-CTA problem grid exceeds the CUDA grid-x limit")
    if max(m*k, n*k, m*n) > sys.maxsize // 2:
        raise argparse.ArgumentTypeError("FP16 allocation exceeds the host addressable limit")
    return dims


def measure(torch, oracle, kernel, inputs, output, warmup, iters, graph_launches):
    """Time a prepared callable with one launch per event pair, then verify it."""
    if graph_launches != 1:
        raise ValueError("comparison requires one direct kernel launch per sample")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Keep initialization, launches, verification and events on one explicit
    # stream. The prepared callable honors the supplied stream argument.
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            kernel(*inputs, output, stream=stream)
        stream.synchronize()
        # Verify the timed output rather than adding a correctness launch.
        # Poison after warmup so a no-op sample cannot reuse warmup's result.
        output.fill_(float("nan"))
        stream.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record(stream)
            kernel(*inputs, output, stream=stream)
            end.record(stream)
            end.synchronize()
        stream.synchronize()
        oracle.verify("fp16", *inputs, output)
    return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))


def main():
    """Compare both implementations using serialized copies of the same inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-bin", type=Path, required=True, help="compiled Rust host executable")
    parser.add_argument("--mnk", type=shape, default=(256, 352, 64))
    parser.add_argument("--has-bias", "--has_bias", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--timing-mode", choices=("direct",), default="direct",
                        help="one kernel per CUDA-event sample; use --rust-only for graph timing")
    parser.add_argument("--graph-launches", type=int, default=1, help="must be 1 for comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=int, help="override Rust persistent cluster count")
    parser.add_argument("--json", type=Path, help="write combined report")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.graph_launches < 1 or (args.clusters is not None and args.clusters < 1):
        parser.error("warmup must be nonnegative; iters, graph-launches and clusters must be positive")
    if args.graph_launches != 1:
        parser.error("comparison requires direct timing and --graph-launches 1; use --rust-only for graph timing")
    binary = args.rust_bin.resolve(strict=True)
    reference_dir = Path(__file__).resolve().parent / "public_reference"
    try:
        import torch
        import numpy as np
        from public_reference import oracle, gemm
    except ImportError as exc:
        raise RuntimeError(
            f"comparison dependencies could not be imported by {sys.executable}: {exc}; "
            "select the configured comparison environment with --python or CUDA_OXIDE_CUTE_PYTHON"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("this comparison requires an SM100 GPU")
    # Both processes select device ordinal zero. CUDA_VISIBLE_DEVICES is the
    # common way to choose a physical GPU without mixing devices in one run.
    torch.cuda.set_device(0)
    inputs, outputs = oracle.prepare_tensors("fp16", mnk=args.mnk, has_bias=args.has_bias, seed=args.seed)
    python_output = outputs[0]
    # Eager compilation stays outside event timing even with --warmup 0.
    callable_kernel = gemm.prepare_callable(
        has_bias=args.has_bias, inputs=inputs, output=python_output
    )
    python_capacity = int(gemm.cluster_capacity())
    python_clusters = min((args.mnk[0] // 256) * ((args.mnk[1] + 351) // 352), python_capacity)
    python_us = measure(torch, oracle, callable_kernel, inputs, python_output,
                        args.warmup, args.iters, args.graph_launches)
    m, n, k = args.mnk
    with tempfile.TemporaryDirectory(prefix="fp16-gemm-compare-") as temporary:
        directory = Path(temporary)
        # The oracle's B is the [1,K,N] view; the kernel stores [N,K].
        tensors = {"a": inputs[0][0], "b": inputs[1][0].transpose(0, 1)}
        if args.has_bias:
            tensors["bias"] = inputs[2]
        digests = {}
        for name, tensor in tensors.items():
            raw = tensor.detach().contiguous().cpu().numpy().astype("<f2", copy=False).tobytes()
            (directory / f"{name}.f16").write_bytes(raw)
            digests[name] = hashlib.sha256(raw).hexdigest()
        rust_output = directory / "c.f16"
        rust_json = directory / "rust.json"
        command = [str(binary), "--rust-only", "--mnk", ",".join(map(str, args.mnk)),
                   "--warmup", str(args.warmup), "--iters", str(args.iters),
                   "--timing-mode", "direct", "--graph-launches", "1",
                   "--input-dir", str(directory), "--output", str(rust_output),
                   "--json", str(rust_json), "--skip-cpu-check"]
        if args.has_bias:
            command.append("--has-bias")
        if args.clusters is not None:
            command.extend(("--clusters", str(args.clusters)))
        subprocess.run(command, check=True, stdout=sys.stderr)
        actual = np.fromfile(rust_output, dtype="<f2")
        if actual.size != m * n:
            raise AssertionError(f"Rust returned {actual.size} values, expected {m*n}")
        actual_cuda = torch.from_numpy(actual.reshape(1, m, n)).to(device="cuda")
        oracle.verify("fp16", *inputs, actual_cuda)
        rust_result = json.loads(rust_json.read_text())
        rust_result["verification"] = "python-oracle-pass"
    report = {
        "mnk": list(args.mnk), "has_bias": args.has_bias, "seed": args.seed,
        "input_sha256": digests, "verification": "both-pass",
        "cuda_oxide": rust_result,
        "python_reference": {"implementation": "CUTLASS (CuTe + CTM)",
            "reference_variant": "original-public-implementation",
            "metric_value": python_us, "metric_unit": "us",
            "tflops": 2*m*n*k/(python_us*1e6), "verification": "python-oracle-pass",
            "clusters": python_clusters, "hardware_cluster_capacity": python_capacity,
            "grid": [2 * python_clusters, 1, 1]},
        "cluster_counts_match": python_clusters == rust_result["provenance"]["clusters"],
        "speedup_python_over_rust": python_us / rust_result["metric_value"],
        "provenance": {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cutlass": getattr(importlib.import_module("cutlass"), "__version__", "unknown"),
            "rust_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256((reference_dir / "gemm.py").read_bytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256((reference_dir / "oracle.py").read_bytes()).hexdigest(),
            "warmup": args.warmup, "iters": args.iters, "graph_launches": args.graph_launches,
            "timing_mode": "direct", "tool": "cuda-events-single-kernel",
            "timing_order": ["python-reference", "cuda-oxide"],
            **oracle.clock_state("cuda:0")},
    }
    rendered = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(f"CUTLASS (CuTe + CTM): {python_us:.3f} us, {report['python_reference']['tflops']:.3f} TFLOP/s")
    print(f"{rust_result['implementation']}: {rust_result['metric_value']:.3f} us, {rust_result['tflops']:.3f} TFLOP/s")


if __name__ == "__main__":
    main()

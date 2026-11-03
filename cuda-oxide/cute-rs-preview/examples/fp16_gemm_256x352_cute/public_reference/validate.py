# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the public FP16 reference on SM100, including stream and graph reuse.

Run with ``python -m public_reference.validate`` from the example directory,
or wrap that command with compute-sanitizer. This runner reports
correctness only; use the example's compare.py for performance measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Case:
    """One GEMM extent, optional row bias, and resident-cluster override."""

    name: str
    """CLI selector for this coverage case."""
    mnk: tuple[int, int, int]
    """Logical M, N, K dimensions, all measured in elements."""
    has_bias: bool = False
    """Whether to add FP16 bias[M] before the final FP16 rounding."""
    clusters: int | None = None
    """Resident cluster count; None lets the launcher select occupancy."""


CASES = (
    Case("full", (256, 352, 64)),
    Case("smalltails", (256, 8, 8), has_bias=True),
    Case("persistent", (512, 712, 72), has_bias=True, clusters=1),
    Case("5stages", (512, 704, 320), clusters=1),
    Case("wrap", (512, 712, 392), has_bias=True, clusters=1),
    Case("multicluster", (768, 1064, 704), has_bias=True, clusters=2),
)
"""Cover full tiles, N/K tails, pipeline depth, wraparound, and cluster reuse."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse case selection without importing CUDA or kernel dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in CASES],
        help="run only this case (repeatable; default: all six)",
    )
    parser.add_argument("--seed", type=int, default=0, help="input seed (default: 0)")
    parser.add_argument(
        "--output", type=Path, help="also save the JSON report to this path",
    )
    return parser.parse_args(argv)


def require_sm100(torch) -> dict:
    """Reject missing CUDA and every capability except 10.0 and 10.3 before JIT imports."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this validation requires SM100 (10.0) or SM103 (10.3)")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"SM100 (10.0) or SM103 (10.3) is required; device {device} is {name} "
            f"with capability {capability[0]}.{capability[1]}"
        )
    return {"index": device, "name": name, "capability": list(capability)}


def validate_case(torch, oracle, prepare_callable, case: Case, seed: int, record: dict):
    """Compile once, then verify a stream launch and two poisoned graph replays.

    Inputs remain owned here until all launches complete. A copy queued on the
    supplied stream is the observed output, so verification consumes that
    stream's result after synchronization. Every invocation must overwrite NaNs.
    """
    inputs, outputs = oracle.prepare_tensors(
        "fp16", mnk=case.mnk, has_bias=case.has_bias, seed=seed,
    )
    output = outputs[0]
    launch = prepare_callable(
        has_bias=case.has_bias, inputs=inputs, output=output, clusters=case.clusters,
    )
    stream = torch.cuda.Stream(device=output.device)
    stream.wait_stream(torch.cuda.current_stream(output.device))

    # Keep the current stream unchanged at the call boundary: launch must use
    # the stream argument. The consumer is queued on that supplied stream.
    with torch.cuda.stream(stream):
        output.fill_(float("nan"))
    launch(*inputs, output, stream=stream)
    with torch.cuda.stream(stream):
        observed = output.clone()
    stream.synchronize()
    oracle.verify("fp16", *inputs, observed)
    record["checks"].append("explicit-stream")

    # Eager compilation and the checked launch above populate cached tensor
    # views before capture. Capture records only the launch, never compilation.
    stream.wait_stream(torch.cuda.current_stream(output.device))
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        output.fill_(float("nan"))
    with torch.cuda.graph(graph, stream=stream):
        launch(*inputs, output, stream=stream)
    for replay in (1, 2):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            graph.replay()
            observed = output.clone()
        stream.synchronize()
        oracle.verify("fp16", *inputs, observed)
        record["checks"].append(f"graph-replay-{replay}")


def main(argv: list[str] | None = None) -> int:
    """Emit one JSON report, returning nonzero on hardware or numerical failure."""
    args = parse_args(argv)
    selected = set(args.case) if args.case else {case.name for case in CASES}
    report = {"status": "failed", "dtype": "fp16", "seed": args.seed, "cases": []}
    active = None
    try:
        import torch

        report["device"] = require_sm100(torch)
        # The architecture check intentionally precedes CuTe and oracle imports.
        from public_reference import oracle, gemm

        for case in CASES:
            if case.name not in selected:
                continue
            active = {
                "name": case.name, "mnk": list(case.mnk), "has_bias": case.has_bias,
                "clusters": case.clusters, "seed": args.seed, "status": "running",
                "checks": [],
            }
            report["cases"].append(active)
            validate_case(torch, oracle, gemm.prepare_callable, case, args.seed, active)
            active["status"] = "passed"
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if active is not None:
            active["status"] = "failed"
            active["error"] = report["error"]

    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    sys.stdout.write(encoded)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

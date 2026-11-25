#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the unmodified CUTLASS v4.6.2 example with one launch per event pair."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute.testing as testing
import torch
import dense_blockscaled_gemm_persistent_cooperative as example

OUT = Path(__file__).parent


def checked(result):
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(result))
    return result[1] if len(result) == 2 else result[1:]


def single_launch_benchmark(fn, *, workspace_generator, workspace_count,
                            stream, warmup_iterations, iterations, **kwargs):
    assert workspace_count == 1 and warmup_iterations == 100 and iterations == 31
    args = workspace_generator()
    checked(cuda.cuStreamSynchronize(stream))
    runs = []
    for round_index in range(3):
        for _ in range(warmup_iterations):
            fn(*args.args, **args.kwargs)
        checked(cuda.cuStreamSynchronize(stream))
        samples = []
        torch.cuda.nvtx.range_push(f"cutedsl_samples_round_{round_index}")
        for _ in range(iterations):
            start = checked(cuda.cuEventCreate(0))
            end = checked(cuda.cuEventCreate(0))
            checked(cuda.cuEventRecord(start, stream))
            fn(*args.args, **args.kwargs)
            checked(cuda.cuEventRecord(end, stream))
            checked(cuda.cuEventSynchronize(end))
            samples.append(checked(cuda.cuEventElapsedTime(start, end)) * 1000)
            checked(cuda.cuEventDestroy(start))
            checked(cuda.cuEventDestroy(end))
        torch.cuda.nvtx.range_pop()
        ordered = sorted(samples)
        run = dict(round=round_index, median_us=statistics.median(samples),
                   p10_us=ordered[3], p90_us=ordered[27], samples_us=samples)
        runs.append(run)
        print(json.dumps(run), flush=True)
    report = dict(implementation="Python CuTeDSL", version=importlib.metadata.version("nvidia-cutlass-dsl"),
                  source_tag="v4.6.2", shape=[2048,2048,1024,1], tile=[128,128,128],
                  input="Float4E2M1FN", scale="Float8E8M0FNU", scale_vector=32,
                  output="Float16", gpu=torch.cuda.get_device_name(),
                  gpu_visible_device=os.getenv("CUDA_VISIBLE_DEVICES"),
                  warmups=100, samples=31, graphs=False, cold_l2=False,
                  timing="one direct launch per CUDA driver event pair",
                  profiler_run=os.getenv("BENCH_PROFILED") == "1",
                  correctness="upstream reference check passed before timing",
                  source_sha256=hashlib.sha256(Path(example.__file__).read_bytes()).hexdigest(),
                  runs=runs, median_us=statistics.median(r["median_us"] for r in runs))
    name = "cutedsl-profiled-events.json" if report["profiler_run"] else "cutedsl-events.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n")
    return report["median_us"]


testing.benchmark = single_launch_benchmark
torch.manual_seed(2026)
example.run_bs(
    (2048,2048,1024,1), cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
    cutlass.Float8E8M0FNU, 32, cutlass.Float16, cutlass.Float32,
    "k", "k", "n", (128,128,128), (128,128), 0.1, 100, 31, False, False,
)
print("PASS", flush=True)

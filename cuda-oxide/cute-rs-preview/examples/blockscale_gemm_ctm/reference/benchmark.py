#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual CUTLASS primitives GEMM, with the Rust CTM fixture and timing contract.

Use compare.py for Nsight kernel-active timings. The CUDA event measurements
reported here include host launch overhead and are a separate metric.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctas", type=int, default=170)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.ctas <= 256:
        parser.error("ctas must be between 1 and 256")
    if args.warmup < 0 or args.samples <= 0:
        parser.error("warmup must be nonnegative and samples must be positive")
    return args


def verify_bits(got, expected):
    if (got.shape != expected.shape or got.dtype != np.uint16
            or expected.dtype != np.uint16):
        raise ValueError("output shape/storage differs from the FP16 bit oracle")
    mismatches = np.flatnonzero(got.reshape(-1) != expected.reshape(-1))
    if mismatches.size:
        first = int(mismatches[0])
        raise ValueError(f"{mismatches.size} FP16 bit mismatches; first at {first}: "
                         f"got {got.flat[first]:#06x}, expected {expected.flat[first]:#06x}")


def main(argv=None):
    args = parse_arguments(argv)
    if args.json:
        args.json.unlink(missing_ok=True)
    # Select the architecture before initializing the DSL. No torch dependency
    # or helper GPU kernels: the trace contains only the GEMM under test.
    os.environ["CUTE_DSL_ARCH"] = "sm_120a"
    import cuda.bindings.driver as driver
    import cutlass
    import cutlass.cute as cute
    from fixture import build_fixture
    from launch import host

    def checked(result):
        if result[0] != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(str(result))
        return result[1] if len(result) == 2 else result[1:]

    checked(driver.cuInit(0))
    device = checked(driver.cuDeviceGet(0))
    capability = list(checked(driver.cuDeviceComputeCapability(device)))
    if capability != [12, 0]:
        raise RuntimeError(f"SM120 required, got {capability}")
    context = checked(driver.cuDevicePrimaryCtxRetain(device))
    with ExitStack() as resources:
        resources.callback(lambda: checked(driver.cuDevicePrimaryCtxRelease(device)))
        checked(driver.cuCtxSetCurrent(context))
        stream = checked(driver.cuStreamCreate(0))
        resources.callback(lambda: checked(driver.cuStreamDestroy(stream)))
        allocations = []
        fixture = build_fixture()
        expected = fixture["expected"]
        output = np.full(expected.shape, 0x7e00, dtype=np.uint16)
        pointers = []
        for data in [fixture[key] for key in ("a", "b", "sa", "sb")] + [output]:
            pointer = checked(driver.cuMemAlloc(data.nbytes))
            resources.callback(lambda p=pointer: checked(driver.cuMemFree(p)))
            allocations.append(pointer)
            checked(driver.cuMemcpyHtoD(pointer, data.ctypes.data, data.nbytes))
            pointers.append(cutlass.Int64(int(pointer)))
        parameters = (*pointers, cutlass.Int32(args.ctas), stream)
        # ExitStack attempts every release even if a driver cleanup call fails.
        resources.callback(lambda: checked(driver.cuStreamSynchronize(stream)))
        compiled = cute.compile(host, *parameters)
        compiled(*parameters)
        checked(driver.cuStreamSynchronize(stream))
        checked(driver.cuMemcpyDtoH(output.ctypes.data, allocations[-1], output.nbytes))
        verify_bits(output, expected)
        print(f"PASS: all {expected.size:,} FP16 output bits; {args.ctas} CTAs", flush=True)
        for _ in range(args.warmup):
            compiled(*parameters)
        checked(driver.cuStreamSynchronize(stream))
        times = []
        for _ in range(args.samples):
            with ExitStack() as events:
                start = checked(driver.cuEventCreate(0))
                events.callback(lambda: checked(driver.cuEventDestroy(start)))
                end = checked(driver.cuEventCreate(0))
                events.callback(lambda: checked(driver.cuEventDestroy(end)))
                checked(driver.cuEventRecord(start, stream))
                compiled(*parameters)
                checked(driver.cuEventRecord(end, stream))
                checked(driver.cuEventSynchronize(end))
                times.append(float(checked(driver.cuEventElapsedTime(start, end))))
        checked(driver.cuMemcpyDtoH(output.ctypes.data, allocations[-1], output.nbytes))
        verify_bits(output, expected)
        ordered = sorted(times)
        median = ordered[len(ordered) // 2]
        report = dict(implementation="CUTLASS experimental primitives",
                      verification="bit-exact-pass", checked_outputs=int(expected.size),
                      mnk=[2048, 2048, 1024], compute_capability=capability,
                      ctas=args.ctas, warmup=args.warmup, samples=args.samples,
                      replay_verified=True, median_ms=median,
                      event_samples_ms=times,
                      event_metric="CUDA events, includes host submission intervals",
                      packages={name: importlib.metadata.version(name) for name in
                                ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13",
                                 "cuda-bindings", "numpy")},
                      sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("*.py")},
                      fixture_sha256={key: hashlib.sha256(data.tobytes()).hexdigest()
                                      for key, data in fixture.items()},
                      gpu_visible_device=os.getenv("CUDA_VISIBLE_DEVICES"))
        from cutlass.experimental.primitives import nvvm_wrapper
        upstream = Path(nvvm_wrapper.__file__)
        report["upstream_primitives"] = dict(
            path=str(upstream), sha256=hashlib.sha256(upstream.read_bytes()).hexdigest())
        cubin = compiled.__cubin__
        ptx = compiled.__ptx__
        if cubin is not None:
            report["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
        if ptx is not None:
            report["ptx_sha256"] = hashlib.sha256(ptx.encode()).hexdigest()
    # Publish success only after every CUDA resource was released successfully.
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: replay exact; CUDA event median {median * 1000:.3f} us", flush=True)


if __name__ == "__main__":
    main()

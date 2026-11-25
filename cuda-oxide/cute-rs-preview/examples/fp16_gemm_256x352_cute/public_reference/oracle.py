# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent FP16 GEMM correctness oracle using public PyTorch operations.

The reference accumulates in FP32, adds optional row bias in FP32, then rounds
once to FP16. Input integers keep ordinary benchmark sums exactly representable
in FP32. This module does not implement or time the CuTe device kernel.
"""

from datetime import datetime, timezone
from numbers import Integral
import subprocess

import torch


def prepare_tensors(family, mnk=(256, 352, 64), has_bias=False, seed=0, device="cuda"):
    """Return owned inputs and one uninitialized output, on ``device``.

    A is contiguous [1,M,K]; B is a [1,K,N] transpose view of contiguous
    [1,N,K] storage. Optional bias is contiguous [M]. All values are FP16
    integers in [-2,2). A private CPU generator preserves caller RNG state
    and gives the same seeded inputs on CPU and CUDA. Output is [1,M,N].
    Only the ``fp16`` family and positive integer dimensions are supported.
    """
    if family != "fp16":
        raise ValueError("only the fp16 family is supported")
    if len(mnk) != 3 or any(not isinstance(x, Integral) or isinstance(x, bool) or x <= 0 for x in mnk):
        raise ValueError("mnk must contain three positive integers")
    m, n, k = mnk
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def integers(shape):
        """Allocate seeded FP16 integer data without mutating global RNG state."""
        return torch.randint(-2, 2, shape, generator=generator).to(device=device, dtype=torch.float16)

    inputs = [integers((1, m, k)), integers((1, n, k)).transpose(1, 2)]
    if has_bias:
        inputs.append(integers((m,)))
    return inputs, [torch.empty((1, m, n), device=device, dtype=torch.float16)]


def verify(family, *tensors):
    """Assert ``A @ B + bias[:,None]`` matches the final tensor.

    Accept A, B, C or A, B, bias, C in FP16 on one CPU/CUDA device; A and B
    have a single batch dimension. Wrong shapes, dtypes, nonfinite values,
    and differences outside atol=0.1/rtol=1e-5 raise AssertionError. TF32 and
    autocast are disabled for FP32 reference accumulation. The caller's
    float32 matmul precision is restored, including when checking fails.
    """
    if family != "fp16":
        raise ValueError("only the fp16 family is supported")
    if len(tensors) not in (3, 4):
        raise AssertionError("expected A, B, [bias,] C")
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise AssertionError("all inputs and output must be tensors")
    a, b = tensors[:2]
    c = tensors[-1]
    bias = tensors[2] if len(tensors) == 4 else None
    if any(t.dtype != torch.float16 or t.device != a.device for t in tensors):
        raise AssertionError("all inputs and output must be FP16 on one device")
    if a.device.type not in ("cpu", "cuda"):
        raise AssertionError("only CPU and CUDA tensors are supported")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != 1 or b.shape[0] != 1:
        raise AssertionError("A and B must have shape [1,M,K] and [1,K,N]")
    _, m, k = a.shape
    n = b.shape[2]
    if min(m, n, k) <= 0 or b.shape[1] != k or c.shape != (1, m, n):
        raise AssertionError("incompatible A, B or C shape")
    if bias is not None and bias.shape != (m,):
        raise AssertionError("bias must have shape [M]")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise AssertionError("inputs and output must contain only finite values")
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad(), torch.autocast(device_type=a.device.type, enabled=False):
            expected = torch.bmm(a.float(), b.float())
            if bias is not None:
                expected = expected + bias.reshape(1, m, 1).float()
            expected = expected.to(dtype=torch.float16)
        if not bool(torch.isfinite(expected).all()):
            raise AssertionError("FP16 reference result is not finite")
        torch.testing.assert_close(c, expected, atol=0.1, rtol=1e-5)
    finally:
        torch.set_float32_matmul_precision(precision)


def clock_state(device="cuda:0"):
    """Return a best-effort post-timing clock snapshot, never a timing claim.

    Query nvidia-smi by the selected device UUID, avoiding physical-index
    assumptions under CUDA_VISIBLE_DEVICES. Missing UUIDs, nvidia-smi, or
    unavailable telemetry produce an explicit unavailable status.
    """
    snapshot = {"phase": "post-timing", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if uuid is None:
            raise RuntimeError("selected GPU UUID is unavailable")
        uuid = str(uuid)
        # PyTorch can expose the bare UUID; nvidia-smi expects its GPU prefix.
        if not uuid.startswith(("GPU-", "MIG-")):
            uuid = "GPU-" + uuid
        result = subprocess.run(
            ["nvidia-smi", "--id=" + uuid,
             "--query-gpu=uuid,clocks.current.sm,clocks.current.memory,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) != 4:
            raise ValueError("unexpected nvidia-smi telemetry format")
        snapshot.update(status="available", gpu_uuid=fields[0],
                        sm_clock_mhz=float(fields[1]), memory_clock_mhz=float(fields[2]),
                        power_w=float(fields[3]))
    except (OSError, RuntimeError, AssertionError, ValueError, subprocess.SubprocessError) as exc:
        snapshot.update(status="unavailable", reason=str(exc))
    return {"clock_snapshot": snapshot}

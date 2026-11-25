# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter around the unchanged public MLA kernel; no replacement device code."""
import hashlib
import importlib.metadata
from pathlib import Path
import sys

import numpy as np

from contract import SOURCE_HASHES, check_close, oracle


class sm100_mla_decode_fp16_cutedsl:
    def __init__(self, cfg, data, clusters=None):
        cfg.validate()
        reference = Path(__file__).with_name("reference")
        source = reference / "cute/blackwell/kernel/attention/mla"
        for name, digest in SOURCE_HASHES.items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"pinned reference changed: {name}")
        if importlib.metadata.version("nvidia-cutlass-dsl") != "4.7.0":
            raise RuntimeError("this comparison requires nvidia-cutlass-dsl==4.7.0")
        sys.path.insert(0, str(reference))
        import torch
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.runtime import from_dlpack
        import cuda.bindings.driver as cuda
        from cute.blackwell.kernel.attention.mla.mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16

        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise RuntimeError("the matched reference profile requires SM100 or SM103")
        self.torch, self.cfg = torch, cfg
        self.stream = torch.cuda.Stream()
        self.buffers = {name: torch.from_numpy(value.copy()).cuda() for name, value in data.items()}
        self.output = torch.full((cfg.batch, 1, 128, 512), float("nan"), device="cuda", dtype=torch.float16)
        self.lse = torch.full((cfg.batch, 1, 128), float("nan"), device="cuda", dtype=torch.float32)
        self.lengths = torch.full((cfg.batch,), cfg.kv_length, dtype=torch.int32, device="cuda")
        self.workspace = (torch.empty(cfg.batch * 128 * cfg.splits * 513 * 4, dtype=torch.int8, device="cuda")
                          if cfg.splits > 1 else None)
        def tensor(t, *, leading=1, order=None):
            result = from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading)
            if order is not None:
                result = result.mark_compact_shape_dynamic(mode=leading, stride_order=order, divisibility=8)
            return result
        q = tensor(self.buffers["q_latent"].view(cfg.batch, 1, 128, 512).permute(2, 3, 1, 0), order=(3, 2, 0, 1))
        qr = tensor(self.buffers["q_rope"].view(cfg.batch, 1, 128, 64).permute(2, 3, 1, 0), order=(3, 2, 0, 1))
        c = tensor(self.buffers["c_latent"].permute(1, 2, 0), order=(2, 0, 1))
        cr = tensor(self.buffers["c_rope"].permute(1, 2, 0), order=(2, 0, 1))
        pages = tensor(self.buffers["page_table"].T, leading=0)
        o = tensor(self.output.permute(2, 3, 1, 0), order=(3, 2, 0, 1))
        lse = tensor(self.lse.permute(2, 1, 0), leading=0)
        workspace = None if self.workspace is None else from_dlpack(self.workspace, assumed_align=32)
        self.arguments = (q, qr, c, cr, pages, o, lse, workspace, cfg.splits,
                          tensor(self.lengths, leading=0), None, cfg.softmax_scale, cfg.output_scale,
                          cuda.CUstream(self.stream.cuda_stream))
        capacity = utils.HardwareInfo().get_max_active_clusters(2)
        self.clusters = min(cfg.batch * cfg.splits, capacity) if clusters is None else clusters
        if self.clusters <= 0 or self.clusters > min(cfg.batch * cfg.splits, capacity):
            raise ValueError(f"clusters must be in 1..{min(cfg.batch * cfg.splits, capacity)}")
        kernel = BlackwellMultiHeadLatentAttentionForwardFP16(
            cutlass.Float32, cutlass.Float32, (128, 128), (128, 256), self.clusters,
            128, 0.0, True, False, False)
        self.compiled = cute.compile(kernel, *self.arguments, options="--opt-level 2")
        torch.cuda.synchronize()
        self.checked = False

    def run(self):
        self.compiled(*self.arguments)

    def read(self):
        self.stream.synchronize()
        return (self.output.cpu().numpy().reshape(self.cfg.batch, 128, 512),
                self.lse.cpu().numpy().reshape(self.cfg.batch, 128))

    def validate(self, data, atol=0.01):
        self.run()
        actual, lse = self.read()
        expected, expected_lse = oracle(self.cfg, data)
        metrics = {"output": check_close(actual, expected, atol), "lse": check_close(lse, expected_lse, atol)}
        self.checked = True
        return metrics

    def benchmark(self, warmup, iterations, rounds):
        if not self.checked:
            raise RuntimeError("reference must pass correctness before timing")
        if warmup < 0 or iterations < 1 or rounds < 1:
            raise ValueError("require nonnegative warmup and positive iterations/rounds")
        samples = []
        for _ in range(rounds):
            for _ in range(warmup):
                self.run()
            self.stream.synchronize()
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            for _ in range(iterations):
                self.run()
            end.record(self.stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
        return samples

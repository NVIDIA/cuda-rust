# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact file interchange and independent, length-masked MLA mathematics."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
SOURCE_HASHES = {
    "mla_decode_fp16.py": "82116d97e611366d3122473ada3fde7e2550f1e06914ba1834fc448954dfd106",
    "mla_helpers.py": "13b0b43a5417be6d79369354ab90dbaddd2c49ec49c982258bf90fe2f285a2ae",
}


@dataclass(frozen=True)
class Config:
    batch: int = 1
    kv_length: int = 1024
    splits: int = 2
    heads: int = 128
    query_length: int = 1
    latent: int = 512
    rope: int = 64
    page_size: int = 128
    softmax_scale: float = 0.0416
    output_scale: float = 1.0

    def validate(self):
        if (self.heads, self.query_length, self.latent, self.rope, self.page_size) != (128, 1, 512, 64, 128):
            raise ValueError("supported profile: H128, Sq1, latent512, RoPE64, page128")
        if not (0 < self.batch <= 65535 and 0 < self.kv_length <= 2**30 and 0 < self.splits <= 256):
            raise ValueError("batch, KV length, and splits must be positive and fit the supported grid")
        if not math.isfinite(self.softmax_scale) or self.softmax_scale <= 0:
            raise ValueError("softmax scale must be finite and positive")
        if not math.isfinite(self.output_scale):
            raise ValueError("output scale must be finite")
        if self.batch * self.splits > (2**31 - 1) // 2:
            raise ValueError("persistent work count overflows i32")
        if self.batch * self.pages_per_batch * 128 > 2**31 - 1:
            raise ValueError("physical cache row coordinates exceed i32")
        for shape, dtype in self.arrays().values():
            if math.prod(shape) * np.dtype(dtype).itemsize > 2**63 - 1:
                raise ValueError("allocation exceeds host addressable size")
        return self

    @property
    def pages_per_batch(self):
        return (self.kv_length + 127) // 128

    def arrays(self):
        return {
            "q_latent": ((self.batch, 128, 512), "<f2"),
            "q_rope": ((self.batch, 128, 64), "<f2"),
            "c_latent": ((self.batch * self.pages_per_batch, 128, 512), "<f2"),
            "c_rope": ((self.batch * self.pages_per_batch, 128, 64), "<f2"),
            "page_table": ((self.batch, self.pages_per_batch), "<i4"),
        }


def split_ranges(length, splits):
    tiles = (length + 127) // 128
    per_split = (tiles + splits - 1) // splits
    return [(min(length, s * per_split * 128), min(length, (s + 1) * per_split * 128))
            for s in range(splits)]


def generate(cfg, seed=1111, pattern="random"):
    cfg.validate()
    rng = np.random.default_rng(seed)
    data = {}
    for name, (shape, dtype) in cfg.arrays().items():
        if name == "page_table":
            data[name] = rng.permutation(math.prod(shape)).astype(dtype).reshape(shape)
        else:
            data[name] = rng.uniform(-2, 2, size=shape).astype(dtype)
    if pattern == "zero-query":
        data["q_latent"].fill(0)
        data["q_rope"].fill(0)
    elif pattern != "random":
        raise ValueError("unknown input pattern")
    return data


def filename(name):
    return name + (".i32" if name == "page_table" else ".f16")


def write_inputs(directory, cfg, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg.validate()
    digests = {}
    for name, (shape, dtype) in cfg.arrays().items():
        value = np.asarray(data[name], dtype=dtype)
        if value.shape != shape:
            raise ValueError(f"incorrect {name} shape")
        payload = value.tobytes(order="C")
        (directory / filename(name)).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    (directory / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (directory / "sha256.json").write_text(json.dumps(digests, indent=2) + "\n")
    return digests


def read_inputs(directory, cfg):
    cfg.validate()
    directory = Path(directory)
    data = {}
    for name, (shape, dtype) in cfg.arrays().items():
        payload = (directory / filename(name)).read_bytes()
        if len(payload) != math.prod(shape) * np.dtype(dtype).itemsize:
            raise ValueError(f"incorrect byte count for {name}")
        data[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if not np.isfinite(data[name]).all():
            raise ValueError(f"nonfinite input in {name}")
    if np.any(data["page_table"] < 0) or np.any(data["page_table"] >= cfg.batch * cfg.pages_per_batch):
        raise ValueError("page table contains an out-of-range physical page")
    return data


def oracle(cfg, data):
    """FP64 mathematical oracle; slice valid keys before normalization."""
    cfg.validate()
    outputs, lses = [], []
    for batch in range(cfg.batch):
        pages = data["page_table"][batch]
        latent = data["c_latent"][pages].reshape(-1, 512)[:cfg.kv_length].astype(np.float64)
        rope = data["c_rope"][pages].reshape(-1, 64)[:cfg.kv_length].astype(np.float64)
        scores = (data["q_latent"][batch].astype(np.float64) @ latent.T
                  + data["q_rope"][batch].astype(np.float64) @ rope.T) * cfg.softmax_scale
        maximum = np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores - maximum)
        denominator = weights.sum(axis=1, keepdims=True)
        outputs.append((weights @ latent) / denominator * cfg.output_scale)
        lses.append(((maximum + np.log(denominator)) / np.log(2)).ravel())
    return np.stack(outputs), np.stack(lses)


def check_close(actual, expected, atol=0.01, rtol=1e-5):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"invalid comparison: shape {actual.shape}, expected {expected.shape}, finite_actual={np.isfinite(actual).all()}, finite_expected={np.isfinite(expected).all()}")
    difference = np.abs(actual - expected)
    metrics = {"max_abs": float(difference.max(initial=0)), "rms": float(np.sqrt(np.mean(difference**2)))}
    if np.any(difference > atol + rtol * np.abs(expected)):
        raise AssertionError(f"numerical mismatch: {metrics}, atol={atol}, rtol={rtol}")
    return metrics

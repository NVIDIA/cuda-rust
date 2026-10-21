# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fixture shared with the Rust CTM block-scaled GEMM example.

No GPU framework is used here. Packed inputs match
``blockscale_gemm_ctm/src/main.rs`` and its ``physical::scale_offset`` layout.
"""

import numpy as np

M = N = 2048
K = 1024
KB = K // 32


def _mix32(value):
    """Apply the Rust fixture's wrapping u32 hash elementwise."""
    value = np.asarray(value, dtype=np.uint32)
    value = value ^ (value >> 16)
    value = np.multiply(value, np.uint32(0x7FEB352D), dtype=np.uint32)
    value = value ^ (value >> 15)
    value = np.multiply(value, np.uint32(0x846CA68B), dtype=np.uint32)
    return value ^ (value >> 16)


def _nibbles(rows, multiplier, seed):
    row = np.arange(rows, dtype=np.uint32)[:, None]
    k = np.arange(K, dtype=np.uint32)[None, :]
    return (_mix32(row * np.uint32(multiplier) ^ k ^ np.uint32(seed)) & 15).astype(
        np.uint8
    )


def _scales(rows, multiplier, seed):
    row = np.arange(rows, dtype=np.uint32)[:, None]
    group = np.arange(KB, dtype=np.uint32)[None, :]
    return (
        126 + _mix32(row * np.uint32(multiplier) ^ group ^ np.uint32(seed)) % 3
    ).astype(np.uint8)


def _pack_nibbles(nibbles):
    # Consecutive low/high nibbles form a byte; pairs of bytes reproduce Rust's
    # little-endian f16 transport carriers without interpreting them as floats.
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)


def _pack_scales(dense):
    rows, groups = dense.shape
    row = np.arange(rows)[:, None]
    group = np.arange(groups)[None, :]
    # Exact physical::scale_offset(row, group, groups), in bytes.
    scale_word = (row & 31) * 4 + ((row >> 5) & 3)
    offset = (
        (row // 128) * (groups // 4) * 512
        + (group // 4) * 512
        + scale_word * 4
        + group % 4
    )
    packed = np.empty(rows * groups, dtype=np.uint8)
    packed[offset] = dense
    words = packed[0::2].astype(np.uint16) | (packed[1::2].astype(np.uint16) << 8)
    return words.reshape((rows // 128) * (groups // 4), 256)


def _scaled_values(nibbles, scales):
    positive = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    values = np.concatenate((positive, -positive))[nibbles]
    powers = np.exp2(scales.astype(np.float32) - 127)
    values.reshape(nibbles.shape[0], KB, 32)[:] *= powers[:, :, None]
    return values


def build_fixture():
    """Return C-contiguous packed A/B, scale atoms, and exact FP16 result bits.

    ``a`` and ``b`` are uint8 arrays of shape (2048, 512), with B's rows
    representing output columns. ``sa`` and ``sb`` are uint16 arrays of shape
    (128, 256), each row holding one 512-byte canonical scale atom. ``expected``
    is a uint16 (2048, 2048) matrix containing FP16 bit patterns.

    CPU float32 matmul is exact for this fixture: scaled e2m1 values are
    multiples of 1/4 with magnitude at most 12. Their products are multiples
    of 1/16, and even the sum of absolute products, 1024 * 144, needs fewer
    than 24 significant bits in those units. Thus every partial sum is exact,
    including the Rust oracle's per-32-element grouping. Only the final FP16
    conversion rounds, using round-to-nearest, ties-to-even.
    """
    a = _nibbles(M, 0x9E3779B9, 0xA511E9B3)
    b = _nibbles(N, 0x85EBCA6B, 0x63D83595)
    sa = _scales(M, 0xC2B2AE35, 0)
    sb = _scales(N, 0x27D4EB2F, 0x165667B1)
    expected = (_scaled_values(a, sa) @ _scaled_values(b, sb).T).astype(np.float16)
    return {
        "a": _pack_nibbles(a),
        "b": _pack_nibbles(b),
        "sa": _pack_scales(sa),
        "sb": _pack_scales(sb),
        "expected": expected.view(np.uint16),
    }

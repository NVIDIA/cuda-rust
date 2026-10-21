#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged public CuTe kernel with the Rust example's exact inputs.

The fixture and scalar oracle require only the standard library. GPU imports
occur only in run(); the large CPU oracle uses PyTorch FP64 on the CPU.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import struct

HEAD_DIM = 64
STATIC_SHAPE = (1024, 1024, 8)  # seq_q, seq_k, heads; batch=1 and head_dim=64
SOFTMAX_SCALE = 0.125
ATOL = 2e-4
RTOL = 2e-3
Q_SEED, K_SEED, V_SEED = 0xA511E9B3, 0x63D83595, 0xC2B2AE35


def mix32(value):
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def make_inputs(seq_q, seq_k, heads, input_scale=1, zero_query=False):
    """Return exact FP16-representable values in flattened BSHD order."""
    def values(length, seed, scale):
        return [((mix32(i ^ seed) & 63) - 32) * scale / 128.0 for i in range(length)]

    q = ([0.0] * (seq_q * heads * HEAD_DIM) if zero_query else
         values(seq_q * heads * HEAD_DIM, Q_SEED, input_scale))
    return (q, values(seq_k * heads * HEAD_DIM, K_SEED, input_scale),
            values(seq_k * heads * HEAD_DIM, V_SEED, 1))


def encode_f16(values):
    return struct.pack(f"<{len(values)}e", *values)


def decode_f16(data):
    if len(data) % 2:
        raise ValueError("FP16 input must contain complete two-byte elements")
    return list(struct.unpack(f"<{len(data) // 2}e", data))


def bits_hash(values):
    result = 0xCBF29CE484222325
    for byte in encode_f16(values):
        result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def reference_cpu(q, k, v, seq_q, seq_k, heads, causal, *, use_torch=False):
    """Stable mathematical attention with FP64 probabilities and accumulation."""
    if len(q) != seq_q * heads * HEAD_DIM or len(k) != seq_k * heads * HEAD_DIM or len(v) != len(k):
        raise ValueError("fixture dimensions do not match tensor lengths")
    if use_torch:
        import torch

        q_cpu = torch.tensor(q, dtype=torch.float64, device="cpu").reshape(seq_q, heads, HEAD_DIM).permute(1, 0, 2)
        k_cpu = torch.tensor(k, dtype=torch.float64, device="cpu").reshape(seq_k, heads, HEAD_DIM).permute(1, 0, 2)
        v_cpu = torch.tensor(v, dtype=torch.float64, device="cpu").reshape(seq_k, heads, HEAD_DIM).permute(1, 0, 2)
        scores = (q_cpu @ k_cpu.transpose(1, 2)) * SOFTMAX_SCALE
        if causal:
            allowed = torch.arange(seq_k)[None, :] <= torch.arange(seq_q)[:, None]
            scores.masked_fill_(~allowed, float("-inf"))
        probabilities = (scores - scores.amax(dim=-1, keepdim=True)).exp()
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
        return (probabilities @ v_cpu).permute(1, 0, 2).contiguous().reshape(-1).tolist()

    output = [0.0] * len(q)
    for head in range(heads):
        for row in range(seq_q):
            q_base = (row * heads + head) * HEAD_DIM
            end = min(row + 1, seq_k) if causal else seq_k
            scores = []
            for column in range(end):
                k_base = (column * heads + head) * HEAD_DIM
                scores.append(sum(q[q_base + d] * k[k_base + d] for d in range(HEAD_DIM)) * SOFTMAX_SCALE)
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            for d in range(HEAD_DIM):
                output[q_base + d] = sum(
                    weight * v[(column * heads + head) * HEAD_DIM + d]
                    for column, weight in enumerate(weights)
                ) / denominator
    return output


def verify(got, expected):
    if len(got) != len(expected) or not got:
        raise ValueError("output and nonempty oracle lengths must match")
    maximum, ratio = 0.0, 0.0
    for index, (actual, want) in enumerate(zip(got, expected)):
        if not math.isfinite(actual) or not math.isfinite(want):
            raise ValueError(f"nonfinite output/oracle at {index}: {actual}, {want}")
        error = abs(actual - want)
        bound = ATOL + RTOL * abs(want)
        if error > bound:
            raise ValueError(f"output mismatch at {index}: actual={actual}, expected={want}, error={error}, bound={bound}")
        maximum = max(maximum, error)
        ratio = max(ratio, error / bound)
    return {"elements": len(got), "max_abs_error": maximum, "max_tolerance_ratio": ratio,
            "atol": ATOL, "rtol": RTOL, "oracle": "cpu-fp64-stable-softmax"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-q", type=int, default=1024)
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--shape-mode", choices=("runtime", "static"), default="runtime",
                        help="static specializes shapes and strides for Sq=Sk=1024, H=8 only")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--causal", action="store_true", dest="causal")
    masks.add_argument("--noncausal", action="store_false", dest="causal")
    parser.set_defaults(causal=True)
    parser.add_argument("--input-scale", type=int, choices=(1, 4, 16), default=1,
                        help="exact power-of-two scaling of Q/K only")
    parser.add_argument("--zero-query", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="ordinal after CUDA_VISIBLE_DEVICES remapping")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=31, help="zero selects verification without timing")
    parser.add_argument("--output", type=Path, help="write checked FP16 BSHD output")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.shape_mode == "static" and (args.seq_q, args.seq_k, args.heads) != STATIC_SHAPE:
        parser.error("static shape mode requires --seq-q 1024 --seq-k 1024 --heads 8")
    if args.seq_q <= 0 or args.seq_k <= 0 or args.heads <= 0:
        parser.error("sequence lengths and heads must be positive")
    if (args.seq_q != 1 and args.seq_q % 64) or args.seq_k % 64:
        parser.error("query length must be 1 or a multiple of 64; KV length must be a multiple of 64")
    if args.causal and args.seq_q != args.seq_k:
        parser.error("initial causal cases require equal sequence lengths")
    if max(args.seq_q, args.seq_k) * args.heads * HEAD_DIM > 0xFFFFFFFF:
        parser.error("tensor element count must fit u32 indexing")
    if args.warmup < 0 or args.iters < 0 or args.device < 0:
        parser.error("device, warmup and iters must be nonnegative")
    args.workload = "prefill-fmha"
    return args


def make_tensor_view(tensor, from_dlpack, shape_mode):
    """Keep static shapes and strides, or apply the original runtime annotations."""
    view = from_dlpack(tensor, assumed_align=16)
    if shape_mode == "runtime":
        view = view.mark_layout_dynamic(leading_dim=3).mark_compact_shape_dynamic(
            mode=3, stride_order=tensor.dim_order(), divisibility=8,
        )
    return view


def run(args):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import torch

    from flash_attention_v2 import FlashAttentionForwardAmpere

    q_values, k_values, v_values = make_inputs(
        args.seq_q, args.seq_k, args.heads, args.input_scale, args.zero_query,
    )
    expected = reference_cpu(q_values, k_values, v_values, args.seq_q, args.seq_k,
                             args.heads, args.causal, use_torch=True)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    tensors = [torch.tensor(values, dtype=torch.float16, device=device).reshape(1, length, args.heads, HEAD_DIM)
               for values, length in ((q_values, args.seq_q), (k_values, args.seq_k), (v_values, args.seq_k))]
    output = torch.full((1, args.seq_q, args.heads, HEAD_DIM), float("nan"), dtype=torch.float16, device=device)
    tensors.append(output)
    views = [make_tensor_view(tensor, from_dlpack, args.shape_mode) for tensor in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    reference = FlashAttentionForwardAmpere(HEAD_DIM, 64, 64, 128, args.causal)
    compiled = cute.compile(reference, *views, SOFTMAX_SCALE, stream)

    def invoke():
        compiled(*views, SOFTMAX_SCALE, stream)

    def read_output():
        return output.detach().cpu().reshape(-1).tolist()

    invoke()
    checked = read_output()
    verification = verify(checked, expected)
    output.fill_(float("nan"))
    invoke()
    replay = read_output()
    verify(replay, expected)
    if encode_f16(replay) != encode_f16(checked):
        raise ValueError("output changed across identical launches")
    for _ in range(args.warmup if args.iters else 0):
        invoke()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(args.iters)]
    samples = []
    for start, end in events:
        start.record()
        invoke()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) * 1000.0
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid event timing: {elapsed}")
        samples.append(elapsed)
    after = read_output()
    verify(after, expected)
    if encode_f16(after) != encode_f16(checked):
        raise ValueError("output changed during benchmark")
    if args.output:
        args.output.write_bytes(encode_f16(after))
    result = {
        "implementation": "cutlass-cute-dsl", "workload": args.workload,
        "shape": [1, args.seq_q, args.seq_k, args.heads, HEAD_DIM],
        "causal": args.causal, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "input_hashes": {name: bits_hash(values) for name, values in zip("qkv", (q_values, k_values, v_values))},
        "output_hash": bits_hash(after), "verification": "cpu-pass", "validation": verification,
        "replay_bitwise_equal": True, "metric_value": statistics.median(samples) if samples else None,
        "metric_unit": "us", "samples_us": samples,
        "provenance": {"gpu": torch.cuda.get_device_name(args.device),
                       "compute_capability": list(torch.cuda.get_device_capability(args.device)),
                       "device": args.device, "tool": "cuda-events-single-kernel", "timing_mode": "direct",
                       "causal_specialization": "static",
                       "shape_specialization": args.shape_mode,
                       "warmup": args.warmup if args.iters else 0, "iters": args.iters,
                       "grid": [(args.seq_q + 63) // 64, 1, args.heads],
                       "threads": 128, "shared_memory_bytes": 24576, "dtype": "float16",
                       "cutlass_version": cutlass.__version__, "torch_version": torch.__version__,
                       "source_commit": "dcf215af68a2d08d305076c152a06f201728cd53"},
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")
    return result


if __name__ == "__main__":
    run(parse_args())

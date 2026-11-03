#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check and time the independently authored CuTeDSL split-KV decode counterpart."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

_shared_path = Path(__file__).resolve().parents[2] / "fmha_prefill_sm120a_cute/reference/launch.py"
_spec = importlib.util.spec_from_file_location("fmha_decode_fixture", _shared_path)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

# Keep fixture and oracle shared; the decode launch sequence is independent.
make_inputs = _shared.make_inputs
reference_cpu = _shared.reference_cpu
verify = _shared.verify
decode_f16 = _shared.decode_f16
encode_f16 = _shared.encode_f16
bits_hash = _shared.bits_hash
HEAD_DIM = 64
KV_CHUNK = 64
PARTIAL_SHARED_BYTES = 17696
STATIC_SHAPES = ((64, 8), (65, 3), (1024, 8), (4097, 8))


def split_count(seq_k):
    if seq_k <= 0:
        raise ValueError("KV length must be positive")
    return (seq_k + KV_CHUNK - 1) // KV_CHUNK


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-q", type=int, default=1)
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--specialization", choices=("auto", "runtime", "static"), default="auto",
                        help="auto specializes the same fixed shape portfolio as Rust")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--causal", action="store_true", dest="causal")
    masks.add_argument("--noncausal", action="store_false", dest="causal")
    parser.set_defaults(causal=False)
    parser.add_argument("--input-scale", type=int, choices=(1, 4, 16), default=1)
    parser.add_argument("--zero-query", action="store_true")
    parser.add_argument("--device", type=int, default=0,
                        help="ordinal after CUDA_VISIBLE_DEVICES remapping")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=31,
                        help="zero selects verification without timing")
    parser.add_argument("--output", type=Path, help="write checked FP16 BSHD output")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.seq_q != 1 or args.causal:
        parser.error("decode requires --seq-q 1 and --noncausal")
    if args.seq_k <= 0 or args.heads <= 0:
        parser.error("KV length and heads must be positive")
    if args.seq_k * args.heads * HEAD_DIM > 0xFFFFFFFF:
        parser.error("tensor element count must fit u32 indexing")
    if args.device < 0 or args.warmup < 0 or args.iters < 0:
        parser.error("device, warmup and iters must be nonnegative")
    static_available = (args.seq_k, args.heads) in STATIC_SHAPES
    if args.specialization == "static" and not static_available:
        parser.error("static specialization requires (seq-k, heads) in " + str(STATIC_SHAPES))
    args.shape_specialization = (
        "static" if static_available and args.specialization != "runtime" else "runtime"
    )
    args.workload = "fmha-decode"
    return args


def run(args):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import torch

    from attention_decode import AttentionDecode

    q_values, k_values, v_values = make_inputs(
        1, args.seq_k, args.heads, args.input_scale, args.zero_query,
    )
    expected = reference_cpu(q_values, k_values, v_values, 1, args.seq_k,
                             args.heads, False, use_torch=True)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    tensors = [torch.tensor(values, dtype=torch.float16, device=device)
               .reshape(1, length, args.heads, HEAD_DIM)
               for values, length in ((q_values, 1), (k_values, args.seq_k), (v_values, args.seq_k))]
    splits = split_count(args.seq_k)
    partial = torch.full((args.heads, splits, 66), float("nan"), dtype=torch.float32, device=device)
    output = torch.full((1, 1, args.heads, HEAD_DIM), float("nan"), dtype=torch.float16, device=device)
    tensors.extend((partial, output))
    # Flat dynamic views prevent accidental shape specialization. The selected
    # kernel mode reconstructs BSHD/state views from constants or runtime values.
    views = [from_dlpack(tensor.reshape(-1), assumed_align=16).mark_layout_dynamic(leading_dim=0)
             for tensor in tensors]
    dimensions = (cutlass.Int32(args.seq_k), cutlass.Int32(args.heads), cutlass.Int32(splits))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    static_shape = (args.seq_k, args.heads) if args.shape_specialization == "static" else (0, 0)
    compiled = cute.compile(AttentionDecode(*static_shape), *views, *dimensions, stream)

    def invoke():
        compiled(*views, *dimensions, stream)

    def read_output():
        return output.detach().cpu().reshape(-1).tolist()

    invoke()
    checked = read_output()
    verification = verify(checked, expected)
    # Poison both final and intermediate storage to expose stale split state.
    output.fill_(float("nan"))
    partial.fill_(float("nan"))
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
    kernel_source = Path(__file__).with_name("attention_decode.py")
    result = {
        "implementation": "cutlass-cute-dsl", "workload": args.workload,
        "shape": [1, 1, args.seq_k, args.heads, HEAD_DIM],
        "causal": False, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "input_hashes": {name: bits_hash(values) for name, values in zip("qkv", (q_values, k_values, v_values))},
        "output_hash": bits_hash(after), "verification": "cpu-pass", "validation": verification,
        "replay_bitwise_equal": True, "metric_value": statistics.median(samples) if samples else None,
        "metric_unit": "us", "samples_us": samples,
        "provenance": {
            "gpu": torch.cuda.get_device_name(args.device),
            "compute_capability": list(torch.cuda.get_device_capability(args.device)),
            "device": args.device, "tool": "cuda-events-kernel-sequence", "timing_mode": "direct",
            "warmup": args.warmup if args.iters else 0, "iters": args.iters,
            "algorithm": "single-query-split-kv", "kv_chunk": KV_CHUNK, "split_count": splits,
            "kernel_count": 2, "kernel_sequence": ["fmha_decode_partial", "fmha_decode_merge"],
            "grids": [[args.heads * splits, 1, 1], [args.heads, 1, 1]],
            "threads": [128, 32], "shared_memory_bytes": [PARTIAL_SHARED_BYTES, 0], "dtype": "float16",
            "cutlass_version": cutlass.__version__, "torch_version": torch.__version__,
            "source_origin": "independently-authored-cuda-oxide-example",
            "shape_specialization": args.shape_specialization,
            "specialization_request": args.specialization,
            "source_file": kernel_source.name,
            "source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
        },
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")
    return result


if __name__ == "__main__":
    run(parse_args())

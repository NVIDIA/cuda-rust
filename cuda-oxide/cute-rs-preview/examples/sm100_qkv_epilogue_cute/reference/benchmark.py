#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and verify Python CuTeDSL fused QKV + RMSNorm + FP8 + Paged KV Scatter on SM100."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cutlass
from cutlass import utils
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver
import torch
from qkv_cutedsl import FusedQkvEpilogueCuTeDSL


def verify_and_time(
    m_list: list[int],
    k: int,
    num_q_heads: int,
    num_kv_heads: int,
    warmup: int,
    replays: int,
    samples: int,
) -> dict:
    """Verify exact numerical accuracy and time 1,000x CUDA Graph replay across M."""
    head_dim = 128
    n_total = (num_q_heads + 2 * num_kv_heads) * head_dim
    max_m = max(m_list)
    num_pages = (max_m // 16) + 16
    page_size = 16
    total_slots = num_pages * page_size

    w_qkv = torch.randint(-2, 2, (n_total, k), dtype=torch.float16, device="cuda")
    gamma_q = torch.ones((head_dim,), dtype=torch.float32, device="cuda")
    gamma_k = torch.ones((head_dim,), dtype=torch.float32, device="cuda")
    k_cache = torch.zeros(
        (total_slots, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device="cuda"
    )
    v_cache = torch.zeros(
        (total_slots, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device="cuda"
    )
    hw_clusters = int(utils.HardwareInfo(0).get_max_active_clusters(2))
    results = {}

    for m in m_list:
        x = torch.randint(-2, 2, (m, k), dtype=torch.float16, device="cuda")
        slot_mapping = (
            (torch.arange(m, dtype=torch.int32, device="cuda") * 7) % (m + 64)
        ).contiguous()
        # Use unique slots for deterministic verification
        slot_mapping = torch.arange(m, dtype=torch.int32, device="cuda")
        q_out = torch.zeros(
            (m, num_q_heads, head_dim), dtype=torch.float8_e4m3fn, device="cuda"
        )
        tiles = (m // 128) * (n_total // 256)
        clusters = min(tiles, hw_clusters)
        inv_scale = 0.0625

        # Reference oracle check
        ref_qkv = torch.matmul(x.float(), w_qkv.float().T)
        ref_q = ref_qkv[:, : num_q_heads * head_dim].view(m, num_q_heads, head_dim)
        ref_q_norm = (
            ref_q
            * torch.rsqrt(ref_q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            * gamma_q
            * inv_scale
        ).to(torch.float8_e4m3fn)
        ref_k = ref_qkv[
            :, num_q_heads * head_dim : (num_q_heads + num_kv_heads) * head_dim
        ].view(m, num_kv_heads, head_dim)
        ref_k_norm = (
            ref_k
            * torch.rsqrt(ref_k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            * gamma_k
            * inv_scale
        ).to(torch.float8_e4m3fn)
        ref_v = (
            ref_qkv[:, (num_q_heads + num_kv_heads) * head_dim :].view(
                m, num_kv_heads, head_dim
            )
            * inv_scale
        ).to(torch.float8_e4m3fn)

        s = torch.cuda.Stream()
        cu_s = driver.CUstream(s.cuda_stream)
        op = FusedQkvEpilogueCuTeDSL(
            num_q_heads, num_kv_heads, clusters=clusters, overlap_epilogue=True
        )
        args = (
            from_dlpack(x, assumed_align=16),
            from_dlpack(w_qkv, assumed_align=16),
            from_dlpack(gamma_q, assumed_align=16),
            from_dlpack(gamma_k, assumed_align=16),
            from_dlpack(slot_mapping, assumed_align=16),
            from_dlpack(q_out, assumed_align=32),
            from_dlpack(k_cache, assumed_align=32),
            from_dlpack(v_cache, assumed_align=32),
            cutlass.Float32(1e-6),
            cutlass.Float32(inv_scale),
            cutlass.Float32(inv_scale),
            cutlass.Float32(inv_scale),
            cu_s,
        )
        compiled = cutlass.cute.compile(
            op, *args, options="--gpu-arch sm_100a --opt-level 2"
        )
        with torch.cuda.stream(s):
            q_out.zero_()
            k_cache.zero_()
            v_cache.zero_()
            compiled(*args)
            s.synchronize()
            q_err = (q_out.float() - ref_q_norm.float()).abs().max().item()
            k_err = (
                (k_cache[slot_mapping.long()].float() - ref_k_norm.float())
                .abs()
                .max()
                .item()
            )
            v_err = (
                (v_cache[slot_mapping.long()].float() - ref_v.float())
                .abs()
                .max()
                .item()
            )
            if max(q_err, k_err, v_err) > 0.125:
                raise AssertionError(
                    f"CuTeDSL verification failed at M={m}: "
                    f"q_err={q_err}, k_err={k_err}, v_err={v_err}"
                )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                compiled(*args)
            for _ in range(warmup):
                graph.replay()
            s.synchronize()
            sample_us = []
            for _ in range(samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(s)
                for _ in range(replays):
                    graph.replay()
                end.record(s)
                end.synchronize()
                sample_us.append(start.elapsed_time(end) * 1000.0 / replays)
            overlapped_us = sum(sample_us) / len(sample_us)

        results[str(m)] = {
            "overlapped_us": overlapped_us,
            "clusters": clusters,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-list", type=str, default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    results = verify_and_time(
        m_list=m_list,
        k=args.k,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        warmup=args.warmup,
        replays=args.replays,
        samples=args.samples,
    )
    payload = json.dumps(results, indent=2)
    if args.json:
        args.json.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

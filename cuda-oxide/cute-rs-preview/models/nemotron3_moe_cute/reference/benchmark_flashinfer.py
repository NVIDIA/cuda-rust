#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer CuTeDSL (w4a4 ReLU²) SOTA CUDA Graph benchmark for Nemotron-3 MoE on B200 (sm_100a)."""

import torch
from flashinfer import fp4_quantize
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe import cute_dsl_fused_moe
from flashinfer.tllm_enums import ActivationType

H = 1024
I = 1024
E = 8
TOP_K = 4
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
ROUTING_PATTERN = [0.125, 0.25, 0.25, 0.375]
UP_FACTORS = [0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DOWN_FACTORS = [1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
TIMING_WARMUPS = 50
TIMING_REPLAYS = 1000
TIMING_SAMPLES = 20


def build_state(device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate and pre-swizzle the 8-expert NVFP4 weights and scales on `device`."""
    w1 = torch.full((E, I, H // 2), 0x22, dtype=torch.uint8, device=device)
    raw_w1_sf = torch.full((E * I, H // 16), 0x38, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    w1_sf = convert_sf_to_mma_layout(raw_w1_sf, m=I, k=H, num_groups=E, sf_vec_size=16)

    w2 = torch.full((E, H, I // 2), 0x22, dtype=torch.uint8, device=device)
    raw_w2_sf = torch.full((E * H, I // 16), 0x38, dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    w2_sf = convert_sf_to_mma_layout(raw_w2_sf, m=H, k=I, num_groups=E, sf_vec_size=16)

    return {
        "w1": w1,
        "w1_sf": w1_sf,
        "w1_alpha": torch.tensor([u / H for u in UP_FACTORS], dtype=torch.float32, device=device),
        "w2": w2,
        "w2_sf": w2_sf,
        "w2_alpha": torch.tensor([d / I for d in DOWN_FACTORS], dtype=torch.float32, device=device),
        "fc2_scale": torch.tensor([1.0], dtype=torch.float32, device=device),
        "global_sf": torch.tensor([1.0], dtype=torch.float32, device=device),
    }


def make_inputs(num_tokens: int, device: torch.device):
    """Build deterministic synthetic inputs matching the Rust CPU reference oracle."""
    mags = [0.375 * m for m in (1.0, 2.0, 4.0)]
    rows = [mags[t % 3] if t % 2 == 0 else -mags[t % 3] for t in range(num_tokens)]
    hidden = (
        torch.tensor(rows, dtype=torch.bfloat16, device=device).unsqueeze(1).expand(num_tokens, H).contiguous()
    )
    expert_ids = torch.tensor(
        [[(t + s) % TOP_K for s in range(TOP_K)] for t in range(num_tokens)],
        dtype=torch.int32,
        device=device,
    )
    routing_weights = torch.tensor(
        [[ROUTING_PATTERN[(t + s) % TOP_K] for s in range(TOP_K)] for t in range(num_tokens)],
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty((num_tokens, H), dtype=torch.bfloat16, device=device)
    return hidden, expert_ids, routing_weights, output


def run_step(st, hidden, expert_ids, routing_weights, output):
    """Execute one end-to-end MoE layer (Quantize + Route + FC1 + ReLU² + FC2 + Finalize)."""
    x_q, x_sf = fp4_quantize(
        hidden, st["global_sf"], sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=False
    )
    cute_dsl_fused_moe(
        x=x_q,
        x_sf=x_sf,
        token_selected_experts=expert_ids,
        token_final_scales=routing_weights,
        w1_weight=st["w1"],
        w1_weight_sf=st["w1_sf"],
        w1_alpha=st["w1_alpha"],
        fc2_input_scale=st["fc2_scale"],
        w2_weight=st["w2"],
        w2_weight_sf=st["w2_sf"],
        w2_alpha=st["w2_alpha"],
        num_experts=E,
        top_k=TOP_K,
        output_dtype=torch.bfloat16,
        use_fused_finalize=False,
        moe_output=output,
        activation_type=ActivationType.Relu2.value,
        quant_mode="w4a4",
    )


def main():
    device = torch.device("cuda:0")
    st = build_state(device)
    print("Batch size | FlashInfer CuTeDSL (us)")
    print("-----------+------------------------")
    for b in BATCH_SIZES:
        hidden, expert_ids, routing_weights, output = make_inputs(b, device)
        for _ in range(5):
            run_step(st, hidden, expert_ids, routing_weights, output)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_step(st, hidden, expert_ids, routing_weights, output)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(TIMING_SAMPLES):
            for _ in range(TIMING_WARMUPS):
                g.replay()
            start.record()
            for _ in range(TIMING_REPLAYS):
                g.replay()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0 / TIMING_REPLAYS)
        mean_us = sum(samples) / len(samples)
        print(f"{b:>10} | {mean_us:>22.3f}")


if __name__ == "__main__":
    main()

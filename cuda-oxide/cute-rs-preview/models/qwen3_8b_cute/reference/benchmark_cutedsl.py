#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL (cutlass.cute) reference benchmark for Qwen/Qwen3-8B (BF16 & NVFP4 E2M1+E4M3)."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import torch.nn.functional as F

HIDDEN = 4096
INTERMEDIATE = 12288
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
Q_WIDTH = Q_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
LAYERS = 36
EPSILON = 1e-6
ROPE_THETA = 1_000_000.0

MMA_TILER_MNK = (128, 1, 64)
SF_VEC_SIZE = 16


class CuTeDslNvfp4GemvKernel:
    """CuTeDSL NVFP4 (Float4E2M1FN + Float8E4M3FN, SF_VEC = 16) block-scaled projection kernel."""

    def __init__(self):
        self.threads_per_cta = 128

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, SF_VEC_SIZE
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, SF_VEC_SIZE
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)
        grid = (
            cute.ceil_div(c_tensor.shape[0], 128),
            1,
            c_tensor.shape[2],
        )
        self.kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA_mkl: cute.Tensor,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(MMA_TILER_MNK, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(MMA_TILER_MNK, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER_MNK, (0, None, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(MMA_TILER_MNK, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(MMA_TILER_MNK, (None, None, 0)), (None, None, None)
        )

        tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
        tCgC = cute.make_tensor(tCgC.iterator, 1)
        res = cute.zeros_like(tCgC, cutlass.Float32)

        k_tile_cnt = gA_mkl.layout[3].shape
        for k_tile in range(k_tile_cnt):
            tAgA = gA_mkl[tidx, None, bidx, k_tile, 0]
            tBgB = gB_nkl[None, None, bidy, k_tile, bidz]
            tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, 0]
            tBgSFB = gSFB_nkl[None, None, bidy, k_tile, bidz]

            a_val_nvfp4 = tAgA.load()
            b_val_nvfp4 = tBgB.load()
            sfa_val_fp8 = tAgSFA.load()
            sfb_val_fp8 = tBgSFB.load()

            vec_num = MMA_TILER_MNK[2] // SF_VEC_SIZE
            a_val = a_val_nvfp4.to(cutlass.Float32)
            b_val = b_val_nvfp4.to(cutlass.Float32)
            sfa_shp = (cute.size(sfa_val_fp8.shape) // vec_num, vec_num)
            sfa_val = sfa_val_fp8.to(cutlass.Float32).reshape(sfa_shp)
            sfb_shp = (cute.size(sfb_val_fp8.shape) // vec_num, vec_num)
            sfb_val = sfb_val_fp8.to(cutlass.Float32).reshape(sfb_shp)

            for i in cutlass.range_constexpr(vec_num):
                for j in cutlass.range_constexpr(SF_VEC_SIZE):
                    res += (
                        a_val[i * SF_VEC_SIZE + j]
                        * sfa_val[0, i]
                        * b_val[i * SF_VEC_SIZE + j]
                        * sfb_val[0, i]
                    )
        tCgC.store(res.to(cutlass.BFloat16))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_cutedsl_args(out_dim: int, in_dim: int, rows: int):
    """Allocate fresh NVFP4 (E2M1 + E4M3, SF_VEC = 16) weight/scale/activation views."""
    a_ref = cutlass_torch.matrix(1, out_dim, in_dim, False, cutlass.Float32)
    b_ref = cutlass_torch.matrix(rows, 1, in_dim, False, cutlass.Float32)
    c_ref = cutlass_torch.matrix(rows, out_dim, 1, True, cutlass.Float32)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_ref, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_ref, cutlass.Float4E2M1FN, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16
    )

    a_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=32)
    b_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=32)
    c_tensor.mark_compact_shape_dynamic(mode=0, stride_order=(2, 1, 0), divisibility=16)

    sf_k = ceil_div(in_dim, SF_VEC_SIZE)
    sfa_mma_shape = (1, ceil_div(out_dim, 128), ceil_div(sf_k, 4), 32, 4, 4)
    sfb_mma_shape = (rows, 1, ceil_div(sf_k, 4), 32, 4, 4)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    sfa_f32_cpu = cutlass_torch.create_and_permute_torch_tensor(
        sfa_mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )
    sfb_f32_cpu = cutlass_torch.create_and_permute_torch_tensor(
        sfb_mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )
    sfa_tensor, _ = cutlass_torch.cute_tensor_like(
        sfa_f32_cpu, cutlass.Float8E4M3FN, is_dynamic_layout=True, assumed_align=16
    )
    sfb_tensor, _ = cutlass_torch.cute_tensor_like(
        sfb_f32_cpu, cutlass.Float8E4M3FN, is_dynamic_layout=True, assumed_align=16
    )
    return (a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor), c_torch


def build_cutedsl_projection(out_dim: int, in_dim: int, rows: int, stream: cuda.CUstream):
    """Compile a CuTeDSL NVFP4 (E2M1 + E4M3, SF_VEC = 16) projection for [rows, out_dim, in_dim]."""
    args, c_torch = make_cutedsl_args(out_dim, in_dim, rows)
    kernel = CuTeDslNvfp4GemvKernel()
    compiled = cute.compile(kernel, *args, stream)
    return compiled, args, c_torch


@torch.compile
def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(variance + EPSILON) * weight.to(torch.float32)).to(
        torch.bfloat16
    )


@torch.compile
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = HEAD_DIM // 2
    x1 = x[..., :half].to(torch.float32)
    x2 = x[..., half:].to(torch.float32)
    c = cos.unsqueeze(1).to(torch.float32)
    s = sin.unsqueeze(1).to(torch.float32)
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return out.to(torch.bfloat16)


@torch.compile
def quantize_activation_step(x: torch.Tensor):
    x_blocks = x.reshape(-1, 16).to(torch.float32)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / 6.0)).clamp(min=1.0 / 512.0, max=448.0).to(torch.float8_e4m3fn)
    scaled = (x_blocks / scale.to(torch.float32)).clamp(-6.0, 6.0)
    return scaled, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "cutedsl_comparison.json"
    )
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print(
        f"CuTeDSL Reference | Qwen/Qwen3-8B | H={HIDDEN}, I={INTERMEDIATE}, "
        f"Q={Q_HEADS}, KV={KV_HEADS}, layers={args.layers}/{LAYERS}"
    )

    shapes = {
        "q_o_4096x4096": (Q_WIDTH, HIDDEN),
        "kv_1024x4096": (KV_WIDTH, HIDDEN),
        "gate_up_12288x4096": (INTERMEDIATE, HIDDEN),
        "down_4096x12288": (HIDDEN, INTERMEDIATE),
    }

    decode_kernels = {}
    prefill_kernels = {}
    projection_bench = {}

    for name, (out_dim, in_dim) in shapes.items():
        compiled_dec, dec_args, dec_out = build_cutedsl_projection(out_dim, in_dim, 1, stream)
        compiled_pre, pre_args, pre_out = build_cutedsl_projection(
            out_dim, in_dim, args.prompt_len, stream
        )
        decode_kernels[name] = (compiled_dec, dec_args, dec_out)
        prefill_kernels[name] = (compiled_pre, pre_args, pre_out)

        for _ in range(20):
            compiled_dec(*dec_args, stream)
            compiled_pre(*pre_args, stream)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            compiled_dec(*dec_args, stream)
        torch.cuda.synchronize()
        dec_us = (time.perf_counter() - t0) * 1e6 / 100.0

        t1 = time.perf_counter()
        for _ in range(50):
            compiled_pre(*pre_args, stream)
        torch.cuda.synchronize()
        pre_us = (time.perf_counter() - t1) * 1e6 / 50.0

        projection_bench[name] = {
            "shape": [out_dim, in_dim],
            "decode_1tok_us": round(dec_us, 2),
            "prefill_32tok_us": round(pre_us, 2),
        }
        print(
            f"  [CuTeDSL NVFP4 Kernel] {name:<20} ({out_dim:>5}x{in_dim:<5}): "
            f"decode (1 tok) = {dec_us:>6.2f} us | prefill ({args.prompt_len} tok) = {pre_us:>7.2f} us"
        )

    norm_w = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    head_norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )

    max_seq = args.prompt_len + args.decode_steps + 16
    k_caches = [
        torch.zeros((max_seq, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
        for _ in range(args.layers)
    ]
    v_caches = [
        torch.zeros((max_seq, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
        for _ in range(args.layers)
    ]

    # Allocate 4 distinct layer weight sets (~414 MiB > 96 MB RTX 5090 L2 cache) so
    # 36-layer execution streams weights from GDDR7 DRAM instead of staying L2-resident.
    num_weight_sets = 4
    decode_layer_sets = []
    prefill_layer_sets = []
    for _ in range(num_weight_sets):
        dec_set = {}
        pre_set = {}
        for name, (out_dim, in_dim) in shapes.items():
            dec_set[name] = (decode_kernels[name][0], *make_cutedsl_args(out_dim, in_dim, 1))
            pre_set[name] = (
                prefill_kernels[name][0],
                *make_cutedsl_args(out_dim, in_dim, args.prompt_len),
            )
        decode_layer_sets.append(dec_set)
        prefill_layer_sets.append(pre_set)

    def run_cutedsl_nvfp4_forward(x: torch.Tensor, start_pos: int, layer_sets: list):
        rows = x.shape[0]
        pos = torch.arange(start_pos, start_pos + rows, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)
        cos = torch.cos(freqs).to(torch.bfloat16)
        sin = torch.sin(freqs).to(torch.bfloat16)

        hidden = x
        for layer_idx in range(args.layers):
            kernel_set = layer_sets[layer_idx % len(layer_sets)]
            q_fn, q_args, q_out = kernel_set["q_o_4096x4096"]
            kv_fn, kv_args, kv_out = kernel_set["kv_1024x4096"]
            gu_fn, gu_args, gu_out = kernel_set["gate_up_12288x4096"]
            dn_fn, dn_args, dn_out = kernel_set["down_4096x12288"]

            normed = rms_norm(hidden, norm_w)
            quantize_activation_step(normed)
            q_fn(*q_args, stream)
            kv_fn(*kv_args, stream)
            kv_fn(*kv_args, stream)

            q = rms_norm(q_out.reshape(rows, Q_HEADS, HEAD_DIM), head_norm_w)
            k = rms_norm(kv_out.reshape(rows, KV_HEADS, HEAD_DIM), head_norm_w)
            v = kv_out.reshape(rows, KV_HEADS, HEAD_DIM)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            k_caches[layer_idx][start_pos : start_pos + rows] = k
            v_caches[layer_idx][start_pos : start_pos + rows] = v

            k_live = k_caches[layer_idx][: start_pos + rows].permute(1, 0, 2).unsqueeze(0)
            v_live = v_caches[layer_idx][: start_pos + rows].permute(1, 0, 2).unsqueeze(0)
            q_live = q.permute(1, 0, 2).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(
                q_live, k_live, v_live, is_causal=(rows > 1), enable_gqa=True
            )
            attn_flat = attn_out.squeeze(0).permute(1, 0, 2).reshape(rows, HIDDEN)

            quantize_activation_step(attn_flat)
            q_fn(*q_args, stream)
            residual = hidden + q_out.reshape(rows, HIDDEN)

            post = rms_norm(residual, norm_w)
            quantize_activation_step(post)
            gu_fn(*gu_args, stream)
            gate = gu_out.reshape(rows, INTERMEDIATE)
            gu_fn(*gu_args, stream)
            up = gu_out.reshape(rows, INTERMEDIATE)
            swiglu = F.silu(gate) * up

            quantize_activation_step(swiglu)
            dn_fn(*dn_args, stream)
            hidden = residual + dn_out.reshape(rows, HIDDEN)

        return rms_norm(hidden, norm_w)

    x_prefill = torch.randn((args.prompt_len, HIDDEN), dtype=torch.bfloat16, device=device)
    x_decode = torch.randn((1, HIDDEN), dtype=torch.bfloat16, device=device)

    _ = run_cutedsl_nvfp4_forward(x_prefill, 0, prefill_layer_sets)
    _ = run_cutedsl_nvfp4_forward(x_decode, args.prompt_len, decode_layer_sets)
    torch.cuda.synchronize()

    t_pre = time.perf_counter()
    _ = run_cutedsl_nvfp4_forward(x_prefill, 0, prefill_layer_sets)
    torch.cuda.synchronize()
    cutedsl_prefill_ms = (time.perf_counter() - t_pre) * 1000.0
    cutedsl_prefill_tps = args.prompt_len / (cutedsl_prefill_ms / 1000.0)

    t_dec = time.perf_counter()
    for step in range(args.decode_steps):
        x_decode = run_cutedsl_nvfp4_forward(x_decode, args.prompt_len + step, decode_layer_sets)
    torch.cuda.synchronize()
    cutedsl_decode_total_ms = (time.perf_counter() - t_dec) * 1000.0
    cutedsl_decode_ms = cutedsl_decode_total_ms / args.decode_steps
    cutedsl_decode_tps = args.decode_steps / (cutedsl_decode_total_ms / 1000.0)

    print(
        f"  [CuTeDSL NVFP4 (E2M1+E4M3, SF=16)] prefill ({args.prompt_len} tok): "
        f"{cutedsl_prefill_ms:7.2f} ms ({cutedsl_prefill_tps:7.1f} tok/s) | "
        f"decode ({args.decode_steps} steps): {cutedsl_decode_ms:6.2f} ms/tok ({cutedsl_decode_tps:6.1f} tok/s)"
    )

    report = {
        "model": "Qwen/Qwen3-8B",
        "gpu": torch.cuda.get_device_name(0),
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch_version": torch.__version__,
        "layers": args.layers,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "cutedsl_nvfp4_projections": projection_bench,
        "cutedsl_nvfp4_e2e": {
            "prefill_ms": round(cutedsl_prefill_ms, 2),
            "prefill_tok_per_sec": round(cutedsl_prefill_tps, 1),
            "decode_ms_per_tok": round(cutedsl_decode_ms, 2),
            "decode_tok_per_sec": round(cutedsl_decode_tps, 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


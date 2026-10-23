# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently authored CuTeDSL counterpart of the cuda-oxide decode example.

This is not an upstream CUTLASS attention kernel. It implements the same
single-query, 64-row split-KV schedule as the Rust example using public CuTeDSL
copy/layout primitives and scalar FP32 arithmetic. No query padding, matrix
multiply instructions, or tensor memory are used.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import SmemAllocator


HEAD_DIM = 64
KV_CHUNK = 64
PARTIAL_THREADS = 128
MERGE_THREADS = 32
PARTIAL_SHARED_BYTES = 17696
SCALE_LOG2 = 0.125 * 1.4426950408889634


class AttentionDecode:
    """FP16 BSHD MHA with B=Sq=1, D=64 and a positive dense KV-cache length."""

    def __init__(self, static_seq_k=0, static_heads=0):
        if (static_seq_k == 0) != (static_heads == 0) or min(static_seq_k, static_heads) < 0:
            raise ValueError("static dimensions must both be positive, or both zero for runtime")
        self.static_seq_k = static_seq_k
        self.static_heads = static_heads

    @cute.jit
    def __call__(
        self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
        partial: cute.Tensor, out: cute.Tensor,
        seq_k: cutlass.Int32, heads: cutlass.Int32, splits: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        shared_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0,
            cute.make_layout((64, 64), stride=(64, 1)),
        )
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float16, num_bits_per_copy=128,
        )
        copy = cute.make_tiled_copy_tv(
            copy_atom, cute.make_layout((16, 8), stride=(8, 1)),
            cute.make_layout((1, 8), stride=(8, 1)),
        )
        self.fmha_decode_partial(q, k, v, partial, seq_k, heads, shared_layout, copy).launch(
            grid=(heads * splits, 1, 1),
            block=(PARTIAL_THREADS, 1, 1), stream=stream,
        )
        self.fmha_decode_merge(partial, out, heads, splits).launch(
            grid=(heads, 1, 1), block=(MERGE_THREADS, 1, 1), stream=stream,
        )

    @cute.kernel
    def fmha_decode_partial(
        self, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
        partial: cute.Tensor, seq_k: cutlass.Int32, heads: cutlass.Int32,
        shared_layout: cute.ComposedLayout, copy: cute.TiledCopy,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        if cutlass.const_expr(self.static_seq_k > 0):
            seq_k = self.static_seq_k
            heads = self.static_heads
            splits = (self.static_seq_k + KV_CHUNK - 1) // KV_CHUNK
            kv_stride = self.static_heads * HEAD_DIM
            state_stride = splits * 66
        else:
            splits = (seq_k + KV_CHUNK - 1) // KV_CHUNK
            kv_stride = cute.assume(cutlass.Int64(heads) * HEAD_DIM, divby=8)
            state_stride = cutlass.Int64(splits) * 66
        head = block // splits
        split = block % splits
        start = split * KV_CHUNK

        # The selected mode matches Rust: specialize both dimensions together,
        # or preserve runtime dimensions, strides and split bounds.
        qo_layout = cute.make_layout((1, 1, heads, 64), stride=(0, 0, 64, 1))
        kv_layout = cute.make_layout(
            (1, seq_k, heads, 64),
            stride=(0, kv_stride, 64, 1),
        )
        state_layout = cute.make_layout(
            (heads, splits, 66), stride=(state_stride, 66, 1),
        )
        q = cute.make_tensor(q.iterator, qo_layout)
        k = cute.make_tensor(k.iterator, kv_layout)
        v = cute.make_tensor(v.iterator, kv_layout)
        partial = cute.make_tensor(partial.iterator, state_layout)

        smem = SmemAllocator()
        sk = smem.allocate_tensor(cutlass.Float16, shared_layout, 1024)
        sv = smem.allocate_tensor(cutlass.Float16, shared_layout, 1024)
        scores = smem.allocate_tensor(cutlass.Float32, cute.make_layout(64), 4)
        warp_max = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), 4)
        warp_sum = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), 4)
        warp_output = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((4, 64), stride=(64, 1)), 4,
        )

        gk = cute.local_tile(k[0, None, head, None], (64, 64), (split, 0))
        gv = cute.local_tile(v[0, None, head, None], (64, 64), (split, 0))
        thr_copy = copy.get_slice(tid)
        tgk = thr_copy.partition_S(gk)
        tgv = thr_copy.partition_S(gv)
        tsk = thr_copy.partition_D(sk)
        tsv = thr_copy.partition_D(sv)
        # Four 16-row groups, each thread moving one aligned 16-byte vector.
        # Invalid rows are zeroed; their global pointers are never read.
        for group in cutlass.range_constexpr(4):
            if start + group * 16 + tid // 8 < seq_k:
                cute.copy(copy, tgk[None, group, None], tsk[None, group, None])
                cute.copy(copy, tgv[None, group, None], tsv[None, group, None])
            else:
                tsk[None, group, None].fill(0.0)
                tsv[None, group, None].fill(0.0)
        cute.arch.cp_async_commit_group()

        rq = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)
        for d in cutlass.range_constexpr(16):
            rq[d] = q[0, 0, head, (tid % 4) * 16 + d].to(cutlass.Float32)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Four threads cooperate on one key; two passes cover the 64-row tile.
        for key_pass in cutlass.range_constexpr(2):
            row = tid // 4 + key_pass * 32
            dot = cutlass.Float32(0.0)
            for d in cutlass.range_constexpr(16):
                kval = sk[row, (tid % 4) * 16 + d].to(cutlass.Float32)
                dot = dot + rq[d] * kval
            dot = dot + cute.arch.shuffle_sync_bfly(dot, offset=1)
            dot = dot + cute.arch.shuffle_sync_bfly(dot, offset=2)
            if tid % 4 == 0:
                if start + row < seq_k:
                    scores[row] = dot
                else:
                    scores[row] = cutlass.Float32(float("-inf"))
        cute.arch.sync_threads()

        score = cutlass.Float32(float("-inf"))
        if tid < KV_CHUNK:
            score = scores[tid]
        maximum = score
        for offset in cutlass.range_constexpr(5):
            maximum = cute.arch.fmax(
                maximum, cute.arch.shuffle_sync_bfly(maximum, offset=16 >> offset),
            )
        if lane == 0:
            warp_max[warp] = maximum
        cute.arch.sync_threads()
        maximum = cute.arch.fmax(
            cute.arch.fmax(warp_max[0], warp_max[1]),
            cute.arch.fmax(warp_max[2], warp_max[3]),
        )
        weight = cute.math.exp2((score - maximum) * SCALE_LOG2, fastmath=True)
        if tid < KV_CHUNK:
            scores[tid] = weight
        denominator = weight
        for offset in cutlass.range_constexpr(5):
            denominator = denominator + cute.arch.shuffle_sync_bfly(
                denominator, offset=16 >> offset,
            )
        if lane == 0:
            warp_sum[warp] = denominator
        cute.arch.sync_threads()
        denominator = warp_sum[0]
        for w in cutlass.range_constexpr(1, 4):
            denominator = denominator + warp_sum[w]

        # Each warp accumulates 16 cache rows; each lane owns two dimensions.
        p0 = cutlass.Float32(0.0)
        p1 = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(16):
            row = warp * 16 + r
            p = scores[row]
            p0 = p0 + p * sv[row, lane * 2].to(cutlass.Float32)
            p1 = p1 + p * sv[row, lane * 2 + 1].to(cutlass.Float32)
        warp_output[warp, lane * 2] = p0
        warp_output[warp, lane * 2 + 1] = p1
        cute.arch.sync_threads()
        if warp == 0:
            p0 = warp_output[0, lane * 2]
            p1 = warp_output[0, lane * 2 + 1]
            for w in cutlass.range_constexpr(1, 4):
                p0 = p0 + warp_output[w, lane * 2]
                p1 = p1 + warp_output[w, lane * 2 + 1]
            partial[head, split, lane * 2 + 2] = p0
            partial[head, split, lane * 2 + 3] = p1
            if lane == 0:
                partial[head, split, 0] = maximum
                partial[head, split, 1] = denominator

    @cute.kernel
    def fmha_decode_merge(
        self, partial: cute.Tensor, out: cute.Tensor,
        heads: cutlass.Int32, splits: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        head, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.static_seq_k > 0):
            heads = self.static_heads
            splits = (self.static_seq_k + KV_CHUNK - 1) // KV_CHUNK
            state_stride = splits * 66
        else:
            state_stride = cutlass.Int64(splits) * 66
        state_layout = cute.make_layout(
            (heads, splits, 66), stride=(state_stride, 66, 1),
        )
        partial = cute.make_tensor(partial.iterator, state_layout)
        out = cute.make_tensor(
            out.iterator, cute.make_layout((1, 1, heads, 64), stride=(0, 0, 64, 1)),
        )
        maximum = cutlass.Float32(float("-inf"))
        for split in cutlass.range(splits):
            maximum = cute.arch.fmax(maximum, partial[head, split, 0])
        denominator = cutlass.Float32(0.0)
        p0 = cutlass.Float32(0.0)
        p1 = cutlass.Float32(0.0)
        for split in cutlass.range(splits):
            weight = cute.math.exp2(
                (partial[head, split, 0] - maximum) * SCALE_LOG2, fastmath=True,
            )
            denominator = denominator + partial[head, split, 1] * weight
            p0 = p0 + partial[head, split, lane * 2 + 2] * weight
            p1 = p1 + partial[head, split, lane * 2 + 3] * weight
        inverse = cute.arch.rcp_approx(denominator)
        out[0, 0, head, lane * 2] = (p0 * inverse).to(cutlass.Float16)
        out[0, 0, head, lane * 2 + 1] = (p1 * inverse).to(cutlass.Float16)

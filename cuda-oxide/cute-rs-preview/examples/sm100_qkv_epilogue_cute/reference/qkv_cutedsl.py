# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 (B200) Python CuTeDSL reference for fused QKV projection + RMSNorm + FP8 quant + Paged KV Scatter.

Computes X [M, K] * W_qkv^T [K, N_q + N_k + N_v] with a persistent two-CTA (M=128, N=256, K=64)
schedule, 8-stage cluster TMA mainloop, and 2-stage TMEM ping-pong (TMEM_0, TMEM_1) overlapping
the per-head RMSNorm + FP8 (e4m3fn) quantization + Q store + Paged KV cache scatter epilogue
with the next tile's Tensor Core MMA.
"""

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import cuda, primitives as prim
from cuda.bindings import driver

_STAGES = 8
_A_BYTES = 64 * 64 * 2     # 8,192 bytes per CTA (M=128 split across 2 CTAs -> 64 rows per CTA)
_B_BYTES = 128 * 64 * 2    # 16,384 bytes per CTA (N=256 split across 2 CTAs -> 128 rows per CTA)
_B_OFFSET = _STAGES * _A_BYTES                        # 65,536
_GAMMA_Q_OFFSET = _B_OFFSET + _STAGES * _B_BYTES      # 196,608
_GAMMA_K_OFFSET = _GAMMA_Q_OFFSET + 512               # 197,120
_CONTROL_OFFSET = _GAMMA_K_OFFSET + 512               # 197,632
SMEM_BYTES = _CONTROL_OFFSET + 256                    # 197,888
_GROUP = prim.CTAGroup.CTA_2


@cute.jit
def _raw(ptr: cute.Pointer):
    """Adapt a CuTe pointer for experimental CTM primitives."""
    return cutlass.Pointer(ptr.to_llvm_ptr(), dtype=ptr.dtype, space=ptr.memspace)


@cute.jit
def _wait(bar: cutlass.Pointer, phase: cutlass.Int32, cluster: cutlass.Constexpr = False):
    """Wait for an mbarrier phase completion with optional cluster acquire ordering."""
    if cutlass.const_expr(cluster):
        while not prim.mbarrier_try_wait_parity(
            bar, phase, scope=prim.MBarrierScope.CLUSTER, order=prim.MemOrder.ACQUIRE
        ):
            pass
    else:
        while not prim.mbarrier_try_wait_parity(bar, phase):
            pass


@cute.jit
def _operand(base: cute.Pointer):
    """Encode a 64-wide K-major FP16 shared-memory tile descriptor with 128B swizzle."""
    return prim.Tcgen05SmemDesc.build(
        _raw(base),
        leading_byte_offset=16,
        stride_byte_offset=8 * 64 * 2,
        layout=prim.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )


@cute.jit
def _swizzle_tile(tile: cutlass.Int32, mtiles: cutlass.Int32, ntiles: cutlass.Int32):
    """Group-M (G=8) 2D tile swizzle keeping 1,024 rows of X resident in B200 L2 cache."""
    group_m = cutlass.Int32(8) if mtiles >= 8 else mtiles
    group_tiles = group_m * ntiles
    group_id = tile // group_tiles
    first_m = group_id * group_m
    rem_m = mtiles - first_m
    actual_m = group_m if rem_m >= group_m else rem_m
    in_group = tile % group_tiles
    m_tile = first_m + (in_group % actual_m)
    n_tile = in_group // actual_m
    return m_tile, n_tile


class FusedQkvEpilogueCuTeDSL:
    """Two-CTA SM100 persistent fused QKV + RMSNorm + FP8 quant + Q store + Paged KV Scatter."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        clusters: int,
        overlap_epilogue: bool = True,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.clusters = clusters
        self.overlap_epilogue = overlap_epilogue

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w_qkv: cute.Tensor,
        gamma_q: cute.Tensor,
        gamma_k: cute.Tensor,
        slot_mapping: cute.Tensor,
        q_out: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        rms_eps: cutlass.Float32,
        inv_scale_q: cutlass.Float32,
        inv_scale_k: cutlass.Float32,
        inv_scale_v: cutlass.Float32,
        stream: driver.CUstream,
    ):
        xd = cuda.create_tensor_map_tiled_from_view(
            x, (64, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        wd = cuda.create_tensor_map_tiled_from_view(
            w_qkv, (128, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        self.kernel(
            xd,
            wd,
            gamma_q,
            gamma_k,
            slot_mapping,
            q_out,
            k_cache,
            v_cache,
            rms_eps,
            inv_scale_q,
            inv_scale_k,
            inv_scale_v,
            x.shape[0],
            w_qkv.shape[0],
            x.shape[1],
        ).launch(
            grid=(2 * self.clusters, 1, 1),
            block=(192, 1, 1),
            cluster=(2, 1, 1),
            smem=SMEM_BYTES,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        xd: cutlass.GridConstant[cuda.TensorMap],
        wd: cutlass.GridConstant[cuda.TensorMap],
        gamma_q: cute.Tensor,
        gamma_k: cute.Tensor,
        slot_mapping: cute.Tensor,
        q_out: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        rms_eps: cutlass.Float32,
        inv_scale_q: cutlass.Float32,
        inv_scale_k: cutlass.Float32,
        inv_scale_v: cutlass.Float32,
        m: cutlass.Constexpr,
        n_total: cutlass.Constexpr,
        k: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = tid // 32
        rank = prim.cluster_ctarank()
        base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
        smem_gq = cute.recast_ptr(base + _GAMMA_Q_OFFSET, dtype=cutlass.Float32)
        smem_gk = cute.recast_ptr(base + _GAMMA_K_OFFSET, dtype=cutlass.Float32)
        full = _raw(cute.recast_ptr(base + _CONTROL_OFFSET, dtype=cutlass.Int64))
        empty = full + _STAGES
        acc_full = empty + _STAGES
        acc_empty = acc_full + 2
        dealloc = acc_empty + 2
        token = cutlass.Pointer(
            (dealloc + 1).to_llvm_ptr(),
            dtype=cutlass.Int32,
            space=cutlass.AddressSpace.smem,
        )

        if tid < 128:
            smem_gq[tid] = gamma_q[tid] * inv_scale_q
            smem_gk[tid] = gamma_k[tid] * inv_scale_k
        if tid == 0:
            for slot in cutlass.range_constexpr(_STAGES):
                prim.mbarrier_init(full + slot, 1)
                prim.mbarrier_init(empty + slot, 1)
            for s in cutlass.range_constexpr(2):
                prim.mbarrier_init(acc_full + s, 1)
                prim.mbarrier_init(acc_empty + s, 2)
            prim.mbarrier_init(dealloc, 32)
        prim.fence_mbarrier_init()
        prim.barrier_cluster_arrive_relaxed_aligned()
        prim.barrier_cluster_wait_aligned()
        if warp == 4:
            if prim.elect_sync():
                prim.prefetch_tensormap(xd.get_ptr())
                prim.prefetch_tensormap(wd.get_ptr())

        first = cute.arch.block_idx()[0] // 2
        stride = cute.arch.grid_dim()[0] // 2
        mtiles = m // 128
        ntiles = n_total // 256
        tile_count = mtiles * ntiles
        ktiles = k // 64
        num_stages = _STAGES if self.overlap_epilogue else 1

        if warp == 5:
            cute.arch.warpgroup_reg_dealloc(40)
            slot, phase, tile = cutlass.Int32(0), cutlass.Int32(1), first
            while tile < tile_count:
                m_tile, n_tile = _swizzle_tile(tile, mtiles, ntiles)
                row = m_tile * 128 + rank * 64
                col = n_tile * 256 + rank * 128
                kt = cutlass.Int32(0)
                while kt < ktiles:
                    _wait(empty + slot, phase, True)
                    if prim.elect_sync():
                        if rank == 0:
                            prim.mbarrier_arrive_expect_tx(
                                full + slot, 2 * (_A_BYTES + _B_BYTES)
                            )
                        prim.cp_async_bulk_tensor_shared_cluster_global(
                            _raw(base + slot * _A_BYTES),
                            xd.get_ptr(),
                            [kt * 64, row],
                            cutlass.Array(full + slot, (1,), dtype=cutlass.Int64),
                            [],
                            group=_GROUP,
                        )
                        prim.cp_async_bulk_tensor_shared_cluster_global(
                            _raw(base + _B_OFFSET + slot * _B_BYTES),
                            wd.get_ptr(),
                            [kt * 64, col],
                            cutlass.Array(full + slot, (1,), dtype=cutlass.Int64),
                            [],
                            group=_GROUP,
                        )
                    slot += 1
                    if slot == num_stages:
                        slot = 0
                        phase ^= 1
                    kt += 1
                tile += stride
            for _ in cutlass.range_constexpr(num_stages - 1):
                slot += 1
                if slot == num_stages:
                    slot = 0
                    phase ^= 1
            _wait(empty + slot, phase, True)

        elif warp == 4:
            cute.arch.warpgroup_reg_dealloc(40)
            prim.barrier_cta_sync(2, thread_count=160)
            tmem = token.load()
            idesc = prim.Tcgen05InstrDesc.build(
                a_dtype=cutlass.Float16,
                b_dtype=cutlass.Float16,
                c_dtype=cutlass.Float32,
                m_dim=128,
                n_dim=256,
            )
            slot, phase = cutlass.Int32(0), cutlass.Int32(0)
            acc_slot, acc_phase, tile = cutlass.Int32(0), cutlass.Int32(1), first
            if rank == 0:
                while tile < tile_count:
                    _wait(acc_empty + acc_slot, acc_phase, True)
                    d_ptr = prim.make_tmem_ptr(tmem + acc_slot * 128, cutlass.Float32)
                    kt = cutlass.Int32(0)
                    while kt < ktiles:
                        _wait(full + slot, phase, True)
                        prim.tcgen05_fence(prim.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prim.elect_sync():
                            da = _operand(base + slot * _A_BYTES)
                            db = _operand(base + _B_OFFSET + slot * _B_BYTES)
                            for part in cutlass.range_constexpr(4):
                                use_d = kt > 0
                                if cutlass.const_expr(part > 0):
                                    use_d = cutlass.Boolean(True)
                                prim.tcgen05_mma(
                                    prim.Tcgen05MMAKind.F16,
                                    _GROUP,
                                    d_ptr,
                                    da.advance_start_address(part * 32),
                                    db.advance_start_address(part * 32),
                                    idesc,
                                    use_d,
                                )
                            prim.tcgen05_commit(
                                empty + slot, multicast_mask=3, group=_GROUP
                            )
                        slot += 1
                        if slot == num_stages:
                            slot = 0
                            phase ^= 1
                        kt += 1
                    if prim.elect_sync():
                        prim.tcgen05_commit(
                            acc_full + acc_slot, multicast_mask=3, group=_GROUP
                        )
                    if cutlass.const_expr(self.overlap_epilogue):
                        acc_slot ^= 1
                        if acc_slot == 0:
                            acc_phase ^= 1
                    else:
                        acc_phase ^= 1
                    tile += stride
                if cutlass.const_expr(self.overlap_epilogue):
                    acc_slot ^= 1
                    if acc_slot == 0:
                        acc_phase ^= 1
                _wait(acc_empty + acc_slot, acc_phase, True)

        else:
            cute.arch.warpgroup_reg_alloc(232)
            if warp == 0:
                prim.tcgen05_alloc(token, 512, group=_GROUP)
                prim.tcgen05_relinquish_alloc_permit(group=_GROUP)
            prim.barrier_cta_sync(2, thread_count=160)
            tmem = token.load()
            acc_slot, acc_phase, tile = cutlass.Int32(0), cutlass.Int32(0), first
            q_heads = cutlass.Int32(self.num_q_heads)
            qk_heads = cutlass.Int32(self.num_q_heads + self.num_kv_heads)
            inv_dim = cutlass.Float32(1.0 / 128.0)
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            while tile < tile_count:
                m_tile, n_tile = _swizzle_tile(tile, mtiles, ntiles)
                row = m_tile * 128 + rank * 64 + (tid % 64)
                head_idx = n_tile * 2 + (tid // 64)
                _wait(acc_full + acc_slot, acc_phase, True)
                prim.tcgen05_fence(prim.Tcgen05Fence.AFTER_THREAD_SYNC)
                tmem_row = tmem + acc_slot * 128 + (warp * 32 << 16)
                c0 = prim.tcgen05_ld(
                    prim.Tcgen05LdStShape.SHAPE_32X32B,
                    prim.make_tmem_ptr(tmem_row, cutlass.Float32),
                    num=32,
                )
                c1 = prim.tcgen05_ld(
                    prim.Tcgen05LdStShape.SHAPE_32X32B,
                    prim.make_tmem_ptr(tmem_row + 32, cutlass.Float32),
                    num=32,
                )
                c2 = prim.tcgen05_ld(
                    prim.Tcgen05LdStShape.SHAPE_32X32B,
                    prim.make_tmem_ptr(tmem_row + 64, cutlass.Float32),
                    num=32,
                )
                c3 = prim.tcgen05_ld(
                    prim.Tcgen05LdStShape.SHAPE_32X32B,
                    prim.make_tmem_ptr(tmem_row + 96, cutlass.Float32),
                    num=32,
                )
                prim.tcgen05_wait(prim.Tcgen05Wait.LOAD)
                if cutlass.const_expr(self.overlap_epilogue):
                    prim.tcgen05_fence(prim.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prim.barrier_cta_sync(1, thread_count=128)
                    if tid == 0:
                        peer_empty = prim.mapa(acc_empty + acc_slot, 0)
                        prim.mbarrier_arrive(peer_empty, scope=prim.MemScope.CLUSTER)

                out_f8 = cute.make_rmem_tensor(128, cutlass.Float8E4M3FN)
                if head_idx < qk_heads:
                    sum_sq = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(32):
                        sum_sq = (
                            sum_sq
                            + c0[j] * c0[j]
                            + c1[j] * c1[j]
                            + c2[j] * c2[j]
                            + c3[j] * c3[j]
                        )
                    inv_rms = cute.math.rsqrt(sum_sq * inv_dim + rms_eps, approx=True)
                    g_off = cutlass.Int32(0) if head_idx < q_heads else cutlass.Int32(128)
                    for j in cutlass.range_constexpr(32):
                        out_f8[j] = (c0[j] * inv_rms * smem_gq[g_off + j]).to(
                            cutlass.Float8E4M3FN
                        )
                        out_f8[32 + j] = (c1[j] * inv_rms * smem_gq[g_off + 32 + j]).to(
                            cutlass.Float8E4M3FN
                        )
                        out_f8[64 + j] = (c2[j] * inv_rms * smem_gq[g_off + 64 + j]).to(
                            cutlass.Float8E4M3FN
                        )
                        out_f8[96 + j] = (c3[j] * inv_rms * smem_gq[g_off + 96 + j]).to(
                            cutlass.Float8E4M3FN
                        )
                else:
                    for j in cutlass.range_constexpr(32):
                        out_f8[j] = (c0[j] * inv_scale_v).to(cutlass.Float8E4M3FN)
                        out_f8[32 + j] = (c1[j] * inv_scale_v).to(cutlass.Float8E4M3FN)
                        out_f8[64 + j] = (c2[j] * inv_scale_v).to(cutlass.Float8E4M3FN)
                        out_f8[96 + j] = (c3[j] * inv_scale_v).to(cutlass.Float8E4M3FN)

                if head_idx < q_heads:
                    dst_view = q_out[row, head_idx, None]
                    dst_view = cute.make_tensor(
                        dst_view.iterator.align(32), dst_view.layout
                    )
                    cute.copy(copy_atom, out_f8, dst_view)
                elif head_idx < qk_heads:
                    slot_idx = slot_mapping[row]
                    kv_head = head_idx - q_heads
                    dst_view = k_cache[slot_idx, kv_head, None]
                    dst_view = cute.make_tensor(
                        dst_view.iterator.align(32), dst_view.layout
                    )
                    cute.copy(copy_atom, out_f8, dst_view)
                else:
                    slot_idx = slot_mapping[row]
                    kv_head = head_idx - qk_heads
                    dst_view = v_cache[slot_idx, kv_head, None]
                    dst_view = cute.make_tensor(
                        dst_view.iterator.align(32), dst_view.layout
                    )
                    cute.copy(copy_atom, out_f8, dst_view)

                if cutlass.const_expr(not self.overlap_epilogue):
                    prim.tcgen05_fence(prim.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prim.barrier_cta_sync(1, thread_count=128)
                    if tid == 0:
                        peer_empty = prim.mapa(acc_empty + acc_slot, 0)
                        prim.mbarrier_arrive(peer_empty, scope=prim.MemScope.CLUSTER)
                    acc_phase ^= 1
                else:
                    acc_slot ^= 1
                    if acc_slot == 0:
                        acc_phase ^= 1
                tile += stride

            if warp == 0:
                prim.barrier_cta_sync(3, thread_count=32)
                peer_dealloc = prim.mapa(dealloc, rank ^ 1)
                prim.mbarrier_arrive(peer_dealloc, scope=prim.MemScope.CLUSTER)
                _wait(dealloc, cutlass.Int32(0), True)
                prim.tcgen05_dealloc(
                    prim.make_tmem_ptr(tmem, cutlass.Float32), 512, group=_GROUP
                )

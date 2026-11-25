# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact physical/scheduling port of blockscale_gemm_ctm using CUTLASS CTM.

CuTe supplies kernel tracing and launch only. All hardware primitives are the
upstream cutlass.experimental.primitives operations; no tensor/layout algebra
or replacement inline PTX is used. Integer offsets match physical.rs.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.experimental import primitives as prims
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers import dsl_user_op

SMEM_BYTES = 86016
STAGES = 3
DATA_BYTES = 8192
SCALE_BYTES = 512
B_OFFSET = 24576
SFA_OFFSET = 49152
SFB_OFFSET = 50688
BARRIERS = 52224
OUTPUT = 53248
TX_BYTES = 17408
N_OFFSETS = (0, 8, 32, 40, 64, 72, 96, 104)


@dsl_user_op
def _mma(c, a, b0, b1, sa, sb, *, loc=None, ip=None):
    """Adapt register vectors to the actual upstream raw-shaped CTM MMA API."""
    f32 = cutlass.Float32.mlir_type
    result = prims.mma_block_scale(
        llvm.StructType.get_literal([f32] * 4),
        ir.Attribute.parse("#nvvm.shape<m = 16, n = 8, k = 64>"),
        prims.ScaleVecSize.X2,
        prims.BlockScaleFormat.UE8M0,
        prims.MMABlockScaleKind.MXF4,
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        [b0.ir_value(loc=loc, ip=ip), b1.ir_value(loc=loc, ip=ip)],
        [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        cutlass.Int32(sa).ir_value(loc=loc, ip=ip),
        cutlass.Int16(0).ir_value(loc=loc, ip=ip),
        cutlass.Int16(0).ir_value(loc=loc, ip=ip),
        cutlass.Int32(sb).ir_value(loc=loc, ip=ip),
        cutlass.Int16(0).ir_value(loc=loc, ip=ip),
        cutlass.Int16(0).ir_value(loc=loc, ip=ip),
        multiplicand_a_ptx_type=ir.Attribute.parse("#nvvm.mma_type<e2m1>"),
        multiplicand_b_ptx_type=ir.Attribute.parse("#nvvm.mma_type<e2m1>"),
        loc=loc,
        ip=ip,
    )
    return cutlass.Vector.from_elements(
        tuple(cutlass.Float32(llvm.extractvalue(f32, result, [i], loc=loc, ip=ip))
              for i in range(4)), dtype=cutlass.Float32, loc=loc, ip=ip)


@cute.jit
def _scale_word(row):
    return (row & 31) * 4 + ((row >> 5) & 3)


@cute.jit
def _matrix_row(atom, half, lane):
    sub = lane // 8
    row = 16 * atom + 8 * (sub % 2) + lane % 8
    column = 16 * half + 8 * (sub // 2)
    plain = row * 64 + column * 2
    return plain ^ ((plain & 0x180) >> 3)


@cute.jit
def _store(output, c, warp, lane, band: cutlass.Constexpr[int],
           slot: cutlass.Constexpr[int]):
    row = 16 * (warp % 4) + 64 * band + (lane & 15)
    column = 16 * (warp // 4) + N_OFFSETS[slot]
    plain = row * 64 + column % 64
    element = (column // 64) * 8192 + (plain ^ ((plain & 0x1C0) >> 3))
    # PTX packs its first input high; Rust's public helper takes low first.
    top = prims.cvt_packfloat_f32(c[1], c[0], 0, prims.CVTPackFloat.F16X2,
                                  rnd=prims.FPRoundingMode.RN)
    bottom = prims.cvt_packfloat_f32(c[3], c[2], 0, prims.CVTPackFloat.F16X2,
                                     rnd=prims.FPRoundingMode.RN)
    prims.stmatrix(output + element, [top, bottom], prims.MMALayout.ROW)


@cute.kernel
def blockscale_gemm(
    a_map: cutlass.GridConstant[cuda.TensorMap],
    b_map: cutlass.GridConstant[cuda.TensorMap],
    sa_map: cutlass.GridConstant[cuda.TensorMap],
    sb_map: cutlass.GridConstant[cuda.TensorMap],
    c_map: cutlass.GridConstant[cuda.TensorMap],
):
    tid, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    stride, _, _ = cute.arch.grid_dim()
    warp = tid // 32
    lane = tid % 32
    storage = cutlass.Array(dtype=cutlass.Uint8, shape=(SMEM_BYTES,),
                            space=cutlass.AddressSpace.smem, alignment=1024)
    base = storage.data_ptr()
    full = cutlass.Pointer(base + BARRIERS, dtype=cutlass.Int64)
    empty = full + STAGES
    if tid == 0:
        for s in cutlass.range_constexpr(STAGES):
            prims.mbarrier_init(full + s, 1)
            prims.mbarrier_init(empty + s, 8)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0, thread_count=288)

    slot = cutlass.Int32(0)
    if warp < 8:
        wm = warp % 4
        wn = warp // 4
        phase = cutlass.Int32(0)
        while tile < 256:
            tile_m = tile % 16
            tile_n = tile // 16

            c00 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c01 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c02 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c03 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c04 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c05 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c06 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c07 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c10 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c11 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c12 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c13 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c14 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c15 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c16 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            c17 = cutlass.full((4,), 0.0, dtype=cutlass.Float32)
            stage = cutlass.Int32(0)
            while stage < 8:
                while not prims.mbarrier_try_wait_parity(full + slot, phase):
                    pass
                a = base + slot * DATA_BYTES
                b = base + B_OFFSET + slot * DATA_BYTES
                sa = cutlass.Pointer(base + SFA_OFFSET + slot * SCALE_BYTES,
                                     dtype=cutlass.Uint32)
                sb = cutlass.Pointer(base + SFB_OFFSET + slot * SCALE_BYTES,
                                     dtype=cutlass.Uint32)
                a_provider = lane // 4 + 8 * (lane & 1)
                sa0 = sa[_scale_word(16 * wm + a_provider)]
                sa1 = sa[_scale_word(16 * wm + a_provider + 64)]
                sb0 = sb[_scale_word(16 * wn + lane // 4 + 0)]
                sb1 = sb[_scale_word(16 * wn + lane // 4 + 8)]
                sb2 = sb[_scale_word(16 * wn + lane // 4 + 32)]
                sb3 = sb[_scale_word(16 * wn + lane // 4 + 40)]
                sb4 = sb[_scale_word(16 * wn + lane // 4 + 64)]
                sb5 = sb[_scale_word(16 * wn + lane // 4 + 72)]
                sb6 = sb[_scale_word(16 * wn + lane // 4 + 96)]
                sb7 = sb[_scale_word(16 * wn + lane // 4 + 104)]
                half = cutlass.Int32(0)
                while half < 2:
                    a0 = prims.ldmatrix(a + _matrix_row(wm, half, lane),
                                       4, prims.MMALayout.ROW)
                    a1 = prims.ldmatrix(a + _matrix_row(wm + 4, half, lane),
                                       4, prims.MMALayout.ROW)
                    scale_a0 = (sa0 >> (16 * half)) & 65535
                    scale_a1 = (sa1 >> (16 * half)) & 65535
                    br = prims.ldmatrix(b + _matrix_row(wn + 0, half, lane),
                                       4, prims.MMALayout.ROW)
                    c00 = _mma(c00, a0, br[0], br[2], scale_a0,
                                 (sb0 >> (16 * half)) & 65535)
                    c01 = _mma(c01, a0, br[1], br[3], scale_a0,
                                 (sb1 >> (16 * half)) & 65535)
                    c10 = _mma(c10, a1, br[0], br[2], scale_a1,
                                 (sb0 >> (16 * half)) & 65535)
                    c11 = _mma(c11, a1, br[1], br[3], scale_a1,
                                 (sb1 >> (16 * half)) & 65535)
                    br = prims.ldmatrix(b + _matrix_row(wn + 2, half, lane),
                                       4, prims.MMALayout.ROW)
                    c02 = _mma(c02, a0, br[0], br[2], scale_a0,
                                 (sb2 >> (16 * half)) & 65535)
                    c03 = _mma(c03, a0, br[1], br[3], scale_a0,
                                 (sb3 >> (16 * half)) & 65535)
                    c12 = _mma(c12, a1, br[0], br[2], scale_a1,
                                 (sb2 >> (16 * half)) & 65535)
                    c13 = _mma(c13, a1, br[1], br[3], scale_a1,
                                 (sb3 >> (16 * half)) & 65535)
                    br = prims.ldmatrix(b + _matrix_row(wn + 4, half, lane),
                                       4, prims.MMALayout.ROW)
                    c04 = _mma(c04, a0, br[0], br[2], scale_a0,
                                 (sb4 >> (16 * half)) & 65535)
                    c05 = _mma(c05, a0, br[1], br[3], scale_a0,
                                 (sb5 >> (16 * half)) & 65535)
                    c14 = _mma(c14, a1, br[0], br[2], scale_a1,
                                 (sb4 >> (16 * half)) & 65535)
                    c15 = _mma(c15, a1, br[1], br[3], scale_a1,
                                 (sb5 >> (16 * half)) & 65535)
                    br = prims.ldmatrix(b + _matrix_row(wn + 6, half, lane),
                                       4, prims.MMALayout.ROW)
                    c06 = _mma(c06, a0, br[0], br[2], scale_a0,
                                 (sb6 >> (16 * half)) & 65535)
                    c07 = _mma(c07, a0, br[1], br[3], scale_a0,
                                 (sb7 >> (16 * half)) & 65535)
                    c16 = _mma(c16, a1, br[0], br[2], scale_a1,
                                 (sb6 >> (16 * half)) & 65535)
                    c17 = _mma(c17, a1, br[1], br[3], scale_a1,
                                 (sb7 >> (16 * half)) & 65535)
                    half += 1
                if lane == 0:
                    prims.mbarrier_arrive(empty + slot)
                slot += 1
                if slot == STAGES:
                    slot = cutlass.Int32(0)
                    phase ^= 1
                stage += 1
            if tid == 0:
                prims.cp_async_bulk_wait_group(0, read=True)
            prims.barrier_cta_sync(2, thread_count=256)
            output = cutlass.Pointer(base + OUTPUT, dtype=cutlass.Float16)
            _store(output, c00, warp, lane, 0, 0)
            _store(output, c01, warp, lane, 0, 1)
            _store(output, c02, warp, lane, 0, 2)
            _store(output, c03, warp, lane, 0, 3)
            _store(output, c04, warp, lane, 0, 4)
            _store(output, c05, warp, lane, 0, 5)
            _store(output, c06, warp, lane, 0, 6)
            _store(output, c07, warp, lane, 0, 7)
            _store(output, c10, warp, lane, 1, 0)
            _store(output, c11, warp, lane, 1, 1)
            _store(output, c12, warp, lane, 1, 2)
            _store(output, c13, warp, lane, 1, 3)
            _store(output, c14, warp, lane, 1, 4)
            _store(output, c15, warp, lane, 1, 5)
            _store(output, c16, warp, lane, 1, 6)
            _store(output, c17, warp, lane, 1, 7)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED,
                              space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(2, thread_count=256)
            if tid == 0:
                prims.cp_async_bulk_tensor_global_shared_cta(
                    c_map.get_ptr(), base + OUTPUT,
                    (tile_n * 128, tile_m * 128))
                prims.cp_async_bulk_tensor_global_shared_cta(
                    c_map.get_ptr(), base + OUTPUT + 16384,
                    (tile_n * 128 + 64, tile_m * 128))
                prims.cp_async_bulk_commit_group()
            tile += stride
        if tid == 0:
            prims.cp_async_bulk_wait_group(0, read=True)
    elif lane == 0:
        phase = cutlass.Int32(1)
        while tile < 256:
            tile_m = tile % 16
            tile_n = tile // 16
            stage = cutlass.Int32(0)
            while stage < 8:
                while not prims.mbarrier_try_wait_parity(empty + slot, phase):
                    pass
                prims.mbarrier_arrive_expect_tx(full + slot, TX_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    base + slot * DATA_BYTES, a_map.get_ptr(),
                    (stage * 64, tile_m * 128), full + slot)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    base + B_OFFSET + slot * DATA_BYTES, b_map.get_ptr(),
                    (stage * 64, tile_n * 128), full + slot)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    base + SFA_OFFSET + slot * SCALE_BYTES, sa_map.get_ptr(),
                    (0, tile_m * 8 + stage), full + slot)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    base + SFB_OFFSET + slot * SCALE_BYTES, sb_map.get_ptr(),
                    (0, tile_n * 8 + stage), full + slot)
                slot += 1
                if slot == STAGES:
                    slot = cutlass.Int32(0)
                    phase ^= 1
                stage += 1
            tile += stride
        tail = cutlass.Int32(0)
        while tail < STAGES:
            while not prims.mbarrier_try_wait_parity(empty + slot, phase):
                pass
            slot += 1
            if slot == STAGES:
                slot = cutlass.Int32(0)
                phase ^= 1
            tail += 1

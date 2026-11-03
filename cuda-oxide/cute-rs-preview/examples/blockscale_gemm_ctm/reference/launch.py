# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host descriptor creation and launch; no CuTe tensor/layout algebra."""
import cuda.bindings.driver as driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda

from kernel import blockscale_gemm


@cute.jit
def host(a: cutlass.Int64, b: cutlass.Int64, sa: cutlass.Int64,
         sb: cutlass.Int64, c: cutlass.Int64, ctas: cutlass.Int32,
         stream: driver.CUstream):
    # The experimental CUDA API expresses strides in 16-byte units.
    a_map = cuda.create_tensor_map_tiled(
        a, cutlass.Uint8, [512, 2048], [32], [64, 128],
        swizzle=cuda.TensorMapSwizzle.s64b,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b)
    b_map = cuda.create_tensor_map_tiled(
        b, cutlass.Uint8, [512, 2048], [32], [64, 128],
        swizzle=cuda.TensorMapSwizzle.s64b,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b)
    sa_map = cuda.create_tensor_map_tiled(
        sa, cutlass.Uint16, [256, 128], [32], [256, 1],
        swizzle=cuda.TensorMapSwizzle.none,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b)
    sb_map = cuda.create_tensor_map_tiled(
        sb, cutlass.Uint16, [256, 128], [32], [256, 1],
        swizzle=cuda.TensorMapSwizzle.none,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b)
    c_map = cuda.create_tensor_map_tiled(
        c, cutlass.Float16, [2048, 2048], [256], [64, 128],
        swizzle=cuda.TensorMapSwizzle.s128b,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b)
    blockscale_gemm(a_map, b_map, sa_map, sb_map, c_map).launch(
        grid=(ctas, 1, 1), block=(288, 1, 1), smem=86016, stream=stream)

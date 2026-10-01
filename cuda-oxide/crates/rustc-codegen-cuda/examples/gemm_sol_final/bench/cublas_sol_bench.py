#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy cublasGemmEx comparison; not the optimization baseline.

Uses ctypes to retain the older API measurements for historical context.
Use cublaslt_bench for the live optimization reference.

Tests multiple configurations:
- FP16 vs BF16 data types
- FP16 vs FP32 accumulation
- Multiple problem sizes (4K, 8K, 16K)
"""

import ctypes
import numpy as np


def main():
    print("Loading CUDA runtime...", flush=True)
    cudart = ctypes.CDLL("libcudart.so")
    cublas = ctypes.CDLL("libcublas.so")

    # Set up function signatures
    setup_signatures(cudart, cublas)

    # GPU info
    val = ctypes.c_int()
    cudart.cudaDeviceGetAttribute(ctypes.byref(val), 16, 0)
    sm_count = val.value
    cudart.cudaDeviceGetAttribute(ctypes.byref(val), 75, 0)
    major = val.value
    cudart.cudaDeviceGetAttribute(ctypes.byref(val), 76, 0)
    minor = val.value
    print(f"GPU: sm_{major}{minor} with {sm_count} SMs\n", flush=True)

    # ─── Test configurations ───
    configs = [
        # (label, dtype_np, cublas_dtype, compute_type, alpha_dtype)
        ("FP16 in, FP16 compute",  np.float16, 2,  64, np.float16),   # CUDA_R_16F, COMPUTE_16F
        ("FP16 in, FP32 compute",  np.float16, 2,  68, np.float32),   # CUDA_R_16F, COMPUTE_32F
        ("BF16 in, FP32 compute",  None,       14, 68, np.float32),   # CUDA_R_16BF, COMPUTE_32F
    ]

    sizes = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 16384),
    ]

    all_results = []

    for label, dtype_np, cublas_dtype, compute_type, alpha_dtype in configs:
        print(f"{'='*60}")
        print(f"  Config: {label}")
        print(f"{'='*60}")

        for M, N, K in sizes:
            tflops, avg_ms = bench_gemm(
                cudart, cublas, M, N, K,
                dtype_np, cublas_dtype, compute_type, alpha_dtype,
                warmup=10, iters=100,
            )
            all_results.append((label, M, N, K, avg_ms, tflops))
        print()

    # ─── Summary ───
    print(f"\n{'='*75}")
    print(f"  SUMMARY — legacy cublasGemmEx (sm_{major}{minor}, {sm_count} SMs)")
    print(f"{'='*75}")
    print(f"  {'Config':<28s}  {'Size':>17s}  {'ms':>8s}  {'TFLOPS':>8s}")
    print(f"  {'-'*28}  {'-'*17}  {'-'*8}  {'-'*8}")
    for label, M, N, K, avg_ms, tflops in all_results:
        print(f"  {label:<28s}  {f'{M}x{N}x{K}':>17s}  {avg_ms:>8.4f}  {tflops:>8.1f}")
    print(f"{'='*75}")


def setup_signatures(cudart, cublas):
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemset.restype = ctypes.c_int
    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamCreate.restype = ctypes.c_int
    cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamSynchronize.restype = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventCreate.restype = ctypes.c_int
    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventRecord.restype = ctypes.c_int
    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int
    cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int
    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]

    cublas.cublasCreate_v2.restype = ctypes.c_int
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas.cublasDestroy_v2.restype = ctypes.c_int
    cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublas.cublasSetStream_v2.restype = ctypes.c_int
    cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cublas.cublasSetMathMode.restype = ctypes.c_int
    cublas.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublas.cublasGemmEx.restype = ctypes.c_int
    cublas.cublasGemmEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]


def make_bf16_array(shape):
    """Create a BF16 array (stored as uint16) from random FP32 data."""
    fp32 = np.random.randn(*shape).astype(np.float32)
    # Convert FP32 to BF16 with round-to-nearest-even.
    bits = fp32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    bf16_bits = ((bits + bias) >> 16).astype(np.uint16)
    return bf16_bits


def bench_gemm(cudart, cublas, M, N, K, dtype_np, cublas_dtype, compute_type, alpha_dtype,
               warmup=10, iters=100):
    """Run a single GEMM benchmark configuration."""

    # Allocate host data
    elem_bytes = 2  # FP16 and BF16 are both 2 bytes
    if dtype_np is not None:
        a_host = np.random.randn(M, K).astype(dtype_np)
        b_host = np.random.randn(K, N).astype(dtype_np)
    else:
        # BF16 path
        a_host = make_bf16_array((M, K))
        b_host = make_bf16_array((K, N))

    nbytes_a = M * K * elem_bytes
    nbytes_b = K * N * elem_bytes
    nbytes_c = M * N * elem_bytes

    # Device alloc
    d_a, d_b, d_c = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(d_a), nbytes_a)
    cudart.cudaMalloc(ctypes.byref(d_b), nbytes_b)
    cudart.cudaMalloc(ctypes.byref(d_c), nbytes_c)
    cudart.cudaMemcpy(d_a, a_host.ctypes.data, nbytes_a, 1)
    cudart.cudaMemcpy(d_b, b_host.ctypes.data, nbytes_b, 1)
    cudart.cudaMemset(d_c, 0, nbytes_c)

    # Stream + handle
    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))
    handle = ctypes.c_void_p()
    cublas.cublasCreate_v2(ctypes.byref(handle))
    cublas.cublasSetStream_v2(handle, stream)

    # Enable tensor op math mode
    CUBLAS_TENSOR_OP_MATH = 1
    cublas.cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)

    alpha = np.array([1.0], dtype=alpha_dtype)
    beta = np.array([0.0], dtype=alpha_dtype)

    OP_N = 0
    GEMM_DEFAULT_TENSOR_OP = 99  # CUBLAS_GEMM_DEFAULT_TENSOR_OP

    def run_gemm():
        return cublas.cublasGemmEx(
            handle, OP_N, OP_N,
            N, M, K,
            alpha.ctypes.data,
            d_b, cublas_dtype, N,
            d_a, cublas_dtype, K,
            beta.ctypes.data,
            d_c, cublas_dtype, N,
            compute_type, GEMM_DEFAULT_TENSOR_OP,
        )

    # Events
    start_ev, end_ev = ctypes.c_void_p(), ctypes.c_void_p()
    cudart.cudaEventCreate(ctypes.byref(start_ev))
    cudart.cudaEventCreate(ctypes.byref(end_ev))

    # Warmup
    for _ in range(warmup):
        status = run_gemm()
        if status != 0:
            print(f"    WARNING: cublasGemmEx returned {status}")
            break
    cudart.cudaStreamSynchronize(stream)

    # Timed
    cudart.cudaEventRecord(start_ev, stream)
    for _ in range(iters):
        run_gemm()
    cudart.cudaEventRecord(end_ev, stream)
    cudart.cudaStreamSynchronize(stream)

    elapsed_ms = ctypes.c_float()
    cudart.cudaEventElapsedTime(ctypes.byref(elapsed_ms), start_ev, end_ev)
    elapsed_ms = elapsed_ms.value

    avg_ms = elapsed_ms / iters
    flops = 2.0 * M * N * K
    tflops = (flops / (avg_ms / 1000.0)) / 1e12

    print(f"  {M:5d}x{N:5d}x{K:5d}  {avg_ms:8.4f} ms  {tflops:8.1f} TFLOPS", flush=True)

    # Cleanup
    cudart.cudaEventDestroy(start_ev)
    cudart.cudaEventDestroy(end_ev)
    cublas.cublasDestroy_v2(handle)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(d_a)
    cudart.cudaFree(d_b)
    cudart.cudaFree(d_c)

    return tflops, avg_ms


if __name__ == "__main__":
    main()

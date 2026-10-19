#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuBLAS + numpy correctness test — matching gemm_sol exactly.

gemm_sol pipeline: FP16 inputs → FP32 accumulation → BF16 output.
cuBLAS pipeline:   FP16 inputs → FP32 compute → FP32 output → truncate to BF16 on host.
(cuBLAS doesn't support FP16→BF16 directly, so we output FP32 and convert.)

Tests:
1. A=1, B=1 → C[i,j] = K
2. A=2, B=3 → C[i,j] = 6K
3. A[i,k]=(i%8+1), B[n,k]=(n%8+1) → C[i,j]=(i%8+1)*(j%8+1)*K
"""

import ctypes
import numpy as np
import struct


def bf16_to_f32(val_u16):
    """Convert a uint16 (bf16 bits) to float32."""
    # BF16 is just the upper 16 bits of FP32
    fp32_bits = int(val_u16) << 16
    return struct.unpack('f', struct.pack('I', fp32_bits))[0]


def f32_to_bf16(val_f32):
    """Convert float32 to bf16 (truncate lower 16 mantissa bits)."""
    fp32_bits = struct.unpack('I', struct.pack('f', val_f32))[0]
    return (fp32_bits >> 16) & 0xFFFF


def fp16_array_to_bf16(arr_fp16):
    """Convert an FP16 numpy array to BF16 (as uint16), preserving values."""
    fp32 = arr_fp16.astype(np.float32)
    bf16_bits = (fp32.view(np.uint32) >> 16).astype(np.uint16)
    return bf16_bits


def main():
    M, N, K = 4096, 4096, 4096

    print("=" * 70)
    print("  Correctness Test: numpy + cuBLAS — matching gemm_sol pipeline")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  GEMM: C(M×N) = A(M×K) @ B_stored(N×K)^T")
    print(f"  Pipeline: FP16 in → FP32 accum → BF16 out (truncate)")
    print(f"  cuBLAS:   FP16 in → FP32 compute → FP32 out → truncate to BF16")
    print("=" * 70)

    # ── Test 1: Uniform A=1, B=1 ──
    print("\n── Test 1: A=1.0, B=1.0 ──")
    a = np.ones((M, K), dtype=np.float16)
    b = np.ones((N, K), dtype=np.float16)
    run_numpy_test(a, b, M, N, K, "A=1, B=1")

    # ── Test 2: Uniform A=2, B=3 ──
    print("\n── Test 2: A=2.0, B=3.0 ──")
    a = np.full((M, K), 2.0, dtype=np.float16)
    b = np.full((N, K), 3.0, dtype=np.float16)
    run_numpy_test(a, b, M, N, K, "A=2, B=3")

    # ── Test 3: Non-uniform ──
    print("\n── Test 3: A[i,k]=(i%8+1), B[n,k]=(n%8+1) ──")
    a = np.zeros((M, K), dtype=np.float16)
    b = np.zeros((N, K), dtype=np.float16)
    for i in range(M):
        a[i, :] = float(i % 8 + 1)
    for n in range(N):
        b[n, :] = float(n % 8 + 1)
    run_numpy_test(a, b, M, N, K, "non-uniform")

    # ── cuBLAS comparison ──
    print("\n" + "=" * 70)
    print("  cuBLAS: FP16 in → FP32 out → truncate to BF16")
    print("=" * 70)

    cudart = ctypes.CDLL("libcudart.so")
    cublas = ctypes.CDLL("libcublas.so")
    setup_signatures(cudart, cublas)

    for test_name, a_fn, b_fn in [
        ("A=1, B=1", lambda: np.ones((M, K), dtype=np.float16),
                      lambda: np.ones((N, K), dtype=np.float16)),
        ("A=2, B=3", lambda: np.full((M, K), 2.0, dtype=np.float16),
                      lambda: np.full((N, K), 3.0, dtype=np.float16)),
        ("non-uniform", lambda: make_nonuniform_a(M, K),
                         lambda: make_nonuniform_b(N, K)),
    ]:
        print(f"\n── cuBLAS: {test_name} ──")
        a = a_fn()
        b = b_fn()
        run_cublas_test(cudart, cublas, a, b, M, N, K, test_name)


def make_nonuniform_a(M, K):
    a = np.zeros((M, K), dtype=np.float16)
    for i in range(M):
        a[i, :] = float(i % 8 + 1)
    return a


def make_nonuniform_b(N, K):
    b = np.zeros((N, K), dtype=np.float16)
    for n in range(N):
        b[n, :] = float(n % 8 + 1)
    return b


def run_numpy_test(a, b, M, N, K, label):
    """Compute C = A @ B^T in f32, convert to bf16, verify."""
    c_f32 = np.dot(a.astype(np.float32), b.astype(np.float32).T)

    # Convert to bf16 (truncate mantissa, same as hardware)
    c_bf16_bits = (c_f32.view(np.uint32) >> 16).astype(np.uint16)

    check_positions = [
        (0, 0), (0, 1), (0, 7), (1, 0), (3, 5),
        (7, 7), (127, 127), (128, 128), (131, 259),
        (M-1, N-1), (M//2, N//2),
    ]

    all_ok = True
    for (row, col) in check_positions:
        if row >= M or col >= N:
            continue
        got_f32 = c_f32[row, col]
        got_bf16 = bf16_to_f32(c_bf16_bits[row, col])
        manual_exp = float(a[row, 0]) * float(b[col, 0]) * K
        tol = abs(manual_exp) * 0.02 + 1.0
        ok = abs(got_bf16 - manual_exp) < tol
        print(f"  C[{row:4d},{col:4d}] f32={got_f32:12.0f}  bf16={got_bf16:12.0f}  (expected {manual_exp:12.0f})  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"  → numpy {label}: ALL MATCH")
    else:
        print(f"  → numpy {label}: SOME MISMATCH")


def run_cublas_test(cudart, cublas_lib, a_host, b_host, M, N, K, label):
    """Run C = A(M×K) @ B_stored(N×K)^T using cuBLAS.

    Matches gemm_sol pipeline exactly: FP16 in → FP32 compute → convert to BF16.
    cuBLAS doesn't support FP16→BF16 directly, so we use FP16→FP32 output,
    then truncate to BF16 on host (same truncation the kernel does via cvt).
    """
    elem_bytes_in = 2   # FP16 = 2 bytes
    elem_bytes_out = 4  # FP32 = 4 bytes
    nbytes_a = M * K * elem_bytes_in
    nbytes_b = N * K * elem_bytes_in
    nbytes_c = M * N * elem_bytes_out

    d_a, d_b, d_c = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    cudart.cudaMalloc(ctypes.byref(d_a), nbytes_a)
    cudart.cudaMalloc(ctypes.byref(d_b), nbytes_b)
    cudart.cudaMalloc(ctypes.byref(d_c), nbytes_c)

    # Upload FP16 inputs directly (no conversion — matches our kernel)
    cudart.cudaMemcpy(d_a, a_host.ctypes.data, nbytes_a, 1)
    cudart.cudaMemcpy(d_b, b_host.ctypes.data, nbytes_b, 1)
    cudart.cudaMemset(d_c, 0, nbytes_c)

    stream = ctypes.c_void_p()
    cudart.cudaStreamCreate(ctypes.byref(stream))
    handle = ctypes.c_void_p()
    cublas_lib.cublasCreate_v2(ctypes.byref(handle))
    cublas_lib.cublasSetStream_v2(handle, stream)
    cublas_lib.cublasSetMathMode(handle, 1)

    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([0.0], dtype=np.float32)

    OP_T = 1
    OP_N = 0
    CUDA_R_16F = 2      # FP16 input
    CUDA_R_32F = 0       # FP32 output
    COMPUTE_32F = 68
    GEMM_DEFAULT_TENSOR_OP = 99

    # cuBLAS: FP16 in → FP32 out (with FP32 compute)
    # Row-major C(M×N) = A(M×K) @ B_stored(N×K)^T
    # Column-major: C^T(N×M) = B_stored(N×K) @ A^T(K×M)
    status = cublas_lib.cublasGemmEx(
        handle, OP_T, OP_N,
        N, M, K,
        alpha.ctypes.data,
        d_b, CUDA_R_16F, K,     # FP16 input (B_stored)
        d_a, CUDA_R_16F, K,     # FP16 input (A)
        beta.ctypes.data,
        d_c, CUDA_R_32F, N,     # FP32 output
        COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP,
    )
    cudart.cudaStreamSynchronize(stream)

    if status != 0:
        print(f"  cublasGemmEx FAILED with status {status}")
        cublas_lib.cublasDestroy_v2(handle)
        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(d_a)
        cudart.cudaFree(d_b)
        cudart.cudaFree(d_c)
        return

    # Read back FP32 output
    c_f32 = np.zeros(M * N, dtype=np.float32)
    cudart.cudaMemcpy(c_f32.ctypes.data, d_c, nbytes_c, 2)
    c_f32 = c_f32.reshape((M, N))

    # Convert F32 → BF16 on host (truncate lower 16 mantissa bits)
    # This matches what our kernel does: cvt_bf16x2_f32
    c_bf16_bits = (c_f32.view(np.uint32) >> 16).astype(np.uint16)

    # Numpy reference in f32
    c_ref = np.dot(a_host.astype(np.float32), b_host.astype(np.float32).T)

    check_positions = [
        (0, 0), (0, 1), (0, 7), (1, 0), (3, 5),
        (7, 7), (127, 127), (128, 128), (131, 259),
        (M-1, N-1), (M//2, N//2),
    ]

    all_ok = True
    for (row, col) in check_positions:
        if row >= M or col >= N:
            continue
        got_f32 = c_f32[row, col]
        got_bf16 = bf16_to_f32(c_bf16_bits[row, col])
        ref = float(c_ref[row, col])
        tol = abs(ref) * 0.02 + 1.0
        ok = abs(got_bf16 - ref) < tol
        print(f"  C[{row:4d},{col:4d}] f32={got_f32:12.0f}  bf16={got_bf16:12.0f}  (ref: {ref:12.0f})  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"  → cuBLAS {label}: ALL PASSED")
    else:
        print(f"  → cuBLAS {label}: SOME FAILED")

    cublas_lib.cublasDestroy_v2(handle)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(d_a)
    cudart.cudaFree(d_b)
    cudart.cudaFree(d_c)


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


if __name__ == "__main__":
    main()

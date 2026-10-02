/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pure CUDA C++ Test Kernels for extern function testing
 *
 * This file contains test kernels that call extern functions from a separate
 * LTOIR file. This helps isolate whether extern function local memory issues
 * are Rust codegen specific or a general nvJitLink/LTOIR limitation.
 */

// Forward declaration of the extern device function (defined in cufftdx_wrappers_funcs.cu)
extern "C" __device__ void debug_extern_double_array(float* data, int n);

// ============================================================================
// Test Kernel: Local memory -> extern function
//
// This kernel tests if a CUDA C++ kernel can pass local memory pointers to 
// extern functions defined in a separate LTOIR file.
// ============================================================================

extern "C" __global__ void cuda_test_local_extern(float* input, float* output) {
    int tid = threadIdx.x;
    int offset = tid * 16;
    
    // Local storage (same as Rust test)
    float local_data[16];
    
    // Load from global memory
    for (int i = 0; i < 16; i++) {
        local_data[i] = input[offset + i];
    }
    
    // Call extern function to double the values
    debug_extern_double_array(local_data, 16);
    
    // Store to global memory
    for (int i = 0; i < 16; i++) {
        output[offset + i] = local_data[i];
    }
}

// ============================================================================
// Test Kernel: Global memory -> extern function (control test)
//
// This should definitely work - just confirms extern linking works at all.
// ============================================================================

extern "C" __global__ void cuda_test_global_extern(float* data) {
    int tid = threadIdx.x;
    int offset = tid * 16;
    
    // Call extern function directly on global memory
    debug_extern_double_array(data + offset, 16);
}

// ============================================================================
// GEMM Test Kernels
//
// These test the cuBLASDx GEMM wrappers from pure CUDA C++ to isolate
// whether GEMM issues are cuda-oxide specific or cuBLASDx wrapper issues.
// ============================================================================

// Forward declarations for cuBLASDx wrappers (defined in cublasdx_wrappers.cu)
extern "C" __device__ void cublasdx_gemm_32x32x32_f32(
    const float* a, const float* b, float* c, char* smem);
extern "C" __device__ void cublasdx_gemm_32x32x32_f32_alphabeta(
    float alpha, const float* a, const float* b, float beta, float* c, char* smem);
extern "C" __device__ int cublasdx_gemm_32x32x32_smem_size();
extern "C" __device__ int cublasdx_gemm_32x32x32_smem_size_ab();

/**
 * Test kernel: 32x32x32 GEMM C = A * B
 *
 * Launch with: grid=(1,1,1), block=(256,1,1), shared_mem>=12KB
 */
extern "C" __global__ void cuda_test_gemm_32x32x32(
    const float* a, const float* b, float* c
) {
    // Dynamic shared memory - must be declared extern with align
    extern __shared__ __align__(128) char smem[];
    
    // All 256 threads call GEMM together
    cublasdx_gemm_32x32x32_f32(a, b, c, smem);
}

/**
 * Test kernel: 32x32x32 GEMM with alpha/beta: C = alpha*A*B + beta*C
 *
 * Launch with: grid=(1,1,1), block=(256,1,1), shared_mem>=16KB
 */
extern "C" __global__ void cuda_test_gemm_32x32x32_alphabeta(
    float alpha, const float* a, const float* b, float beta, float* c
) {
    extern __shared__ __align__(128) char smem[];
    
    cublasdx_gemm_32x32x32_f32_alphabeta(alpha, a, b, beta, c, smem);
}

/**
 * Query kernel: Get GEMM shared memory requirements
 * Useful for debugging - writes smem sizes to output array
 */
extern "C" __global__ void cuda_test_gemm_query(int* out) {
    if (threadIdx.x == 0) {
        out[0] = cublasdx_gemm_32x32x32_smem_size();     // Full smem (A,B,C)
        out[1] = cublasdx_gemm_32x32x32_smem_size_ab(); // Partial smem (A,B only)
    }
}

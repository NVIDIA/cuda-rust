/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Host-side test for separate LTOIR compilation
 * 
 * Compile and run:
 *   nvcc -arch=sm_120 test_separate.cu -o test_separate -lcuda
 *   ./test_separate
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== Separate LTOIR Test ===\n\n");
    
    // Initialize CUDA Driver API
    CHECK_CUDA(cuInit(0));
    
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    
    char name[256];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), device));
    printf("Device: %s\n\n", name);
    
    CUcontext ctx;
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&ctx, device));
    CHECK_CUDA(cuCtxSetCurrent(ctx));
    
    // Load the cubin
    CUmodule module;
    CUresult loadResult = cuModuleLoad(&module, "test_separate.cubin");
    if (loadResult != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(loadResult, &errStr);
        fprintf(stderr, "Failed to load cubin: %s\n", errStr);
        return 1;
    }
    printf("Loaded test_separate.cubin\n\n");
    
    // Test parameters
    const int num_threads = 4;
    const int floats_per_thread = 16;
    const int total_floats = num_threads * floats_per_thread;
    const size_t size = total_floats * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    
    // Initialize input: 0, 1, 2, ..., 15, 0, 1, 2, ..., 15, ...
    for (int i = 0; i < total_floats; i++) {
        h_input[i] = (float)(i % floats_per_thread);
    }
    
    // Allocate device memory
    CUdeviceptr d_input, d_output;
    CHECK_CUDA(cuMemAlloc(&d_input, size));
    CHECK_CUDA(cuMemAlloc(&d_output, size));
    
    // =========================================================================
    // Test 1: cuda_test_global_extern (global memory -> extern function)
    // =========================================================================
    printf("--- Test 1: cuda_test_global_extern ---\n");
    printf("    Testing extern function with GLOBAL memory pointer\n");
    
    // Copy input to device
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, size));
    
    CUfunction globalKernel;
    CUresult funcResult = cuModuleGetFunction(&globalKernel, module, "cuda_test_global_extern");
    if (funcResult != CUDA_SUCCESS) {
        printf("    SKIPPED (kernel not found)\n\n");
    } else {
        void* args[] = { &d_input };
        CHECK_CUDA(cuLaunchKernel(globalKernel,
            1, 1, 1,              // grid dim
            num_threads, 1, 1,    // block dim
            0, 0,                 // shared mem, stream
            args, NULL));
        CHECK_CUDA(cuCtxSynchronize());
        
        // Copy result back
        CHECK_CUDA(cuMemcpyDtoH(h_output, d_input, size));
        
        // Verify: output should be input * 2
        float max_error = 0.0f;
        for (int i = 0; i < total_floats; i++) {
            float expected = h_input[i] * 2.0f;
            float error = fabsf(h_output[i] - expected);
            if (error > max_error) max_error = error;
        }
        
        if (max_error < 1e-6f) {
            printf("    PASSED (max_error=%.6f)\n\n", max_error);
        } else {
            printf("    FAILED (max_error=%.6f)\n", max_error);
            printf("    First 4 values: input=[%.1f,%.1f,%.1f,%.1f], output=[%.1f,%.1f,%.1f,%.1f]\n\n",
                   h_input[0], h_input[1], h_input[2], h_input[3],
                   h_output[0], h_output[1], h_output[2], h_output[3]);
        }
    }
    
    // =========================================================================
    // Test 2: cuda_test_local_extern (local memory -> extern function)
    // =========================================================================
    printf("--- Test 2: cuda_test_local_extern ---\n");
    printf("    Testing extern function with LOCAL memory pointer\n");
    
    // Reset input on device
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, size));
    memset(h_output, 0, size);
    CHECK_CUDA(cuMemcpyHtoD(d_output, h_output, size));
    
    CUfunction localKernel;
    funcResult = cuModuleGetFunction(&localKernel, module, "cuda_test_local_extern");
    if (funcResult != CUDA_SUCCESS) {
        printf("    SKIPPED (kernel not found)\n\n");
    } else {
        void* args[] = { &d_input, &d_output };
        CHECK_CUDA(cuLaunchKernel(localKernel,
            1, 1, 1,              // grid dim
            num_threads, 1, 1,    // block dim
            0, 0,                 // shared mem, stream
            args, NULL));
        CHECK_CUDA(cuCtxSynchronize());
        
        // Copy result back
        CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, size));
        
        // Verify: output should be input * 2
        float max_error = 0.0f;
        for (int i = 0; i < total_floats; i++) {
            float expected = h_input[i] * 2.0f;
            float error = fabsf(h_output[i] - expected);
            if (error > max_error) max_error = error;
        }
        
        if (max_error < 1e-6f) {
            printf("    PASSED (max_error=%.6f)\n", max_error);
            printf("    >>> CUDA C++ local->extern WORKS with separate LTOIR!\n\n");
        } else {
            printf("    FAILED (max_error=%.6f)\n", max_error);
            printf("    First 4 values: input=[%.1f,%.1f,%.1f,%.1f], output=[%.1f,%.1f,%.1f,%.1f]\n",
                   h_input[0], h_input[1], h_input[2], h_input[3],
                   h_output[0], h_output[1], h_output[2], h_output[3]);
            printf("    >>> CUDA C++ also fails! Issue is nvJitLink/LTOIR-level.\n\n");
        }
    }
    
    // =========================================================================
    // Test 3: GEMM 32x32x32 (C = A * B)
    // =========================================================================
    printf("--- Test 3: cuda_test_gemm_32x32x32 ---\n");
    printf("    Testing cuBLASDx GEMM: C = A * B (identity matrices)\n");
    
    {
        const int m = 32, n = 32, k = 32;
        const size_t matrix_size = m * k * sizeof(float);
        
        // Allocate host matrices
        float* h_a = (float*)malloc(matrix_size);
        float* h_b = (float*)malloc(matrix_size);
        float* h_c = (float*)malloc(matrix_size);
        
        // Initialize A = identity (row-major)
        memset(h_a, 0, matrix_size);
        for (int i = 0; i < m && i < k; i++) {
            h_a[i * k + i] = 1.0f;
        }
        
        // Initialize B = identity (col-major: B[i,j] at j*k + i)
        memset(h_b, 0, matrix_size);
        for (int i = 0; i < k && i < n; i++) {
            h_b[i * k + i] = 1.0f;
        }
        
        // C = zeros
        memset(h_c, 0, matrix_size);
        
        // Allocate device memory
        CUdeviceptr d_a, d_b, d_c;
        CHECK_CUDA(cuMemAlloc(&d_a, matrix_size));
        CHECK_CUDA(cuMemAlloc(&d_b, matrix_size));
        CHECK_CUDA(cuMemAlloc(&d_c, matrix_size));
        
        CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, matrix_size));
        CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, matrix_size));
        CHECK_CUDA(cuMemcpyHtoD(d_c, h_c, matrix_size));
        
        CUfunction gemmKernel;
        funcResult = cuModuleGetFunction(&gemmKernel, module, "cuda_test_gemm_32x32x32");
        if (funcResult != CUDA_SUCCESS) {
            printf("    SKIPPED (kernel not found)\n\n");
        } else {
            void* args[] = { &d_a, &d_b, &d_c };
            unsigned int smem_bytes = 16 * 1024;  // 16KB should be enough
            
            CHECK_CUDA(cuLaunchKernel(gemmKernel,
                1, 1, 1,              // grid dim
                256, 1, 1,            // block dim (cuBLASDx requires 256)
                smem_bytes, 0,        // shared mem, stream
                args, NULL));
            CHECK_CUDA(cuCtxSynchronize());
            
            // Copy result back
            CHECK_CUDA(cuMemcpyDtoH(h_c, d_c, matrix_size));
            
            // Verify: C = I * I = I (identity)
            float max_error = 0.0f;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float expected = (i == j) ? 1.0f : 0.0f;
                    float error = fabsf(h_c[i * n + j] - expected);
                    if (error > max_error) max_error = error;
                }
            }
            
            if (max_error < 1e-4f) {
                printf("    PASSED (max_error=%.6f)\n", max_error);
                printf("    >>> cuBLASDx GEMM works from CUDA C++!\n\n");
            } else {
                printf("    FAILED (max_error=%.6f)\n", max_error);
                printf("    C[0,0]=%.1f, C[1,1]=%.1f, C[0,1]=%.1f\n",
                       h_c[0], h_c[1*n+1], h_c[0*n+1]);
                printf("    >>> cuBLASDx GEMM fails from CUDA C++ too!\n\n");
            }
        }
        
        cuMemFree(d_a);
        cuMemFree(d_b);
        cuMemFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
    }
    
    // =========================================================================
    // Test 4: GEMM 32x32x32 with alpha/beta (C = alpha*A*B + beta*C)
    // =========================================================================
    printf("--- Test 4: cuda_test_gemm_32x32x32_alphabeta ---\n");
    printf("    Testing cuBLASDx GEMM: C = 2*A*B + 1*C\n");
    
    {
        const int m = 32, n = 32, k = 32;
        const size_t matrix_size = m * k * sizeof(float);
        
        float* h_a = (float*)malloc(matrix_size);
        float* h_b = (float*)malloc(matrix_size);
        float* h_c = (float*)malloc(matrix_size);
        
        // A = identity, B = identity, C = identity
        memset(h_a, 0, matrix_size);
        memset(h_b, 0, matrix_size);
        memset(h_c, 0, matrix_size);
        for (int i = 0; i < 32; i++) {
            h_a[i * k + i] = 1.0f;
            h_b[i * k + i] = 1.0f;
            h_c[i * n + i] = 1.0f;
        }
        
        CUdeviceptr d_a, d_b, d_c;
        CHECK_CUDA(cuMemAlloc(&d_a, matrix_size));
        CHECK_CUDA(cuMemAlloc(&d_b, matrix_size));
        CHECK_CUDA(cuMemAlloc(&d_c, matrix_size));
        
        CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, matrix_size));
        CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, matrix_size));
        CHECK_CUDA(cuMemcpyHtoD(d_c, h_c, matrix_size));
        
        CUfunction gemmKernel;
        funcResult = cuModuleGetFunction(&gemmKernel, module, "cuda_test_gemm_32x32x32_alphabeta");
        if (funcResult != CUDA_SUCCESS) {
            printf("    SKIPPED (kernel not found)\n\n");
        } else {
            float alpha = 2.0f;
            float beta = 1.0f;
            void* args[] = { &alpha, &d_a, &d_b, &beta, &d_c };
            unsigned int smem_bytes = 16 * 1024;
            
            CHECK_CUDA(cuLaunchKernel(gemmKernel,
                1, 1, 1,
                256, 1, 1,
                smem_bytes, 0,
                args, NULL));
            CHECK_CUDA(cuCtxSynchronize());
            
            CHECK_CUDA(cuMemcpyDtoH(h_c, d_c, matrix_size));
            
            // Expected: C = 2*I*I + 1*I = 2*I + I = 3*I
            float max_error = 0.0f;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float expected = (i == j) ? 3.0f : 0.0f;
                    float error = fabsf(h_c[i * n + j] - expected);
                    if (error > max_error) max_error = error;
                }
            }
            
            if (max_error < 1e-4f) {
                printf("    PASSED (max_error=%.6f)\n", max_error);
                printf("    >>> cuBLASDx GEMM alpha/beta works from CUDA C++!\n\n");
            } else {
                printf("    FAILED (max_error=%.6f)\n", max_error);
                printf("    C[0,0]=%.1f (expected 3.0)\n", h_c[0]);
                printf("    >>> cuBLASDx GEMM alpha/beta fails from CUDA C++ too!\n\n");
            }
        }
        
        cuMemFree(d_a);
        cuMemFree(d_b);
        cuMemFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
    }
    
    // Cleanup
    cuMemFree(d_input);
    cuMemFree(d_output);
    free(h_input);
    free(h_output);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);
    
    printf("=== Test Complete ===\n");
    return 0;
}

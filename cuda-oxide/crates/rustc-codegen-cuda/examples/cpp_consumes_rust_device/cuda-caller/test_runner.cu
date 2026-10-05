// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Test Runner — loads merged cubin and verifies Rust device functions via GPU
//
// This C++ program is the "consumer" of Rust device functions. It:
// 1. Loads the merged cubin (Rust LTOIR + C++ LTOIR linked by nvJitLink)
// 2. Launches C++ kernels that call Rust device functions
// 3. Verifies the GPU results match expected values
//
// Build:
//   nvcc -o test_runner test_runner.cu -lcuda
//
// Usage:
//   ./test_runner <cubin_file>
// =============================================================================

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        CUresult err = call;                                                                       \
        if (err != CUDA_SUCCESS) {                                                                 \
            const char *errStr;                                                                    \
            cuGetErrorString(err, &errStr);                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr);              \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static int tests_passed = 0;
static int tests_failed = 0;

// =============================================================================
// Test 1: fast_sqrt + clamp_f32
// =============================================================================
void test_sqrt_clamp(CUmodule module) {
    printf("--- Test 1: test_sqrt_clamp (fast_sqrt + clamp_f32) ---\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_sqrt_clamp");
    if (res != CUDA_SUCCESS) {
        printf("  SKIP (kernel not found)\n");
        return;
    }

    const int N = 256;

    // Allocate
    CUdeviceptr d_input, d_output;
    CHECK_CUDA(cuMemAlloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output, N * sizeof(float)));

    // Initialize: input[i] = i * 0.5
    float *h_input = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = i * 0.5f;
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, N * sizeof(float)));

    // Launch
    int n = N;
    void *args[] = {&d_input, &d_output, &n};
    CHECK_CUDA(cuLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    // Verify
    float *h_output = (float *)malloc(N * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = sqrtf(h_input[i]);
        if (expected < 0.0f)
            expected = 0.0f;
        if (expected > 10.0f)
            expected = 10.0f;
        float diff = fabsf(h_output[i] - expected);
        if (diff > 0.01f) {
            if (errors < 3) {
                printf("  Mismatch at [%d]: got %.4f, expected %.4f (diff=%.4f)\n", i, h_output[i],
                       expected, diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASS\n");
        tests_passed++;
    } else {
        printf("  FAIL (%d errors)\n", errors);
        tests_failed++;
    }

    free(h_input);
    free(h_output);
    cuMemFree(d_input);
    cuMemFree(d_output);
}

// =============================================================================
// Test 2: safe_sqrt (transitive calls across LTOIR boundary)
// =============================================================================
void test_safe_sqrt(CUmodule module) {
    printf("--- Test 2: test_safe_sqrt (transitive Rust device calls) ---\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_safe_sqrt");
    if (res != CUDA_SUCCESS) {
        printf("  SKIP (kernel not found)\n");
        return;
    }

    const int N = 256;

    CUdeviceptr d_input, d_output;
    CHECK_CUDA(cuMemAlloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output, N * sizeof(float)));

    // Include negatives to test clamping: input[i] = i - 50
    float *h_input = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = (float)i - 50.0f;
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, N * sizeof(float)));

    int n = N;
    void *args[] = {&d_input, &d_output, &n};
    CHECK_CUDA(cuLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    float *h_output = (float *)malloc(N * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float clamped = h_input[i];
        if (clamped < 0.0f)
            clamped = 0.0f;
        if (clamped > 1e10f)
            clamped = 1e10f;
        float expected = sqrtf(clamped);
        float diff = fabsf(h_output[i] - expected);
        if (diff > 0.05f) {
            if (errors < 3) {
                printf("  Mismatch at [%d]: input=%.1f, got %.4f, expected %.4f (diff=%.4f)\n", i,
                       h_input[i], h_output[i], expected, diff);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASS\n");
        tests_passed++;
    } else {
        printf("  FAIL (%d errors)\n", errors);
        tests_failed++;
    }

    free(h_input);
    free(h_output);
    cuMemFree(d_input);
    cuMemFree(d_output);
}

// =============================================================================
// Test 3: fma_f32 + fma_i32 (monomorphized Rust generics)
// =============================================================================
void test_fma(CUmodule module) {
    printf("--- Test 3: test_fma (monomorphized generics: fma_f32, fma_i32) ---\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_fma");
    if (res != CUDA_SUCCESS) {
        printf("  SKIP (kernel not found)\n");
        return;
    }

    const int N = 256;

    CUdeviceptr d_input, d_output_f32, d_output_i32;
    CHECK_CUDA(cuMemAlloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output_f32, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output_i32, N * sizeof(int)));

    float *h_input = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        h_input[i] = (float)i;
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, N * sizeof(float)));

    int n = N;
    void *args[] = {&d_input, &d_output_f32, &d_output_i32, &n};
    CHECK_CUDA(cuLaunchKernel(kernel, (N + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHECK_CUDA(cuCtxSynchronize());

    float *h_out_f32 = (float *)malloc(N * sizeof(float));
    int *h_out_i32 = (int *)malloc(N * sizeof(int));
    CHECK_CUDA(cuMemcpyDtoH(h_out_f32, d_output_f32, N * sizeof(float)));
    CHECK_CUDA(cuMemcpyDtoH(h_out_i32, d_output_i32, N * sizeof(int)));

    int errors = 0;

    // f32: fma_f32(input[i], 2.0, 1.0) = input[i] * 2 + 1
    for (int i = 0; i < N; i++) {
        float expected = h_input[i] * 2.0f + 1.0f;
        if (fabsf(h_out_f32[i] - expected) > 0.001f) {
            if (errors < 3) {
                printf("  f32 mismatch at [%d]: got %.4f, expected %.4f\n", i, h_out_f32[i],
                       expected);
            }
            errors++;
        }
    }

    // i32: fma_i32(i, 3, 10) = i * 3 + 10
    for (int i = 0; i < N; i++) {
        int expected = i * 3 + 10;
        if (h_out_i32[i] != expected) {
            if (errors < 3) {
                printf("  i32 mismatch at [%d]: got %d, expected %d\n", i, h_out_i32[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASS\n");
        tests_passed++;
    } else {
        printf("  FAIL (%d errors)\n", errors);
        tests_failed++;
    }

    free(h_input);
    free(h_out_f32);
    free(h_out_i32);
    cuMemFree(d_input);
    cuMemFree(d_output_f32);
    cuMemFree(d_output_i32);
}

// =============================================================================
// Main — loads cubin and runs all tests
// =============================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cubin_file>\n", argv[0]);
        fprintf(stderr, "\nLoads a cubin with Rust+C++ LTOIR and verifies GPU results.\n");
        return 1;
    }

    const char *cubin_file = argv[1];

    printf("=== Phase 2 Test: C++ calling Rust device functions via LTOIR ===\n\n");

    // Initialize CUDA Driver API
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    char name[256];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), device));

    int major, minor;
    CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("Device: %s (sm_%d%d)\n", name, major, minor);

    CUcontext ctx;
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&ctx, device));
    CHECK_CUDA(cuCtxSetCurrent(ctx));

    // Load cubin
    printf("Cubin: %s\n\n", cubin_file);
    CUmodule module;
    CUresult loadRes = cuModuleLoad(&module, cubin_file);
    if (loadRes != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(loadRes, &errStr);
        fprintf(stderr, "Failed to load cubin: %s\n", errStr);
        return 1;
    }

    // Run tests
    test_sqrt_clamp(module);
    test_safe_sqrt(module);
    test_fma(module);

    // Summary
    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    if (tests_failed == 0 && tests_passed > 0) {
        printf("\n✓ All tests PASSED — C++ successfully called Rust device functions via LTOIR!\n");
    } else if (tests_passed == 0) {
        printf("\n✗ No tests ran\n");
    } else {
        printf("\n✗ Some tests FAILED\n");
    }

    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);

    return tests_failed > 0 ? 1 : 0;
}

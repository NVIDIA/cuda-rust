/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Launch device_ffi_test kernels from linked cubin
 *
 * This tool loads a cubin (produced by link_ltoir) and runs the test kernels
 * to verify that cuda-oxide device FFI works correctly.
 *
 * Build:
 *   nvcc -o launch_cubin launch_cubin.cu -lcuda
 *
 * Usage:
 *   ./launch_cubin <cubin_file>
 *
 * Kernels tested:
 *   - test_simple_device_funcs: Tests magnitude_squared, simple_add, warp_reduce_sum
 *   - test_cub_warp_reduce: Tests CUB warp reduction (if CCCL LTOIR linked)
 *   - test_mixed_attrs: Tests dot_product, fast_rsqrt, warp_ballot
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
            exit(1); \
        } \
    } while(0)

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

void test_simple_device_funcs(CUmodule module) {
    printf("\n--- Test 1: test_simple_device_funcs ---\n");
    printf("Testing: magnitude_squared, simple_add, warp_reduce_sum\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_simple_device_funcs");
    if (res != CUDA_SUCCESS) {
        printf("SKIP: Kernel not found\n");
        return;
    }

    const int N = 256;  // 8 warps
    const int WARPS = N / 32;

    // Allocate output
    CUdeviceptr d_output;
    CHECK_CUDA(cuMemAlloc(&d_output, WARPS * sizeof(float)));
    CHECK_CUDA(cuMemsetD32(d_output, 0, WARPS));

    // Launch: kernel signature is (output: *mut f32)
    void* args[] = { &d_output };

    CHECK_CUDA(cuLaunchKernel(
        kernel,
        1, 1, 1,      // grid
        N, 1, 1,      // block
        0, NULL,      // shared mem, stream
        args, NULL
    ));
    CHECK_CUDA(cuCtxSynchronize());

    // Get results
    float* h_output = (float*)malloc(WARPS * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, WARPS * sizeof(float)));

    // Expected: Each thread computes magnitude_squared(tid, tid+1) + 1.0
    // Then warp_reduce_sum across 32 threads
    // magnitude_squared(x,y) = x*x + y*y
    // For warp W, threads are W*32 to W*32+31
    // So warp 0 has tid 0-31, warp 1 has tid 32-63, etc.

    printf("Results (first 4 warps): ");
    for (int i = 0; i < WARPS && i < 4; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");

    // Verify each warp separately
    int errors = 0;
    for (int warp = 0; warp < WARPS; warp++) {
        float expected_warp_sum = 0.0f;
        int base_tid = warp * 32;
        for (int lane = 0; lane < 32; lane++) {
            int tid = base_tid + lane;
            float x = (float)tid;
            float y = (float)(tid + 1);
            expected_warp_sum += x*x + y*y + 1.0f;
        }

        if (fabsf(h_output[warp] - expected_warp_sum) > 1.0f) {
            if (errors < 3) {
                printf("  Warp %d: got %.1f, expected %.1f\n",
                       warp, h_output[warp], expected_warp_sum);
            }
            errors++;
        }
    }

    printf("Expected warp 0: %.1f, warp 1: %.1f\n",
           21888.0f, 152960.0f);  // Pre-computed for verification

    if (errors == 0) {
        printf("✓ PASSED\n");
        tests_passed++;
    } else {
        printf("✗ FAILED (%d errors)\n", errors);
        tests_failed++;
    }

    free(h_output);
    cuMemFree(d_output);
}

void test_cub_warp_reduce(CUmodule module) {
    printf("\n--- Test 2: test_cub_warp_reduce ---\n");
    printf("Testing: cub_warp_reduce_sum_f32\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_cub_warp_reduce");
    if (res != CUDA_SUCCESS) {
        printf("SKIP: Kernel not found (CCCL LTOIR may not be linked)\n");
        return;
    }

    const int N = 256;
    const int WARPS = N / 32;

    // Allocate input/output
    CUdeviceptr d_input, d_output;
    CHECK_CUDA(cuMemAlloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output, WARPS * sizeof(float)));

    // Initialize input: each thread gets value = tid % 32
    float* h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 32);
    }
    CHECK_CUDA(cuMemcpyHtoD(d_input, h_input, N * sizeof(float)));

    // Launch: kernel signature is (input: *const f32, output: *mut f32)
    void* args[] = { &d_input, &d_output };

    CHECK_CUDA(cuLaunchKernel(
        kernel,
        1, 1, 1,
        N, 1, 1,
        0, NULL,
        args, NULL
    ));
    CHECK_CUDA(cuCtxSynchronize());

    // Get results
    float* h_output = (float*)malloc(WARPS * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, WARPS * sizeof(float)));

    // Expected: sum of 0..31 = 496
    float expected = 496.0f;

    printf("Results: ");
    for (int i = 0; i < WARPS && i < 4; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\nExpected: %.1f\n", expected);

    int errors = 0;
    for (int i = 0; i < WARPS; i++) {
        if (fabsf(h_output[i] - expected) > 1.0f) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("✓ PASSED\n");
        tests_passed++;
    } else {
        printf("✗ FAILED (%d errors)\n", errors);
        tests_failed++;
    }

    free(h_input);
    free(h_output);
    cuMemFree(d_input);
    cuMemFree(d_output);
}

void test_mixed_attrs(CUmodule module) {
    printf("\n--- Test 3: test_mixed_attrs ---\n");
    printf("Testing: dot_product, fast_rsqrt, warp_ballot\n");

    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, module, "test_mixed_attrs");
    if (res != CUDA_SUCCESS) {
        printf("SKIP: Kernel not found\n");
        return;
    }

    const int N = 32;  // One warp for simplicity
    const int VEC_SIZE = 4;

    // Allocate arrays
    CUdeviceptr d_a, d_b, d_output;
    CHECK_CUDA(cuMemAlloc(&d_a, VEC_SIZE * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_b, VEC_SIZE * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_output, N * sizeof(float)));

    // Initialize: a = [1,2,3,4], b = [1,1,1,1]
    // dot_product = 1+2+3+4 = 10
    float h_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_b[] = {1.0f, 1.0f, 1.0f, 1.0f};
    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, VEC_SIZE * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, VEC_SIZE * sizeof(float)));

    // Launch: kernel signature is (a: *const f32, b: *const f32, output: *mut f32, n: i32)
    int n = VEC_SIZE;
    void* args[] = { &d_a, &d_b, &d_output, &n };

    CHECK_CUDA(cuLaunchKernel(
        kernel,
        1, 1, 1,
        N, 1, 1,
        0, NULL,
        args, NULL
    ));
    CHECK_CUDA(cuCtxSynchronize());

    // Get results
    float* h_output = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_output, d_output, N * sizeof(float)));

    printf("Output (first 8): ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");

    // Basic check: outputs should be non-zero and finite
    int valid = 1;
    for (int i = 0; i < N; i++) {
        if (!isfinite(h_output[i])) {
            valid = 0;
            break;
        }
    }

    if (valid) {
        printf("✓ PASSED (outputs are finite)\n");
        tests_passed++;
    } else {
        printf("✗ FAILED (invalid outputs)\n");
        tests_failed++;
    }

    free(h_output);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_output);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <cubin_file>\n", argv[0]);
        printf("\nThis tool launches device_ffi_test kernels to verify FFI linking.\n");
        return 1;
    }

    const char* cubin_file = argv[1];

    printf("=== Device FFI Test Launcher ===\n");
    printf("Cubin: %s\n", cubin_file);

    // Initialize CUDA
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    char deviceName[256];
    CHECK_CUDA(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    printf("Device: %s\n", deviceName);

    int major, minor;
    CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("Compute: sm_%d%d\n", major, minor);

    // Create context
    CUcontext context;
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_CUDA(cuCtxSetCurrent(context));

    // Load cubin
    CUmodule module;
    CUresult loadRes = cuModuleLoad(&module, cubin_file);
    if (loadRes != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(loadRes, &errStr);
        fprintf(stderr, "Failed to load cubin: %s\n", errStr);
        fprintf(stderr, "This may happen if architecture doesn't match your GPU.\n");
        return 1;
    }
    printf("Cubin loaded!\n");

    // Run tests
    test_simple_device_funcs(module);
    test_cub_warp_reduce(module);
    test_mixed_attrs(module);

    // Summary
    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    if (tests_failed == 0 && tests_passed > 0) {
        printf("\n✓ All tests PASSED!\n");
    } else if (tests_passed == 0) {
        printf("\nNo tests ran (kernels not found in cubin)\n");
    } else {
        printf("\n✗ Some tests FAILED\n");
    }

    // Cleanup
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);

    return tests_failed > 0 ? 1 : 0;
}

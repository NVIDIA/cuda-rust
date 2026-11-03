// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// C++ Kernels That Call Rust Device Functions via LTOIR
//
// This file is compiled to LTOIR by nvcc and linked with Rust-generated LTOIR.
// The Rust device functions are resolved at link time by nvJitLink.
//
// This is Phase 2 of Device FFI: C++ consuming Rust device functions.
// (Phase 3 was the reverse: Rust consuming C++ device functions.)
// =============================================================================

// ---------------------------------------------------------------------------
// Declarations of Rust device functions
//
// These symbols come from cuda-oxide compiled LTOIR. The clean export names
// (no reserved cuda_oxide_device_<hash>_ prefix) match what cuda-oxide produces.
// ---------------------------------------------------------------------------

extern "C" __device__ float fast_sqrt(float x);
extern "C" __device__ float clamp_f32(float val, float min_val, float max_val);
extern "C" __device__ float safe_sqrt(float x);
extern "C" __device__ float fma_f32(float a, float b, float c);
extern "C" __device__ int fma_i32(int a, int b, int c);

// ---------------------------------------------------------------------------
// Test Kernel 1: fast_sqrt + clamp_f32
//
// output[i] = clamp_f32(fast_sqrt(input[i]), 0.0, 10.0)
// ---------------------------------------------------------------------------
extern "C" __global__ void test_sqrt_clamp(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sq = fast_sqrt(input[idx]);
        output[idx] = clamp_f32(sq, 0.0f, 10.0f);
    }
}

// ---------------------------------------------------------------------------
// Test Kernel 2: safe_sqrt (transitive device fn calls across LTOIR)
//
// safe_sqrt internally calls clamp_f32 then fast_sqrt.
// Tests that transitive Rust-to-Rust device calls work through LTOIR linking.
// ---------------------------------------------------------------------------
extern "C" __global__ void test_safe_sqrt(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = safe_sqrt(input[idx]);
    }
}

// ---------------------------------------------------------------------------
// Test Kernel 3: fma_f32 + fma_i32 (monomorphized generics)
//
// Tests that Rust generic device functions (instantiated as concrete types)
// link correctly through LTOIR.
//
// output_f32[i] = fma_f32(input[i], 2.0, 1.0)  =>  input[i] * 2 + 1
// output_i32[i] = fma_i32(i, 3, 10)             =>  i * 3 + 10
// ---------------------------------------------------------------------------
extern "C" __global__ void test_fma(const float *input, float *output_f32, int *output_i32, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output_f32[idx] = fma_f32(input[idx], 2.0f, 1.0f);
        output_i32[idx] = fma_i32(idx, 3, 10);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SIMT implementation of the static test tile library. Each function takes
// one pointer per fragment of its implementation.mlir entry, except a
// rank-0 local `ptr` fragment, which arrives by value.

extern "C" __device__ __noinline__ void cutile_test_add_one(float* x) {
    x[0] += 1.0f;
}

extern "C" __device__ __noinline__ void cutile_test_add_two(float* x) {
    x[0] += 2.0f;
}

// Writes base plus each element's index: the fragment map gives thread i
// element i.
extern "C" __device__ __noinline__ void cutile_test_offset_ids(const int* base, int* ids) {
    ids[0] = base[0] + static_cast<int>(threadIdx.x % 32);
}

extern "C" __device__ __noinline__ void cutile_test_add_scalar(float* x, const float* s) {
    x[0] += s[0];
}

extern "C" __device__ __noinline__ void cutile_test_twice(const int* s, int* r) {
    r[0] = 2 * s[0];
}

// Takes no input: writes each element's index.
extern "C" __device__ __noinline__ void cutile_test_lane_ids(int* ids) {
    ids[0] = static_cast<int>(threadIdx.x % 32);
}

// Writes twice each element of `x` to global memory through `dst`.
extern "C" __device__ __noinline__ void cutile_test_store_to(const float* x, float* dst) {
    dst[threadIdx.x % 32] = 2.0f * x[0];
}

// Reads 32 elements from global memory through `src`; returns each plus one.
extern "C" __device__ __noinline__ void cutile_test_load_from(const float* src, float* out) {
    out[0] = src[threadIdx.x % 32] + 1.0f;
}

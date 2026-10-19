/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * External Device Functions for cuda-oxide FFI Testing
 *
 * This file contains simple device functions that will be compiled to LTOIR
 * and linked with cuda-oxide kernels at runtime via nvJitLink.
 *
 * Compile to LTOIR:
 *   nvcc -arch=sm_120 -dc -dlto external_device_funcs.cu -o external_device_funcs.o
 *   # The .ltoir is embedded in the .o file, extract with:
 *   nvcc -arch=sm_120 -dc -dlto --keep external_device_funcs.cu
 *   # This creates external_device_funcs.ltoir
 */

// ============================================================================
// Example 1: Pure Math Functions
//
// These are pure functions (no side effects, result depends only on inputs).
// NVCC automatically emits appropriate LLVM attributes in the LTOIR.
// ============================================================================

/**
 * Compute the squared magnitude of a 2D vector.
 * Pure function - no memory access, no side effects.
 */
extern "C" __device__ float magnitude_squared(float x, float y) {
    return x * x + y * y;
}

/**
 * Fast approximate inverse square root (Quake III style).
 * Pure function - no memory access.
 */
extern "C" __device__ float fast_rsqrt(float x) {
    float xhalf = 0.5f * x;
    int i = __float_as_int(x);
    i = 0x5f3759df - (i >> 1);
    x = __int_as_float(i);
    x = x * (1.5f - xhalf * x * x);
    return x;
}

/**
 * Simple addition - basic device function.
 * Pure function - no memory access.
 */
extern "C" __device__ float simple_add(float a, float b) {
    return a + b;
}

/**
 * Uppercase one Unicode scalar value in the ASCII range.
 *
 * Rust `char` lowers to the same plain 32-bit slot as `unsigned int`. Keep the
 * result in Rust's Unicode scalar range: an out-of-range value would violate
 * the Rust caller's validity contract.
 */
extern "C" __device__ unsigned int char_to_upper(unsigned int c) {
    return (c >= 'a' && c <= 'z') ? c - 32u : c;
}

/** Store one valid Unicode scalar through the pointer form of the same ABI. */
extern "C" __device__ void char_store(unsigned int *output, unsigned int c) {
    *output = c;
}

/**
 * Clamp a value to a range.
 * Pure function - no memory access.
 */
extern "C" __device__ float clamp_value(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

// ============================================================================
// Example 2: Read-Only Functions
//
// These functions only read memory (don't write). NVCC infers appropriate
// memory attributes from the __restrict__ qualifiers and access patterns.
// ============================================================================

/**
 * Lookup a value from a constant table.
 * Read-only function - reads from global memory but doesn't write.
 */
extern "C" __device__ float lookup_table(const float* __restrict__ table, int idx) {
    return table[idx];
}

/**
 * Compute dot product of two vectors.
 * Read-only - reads from both arrays but doesn't modify them.
 */
extern "C" __device__ float dot_product(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ============================================================================
// Example 3: Regular Device Function (no special attributes)
// ============================================================================

/**
 * Multiply-add operation that writes to output.
 * Regular function - has memory side effects.
 */
extern "C" __device__ void fused_multiply_add(
    float* output,
    float a,
    float b,
    float c
) {
    *output = a * b + c;
}

/**
 * Swap two values in memory.
 * Regular function - modifies memory.
 */
extern "C" __device__ void swap_floats(float* a, float* b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

// ============================================================================
// Example 4: Convergent Functions (Warp/Block Synchronous)
//
// These use warp shuffle or __syncthreads() internally, so NVCC marks them
// as convergent in the LTOIR. The convergent attribute prevents the optimizer
// from moving these operations across divergent control flow.
// ============================================================================

/**
 * Simple warp-level ballot - all threads must participate.
 * Convergent function - cannot be hoisted out of conditionals.
 */
extern "C" __device__ unsigned int warp_ballot(int predicate) {
    return __ballot_sync(0xffffffff, predicate);
}

/**
 * Warp-level reduction using shuffle.
 * Convergent function - all threads in warp must execute together.
 */
extern "C" __device__ float warp_reduce_sum(float val) {
    // Butterfly reduction pattern
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level barrier with a value exchange.
 * Convergent function - wraps __syncthreads().
 */
extern "C" __device__ void block_sync_and_store(float* shared_mem, int tid, float value) {
    shared_mem[tid] = value;
    __syncthreads();
}

// ============================================================================
// Example 5: Inline Hints
//
// CUDA provides __forceinline__ and __noinline__ qualifiers which emit
// alwaysinline and noinline attributes in the LTOIR respectively.
// ============================================================================

/**
 * Small helper that should always be inlined.
 */
extern "C" __device__ __forceinline__ float clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

/**
 * Large function that should not be inlined.
 */
extern "C" __device__ __noinline__ float complex_computation(
    const float* data,
    int n,
    float threshold
) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = data[i];
        if (val > threshold) {
            result += val * val;
        } else {
            result += val;
        }
    }
    return result;
}

// ============================================================================
// Example 6: Dynamic Shared Memory with Alignment
//
// Tests what happens when extern C++ function declares dynamic shared memory
// with a specific alignment requirement. The caller (Rust) may declare a
// different alignment - the linker should take the maximum.
//
// PTX generated by this function:
//   .extern .shared .align 128 .b8 __cuda_extern_shared_dynamic_smem[];
// ============================================================================

/**
 * 128-byte aligned type for TMA-compatible shared memory access.
 */
struct __align__(128) TmaAligned128 {
    float data[32];  // 128 bytes
};

/**
 * Write to dynamic shared memory at a specific offset.
 * This function expects 128-byte alignment for TMA operations.
 *
 * The extern __shared__ declaration here uses TmaAligned128,
 * which has 128-byte alignment.
 *
 * @param offset Byte offset into shared memory
 * @param value Value to write
 */
extern "C" __device__ void smem_write_aligned_128(int offset, float value) {
    extern __shared__ TmaAligned128 smem_128[];
    float* base = reinterpret_cast<float*>(smem_128);
    base[offset] = value;
}

/**
 * Read from dynamic shared memory at a specific offset.
 * This function expects 128-byte alignment for TMA operations.
 *
 * @param offset Byte offset into shared memory
 * @return Value at offset
 */
extern "C" __device__ float smem_read_aligned_128(int offset) {
    extern __shared__ TmaAligned128 smem_128[];
    float* base = reinterpret_cast<float*>(smem_128);
    return base[offset];
}

/**
 * Returns the base address of dynamic shared memory (for debugging).
 * Cast to uint64_t for printing.
 */
extern "C" __device__ unsigned long long smem_get_base_addr() {
    extern __shared__ char smem[];
    return reinterpret_cast<unsigned long long>(smem);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Small scalar ABI device functions for cuda-oxide FFI testing
 *
 * These functions are compiled to LTOIR and linked with cuda-oxide kernels
 * using nvJitLink.
 *
 * Compilation:
 *   nvcc -arch=sm_120 -dc -dlto --keep small_type_funcs.cu -o small_type_funcs.o
 *   # This creates small_type_funcs.ltoir
 */

#include <cuda_fp16.h>
#include <stdint.h>

// ============================================================================
// Small Scalar ABI Functions (i8/u8/i16/u16/bool/f16)
//
// Sub-32-bit scalars cross the extern "C" boundary as narrow LLVM types
// (i8/i16/i1/half) with signext/zeroext parameter and return attributes.
// cuda-oxide must emit byte-for-byte matching declarations for nvJitLink LTO
// to resolve these against the definitions below.
// ============================================================================

/** Widen a signed byte; a negative argument checks the caller's signext. */
extern "C" __device__ int32_t small_widen_i8(int8_t x) {
    return (int32_t)x;
}

/** Widen an unsigned short; a high-bit argument checks the caller's zeroext. */
extern "C" __device__ uint32_t small_widen_u16(uint16_t x) {
    return (uint32_t)x;
}

/** i8 round-trip: small return value with signext. */
extern "C" __device__ int8_t small_scale_i8(int8_t x) {
    return (int8_t)(x * 2);
}

/** u8 round-trip: small return value with zeroext. */
extern "C" __device__ uint8_t small_add_u8(uint8_t a, uint8_t b) {
    return (uint8_t)(a + b);
}

/** i16 round-trip. */
extern "C" __device__ int16_t small_scale_i16(int16_t x) {
    return (int16_t)(x * 3);
}

/** u16 round-trip. */
extern "C" __device__ uint16_t small_add_u16(uint16_t a, uint16_t b) {
    return (uint16_t)(a + b);
}

/** bool round-trip: i1 with zeroext in both positions. */
extern "C" __device__ bool small_not_bool(bool b) {
    return !b;
}

/** f16 round-trip: passed and returned directly as LLVM `half`. */
extern "C" __device__ __half small_half_add(__half a, __half b) {
    return __hadd(a, b);
}

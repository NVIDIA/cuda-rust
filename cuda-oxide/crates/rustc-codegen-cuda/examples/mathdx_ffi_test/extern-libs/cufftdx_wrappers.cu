/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * cuFFTDx Wrapper Functions for cuda-oxide FFI Testing
 *
 * This file wraps cuFFTDx template functions as extern "C" device functions
 * that can be called from cuda-oxide via LTOIR linking.
 *
 * Based on the official cuFFTDx simple_fft_thread example.
 *
 * Compile to LTOIR:
 *   nvcc -arch=sm_120 -dc -dlto --keep \
 *        -I/path/to/mathdx/include \
 *        -I/path/to/mathdx/external/cutlass/include \
 *        -std=c++17 \
 *        cufftdx_wrappers.cu
 */

#include <cufftdx.hpp>

// ============================================================================
// 8-point Thread-level FFT (Complex-to-Complex, Single Precision)
//
// Each thread computes a complete 8-point FFT independently.
// This is the simplest FFT configuration for testing.
// ============================================================================

using FFT_8_C2C_F32_FWD = decltype(
    cufftdx::Thread() +
    cufftdx::Size<8>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::forward>() +
    cufftdx::Precision<float>()
);

using FFT_8_C2C_F32_INV = decltype(
    cufftdx::Thread() +
    cufftdx::Size<8>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::inverse>() +
    cufftdx::Precision<float>()
);

/**
 * Get the storage size (number of complex elements) required for 8-point FFT.
 */
extern "C" __device__ int cufftdx_fft_8_storage_size() {
    return FFT_8_C2C_F32_FWD::storage_size;
}

/**
 * Get elements per thread for 8-point thread-level FFT.
 */
extern "C" __device__ int cufftdx_fft_8_elements_per_thread() {
    return FFT_8_C2C_F32_FWD::elements_per_thread;
}

/**
 * Execute an 8-point forward FFT in-place.
 * 
 * @param data  Pointer to complex data (8 complex floats = 16 floats).
 *              Data must be in thread-local memory (registers/local).
 *              Result is written back to the same location.
 */
extern "C" __device__ void cufftdx_fft_8_c2c_f32_forward(float* data) {
    using FFT = FFT_8_C2C_F32_FWD;
    using complex_type = typename FFT::value_type;
    
    // cuFFTDx expects complex_type array (cuda::std::complex<float>)
    // which has same memory layout as float2 or float[2]
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    
    // Execute in-place FFT
    FFT().execute(cdata);
}

/**
 * Execute an 8-point inverse FFT in-place.
 */
extern "C" __device__ void cufftdx_fft_8_c2c_f32_inverse(float* data) {
    using FFT = FFT_8_C2C_F32_INV;
    using complex_type = typename FFT::value_type;
    
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    FFT().execute(cdata);
}

// ============================================================================
// 16-point Thread-level FFT (Complex-to-Complex, Single Precision)
// ============================================================================

using FFT_16_C2C_F32_FWD = decltype(
    cufftdx::Thread() +
    cufftdx::Size<16>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::forward>() +
    cufftdx::Precision<float>()
);

using FFT_16_C2C_F32_INV = decltype(
    cufftdx::Thread() +
    cufftdx::Size<16>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::inverse>() +
    cufftdx::Precision<float>()
);

extern "C" __device__ int cufftdx_fft_16_storage_size() {
    return FFT_16_C2C_F32_FWD::storage_size;
}

extern "C" __device__ int cufftdx_fft_16_elements_per_thread() {
    return FFT_16_C2C_F32_FWD::elements_per_thread;
}

extern "C" __device__ void cufftdx_fft_16_c2c_f32_forward(float* data) {
    using FFT = FFT_16_C2C_F32_FWD;
    using complex_type = typename FFT::value_type;
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    FFT().execute(cdata);
}

extern "C" __device__ void cufftdx_fft_16_c2c_f32_inverse(float* data) {
    using FFT = FFT_16_C2C_F32_INV;
    using complex_type = typename FFT::value_type;
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    FFT().execute(cdata);
}

// ============================================================================
// 32-point Thread-level FFT (Complex-to-Complex, Single Precision)
// ============================================================================

using FFT_32_C2C_F32_FWD = decltype(
    cufftdx::Thread() +
    cufftdx::Size<32>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() +
    cufftdx::Direction<cufftdx::fft_direction::forward>() +
    cufftdx::Precision<float>()
);

extern "C" __device__ int cufftdx_fft_32_storage_size() {
    return FFT_32_C2C_F32_FWD::storage_size;
}

extern "C" __device__ void cufftdx_fft_32_c2c_f32_forward(float* data) {
    using FFT = FFT_32_C2C_F32_FWD;
    using complex_type = typename FFT::value_type;
    complex_type* cdata = reinterpret_cast<complex_type*>(data);
    FFT().execute(cdata);
}

// ============================================================================
// Debug Test Function
//
// Simple function to verify extern pointer writes work correctly.
// Just doubles each value in the array.
// ============================================================================

extern "C" __device__ void debug_extern_double_array(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
}

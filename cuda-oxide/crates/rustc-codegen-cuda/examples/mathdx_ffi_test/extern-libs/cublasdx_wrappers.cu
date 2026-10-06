/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * cuBLASDx Wrapper Functions for cuda-oxide FFI Testing
 *
 * This file wraps cuBLASDx template functions as extern "C" device functions
 * that can be called from cuda-oxide via LTOIR linking.
 *
 * Based on the official cuBLASDx introduction example.
 *
 * Compile to LTOIR:
 *   nvcc -arch=sm_90 -dc -dlto --keep \
 *        -I/path/to/mathdx/include \
 *        -I/path/to/mathdx/external/cutlass/include \
 *        -std=c++17 \
 *        cublasdx_wrappers.cu
 */

#include <cublasdx.hpp>

// ============================================================================
// GEMM Type Definitions
// ============================================================================

// 32x32x32 GEMM with 256 threads, matching official example
// Note: Must use actual SM architecture value at compile time
template<unsigned int Arch>
using GEMM_32x32x32_F32_T = decltype(
    cublasdx::Size<32, 32, 32>() + 
    cublasdx::Precision<float>() + 
    cublasdx::Type<cublasdx::type::real>() +
    cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
    cublasdx::Function<cublasdx::function::MM>() + 
    cublasdx::SM<Arch>() + 
    cublasdx::Block() +
    cublasdx::BlockDim<256>()
);

// ============================================================================
// Internal template implementation
// ============================================================================

template<class GEMM>
__device__ void gemm_registers_impl(
    const typename GEMM::a_value_type* a,
    const typename GEMM::b_value_type* b,
    typename GEMM::c_value_type* c,
    char* smem
) {
    // Make global memory tensors
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());
    
    // Make shared memory tensors for A and B only
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    
    // Load data from global memory to shared memory
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();
    
    // Execute GEMM and get accumulator
    auto accumulator = GEMM().execute(a_shared_tensor, b_shared_tensor);
    
    // Store results to global memory
    accumulator.partition_and_store(c_global_tensor);
}

template<class GEMM>
__device__ void gemm_shared_impl(
    typename GEMM::c_value_type alpha,
    const typename GEMM::a_value_type* a,
    const typename GEMM::b_value_type* b,
    typename GEMM::c_value_type beta,
    typename GEMM::c_value_type* c,
    char* smem
) {
    // Make global memory tensors
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());
    
    // Make shared memory tensors
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());
    
    // Load data from global memory to shared memory
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();
    
    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    __syncthreads();
    
    // Store data from shared memory to global memory
    cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
}

// ============================================================================
// Extern "C" Wrappers
// ============================================================================

// NOTE: cuBLASDx does NOT work on Blackwell (sm_120) as of MathDx 25.12
// The copy templates generate empty PTX. See BUG-004 in bugs/ folder.
// Using SM<1200> here for documentation purposes, but GEMM will fail.
using GEMM_32x32x32_F32 = GEMM_32x32x32_F32_T<1200>;

/**
 * Get shared memory size required for 32x32x32 GEMM (A, B, and C in smem).
 */
extern "C" __device__ int cublasdx_gemm_32x32x32_smem_size() {
    return cublasdx::get_shared_storage_size<GEMM_32x32x32_F32>();
}

/**
 * Get shared memory size for 32x32x32 GEMM (A and B only, C in registers).
 */
extern "C" __device__ int cublasdx_gemm_32x32x32_smem_size_ab() {
    return cublasdx::get_shared_storage_size_ab<GEMM_32x32x32_F32>();
}

/**
 * Get the required block dimension (x) for 32x32x32 GEMM.
 */
extern "C" __device__ int cublasdx_gemm_32x32x32_block_dim_x() {
    return GEMM_32x32x32_F32::block_dim.x;
}

/**
 * Get the required block dimension (y) for 32x32x32 GEMM.
 */
extern "C" __device__ int cublasdx_gemm_32x32x32_block_dim_y() {
    return GEMM_32x32x32_F32::block_dim.y;
}

/**
 * Get the required block dimension (z) for 32x32x32 GEMM.
 */
extern "C" __device__ int cublasdx_gemm_32x32x32_block_dim_z() {
    return GEMM_32x32x32_F32::block_dim.z;
}

/**
 * Execute 32x32x32 GEMM: C = A * B
 *
 * Uses shared memory for A and B, accumulator in registers.
 * All threads in the block (256) must participate.
 *
 * @param a      Pointer to A matrix (32x32, row-major, global memory)
 * @param b      Pointer to B matrix (32x32, col-major, global memory)
 * @param c      Pointer to C matrix (32x32, row-major, global memory, output)
 * @param smem   Shared memory buffer (at least cublasdx_gemm_32x32x32_smem_size_ab bytes)
 *
 * Note: Caller must launch with 256 threads and provide adequate shared memory.
 */
extern "C" __device__ void cublasdx_gemm_32x32x32_f32(
    const float* a,
    const float* b,
    float* c,
    char* smem
) {
    gemm_registers_impl<GEMM_32x32x32_F32>(a, b, c, smem);
}

/**
 * Execute 32x32x32 GEMM with alpha/beta: C = alpha * A * B + beta * C
 *
 * Uses shared memory for A, B, and C.
 * All threads in the block (256) must participate.
 *
 * @param alpha  Scalar multiplier for A*B
 * @param a      Pointer to A matrix (32x32, row-major)
 * @param b      Pointer to B matrix (32x32, col-major)
 * @param beta   Scalar multiplier for C
 * @param c      Pointer to C matrix (32x32, row-major, in/out)
 * @param smem   Shared memory buffer (at least cublasdx_gemm_32x32x32_smem_size bytes)
 */
extern "C" __device__ void cublasdx_gemm_32x32x32_f32_alphabeta(
    float alpha,
    const float* a,
    const float* b,
    float beta,
    float* c,
    char* smem
) {
    gemm_shared_impl<GEMM_32x32x32_F32>(alpha, a, b, beta, c, smem);
}

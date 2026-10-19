/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * CCCL (CUB/Thrust) Wrapper Functions for cuda-oxide FFI Testing
 *
 * This file wraps CCCL template functions as extern "C" device functions
 * that can be called from cuda-oxide via LTOIR linking.
 *
 * Compile to LTOIR:
 *   nvcc -arch=sm_120 -dc -dlto --keep -I/usr/local/cuda/include/cccl cccl_wrappers.cu
 */

#include <cccl/cub/block/block_reduce.cuh>
#include <cccl/cub/block/block_scan.cuh>
#include <cccl/cub/warp/warp_reduce.cuh>
#include <cccl/cub/warp/warp_scan.cuh>

// ============================================================================
// Block Reduce (256 threads)
//
// Block-level reductions require all threads in the block to participate.
// CUB internally uses shuffle and shared memory ops that emit convergent
// attributes in the LTOIR.
// ============================================================================

/// Temporary storage size for BlockReduce<float, 256>
constexpr int BLOCK_REDUCE_F32_256_TEMP_SIZE =
    sizeof(typename cub::BlockReduce<float, 256>::TempStorage);

/**
 * Block-level sum reduction for 256 threads.
 *
 * @param input    Value from each thread
 * @param temp     Shared memory for temporary storage (BLOCK_REDUCE_F32_256_TEMP_SIZE bytes)
 * @return         Sum of all inputs (only valid in thread 0)
 */
extern "C" __device__ float cub_block_reduce_sum_f32_256(float input, void* temp) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(temp);
    return BlockReduce(temp_storage).Sum(input);
}

/**
 * Block-level max reduction for 256 threads.
 */
extern "C" __device__ float cub_block_reduce_max_f32_256(float input, void* temp) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(temp);
    return BlockReduce(temp_storage).Reduce(input, ::cuda::maximum<float>{});
}

/**
 * Block-level min reduction for 256 threads.
 */
extern "C" __device__ float cub_block_reduce_min_f32_256(float input, void* temp) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(temp);
    return BlockReduce(temp_storage).Reduce(input, ::cuda::minimum<float>{});
}

// ============================================================================
// Block Reduce (128 threads) - different block size
// ============================================================================

constexpr int BLOCK_REDUCE_F32_128_TEMP_SIZE =
    sizeof(typename cub::BlockReduce<float, 128>::TempStorage);

extern "C" __device__ float cub_block_reduce_sum_f32_128(float input, void* temp) {
    using BlockReduce = cub::BlockReduce<float, 128>;
    auto& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(temp);
    return BlockReduce(temp_storage).Sum(input);
}

// ============================================================================
// Block Scan (Prefix Sum)
//
// Block-level scans require all threads in the block to participate.
// ============================================================================

constexpr int BLOCK_SCAN_F32_256_TEMP_SIZE =
    sizeof(typename cub::BlockScan<float, 256>::TempStorage);

/**
 * Block-level exclusive prefix sum for 256 threads.
 *
 * @param input       Value from each thread
 * @param output      Receives exclusive prefix sum for this thread
 * @param block_sum   Receives total sum (only valid in thread 0, optional)
 * @param temp        Shared memory for temporary storage
 */
extern "C" __device__ void cub_block_scan_exclusive_sum_f32_256(
    float input,
    float* output,
    float* block_sum,
    void* temp
) {
    using BlockScan = cub::BlockScan<float, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockScan::TempStorage*>(temp);

    float sum;
    BlockScan(temp_storage).ExclusiveSum(input, *output, sum);
    if (block_sum) {
        *block_sum = sum;
    }
}

/**
 * Block-level inclusive prefix sum for 256 threads.
 */
extern "C" __device__ void cub_block_scan_inclusive_sum_f32_256(
    float input,
    float* output,
    void* temp
) {
    using BlockScan = cub::BlockScan<float, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockScan::TempStorage*>(temp);
    BlockScan(temp_storage).InclusiveSum(input, *output);
}

// ============================================================================
// Warp Reduce (32 threads, no shared memory needed)
//
// Warp-level reductions require all threads in the warp to participate.
// ============================================================================

/**
 * Warp-level sum reduction.
 * No temporary storage needed - uses shuffle instructions.
 *
 * @param input  Value from each thread in the warp
 * @return       Sum of all inputs (valid in all threads)
 */
extern "C" __device__ float cub_warp_reduce_sum_f32(float input) {
    using WarpReduce = cub::WarpReduce<float>;
    typename WarpReduce::TempStorage temp;
    return WarpReduce(temp).Sum(input);
}

/**
 * Warp-level max reduction.
 */
extern "C" __device__ float cub_warp_reduce_max_f32(float input) {
    using WarpReduce = cub::WarpReduce<float>;
    typename WarpReduce::TempStorage temp;
    return WarpReduce(temp).Reduce(input, ::cuda::maximum<float>{});
}

// ============================================================================
// Warp Scan (Prefix Sum)
//
// Warp-level scans require all threads in the warp to participate.
// ============================================================================

/**
 * Warp-level exclusive prefix sum.
 * No temporary storage needed.
 */
extern "C" __device__ float cub_warp_scan_exclusive_sum_f32(float input) {
    using WarpScan = cub::WarpScan<float>;
    typename WarpScan::TempStorage temp;
    float output;
    WarpScan(temp).ExclusiveSum(input, output);
    return output;
}

/**
 * Warp-level inclusive prefix sum.
 */
extern "C" __device__ float cub_warp_scan_inclusive_sum_f32(float input) {
    using WarpScan = cub::WarpScan<float>;
    typename WarpScan::TempStorage temp;
    float output;
    WarpScan(temp).InclusiveSum(input, output);
    return output;
}

// ============================================================================
// Integer variants
// ============================================================================

extern "C" __device__ int cub_block_reduce_sum_i32_256(int input, void* temp) {
    using BlockReduce = cub::BlockReduce<int, 256>;
    auto& temp_storage = *reinterpret_cast<typename BlockReduce::TempStorage*>(temp);
    return BlockReduce(temp_storage).Sum(input);
}

extern "C" __device__ int cub_warp_reduce_sum_i32(int input) {
    using WarpReduce = cub::WarpReduce<int>;
    typename WarpReduce::TempStorage temp;
    return WarpReduce(temp).Sum(input);
}

// ============================================================================
// Utility: Get temp storage sizes
// These are constexpr but exposed as device functions for Rust to query
// ============================================================================

extern "C" __device__ int cub_get_block_reduce_temp_size_256() {
    return BLOCK_REDUCE_F32_256_TEMP_SIZE;
}

extern "C" __device__ int cub_get_block_reduce_temp_size_128() {
    return BLOCK_REDUCE_F32_128_TEMP_SIZE;
}

extern "C" __device__ int cub_get_block_scan_temp_size_256() {
    return BLOCK_SCAN_F32_256_TEMP_SIZE;
}

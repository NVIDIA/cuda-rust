/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A Tile C++ kernel that calls the three Rust exports of tilelib_relu.tilelib.
//
//   nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
//        -std=c++20 -arch=sm_120 -o tile_caller cpp/tile_caller.cu
//
// Needs a toolkit whose nvcc knows `--enable-tile` and whose tileiras built
// the library (CUDA 13.5 or newer). Each foreign declaration mirrors one
// declaration in the library's interface: parameters the library reads carry
// no attribute, parameters it writes carry `[[cutile::experimental_inout]]`
// and come back as the result. The kernel runs one 128-element tile through
// each export; the host checks every element bit for bit.

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace ct = cuda::tiles;

inline constexpr int tile_size = 128;
using tile_f32 = ct::tile<float, ct::shape<tile_size>>;

// @relu(%x: tile<128xf32> in, %relu_out: tile<128xf32> out) -> tile<128xf32>
__experimental_foreign_decl__ __experimental_foreign_name__("relu")
    tile_f32 relu(tile_f32 x, [[cutile::experimental_inout]] tile_f32 relu_out);

// @scale_in_place(%x: tile<128xf32> inout, %factor: tile<128xf32> in) -> tile<128xf32>
__experimental_foreign_decl__ __experimental_foreign_name__("scale_in_place")
    tile_f32 scale_in_place([[cutile::experimental_inout]] tile_f32 x, tile_f32 factor);

// @warp_sum(%x: tile<128xf32> in, %warp_sum_out: tile<128xf32> out) -> tile<128xf32>,
// implemented with synchronization<warp>: 32 lanes call the Rust function together.
__experimental_foreign_decl__ __experimental_foreign_name__("warp_sum")
    tile_f32 warp_sum(tile_f32 x, [[cutile::experimental_inout]] tile_f32 warp_sum_out);

[[cutile::hint(0, num_worker_warps_per_cta = 4)]] __tile_global__ void call_exports(
    float* ramp, float* factor, float* lanes, float* relu_out, float* scale_out, float* warp_out) {
    auto view = [](float* data) {
        return ct::partition_view { ct::tensor_span { data, ct::shape<tile_size> {} }, ct::shape<tile_size> {} };
    };
    auto relu_view = view(relu_out);
    auto scale_view = view(scale_out);
    auto warp_view = view(warp_out);

    relu_view.store(relu(view(ramp).load(0), relu_view.load(0)), 0);
    scale_view.store(scale_in_place(view(ramp).load(0), view(factor).load(0)), 0);
    warp_view.store(warp_sum(view(lanes).load(0), warp_view.load(0)), 0);
}

#define CHECK_CUDA(expression)                                                                     \
    do {                                                                                           \
        const cudaError_t cuda_status = (expression);                                              \
        if (cuda_status != cudaSuccess) {                                                          \
            std::fprintf(stderr, "%s failed: %s\n", #expression, cudaGetErrorString(cuda_status)); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

int main() {
    std::array<float, tile_size> ramp {};
    std::array<float, tile_size> factor {};
    std::array<float, tile_size> lanes {};
    for (int i = 0; i < tile_size; ++i) {
        ramp[i] = -4.0f + 8.0f * float(i) / float(tile_size - 1);
        factor[i] = 0.5f;
        lanes[i] = float(i);
    }

    float* device[6] = {};
    for (float*& pointer : device) CHECK_CUDA(cudaMalloc(&pointer, sizeof(ramp)));
    CHECK_CUDA(cudaMemcpy(device[0], ramp.data(), sizeof(ramp), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device[1], factor.data(), sizeof(factor), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device[2], lanes.data(), sizeof(lanes), cudaMemcpyHostToDevice));
    for (int i = 3; i < 6; ++i) CHECK_CUDA(cudaMemset(device[i], 0, sizeof(ramp)));

    call_exports<<<1>>>(device[0], device[1], device[2], device[3], device[4], device[5]);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::array<float, tile_size> relu_out {};
    std::array<float, tile_size> scale_out {};
    std::array<float, tile_size> warp_out {};
    CHECK_CUDA(cudaMemcpy(relu_out.data(), device[3], sizeof(relu_out), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(scale_out.data(), device[4], sizeof(scale_out), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(warp_out.data(), device[5], sizeof(warp_out), cudaMemcpyDeviceToHost));

    int failures = 0;
    for (int i = 0; i < tile_size; ++i) {
        const float expected_relu = ramp[i] > 0.0f ? ramp[i] : 0.0f;
        const float expected_scale = ramp[i] * 0.5f;
        // Lane values 32w .. 32w+31 sum to 32 * 32w + 496.
        const float expected_warp = float(32 * 32 * (i / 32) + 496);
        if (relu_out[i] != expected_relu) {
            std::fprintf(stderr, "relu mismatch at %d: got %f expected %f\n", i, relu_out[i], expected_relu);
            ++failures;
        }
        if (scale_out[i] != expected_scale) {
            std::fprintf(stderr, "scale_in_place mismatch at %d: got %f expected %f\n", i, scale_out[i], expected_scale);
            ++failures;
        }
        if (warp_out[i] != expected_warp) {
            std::fprintf(stderr, "warp_sum mismatch at %d: got %f expected %f\n", i, warp_out[i], expected_warp);
            ++failures;
        }
    }
    for (float* pointer : device) CHECK_CUDA(cudaFree(pointer));
    if (failures != 0) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASS relu: 128 elements through the tile library\n");
    std::printf("PASS scale_in_place: 128 elements through the tile library\n");
    std::printf("PASS warp_sum: 128 elements through the tile library, every lane holding its warp's total\n");
    std::printf("SUCCESS: 3/3 tile exports called from a Tile C++ kernel\n");
    return EXIT_SUCCESS;
}

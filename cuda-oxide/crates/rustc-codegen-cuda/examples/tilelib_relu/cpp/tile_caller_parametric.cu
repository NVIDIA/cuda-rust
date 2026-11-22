/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A Tile C++ kernel that calls the parametric exports of tilelib_relu at three
// tile extents, through the provider form of the library:
//
//   cargo oxide tilelib tilelib_relu --arch sm_120 --provider
//   nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
//        -std=c++20 -arch=sm_120 -o tile_caller_parametric cpp/tile_caller_parametric.cu
//
// `relu_any` and `warp_sum_any` name no `tile_shape`, so their declarations
// are `@relu_any<[N: index]>` and `@warp_sum_any<[N: index]>`. Each foreign
// declaration here is a function template whose extent parameter is that N;
// every distinct extent a call uses is one instantiation, and the provider
// compiles the implementation symbol for it while this file compiles.

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace ct = cuda::tiles;

template <size_t N>
using tile_f32 = ct::tile<float, ct::shape<N>>;

// @relu_any<[N: index]>(%x: tile<[N]xf32> in, %relu_any_out: tile<[N]xf32> out) -> tile<[N]xf32>
template <size_t N>
__experimental_foreign_decl__ __experimental_foreign_name__("relu_any")
    tile_f32<N> relu_any(tile_f32<N> x, [[cutile::experimental_inout]] tile_f32<N> relu_any_out);

// @warp_sum_any<[N: index]>(%x: tile<[N]xf32> in, %warp_sum_any_out: tile<[N]xf32> out)
// -> tile<[N]xf32>, with synchronization<warp>: N must be a multiple of 32.
template <size_t N>
__experimental_foreign_decl__ __experimental_foreign_name__("warp_sum_any")
    tile_f32<N> warp_sum_any(tile_f32<N> x, [[cutile::experimental_inout]] tile_f32<N> warp_sum_any_out);

template <size_t N>
__tile__ auto view(float* data) {
    return ct::partition_view { ct::tensor_span { data, ct::shape<N> {} }, ct::shape<N> {} };
}

[[cutile::hint(0, num_worker_warps_per_cta = 4)]] __tile_global__ void call_exports(
    float* ramp128, float* ramp256, float* lanes64, float* relu128_out, float* relu256_out, float* warp64_out) {
    auto relu128 = view<128>(relu128_out);
    relu128.store(relu_any<128>(view<128>(ramp128).load(0), relu128.load(0)), 0);

    auto relu256 = view<256>(relu256_out);
    relu256.store(relu_any<256>(view<256>(ramp256).load(0), relu256.load(0)), 0);

    auto warp64 = view<64>(warp64_out);
    warp64.store(warp_sum_any<64>(view<64>(lanes64).load(0), warp64.load(0)), 0);
}

#define CHECK_CUDA(expression)                                                                     \
    do {                                                                                           \
        const cudaError_t cuda_status = (expression);                                              \
        if (cuda_status != cudaSuccess) {                                                          \
            std::fprintf(stderr, "%s failed: %s\n", #expression, cudaGetErrorString(cuda_status)); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

template <size_t N>
static void fill_ramp(std::array<float, N>& ramp) {
    for (size_t i = 0; i < N; ++i) ramp[i] = -4.0f + 8.0f * float(i) / float(N - 1);
}

int main() {
    std::array<float, 128> ramp128 {};
    std::array<float, 256> ramp256 {};
    std::array<float, 64> lanes64 {};
    fill_ramp(ramp128);
    fill_ramp(ramp256);
    for (size_t i = 0; i < lanes64.size(); ++i) lanes64[i] = float(i);

    float *d_ramp128, *d_ramp256, *d_lanes64, *d_relu128, *d_relu256, *d_warp64;
    CHECK_CUDA(cudaMalloc(&d_ramp128, sizeof(ramp128)));
    CHECK_CUDA(cudaMalloc(&d_ramp256, sizeof(ramp256)));
    CHECK_CUDA(cudaMalloc(&d_lanes64, sizeof(lanes64)));
    CHECK_CUDA(cudaMalloc(&d_relu128, sizeof(ramp128)));
    CHECK_CUDA(cudaMalloc(&d_relu256, sizeof(ramp256)));
    CHECK_CUDA(cudaMalloc(&d_warp64, sizeof(lanes64)));
    CHECK_CUDA(cudaMemcpy(d_ramp128, ramp128.data(), sizeof(ramp128), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ramp256, ramp256.data(), sizeof(ramp256), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lanes64, lanes64.data(), sizeof(lanes64), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_relu128, 0, sizeof(ramp128)));
    CHECK_CUDA(cudaMemset(d_relu256, 0, sizeof(ramp256)));
    CHECK_CUDA(cudaMemset(d_warp64, 0, sizeof(lanes64)));

    call_exports<<<1>>>(d_ramp128, d_ramp256, d_lanes64, d_relu128, d_relu256, d_warp64);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::array<float, 128> relu128 {};
    std::array<float, 256> relu256 {};
    std::array<float, 64> warp64 {};
    CHECK_CUDA(cudaMemcpy(relu128.data(), d_relu128, sizeof(relu128), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(relu256.data(), d_relu256, sizeof(relu256), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(warp64.data(), d_warp64, sizeof(warp64), cudaMemcpyDeviceToHost));

    int failures = 0;
    for (size_t i = 0; i < 128; ++i) {
        const float expected = ramp128[i] > 0.0f ? ramp128[i] : 0.0f;
        if (relu128[i] != expected) {
            std::fprintf(stderr, "relu_any<128> mismatch at %zu: got %f expected %f\n", i, relu128[i], expected);
            ++failures;
        }
    }
    for (size_t i = 0; i < 256; ++i) {
        const float expected = ramp256[i] > 0.0f ? ramp256[i] : 0.0f;
        if (relu256[i] != expected) {
            std::fprintf(stderr, "relu_any<256> mismatch at %zu: got %f expected %f\n", i, relu256[i], expected);
            ++failures;
        }
    }
    for (size_t i = 0; i < 64; ++i) {
        // Lane values 32w .. 32w+31 sum to 32 * 32w + 496.
        const float expected = float(32 * 32 * (i / 32) + 496);
        if (warp64[i] != expected) {
            std::fprintf(stderr, "warp_sum_any<64> mismatch at %zu: got %f expected %f\n", i, warp64[i], expected);
            ++failures;
        }
    }
    for (float* pointer : {d_ramp128, d_ramp256, d_lanes64, d_relu128, d_relu256, d_warp64}) {
        CHECK_CUDA(cudaFree(pointer));
    }
    if (failures != 0) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASS relu_any<128>: 128 elements through the provider\n");
    std::printf("PASS relu_any<256>: 256 elements through the provider\n");
    std::printf("PASS warp_sum_any<64>: 64 elements through the provider, every lane holding its warp's total\n");
    std::printf("SUCCESS: 3/3 parametric calls served by the tile library provider\n");
    return EXIT_SUCCESS;
}

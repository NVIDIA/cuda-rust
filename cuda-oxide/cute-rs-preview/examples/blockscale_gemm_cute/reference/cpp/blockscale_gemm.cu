// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Instruction-level CUDA C++ counterpart to blockscale_gemm_ctm.
// Physical addresses, issue order, barrier phases, and ownership stay explicit.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "layout.cuh"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace primitives {
struct Mxf4A { std::uint32_t x[4]; };
struct Mxf4B { std::uint32_t x[2]; };
struct ScaleA { std::uint16_t bits; };
struct ScaleB { std::uint16_t bits; };
struct Accumulator { float x[4]; };

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}
__device__ __forceinline__ void init(std::uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::
               "r"(shared_address(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void wait(std::uint64_t* bar, unsigned phase) {
  unsigned done;
  do {
    asm volatile("{ .reg .pred p; "
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
                 "selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(shared_address(bar)), "r"(phase) : "memory");
  } while (!done);
}
__device__ __forceinline__ void arrive(std::uint64_t* bar) {
  [[maybe_unused]] std::uint64_t token;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(token) :
               "r"(shared_address(bar)) : "memory");
  (void)token;
}
__device__ __forceinline__ void expect(std::uint64_t* bar, unsigned bytes) {
  [[maybe_unused]] std::uint64_t token;
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;" : "=l"(token) :
               "r"(shared_address(bar)), "r"(bytes) : "memory");
  (void)token;
}
__device__ __forceinline__ void load(void* destination, const CUtensorMap* map,
                                    int x, int y, std::uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile."
               "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
               "r"(shared_address(destination)), "l"(map), "r"(x), "r"(y),
               "r"(shared_address(bar)) : "memory");
}
__device__ __forceinline__ void store(const void* source, const CUtensorMap* map,
                                     int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2}], [%3];" :: "l"(map), "r"(x), "r"(y),
               "r"(shared_address(source)) : "memory");
}
__device__ __forceinline__ Mxf4A ldmatrix(const void* source) {
  Mxf4A result;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(result.x[0]), "=r"(result.x[1]), "=r"(result.x[2]), "=r"(result.x[3])
               : "r"(shared_address(source)) : "memory");
  return result;
}
__device__ __forceinline__ void mma(Accumulator& c, Mxf4A a, Mxf4B b,
                                   ScaleA sa, ScaleB sb) {
  // Two UE8M0 bytes per lane represent consecutive K32 scale groups.
  const std::uint16_t selector = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X."
      "f32.e2m1.e2m1.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {%12, %12}, %11, {%12, %12};"
      : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
        "r"(b.x[0]), "r"(b.x[1]), "r"(static_cast<unsigned>(sa.bits)),
        "r"(static_cast<unsigned>(sb.bits)), "h"(selector));
}
__device__ __forceinline__ void stmatrix(void* destination, const Accumulator& c) {
  unsigned top, bottom;
  // PTX takes the high half first; the accumulator is low-column first.
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(top) : "f"(c.x[1]), "f"(c.x[0]));
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(bottom) : "f"(c.x[3]), "f"(c.x[2]));
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" ::
               "r"(shared_address(destination)), "r"(top), "r"(bottom) : "memory");
}
__device__ __forceinline__ void wait_output() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void output_barrier() {
  asm volatile("bar.sync 2, 256;" ::: "memory");
}
}  // namespace primitives

__global__ __launch_bounds__(288) void blockscale_gemm(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap sa_map,
    const __grid_constant__ CUtensorMap sb_map,
    const __grid_constant__ CUtensorMap c_map) {
  using namespace blockscale;
  using namespace primitives;
  extern __shared__ __align__(1024) unsigned char shared[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  auto* full = reinterpret_cast<std::uint64_t*>(shared + BARRIERS);
  auto* empty = full + STAGES;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      init(full + s, 1);
      init(empty + s, 8);
    }
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  int tile = blockIdx.x;
  const int stride = gridDim.x;
  int slot = 0;
  if (warp < 8) {
    const int wm = warp % 4, wn = warp / 4;
    unsigned phase = 0;
    for (; tile < TOTAL_TILES; tile += stride) {
      const int tile_m = tile % (M / 128), tile_n = tile / (M / 128);
      Accumulator c[2][8]{};
      for (int stage = 0; stage < K_STAGES; ++stage) {
        wait(full + slot, phase);
        const auto* a = shared + slot * DATA_BYTES;
        const auto* b = shared + B_OFFSET + slot * DATA_BYTES;
        const auto* sa = reinterpret_cast<const unsigned*>(shared + SFA_OFFSET + slot * SCALE_BYTES);
        const auto* sb = reinterpret_cast<const unsigned*>(shared + SFB_OFFSET + slot * SCALE_BYTES);
        const int provider = lane / 4 + 8 * (lane & 1);
        const unsigned sa0 = sa[scale_word(16 * wm + provider)];
        const unsigned sa1 = sa[scale_word(16 * wm + provider + 64)];
        unsigned scales_b[8];
#pragma unroll
        for (int n = 0; n < 8; ++n)
          scales_b[n] = sb[scale_word(16 * wn + lane / 4 + n_offset(n))];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const auto a0 = ldmatrix(a + matrix_row(wm, half, lane));
          const auto a1 = ldmatrix(a + matrix_row(wm + 4, half, lane));
          const ScaleA as0{static_cast<std::uint16_t>(sa0 >> (16 * half))};
          const ScaleA as1{static_cast<std::uint16_t>(sa1 >> (16 * half))};
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const auto loaded_b = ldmatrix(b + matrix_row(wn + 2 * pair, half, lane));
            const Mxf4B b0{{loaded_b.x[0], loaded_b.x[2]}};
            const Mxf4B b1{{loaded_b.x[1], loaded_b.x[3]}};
            const ScaleB bs0{static_cast<std::uint16_t>(scales_b[2 * pair] >> (16 * half))};
            const ScaleB bs1{static_cast<std::uint16_t>(scales_b[2 * pair + 1] >> (16 * half))};
            mma(c[0][2 * pair], a0, b0, as0, bs0);
            mma(c[0][2 * pair + 1], a0, b1, as0, bs1);
            mma(c[1][2 * pair], a1, b0, as1, bs0);
            mma(c[1][2 * pair + 1], a1, b1, as1, bs1);
          }
        }
        if (lane == 0) arrive(empty + slot);
        if (++slot == STAGES) { slot = 0; phase ^= 1; }
      }
      // The producer does not participate in these 256-thread barriers.
      if (tid == 0) wait_output();
      output_barrier();
      auto* output = reinterpret_cast<__half*>(shared + OUTPUT);
#pragma unroll
      for (int band = 0; band < 2; ++band) {
#pragma unroll
        for (int n = 0; n < 8; ++n)
          stmatrix(output + output_row(warp, lane, band, n), c[band][n]);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      output_barrier();
      if (tid == 0) {
        store(shared + OUTPUT, &c_map, tile_n * 128, tile_m * 128);
        store(shared + OUTPUT + 16384, &c_map, tile_n * 128 + 64, tile_m * 128);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
    if (tid == 0) wait_output();
  } else if (lane == 0) {
    unsigned phase = 1;
    for (; tile < TOTAL_TILES; tile += stride) {
      const int tile_m = tile % (M / 128), tile_n = tile / (M / 128);
      for (int stage = 0; stage < K_STAGES; ++stage) {
        wait(empty + slot, phase);
        expect(full + slot, TX_BYTES);
        load(shared + slot * DATA_BYTES, &a_map, stage * 64, tile_m * 128, full + slot);
        load(shared + B_OFFSET + slot * DATA_BYTES, &b_map, stage * 64, tile_n * 128, full + slot);
        load(shared + SFA_OFFSET + slot * SCALE_BYTES, &sa_map, 0,
             tile_m * K_STAGES + stage, full + slot);
        load(shared + SFB_OFFSET + slot * SCALE_BYTES, &sb_map, 0,
             tile_n * K_STAGES + stage, full + slot);
        if (++slot == STAGES) { slot = 0; phase ^= 1; }
      }
    }
    for (int tail = 0; tail < STAGES; ++tail) {
      wait(empty + slot, phase);
      if (++slot == STAGES) { slot = 0; phase ^= 1; }
    }
  }
}

namespace {
void check_cuda(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
void check_driver(CUresult status, const char* expression) {
  if (status != CUDA_SUCCESS) {
    const char* text = nullptr;
    cuGetErrorString(status, &text);
    throw std::runtime_error(std::string(expression) + ": " + (text ? text : "CUDA driver error"));
  }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression)
#define DRIVER_CHECK(expression) check_driver((expression), #expression)

template <class T> struct DeviceBuffer {
  T* pointer = nullptr;
  explicit DeviceBuffer(std::size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
  }
  ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Event {
  cudaEvent_t value{};
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

CUtensorMap encode(void* pointer, CUtensorMapDataType type, std::uint64_t columns,
                   std::uint64_t rows, std::uint64_t row_bytes,
                   std::uint32_t tile_columns, std::uint32_t tile_rows,
                   CUtensorMapSwizzle swizzle) {
  alignas(128) CUtensorMap map{};
  const std::uint64_t dimensions[] = {columns, rows};
  const std::uint64_t strides[] = {row_bytes};
  const std::uint32_t box[] = {tile_columns, tile_rows}, element_strides[] = {1, 1};
  DRIVER_CHECK(cuTensorMapEncodeTiled(&map, type, 2, pointer, dimensions, strides,
                                     box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}
std::uint32_t mix32(std::uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU;
  return x ^ (x >> 16);
}
float decode(std::uint8_t nibble) {
  constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (nibble & 8) ? -values[nibble & 7] : values[nibble & 7];
}
std::vector<std::uint16_t> pack(const std::vector<std::uint8_t>& nibbles) {
  std::vector<std::uint16_t> result(nibbles.size() / 4);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = nibbles[4 * i] | (nibbles[4 * i + 1] << 4) |
                (nibbles[4 * i + 2] << 8) | (nibbles[4 * i + 3] << 12);
  return result;
}
std::vector<std::uint8_t> pack_scales(const std::vector<std::uint8_t>& dense, int rows) {
  std::vector<std::uint8_t> result(dense.size());
  for (int row = 0; row < rows; ++row)
    for (int group = 0; group < blockscale::K_GROUPS; ++group)
      result[blockscale::scale_offset(row, group)] = dense[row * blockscale::K_GROUPS + group];
  return result;
}
unsigned parse_count(const std::string& argument, const std::string& value) {
  unsigned result = 0;
  auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::runtime_error(argument + " requires an unsigned integer");
  return result;
}
struct Options {
  unsigned ctas = 0, warmup = 100, samples = 31;
  std::filesystem::path json;
};
Options options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument != "--ctas" && argument != "--warmup" && argument != "--samples" && argument != "--json")
      throw std::runtime_error("unknown option " + argument + "; use --ctas, --warmup, --samples, --json");
    if (++i == argc) throw std::runtime_error("missing value for " + argument);
    if (argument == "--json") {
      result.json = argv[i];
      if (result.json.empty()) throw std::runtime_error("--json path cannot be empty");
    } else {
      unsigned count = parse_count(argument, argv[i]);
      if (argument == "--ctas") {
        if (!count || count > blockscale::TOTAL_TILES)
          throw std::runtime_error("--ctas must be between 1 and 256");
        result.ctas = count;
      } else if (argument == "--samples") {
        if (!count) throw std::runtime_error("--samples must be positive");
        result.samples = count;
      } else result.warmup = count;
    }
  }
  return result;
}
float percentile(const std::vector<float>& sorted, std::size_t numerator, std::size_t denominator) {
  return sorted[((sorted.size() - 1) * numerator + denominator / 2) / denominator];
}

int run(int argc, char** argv) {
  using namespace blockscale;
  const auto config = options(argc, argv);
  if (!config.json.empty()) std::filesystem::remove(config.json);
  CUDA_CHECK(cudaSetDevice(0));  // CUDA_VISIBLE_DEVICES controls the physical GPU.
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 12 || properties.minor != 0)
    throw std::runtime_error("mxf4 primitives require an SM120 GPU");
  std::cout << "=== Block-scaled MXFP4 GEMM with C++ instruction primitives ===\n"
            << "C(2048x2048) = A(2048x1024 e2m1) x B(1024x2048 e2m1), "
               "ue8m0 scales per 32-element block\n";

  std::vector<std::uint8_t> a(M * K), b(N * K), sa(M * K_GROUPS), sb(N * K_GROUPS);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k)
      a[m * K + k] = mix32(std::uint32_t(m) * 0x9e3779b9U ^ std::uint32_t(k) ^ 0xa511e9b3U) & 15;
    for (int group = 0; group < K_GROUPS; ++group)
      sa[m * K_GROUPS + group] = 126 + mix32(std::uint32_t(m) * 0xc2b2ae35U ^ std::uint32_t(group)) % 3;
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k)
      b[n * K + k] = mix32(std::uint32_t(n) * 0x85ebca6bU ^ std::uint32_t(k) ^ 0x63d83595U) & 15;
    for (int group = 0; group < K_GROUPS; ++group)
      sb[n * K_GROUPS + group] = 126 + mix32(std::uint32_t(n) * 0x27d4eb2fU ^ std::uint32_t(group) ^ 0x165667b1U) % 3;
  }
  // Dense CPU oracle independent of shared/register layouts. Products and sums
  // are exact multiples of 1/16 within FP32's exact range for this fixture.
  std::cout << "computing exact host reference (2048x2048x1024)...\n" << std::flush;
  std::vector<float> decoded_a(a.size()), decoded_b(b.size());
  std::transform(a.begin(), a.end(), decoded_a.begin(), decode);
  std::transform(b.begin(), b.end(), decoded_b.begin(), decode);
  std::vector<std::uint16_t> expected(M * N);
  std::size_t nonzero = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float total = 0;
      for (int group = 0; group < K_GROUPS; ++group) {
        float dot = 0;
        for (int j = 0; j < 32; ++j) {
          int k = group * 32 + j;
          dot += decoded_a[m * K + k] * decoded_b[n * K + k];
        }
        total += std::ldexp(dot, int(sa[m * K_GROUPS + group]) + int(sb[n * K_GROUPS + group]) - 254);
      }
      nonzero += total != 0;
      expected[m * N + n] = __half_as_ushort(__float2half_rn(total));
    }
  }
  auto packed_a = pack(a), packed_b = pack(b);
  auto packed_sa = pack_scales(sa, M), packed_sb = pack_scales(sb, N);
  DeviceBuffer<std::uint16_t> da(packed_a.size()), db(packed_b.size()), dc(M * N);
  DeviceBuffer<std::uint8_t> dsa(packed_sa.size()), dsb(packed_sb.size());
  CUDA_CHECK(cudaMemcpy(da.pointer, packed_a.data(), packed_a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.pointer, packed_b.data(), packed_b.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsa.pointer, packed_sa.data(), packed_sa.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsb.pointer, packed_sb.data(), packed_sb.size(), cudaMemcpyHostToDevice));
  std::vector<std::uint16_t> got(M * N, 0x7e00);  // FP16 quiet NaN poison.
  CUDA_CHECK(cudaMemcpy(dc.pointer, got.data(), got.size() * 2, cudaMemcpyHostToDevice));
  const auto a_map = encode(da.pointer, CU_TENSOR_MAP_DATA_TYPE_UINT8, K / 2, M, K / 2,
                            64, 128, CU_TENSOR_MAP_SWIZZLE_64B);
  const auto b_map = encode(db.pointer, CU_TENSOR_MAP_DATA_TYPE_UINT8, K / 2, N, K / 2,
                            64, 128, CU_TENSOR_MAP_SWIZZLE_64B);
  const auto sa_map = encode(dsa.pointer, CU_TENSOR_MAP_DATA_TYPE_UINT16, 256, (M / 128) * K_STAGES,
                             512, 256, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const auto sb_map = encode(dsb.pointer, CU_TENSOR_MAP_DATA_TYPE_UINT16, 256, (N / 128) * K_STAGES,
                             512, 256, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const auto c_map = encode(dc.pointer, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, N, M, N * 2,
                            64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  CUDA_CHECK(cudaFuncSetAttribute(blockscale_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  DYNAMIC_SMEM_BYTES));
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, blockscale_gemm, THREADS,
                                                         DYNAMIC_SMEM_BYTES));
  if (!active) throw std::runtime_error("kernel has no resident CTAs on this GPU");
  const unsigned ctas = config.ctas ? config.ctas :
      static_cast<unsigned>(std::max(1, std::min(TOTAL_TILES, active * properties.multiProcessorCount)));
  std::cout << "Persistent grid: " << ctas << " CTAs\n";
  auto launch = [&] {
    blockscale_gemm<<<ctas, THREADS, DYNAMIC_SMEM_BYTES>>>(a_map, b_map, sa_map, sb_map, c_map);
    CUDA_CHECK(cudaGetLastError());
  };
  auto verify = [&] {
    CUDA_CHECK(cudaMemcpy(got.data(), dc.pointer, got.size() * 2, cudaMemcpyDeviceToHost));
    std::size_t errors = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != expected[i]) {
        if (errors < 5)
          std::cerr << "C(" << i / N << ',' << i % N << "): expected 0x" << std::hex
                    << expected[i] << ", got 0x" << got[i] << std::dec << '\n';
        ++errors;
      }
    }
    if (errors) throw std::runtime_error(std::to_string(errors) + " FP16 bit mismatches");
  };
  launch();
  verify();
  std::cout << "Numeric check: all " << M * N << " FP16 output bit patterns exact ("
            << nonzero << " nonzero FP32 accumulators), " << K_GROUPS << " K-blocks\n";
  for (unsigned i = 0; i < config.warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> times;
  times.reserve(config.samples);
  for (unsigned i = 0; i < config.samples; ++i) {
    Event start, end;
    CUDA_CHECK(cudaEventRecord(start.value));
    launch();
    CUDA_CHECK(cudaEventRecord(end.value));
    CUDA_CHECK(cudaEventSynchronize(end.value));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, end.value));
    times.push_back(ms);
  }
  verify();
  std::sort(times.begin(), times.end());
  const float median = percentile(times, 1, 2), p10 = percentile(times, 1, 10), p90 = percentile(times, 9, 10);
  const double tflops = 2.0 * M * N * K / (median * 1.e9);
  std::cout << std::fixed << std::setprecision(6) << "Single-kernel device time: median=" << median
            << " ms, p10=" << p10 << " ms, p90=" << p90 << " ms (" << config.samples
            << " samples, " << config.warmup << " warmups); " << tflops << " TFLOP/s\n";
  if (!config.json.empty()) {
    std::ofstream report(config.json);
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report << std::setprecision(9)
           << "{\n  \"verification\": \"bit-exact-pass\",\n  \"mnk\": [2048, 2048, 1024],\n"
              "  \"compute_capability\": [12, 0],\n  \"ctas\": " << ctas
           << ",\n  \"warmup\": " << config.warmup << ",\n  \"samples\": " << config.samples
           << ",\n  \"median_ms\": " << median << ",\n  \"p10_ms\": " << p10
           << ",\n  \"p90_ms\": " << p90 << ",\n  \"tflops\": " << tflops
           << ",\n  \"checked_outputs\": 4194304,\n  \"replay_verified\": true\n}\n";
    report.close();
  }
  std::cout << "SUCCESS: block-scaled MXFP4 GEMM is bit-exact through C++ instruction primitives\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarmup = 5;
constexpr int kIters = 3;
constexpr int kSamples = 10;
constexpr int kAlgoCount = 256;
constexpr size_t kDefaultWorkspaceBytes = 1ull << 30;

__global__ void fill_half_kernel(__half* data, size_t n, __half value) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value;
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::ostringstream oss;                                                  \
      oss << "CUDA error " << cudaGetErrorString(status) << " at " << __FILE__ \
          << ":" << __LINE__;                                                  \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::ostringstream oss;                                                  \
      oss << "cuBLASLt error " << static_cast<int>(status) << " at "           \
          << __FILE__ << ":" << __LINE__;                                      \
      throw std::runtime_error(oss.str());                                     \
    }                                                                          \
  } while (0)

struct Timing {
  double median_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  double stdev_s = 0.0;
  double tflops = 0.0;
};

struct Row {
  int size = 0;
  int heuristic_rank = 0;
  cublasStatus_t heuristic_state = CUBLAS_STATUS_SUCCESS;
  size_t workspace_bytes = 0;
  float waves_count = 0.0f;
  Timing timing;
};

std::vector<int> default_sizes() {
  return {1024, 2048, 4096, 8192, 16384, 32768};
}

std::string getenv_or(const char* key, const std::string& fallback) {
  const char* value = std::getenv(key);
  return value && value[0] ? std::string(value) : fallback;
}

size_t getenv_size(const char* key, size_t fallback) {
  const char* value = std::getenv(key);
  if (!value || !value[0]) return fallback;
  return static_cast<size_t>(std::strtoull(value, nullptr, 10));
}

std::vector<int> parse_sizes(const std::string& spec) {
  if (spec.empty()) return default_sizes();
  std::vector<int> sizes;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) sizes.push_back(std::stoi(item));
  }
  return sizes.empty() ? default_sizes() : sizes;
}

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

Timing benchmark_algo(cublasLtHandle_t handle, cublasLtMatmulDesc_t op_desc,
                      cublasLtMatrixLayout_t a_desc,
                      cublasLtMatrixLayout_t b_desc,
                      cublasLtMatrixLayout_t c_desc, const void* a,
                      const void* b, void* c,
                      const cublasLtMatmulAlgo_t& algo, void* workspace,
                      size_t workspace_bytes, cudaStream_t stream, int size) {
  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);

  auto launch = [&]() {
    return cublasLtMatmul(handle, op_desc, &alpha, a, a_desc, b, b_desc, &beta,
                          c, c_desc, c, c_desc, &algo, workspace,
                          workspace_bytes, stream);
  };

  for (int i = 0; i < kWarmup; ++i) {
    CHECK_CUBLAS(launch());
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<double> samples;
  samples.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      CHECK_CUBLAS(launch());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto end = std::chrono::steady_clock::now();
    double elapsed_s =
        std::chrono::duration<double>(end - start).count() / kIters;
    samples.push_back(elapsed_s);
  }

  std::sort(samples.begin(), samples.end());
  Timing out;
  out.median_s = samples[samples.size() / 2];
  out.min_s = samples.front();
  out.max_s = samples.back();
  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                static_cast<double>(samples.size());
  double accum = 0.0;
  for (double x : samples) {
    accum += (x - mean) * (x - mean);
  }
  out.stdev_s = samples.size() > 1
                    ? std::sqrt(accum / static_cast<double>(samples.size() - 1))
                    : 0.0;
  double flops = 2.0 * static_cast<double>(size) * size * size;
  out.tflops = flops / out.median_s / 1.0e12;
  return out;
}

void write_rows(const std::string& path, const std::vector<Row>& rows) {
  std::ofstream out(path);
  out << "config,M,N,K,warmup,samples,iters,heuristic_rank,heuristic_state,"
         "workspace_bytes,waves_count,median_s,min_s,max_s,stdev_s,tflops\n";
  for (const Row& row : rows) {
    out << "cublaslt_cpp," << row.size << "," << row.size << "," << row.size
        << "," << kWarmup << "," << kSamples << "," << kIters << ","
        << row.heuristic_rank << "," << static_cast<int>(row.heuristic_state)
        << "," << row.workspace_bytes << "," << row.waves_count << ","
        << row.timing.median_s << "," << row.timing.min_s << ","
        << row.timing.max_s << "," << row.timing.stdev_s << ","
        << row.timing.tflops << "\n";
  }
}

std::vector<Row> best_rows(const std::vector<Row>& rows) {
  std::vector<Row> best;
  for (const Row& row : rows) {
    auto it = std::find_if(best.begin(), best.end(),
                           [&](const Row& x) { return x.size == row.size; });
    if (it == best.end()) {
      best.push_back(row);
    } else if (row.timing.tflops > it->timing.tflops) {
      *it = row;
    }
  }
  std::sort(best.begin(), best.end(),
            [](const Row& a, const Row& b) { return a.size < b.size; });
  return best;
}

}  // namespace

int main() {
  try {
    std::string result_dir = getenv_or("RESULTS_DIR", ".");
    std::vector<int> sizes = parse_sizes(getenv_or("CUBLASLT_CPP_SIZES", ""));
    size_t workspace_bytes =
        getenv_size("CUBLASLT_WORKSPACE_BYTES", kDefaultWorkspaceBytes);
    int max_algos = static_cast<int>(
        getenv_size("CUBLASLT_MAX_ALGOS", static_cast<size_t>(kAlgoCount)));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasLtHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    void* workspace = nullptr;
    if (workspace_bytes > 0) {
      CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    }

    std::vector<Row> rows;
    std::cout << "=== cuBLASLt C++ GEMM sanity ===\n";
    std::cout << "warmup=" << kWarmup << " samples=" << kSamples
              << " iters=" << kIters
              << " workspace_bytes=" << workspace_bytes << "\n";

    for (int size : sizes) {
      size_t elems = static_cast<size_t>(size) * size;
      size_t bytes = elems * sizeof(__half);
      __half* a = nullptr;
      __half* b = nullptr;
      __half* c = nullptr;
      CHECK_CUDA(cudaMalloc(&a, bytes));
      CHECK_CUDA(cudaMalloc(&b, bytes));
      CHECK_CUDA(cudaMalloc(&c, bytes));
      int threads = 256;
      int blocks = static_cast<int>((elems + threads - 1) / threads);
      fill_half_kernel<<<blocks, threads, 0, stream>>>(a, elems,
                                                       __float2half(1.0f));
      fill_half_kernel<<<blocks, threads, 0, stream>>>(b, elems,
                                                       __float2half(1.0f));
      CHECK_CUDA(cudaMemset(c, 0, bytes));
      CHECK_CUDA(cudaStreamSynchronize(stream));

      cublasLtMatmulDesc_t op_desc = nullptr;
      cublasLtMatrixLayout_t a_desc = nullptr;
      cublasLtMatrixLayout_t b_desc = nullptr;
      cublasLtMatrixLayout_t c_desc = nullptr;
      cublasLtMatmulPreference_t pref = nullptr;

      CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_16F,
                                            CUDA_R_16F));
      cublasOperation_t trans = CUBLAS_OP_N;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, size, size,
                                              size));
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, size, size,
                                              size));
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, size, size,
                                              size));
      set_row_major(a_desc);
      set_row_major(b_desc);
      set_row_major(c_desc);

      CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
      CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
          pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
          sizeof(workspace_bytes)));

      std::vector<cublasLtMatmulHeuristicResult_t> heuristics(max_algos);
      int returned = 0;
      CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
          handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, max_algos,
          heuristics.data(), &returned));
      std::cout << "size=" << size << " returned_algorithms=" << returned
                << "\n";

      for (int i = 0; i < returned; ++i) {
        const auto& h = heuristics[i];
        if (h.state != CUBLAS_STATUS_SUCCESS ||
            h.workspaceSize > workspace_bytes) {
          continue;
        }
        Row row;
        row.size = size;
        row.heuristic_rank = i;
        row.heuristic_state = h.state;
        row.workspace_bytes = h.workspaceSize;
        row.waves_count = h.wavesCount;
        try {
          row.timing =
              benchmark_algo(handle, op_desc, a_desc, b_desc, c_desc, a, b, c,
                             h.algo, workspace, workspace_bytes, stream, size);
          rows.push_back(row);
          std::cout << "  rank=" << i << " " << row.timing.tflops
                    << " TFLOP/s (" << row.timing.median_s * 1e6
                    << " us, workspace=" << row.workspace_bytes
                    << ", waves=" << row.waves_count << ")\n";
        } catch (const std::exception& e) {
          std::cout << "  rank=" << i << " failed: " << e.what() << "\n";
        }
      }

      CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
      CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
      CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
      CHECK_CUDA(cudaFree(c));
      CHECK_CUDA(cudaFree(b));
      CHECK_CUDA(cudaFree(a));
    }

    std::string all_path = result_dir + "/gemm_cublaslt_cpp_algos.csv";
    std::string best_path = result_dir + "/gemm_cublaslt_cpp_best.csv";
    write_rows(all_path, rows);
    write_rows(best_path, best_rows(rows));
    std::cout << "all algorithms CSV: " << all_path << "\n";
    std::cout << "best algorithms CSV: " << best_path << "\n";

    if (workspace) CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Measure an empirical local device-memory bandwidth reference.

The default path measures local memory movement with an SM-issued
vectorized copy kernel. Copy traffic is counted as one full-buffer read
plus one full-buffer write. The helper sweeps a small set of launch
geometries and reports the best median bandwidth as the empirical
data-movement reference.

The elemwise add benchmark reports useful/requested bandwidth as
  (2 reads + 1 write) * N * sizeof(f16) / kernel_time.
The CUDA f16-add reference remains available as --method cuda-f16-add
for workload-shaped comparisons, and the D2D cudaMemcpy reference
remains available as --method cuda-d2d-memcpy.

This deliberately avoids a DRAM-spec fallback: if no measurement path is
available, the script fails unless MEM_ROOFLINE_GB_S or
ELEM_MEM_ROOFLINE_GB_S provides an externally measured value.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def _numbers_from_table(text: str) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        lower = line.lower()
        if not line.strip():
            continue
        if "coefficient" in lower or "note:" in lower:
            continue
        if lower.startswith("sum "):
            continue
        if not re.match(r"^\s*(?:\d+|gpu|cpu|n/a|-)", lower):
            continue
        for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", line):
            try:
                v = float(token)
            except ValueError:
                continue
            if v > 0:
                values.append(v)
    return values


def _bandwidths_from_json(obj) -> list[float]:
    values: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if isinstance(v, (int, float)) and ("bandwidth" in lk or lk in {"gb/s", "gbps"}):
                if v > 0:
                    values.append(float(v))
            else:
                values.extend(_bandwidths_from_json(v))
    elif isinstance(obj, list):
        for v in obj:
            values.extend(_bandwidths_from_json(v))
    return values


def _parse_nvbandwidth(stdout: str, stderr: str) -> float | None:
    text = stdout + "\n" + stderr
    try:
        values = _bandwidths_from_json(json.loads(stdout))
        if values:
            return max(values)
    except json.JSONDecodeError:
        pass

    # Plain output is usually a matrix of GB/s numbers. For a single visible
    # GPU this is the local measurement; for multi-GPU runs, using max cell is
    # intentionally more conservative than SUM, which is aggregate fabric BW.
    values = _numbers_from_table(text)
    if values:
        return max(values)
    return None


def _parse_bandwidth_test(stdout: str, stderr: str) -> float | None:
    text = stdout + "\n" + stderr
    vals = []
    for m in re.finditer(r"Bandwidth\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*GB/s", text):
        vals.append(float(m.group(1)))
    if vals:
        return max(vals)

    # CSV/table fallback: keep D2D/device rows only.
    for line in text.splitlines():
        lower = line.lower()
        if "d2d" not in lower and "device to device" not in lower:
            continue
        for token in re.findall(r"([0-9]+(?:\.[0-9]+)?)", line):
            vals.append(float(token))
    return max(vals) if vals else None


CUDA_F16_ADD_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                           \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                   __FILE__, __LINE__, cudaGetErrorString(err__));      \
      return 1;                                                         \
    }                                                                   \
  } while (0)

__global__ void add_f16_kernel(const __half2* __restrict__ x,
                               const __half2* __restrict__ y,
                               __half2* __restrict__ z,
                               size_t n2) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = tid; i < n2; i += stride) {
    z[i] = __hadd2(x[i], y[i]);
  }
}

static size_t parse_size(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') {
    std::fprintf(stderr, "invalid integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<size_t>(v);
}

static int parse_int(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v <= 0) {
    std::fprintf(stderr, "invalid positive integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<int>(v);
}

int main(int argc, char** argv) {
  size_t n = 268435456ULL;
  int samples = 25;
  int warmup = 20;
  int iters = 3;
  int block = 256;
  int cta_per_sm = 32;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n = parse_size(argv[++i]);
    } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
      samples = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
      block = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--cta-per-sm") == 0 && i + 1 < argc) {
      cta_per_sm = parse_int(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--n N] [--samples S] [--warmup W] "
                   "[--iters I] [--block B] [--cta-per-sm C]\n",
                   argv[0]);
      return 2;
    }
  }

  if (n < 2) {
    std::fprintf(stderr, "N must be at least 2\n");
    return 2;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  size_t n2 = (n + 1) / 2;
  size_t alloc_bytes = n2 * sizeof(__half2);
  double useful_bytes = 3.0 * static_cast<double>(n) * sizeof(__half);

  __half2* x = nullptr;
  __half2* y = nullptr;
  __half2* z = nullptr;
  CHECK_CUDA(cudaMalloc(&x, alloc_bytes));
  CHECK_CUDA(cudaMalloc(&y, alloc_bytes));
  CHECK_CUDA(cudaMalloc(&z, alloc_bytes));
  CHECK_CUDA(cudaMemset(x, 1, alloc_bytes));
  CHECK_CUDA(cudaMemset(y, 2, alloc_bytes));
  CHECK_CUDA(cudaMemset(z, 0, alloc_bytes));

  int blocks = prop.multiProcessorCount * cta_per_sm;
  size_t enough_blocks = (n2 + static_cast<size_t>(block) - 1) /
                         static_cast<size_t>(block);
  if (enough_blocks < static_cast<size_t>(blocks)) {
    blocks = static_cast<int>(enough_blocks);
  }
  blocks = std::max(blocks, 1);

  for (int i = 0; i < warmup; ++i) {
    add_f16_kernel<<<blocks, block>>>(x, y, z, n2);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> sample_us;
  sample_us.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
      add_f16_kernel<<<blocks, block>>>(x, y, z, n2);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    sample_us.push_back(ms * 1000.0f / static_cast<float>(iters));
  }

  std::sort(sample_us.begin(), sample_us.end());
  double median_us = sample_us[sample_us.size() / 2];
  double gb_s = useful_bytes / (median_us * 1.0e-6) / 1.0e9;

  std::printf("gb_per_s=%.6f median_us=%.6f n=%zu useful_bytes=%.0f "
              "samples=%d iters=%d blocks=%d block=%d device=\"%s\"\n",
              gb_s, median_us, n, useful_bytes, samples, iters, blocks,
              block, prop.name);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaFree(z));
  return 0;
}
"""


CUDA_D2D_MEMCPY_SOURCE = r"""
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                           \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                   __FILE__, __LINE__, cudaGetErrorString(err__));      \
      return 1;                                                         \
    }                                                                   \
  } while (0)

static size_t parse_size(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') {
    std::fprintf(stderr, "invalid integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<size_t>(v);
}

static int parse_int(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v <= 0) {
    std::fprintf(stderr, "invalid positive integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<int>(v);
}

int main(int argc, char** argv) {
  size_t copy_bytes = 536870912ULL;
  int samples = 25;
  int warmup = 20;
  int iters = 10;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      copy_bytes = parse_size(argv[++i]);
    } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
      samples = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = parse_int(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--bytes B] [--samples S] "
                   "[--warmup W] [--iters I]\n",
                   argv[0]);
      return 2;
    }
  }

  if (copy_bytes == 0) {
    std::fprintf(stderr, "copy size must be nonzero\n");
    return 2;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  void* src = nullptr;
  void* dst = nullptr;
  CHECK_CUDA(cudaMalloc(&src, copy_bytes));
  CHECK_CUDA(cudaMalloc(&dst, copy_bytes));
  CHECK_CUDA(cudaMemset(src, 1, copy_bytes));
  CHECK_CUDA(cudaMemset(dst, 0, copy_bytes));

  cudaStream_t stream{};
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  for (int i = 0; i < warmup; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, copy_bytes,
                               cudaMemcpyDeviceToDevice, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> sample_us;
  sample_us.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      CHECK_CUDA(cudaMemcpyAsync(dst, src, copy_bytes,
                                 cudaMemcpyDeviceToDevice, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    sample_us.push_back(ms * 1000.0f / static_cast<float>(iters));
  }

  std::sort(sample_us.begin(), sample_us.end());
  double median_us = sample_us[sample_us.size() / 2];
  double traffic_bytes = 2.0 * static_cast<double>(copy_bytes);
  double gb_s = traffic_bytes / (median_us * 1.0e-6) / 1.0e9;

  std::printf("gb_per_s=%.6f median_us=%.6f copy_bytes=%zu "
              "traffic_bytes=%.0f samples=%d iters=%d device=\"%s\"\n",
              gb_s, median_us, copy_bytes, traffic_bytes, samples, iters,
              prop.name);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(src));
  CHECK_CUDA(cudaFree(dst));
  return 0;
}
"""


CUDA_VECTOR_COPY_SOURCE = r"""
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do {                                           \
    cudaError_t err__ = (call);                                         \
    if (err__ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                   __FILE__, __LINE__, cudaGetErrorString(err__));      \
      return 1;                                                         \
    }                                                                   \
  } while (0)

static size_t parse_size(const char* s) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') {
    std::fprintf(stderr, "invalid integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<size_t>(v);
}

static int parse_int(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v <= 0) {
    std::fprintf(stderr, "invalid positive integer: %s\n", s);
    std::exit(2);
  }
  return static_cast<int>(v);
}

template <int UNROLL>
__global__ void copy_u128_kernel(const uint4* __restrict__ src,
                                 uint4* __restrict__ dst,
                                 size_t n_vec) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  size_t base = tid * UNROLL;
  size_t step = stride * UNROLL;

  for (size_t i = base; i < n_vec; i += step) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      size_t j = i + u;
      if (j < n_vec) {
        uint4 v = src[j];
        dst[j] = v;
      }
    }
  }
}

static void launch_copy(const uint4* src,
                        uint4* dst,
                        size_t n_vec,
                        int blocks,
                        int block,
                        int unroll,
                        cudaStream_t stream) {
  switch (unroll) {
    case 1:
      copy_u128_kernel<1><<<blocks, block, 0, stream>>>(src, dst, n_vec);
      break;
    case 2:
      copy_u128_kernel<2><<<blocks, block, 0, stream>>>(src, dst, n_vec);
      break;
    case 4:
      copy_u128_kernel<4><<<blocks, block, 0, stream>>>(src, dst, n_vec);
      break;
    case 8:
      copy_u128_kernel<8><<<blocks, block, 0, stream>>>(src, dst, n_vec);
      break;
    default:
      std::fprintf(stderr, "unsupported unroll %d; use 1, 2, 4, or 8\n", unroll);
      std::exit(2);
  }
}

int main(int argc, char** argv) {
  size_t copy_bytes = 536870912ULL;
  int samples = 25;
  int warmup = 20;
  int iters = 10;
  int block = 512;
  int cta_per_sm = 32;
  int unroll = 4;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      copy_bytes = parse_size(argv[++i]);
    } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
      samples = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
      block = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--cta-per-sm") == 0 && i + 1 < argc) {
      cta_per_sm = parse_int(argv[++i]);
    } else if (std::strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) {
      unroll = parse_int(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--bytes B] [--samples S] [--warmup W] "
                   "[--iters I] [--block B] [--cta-per-sm C] [--unroll U]\n",
                   argv[0]);
      return 2;
    }
  }

  copy_bytes = (copy_bytes / sizeof(uint4)) * sizeof(uint4);
  if (copy_bytes == 0) {
    std::fprintf(stderr, "copy size must be at least %zu bytes\n", sizeof(uint4));
    return 2;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  size_t n_vec = copy_bytes / sizeof(uint4);
  double traffic_bytes = 2.0 * static_cast<double>(copy_bytes);

  uint4* src = nullptr;
  uint4* dst = nullptr;
  CHECK_CUDA(cudaMalloc(&src, copy_bytes));
  CHECK_CUDA(cudaMalloc(&dst, copy_bytes));
  CHECK_CUDA(cudaMemset(src, 1, copy_bytes));
  CHECK_CUDA(cudaMemset(dst, 0, copy_bytes));

  int blocks = prop.multiProcessorCount * cta_per_sm;
  size_t enough_blocks = (n_vec + static_cast<size_t>(block) - 1) /
                         static_cast<size_t>(block);
  if (enough_blocks < static_cast<size_t>(blocks)) {
    blocks = static_cast<int>(enough_blocks);
  }
  blocks = std::max(blocks, 1);

  cudaStream_t stream{};
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  for (int i = 0; i < warmup; ++i) {
    launch_copy(src, dst, n_vec, blocks, block, unroll, stream);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> sample_us;
  sample_us.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      launch_copy(src, dst, n_vec, blocks, block, unroll, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    sample_us.push_back(ms * 1000.0f / static_cast<float>(iters));
  }

  std::sort(sample_us.begin(), sample_us.end());
  double median_us = sample_us[sample_us.size() / 2];
  double gb_s = traffic_bytes / (median_us * 1.0e-6) / 1.0e9;

  std::printf("gb_per_s=%.6f median_us=%.6f copy_bytes=%zu "
              "traffic_bytes=%.0f samples=%d iters=%d blocks=%d block=%d "
              "cta_per_sm=%d unroll=%d device=\"%s\"\n",
              gb_s, median_us, copy_bytes, traffic_bytes, samples, iters,
              blocks, block, cta_per_sm, unroll, prop.name);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(src));
  CHECK_CUDA(cudaFree(dst));
  return 0;
}
"""


def _find_bin(env_name: str, names: list[str], candidates: list[str] | None = None) -> str | None:
    override = os.environ.get(env_name)
    if override:
        return override
    for name in names:
        p = shutil.which(name)
        if p:
            return p
    for p in candidates or []:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return None


def measure_nvbandwidth(timeout: int) -> tuple[float, str, str] | None:
    exe = _find_bin("NVBANDWIDTH_BIN", ["nvbandwidth"])
    if not exe:
        return None

    buf = os.environ.get("NVBANDWIDTH_BUFFER_MIB", "1024")
    samples = os.environ.get("NVBANDWIDTH_SAMPLES", "5")
    tests = os.environ.get(
        "NVBANDWIDTH_TESTS",
        "device_to_device_memcpy_read_sm,"
        "device_to_device_memcpy_write_sm",
    ).split(",")

    for test in [t.strip() for t in tests if t.strip()]:
        cmd = [exe, "-t", test, "-b", buf, "-i", samples, "-j"]
        rc, out, err = _run(cmd, timeout)
        if rc != 0:
            cmd = [exe, "-t", test, "-b", buf, "-i", samples]
            rc, out, err = _run(cmd, timeout)
        bw = _parse_nvbandwidth(out, err)
        if rc == 0 and bw:
            return bw, f"nvbandwidth:{test}", " ".join(cmd)
    return None


def measure_bandwidth_test(timeout: int) -> tuple[float, str, str] | None:
    exe = _find_bin("BANDWIDTHTEST_BIN", ["bandwidthTest"], [
        "/usr/local/cuda/extras/demo_suite/bandwidthTest",
        "/usr/local/cuda-13.2/extras/demo_suite/bandwidthTest",
        "/usr/local/cuda-13.1/extras/demo_suite/bandwidthTest",
        "/usr/local/cuda-13.0/extras/demo_suite/bandwidthTest",
        "/usr/local/cuda-12.8/extras/demo_suite/bandwidthTest",
    ])
    if not exe:
        return None
    cmd = [exe, "--mode=quick", "--dtod", "--csv"]
    rc, out, err = _run(cmd, timeout)
    bw = _parse_bandwidth_test(out, err)
    if rc == 0 and bw:
        return bw, "bandwidthTest:D2D", " ".join(cmd)
    return None


def _parse_cuda_add(stdout: str, stderr: str) -> float | None:
    m = re.search(r"\bgb_per_s=([0-9]+(?:\.[0-9]+)?)", stdout + "\n" + stderr)
    return float(m.group(1)) if m else None


DEFAULT_VECTOR_CONFIGS = (
    "256,16,4 256,32,4 256,64,4 "
    "512,16,4 512,32,4 512,64,4 "
    "1024,8,4 1024,16,4 1024,32,4 "
    "256,32,8 512,32,8 1024,16,8"
)


def _parse_vector_configs(text: str) -> list[tuple[int, int, int]]:
    configs: list[tuple[int, int, int]] = []
    for part in text.replace(";", " ").split():
        fields = part.split(",")
        if len(fields) != 3:
            raise ValueError(f"invalid vector config {part!r}; expected block,cta_per_sm,unroll")
        block, cta_per_sm, unroll = (int(x) for x in fields)
        if block <= 0 or cta_per_sm <= 0 or unroll not in {1, 2, 4, 8}:
            raise ValueError(f"invalid vector config {part!r}")
        configs.append((block, cta_per_sm, unroll))
    if not configs:
        raise ValueError("at least one vector config is required")
    return configs


def _find_nvcc() -> str | None:
    return _find_bin("NVCC", ["nvcc"], [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
    ])


def _compile_cuda_source(
    source: str,
    source_name: str,
    binary_name: str,
    timeout: int,
) -> Path | None:
    exe = _find_nvcc()
    if not exe:
        return None

    build_dir = Path(os.environ.get("MEM_ROOFLINE_BUILD_DIR", "/tmp"))
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / source_name
    bin_path = build_dir / binary_name
    src.write_text(source)

    compile_cmds: list[list[str]] = []
    arch = os.environ.get("CUDA_ARCH", "native").strip()
    if arch:
        compile_cmds.append([
            exe,
            "-O3",
            "-std=c++17",
            f"-arch={arch}",
            str(src),
            "-o",
            str(bin_path),
        ])
    compile_cmds.append([exe, "-O3", "-std=c++17", str(src), "-o", str(bin_path)])

    compile_errors = []
    for cmd in compile_cmds:
        rc, out, err = _run(cmd, timeout)
        if rc == 0:
            return bin_path
        compile_errors.append((cmd, out, err))

    for cmd, out, err in compile_errors:
        print(f"CUDA compile failed: {' '.join(cmd)}", file=sys.stderr)
        if out.strip():
            print(out.strip(), file=sys.stderr)
        if err.strip():
            print(err.strip(), file=sys.stderr)
    return None


def measure_cuda_vector_copy(
    timeout: int,
    n: int,
    samples: int,
    warmup: int,
    iters: int,
    copy_bytes: int | None,
    vector_configs: str | None,
) -> tuple[float, str, str] | None:
    bin_path = _compile_cuda_source(
        CUDA_VECTOR_COPY_SOURCE,
        "cutile_vector_copy_roofline.cu",
        "cutile_vector_copy_roofline",
        timeout,
    )
    if not bin_path:
        return None

    if copy_bytes is None:
        copy_bytes = int(os.environ.get("MEM_ROOFLINE_COPY_BYTES", str(n * 2)))
    config_text = (
        vector_configs
        or os.environ.get("MEM_ROOFLINE_VECTOR_CONFIGS")
        or DEFAULT_VECTOR_CONFIGS
    )
    try:
        configs = _parse_vector_configs(config_text)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return None

    best: tuple[float, str, str] | None = None
    for block, cta_per_sm, unroll in configs:
        run_cmd = [
            str(bin_path),
            "--bytes",
            str(copy_bytes),
            "--samples",
            str(samples),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--block",
            str(block),
            "--cta-per-sm",
            str(cta_per_sm),
            "--unroll",
            str(unroll),
        ]
        rc, out, err = _run(run_cmd, timeout)
        bw = _parse_cuda_add(out, err)
        if out.strip():
            print(out.strip(), file=sys.stderr)
        if err.strip():
            print(err.strip(), file=sys.stderr)
        if rc == 0 and bw:
            command = " ".join(run_cmd)
            if best is None or bw > best[0]:
                best = (bw, "cuda-vector-copy", command)
    return best


def measure_cuda_d2d_memcpy(
    timeout: int,
    n: int,
    samples: int,
    warmup: int,
    iters: int,
    copy_bytes: int | None,
) -> tuple[float, str, str] | None:
    bin_path = _compile_cuda_source(
        CUDA_D2D_MEMCPY_SOURCE,
        "cutile_d2d_memcpy_roofline.cu",
        "cutile_d2d_memcpy_roofline",
        timeout,
    )
    if not bin_path:
        return None

    if copy_bytes is None:
        copy_bytes = int(os.environ.get("MEM_ROOFLINE_COPY_BYTES", str(n * 2)))
    run_cmd = [
        str(bin_path),
        "--bytes",
        str(copy_bytes),
        "--samples",
        str(samples),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    rc, out, err = _run(run_cmd, timeout)
    bw = _parse_cuda_add(out, err)
    if rc == 0 and bw:
        return bw, "cuda-d2d-memcpy", " ".join(run_cmd)
    if out.strip():
        print(out.strip(), file=sys.stderr)
    if err.strip():
        print(err.strip(), file=sys.stderr)
    return None


def measure_cuda_f16_add(
    timeout: int,
    n: int,
    samples: int,
    warmup: int,
    iters: int,
) -> tuple[float, str, str] | None:
    bin_path = _compile_cuda_source(
        CUDA_F16_ADD_SOURCE,
        "cutile_elemwise_mem_roofline.cu",
        "cutile_elemwise_mem_roofline",
        timeout,
    )
    if not bin_path:
        return None

    run_cmd = [
        str(bin_path),
        "--n",
        str(n),
        "--samples",
        str(samples),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    rc, out, err = _run(run_cmd, timeout)
    bw = _parse_cuda_add(out, err)
    if rc == 0 and bw:
        return bw, "cuda-f16-add", " ".join(run_cmd)
    if out.strip():
        print(out.strip(), file=sys.stderr)
    if err.strip():
        print(err.strip(), file=sys.stderr)
    return None


def write_csv(path: Path, gb_s: float, source: str, command: str) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if needs_header:
            w.writerow(["timestamp", "gb_per_s", "tb_per_s", "source", "command"])
        w.writerow([
            _dt.datetime.now().isoformat(timespec="seconds"),
            f"{gb_s:.6f}",
            f"{gb_s / 1000.0:.6f}",
            source,
            command,
        ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument(
        "--method",
        choices=[
            "cuda-vector-copy",
            "cuda-d2d-memcpy",
            "cuda-f16-add",
            "nvbandwidth",
            "bandwidthTest",
            "auto",
        ],
        default="cuda-vector-copy",
        help="measurement method; default is an SM-issued vectorized copy sweep",
    )
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--n", type=int, default=268435456)
    ap.add_argument(
        "--copy-bytes",
        type=int,
        default=None,
        help="bytes copied per iteration; default is 2*N bytes",
    )
    ap.add_argument(
        "--vector-configs",
        default=None,
        help=(
            "space-separated block,cta_per_sm,unroll triples for "
            "cuda-vector-copy; default is a small built-in sweep"
        ),
    )
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    forced_name = "MEM_ROOFLINE_GB_S"
    forced = os.environ.get(forced_name)
    if not forced:
        forced_name = "ELEM_MEM_ROOFLINE_GB_S"
        forced = os.environ.get(forced_name)
    if forced:
        gb_s = float(forced)
        source = f"env:{forced_name}"
        command = ""
    else:
        methods = {
            "cuda-vector-copy": lambda: measure_cuda_vector_copy(
                args.timeout,
                args.n,
                args.samples,
                args.warmup,
                args.iters,
                args.copy_bytes,
                args.vector_configs,
            ),
            "cuda-d2d-memcpy": lambda: measure_cuda_d2d_memcpy(
                args.timeout,
                args.n,
                args.samples,
                args.warmup,
                args.iters,
                args.copy_bytes,
            ),
            "cuda-f16-add": lambda: measure_cuda_f16_add(
                args.timeout,
                args.n,
                args.samples,
                args.warmup,
                args.iters,
            ),
            "nvbandwidth": lambda: measure_nvbandwidth(args.timeout),
            "bandwidthTest": lambda: measure_bandwidth_test(args.timeout),
        }
        if args.method == "auto":
            result = (
                methods["cuda-vector-copy"]()
                or methods["nvbandwidth"]()
                or methods["cuda-d2d-memcpy"]()
                or methods["bandwidthTest"]()
            )
        else:
            result = methods[args.method]()
        if result:
            gb_s, source, command = result
        else:
            print(
                "error: no measured local memory-bandwidth reference was available. "
                "Provide nvcc for the local CUDA vector-copy reference, install "
                "nvbandwidth or CUDA bandwidthTest, run with a different "
                "--method, or set "
                "MEM_ROOFLINE_GB_S to an externally measured GB/s value.",
                file=sys.stderr,
            )
            return 1

    if args.csv:
        write_csv(args.csv, gb_s, source, command)

    print(f"{gb_s:.6f}")
    print(f"memory reference: {gb_s:.2f} GB/s ({source})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

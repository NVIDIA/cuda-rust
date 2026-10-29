// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(__CUDACC__)
#define BLOCKSCALE_HD __host__ __device__
#else
#define BLOCKSCALE_HD
#endif

namespace blockscale {
constexpr int M = 2048, N = 2048, K = 1024;
constexpr int K_GROUPS = K / 32, K_STAGES = K / 128;
constexpr int THREADS = 288, TOTAL_TILES = (M / 128) * (N / 128);
constexpr int STAGES = 3, DATA_BYTES = 8192, SCALE_BYTES = 512;
constexpr int B_OFFSET = 24576, SFA_OFFSET = 49152, SFB_OFFSET = 50688;
constexpr int BARRIERS = 52224, OUTPUT = 53248, DYNAMIC_SMEM_BYTES = 86016;
constexpr unsigned TX_BYTES = 2 * DATA_BYTES + 2 * SCALE_BYTES;
static_assert(B_OFFSET == STAGES * DATA_BYTES);
static_assert(SFA_OFFSET == B_OFFSET + STAGES * DATA_BYTES);
static_assert(SFB_OFFSET == SFA_OFFSET + STAGES * SCALE_BYTES);
static_assert(BARRIERS == SFB_OFFSET + STAGES * SCALE_BYTES);
static_assert(OUTPUT >= BARRIERS + 2 * STAGES * 8);
static_assert(OUTPUT % 1024 == 0 && OUTPUT + 2 * 16384 == DYNAMIC_SMEM_BYTES);
static_assert(DATA_BYTES % 512 == 0 && SCALE_BYTES % 128 == 0);

BLOCKSCALE_HD constexpr int scale_word(int row) {
  return (row & 31) * 4 + ((row >> 5) & 3);
}
BLOCKSCALE_HD constexpr int scale_offset(int row, int group) {
  return (row / 128) * (K_GROUPS / 4) * 512 + (group / 4) * 512 +
         scale_word(row) * 4 + group % 4;
}
BLOCKSCALE_HD constexpr int matrix_row(int atom, int half, int lane) {
  int sub = lane / 8;
  int row = 16 * atom + 8 * (sub % 2) + lane % 8;
  int column = 16 * half + 8 * (sub / 2);
  int plain = row * 64 + column * 2;
  return plain ^ ((plain & 0x180) >> 3);
}
BLOCKSCALE_HD constexpr int n_offset(int slot) {
  return 32 * (slot / 2) + 8 * (slot % 2);
}
// Return FP16 element offsets; matrix_row returns byte offsets.
BLOCKSCALE_HD constexpr int output_element(int row, int column) {
  int plain = row * 64 + column % 64;
  return (column / 64) * 8192 + (plain ^ ((plain & 0x1c0) >> 3));
}
BLOCKSCALE_HD constexpr int output_row(int warp, int lane, int band, int slot) {
  return output_element(16 * (warp % 4) + 64 * band + (lane & 15),
                        16 * (warp / 4) + n_offset(slot));
}
}  // namespace blockscale
#undef BLOCKSCALE_HD

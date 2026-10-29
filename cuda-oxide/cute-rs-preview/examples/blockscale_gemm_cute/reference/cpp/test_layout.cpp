// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "layout.cuh"

#include <array>
#include <cassert>
#include <iostream>

int main() {
  using namespace blockscale;
  // The complete scale tensor must be a permutation of the dense input bytes,
  // including boundaries between 128-row tiles and four-K-group atoms.
  std::array<bool, M * (K / 32)> scales{};
  for (int row = 0; row < M; ++row) {
    for (int group = 0; group < K / 32; ++group) {
      auto offset = scale_offset(row, group);
      assert(offset >= 0 && static_cast<std::size_t>(offset) < scales.size() && !scales[offset]);
      scales[offset] = true;
    }
  }
  for (bool visited : scales) assert(visited);
  assert(scale_offset(0, 0) == 0);
  assert(scale_offset(32, 0) == 4);
  assert(scale_offset(128, 0) == 4096);

  // Each ldmatrix address provider names one distinct aligned 16-byte row.
  for (int half = 0; half < 2; ++half) {
    std::array<bool, DATA_BYTES / 16> matrix{};
    for (int atom = 0; atom < 8; ++atom) {
      for (int lane = 0; lane < 32; ++lane) {
        auto offset = matrix_row(atom, half, lane);
        assert(offset < DATA_BYTES && offset % 16 == 0);
        assert(!matrix[offset / 16]);
        matrix[offset / 16] = true;
      }
    }
  }

  // Independently enumerate the PTX accumulator's logical (row,column)
  // ownership and ensure the output swizzle neither aliases nor loses values.
  std::array<bool, 128 * 128> output{};
  constexpr int columns[] = {0, 8, 32, 40, 64, 72, 96, 104};
  for (int warp = 0; warp < 8; ++warp) {
    for (int lane = 0; lane < 32; ++lane) {
      for (int band = 0; band < 2; ++band) {
        for (int slot = 0; slot < 8; ++slot) {
          for (int reg = 0; reg < 4; ++reg) {
            int row = 16 * (warp % 4) + 64 * band + lane / 4 + 8 * (reg / 2);
            int col = 16 * (warp / 4) + columns[slot] + 2 * (lane % 4) + reg % 2;
            assert(row < 128 && col < 128);
            auto offset = output_element(row, col);
            assert(offset >= 0 && static_cast<std::size_t>(offset) < output.size() && !output[offset]);
            output[offset] = true;
          }
          int provider_row = 16 * (warp % 4) + 64 * band + (lane & 15);
          int provider_col = 16 * (warp / 4) + columns[slot];
          assert(output_row(warp, lane, band, slot) ==
                 output_element(provider_row, provider_col));
        }
      }
    }
  }
  for (bool visited : output) assert(visited);
  std::cout << "PASS physical scale, matrix, and accumulator mappings\n";
}

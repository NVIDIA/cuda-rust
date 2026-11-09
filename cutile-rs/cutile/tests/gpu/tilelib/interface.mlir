// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Interface of the static test tile library: the declarations that
// `tileiras -l` splices into a kernel that calls them.
cuda_tile.module @cutile_test_lib {
  preview$func extern @add_one(
      %x: tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>}
    ) -> (tile<32xf32>)
  preview$func extern @add_n<[N: index]>(
      %x: tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>}
    ) -> (tile<32xf32>)
  preview$func extern @offset_ids(
      %base: tile<32xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
      %ids: tile<32xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> (tile<32xi32>)
  preview$func extern @add_scalar(
      %x: tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>},
      %s: tile<f32> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>}
    ) -> (tile<32xf32>)
  preview$func extern @twice(
      %s: tile<i32> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
      %r: tile<i32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> (tile<i32>)
  preview$func extern @lane_ids(
      %ids: tile<32xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> (tile<32xi32>)
}

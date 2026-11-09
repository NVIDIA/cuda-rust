// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Implementation catalog of the static test tile library: each entry names
// the SIMT symbol in test_lib.cu that implements a declaration of
// interface.mlir, and how the tile is spread over the threads that call it.
cuda_tile.module @cutile_test_lib {
  // One element per thread, in registers, for 32 threads; no synchronization.
  preview$extern_impl @cutile_test_add_one implements @add_one(
      %x: preview$fragment<1xf32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
  // `add_n` is implemented only for N = 2; the SIMT function takes no
  // constant arguments (`<[?]>`).
  preview$extern_impl @cutile_test_add_two<[?]> implements @add_n<[2]>(
      %x: preview$fragment<1xf32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
  preview$extern_impl @cutile_test_offset_ids implements @offset_ids(
      %base: preview$fragment<1xi32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
      %ids: preview$fragment<1xi32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
  // A rank-0 tile is one element that every thread holds.
  preview$extern_impl @cutile_test_add_scalar implements @add_scalar(
      %x: preview$fragment<1xf32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>},
      %s: preview$fragment<1xf32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<in>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
  preview$extern_impl @cutile_test_twice implements @twice(
      %s: preview$fragment<1xi32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
      %r: preview$fragment<1xi32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
  preview$extern_impl @cutile_test_lane_ids implements @lane_ids(
      %ids: preview$fragment<1xi32, local, mem_layout = <(1):(1)>, fragmap = <(1, 32, 1):(0, 1, 0)>>
        {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
    ) -> ()
    synchronization = #cuda_tile<preview$synchronization<none>>
}

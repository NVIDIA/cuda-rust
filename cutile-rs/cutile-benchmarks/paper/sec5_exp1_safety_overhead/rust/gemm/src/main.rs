/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GEMM benchmark for cuTile Rust — paper Exp 1.
//
// Mirrors `gemm_python.py` measurement methodology exactly so both produce
// comparable numbers:
//   - fixed N_WARMUP launches
//   - N_SAMPLES samples, each wrapping fixed ITERS launches between two
//     stream syncs, timed by Instant
//   - median of per-launch times reported
//
// Kernels are verbatim copies of cutile-benchmarks/benches/gemm.rs (the
// source of the paper's Rust baseline); keeping them here locks the paper
// to a stable reference independent of upstream changes.

use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, IntoResult, Stream};
use cutile::api;
use cutile::core::f16;
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, PartitionOp, TileKernel};
use kernels::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    unsafe fn get_tensor_2d<T: ElementType>(
        ptr: *mut T,
        dim_0: i32,
        dim_1: i32,
        stride_0: i32,
    ) -> Tensor<T, { [-1, -1] }> {
        let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[dim_0, dim_1],
        };
        let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[stride_0] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    fn ceil_div_i32(a: i32, b: i32) -> i32 {
        (a + b - 1i32) / b
    }

    fn swizzled_bid(tile_id: i32, num_bid_m: i32, num_bid_n: i32, group_size_m: i32) -> (i32, i32) {
        let num_bid_in_group = group_size_m * num_bid_n;
        let group_id = tile_id / num_bid_in_group;
        let first_bid_m = group_id * group_size_m;
        let remaining_m = num_bid_m - first_bid_m;
        let actual_group_size_m = if remaining_m < group_size_m {
            remaining_m
        } else {
            group_size_m
        };
        let bid_m = first_bid_m + (tile_id % actual_group_size_m);
        let bid_n = (tile_id % num_bid_in_group) / actual_group_size_m;
        (bid_m, bid_n)
    }

    fn linear_bid(tile_id: i32, num_bid_n: i32) -> (i32, i32) {
        let bid_m = tile_id / num_bid_n;
        let bid_n = tile_id % num_bid_n;
        (bid_m, bid_n)
    }

    #[cutile::entry(unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    unsafe fn gemm<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry(
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
            sm_100 = (num_cta_in_cga = 2,),
        )
    )]
    fn gemm_persistent<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let k = Dim::new(x.shape()[1] / BK);

        let part_x = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let part_y = y.partition(const_shape![BK, BN]).with_bounds((k, n));

        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in k {
                let tile_x = part_x.load(coord((bid_m, k_tile)));
                let tile_y = part_y.load(coord((k_tile, bid_n)));
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, out_idx);
        }
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
            sm_100 = (num_cta_in_cga = 2,),
        )
    )]
    unsafe fn gemm_persistent_unchecked<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let k = Dim::new(x.shape()[1] / BK);

        let part_x = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let part_y = y.partition(const_shape![BK, BN]).with_bounds((k, n));

        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in k {
                let tile_x = part_x.load(coord((bid_m, k_tile)));
                let tile_y = part_y.load(coord((k_tile, bid_n)));
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, out_idx);
        }
    }

    #[cutile::entry(unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    unsafe fn gemm_persistent_raw<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const SWIZZLE: i32,
    >(
        z_ptr: *mut T,
        z_dim_0: i32,
        z_dim_1: i32,
        z_stride_0: i32,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        m: i32,
        n: i32,
        k: i32,
        group_size_m: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let num_bid_m = ceil_div_i32(m, BM);
        let num_bid_n = ceil_div_i32(n, BN);
        let total_tiles = num_bid_m * num_bid_n;
        let k_tiles = ceil_div_i32(k, BK);

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let mut z = get_tensor_2d(z_ptr, z_dim_0, z_dim_1, z_stride_0);
        let mut part_z = z.partition_mut(const_shape![BM, BN]);

        for tile_id in (pid.0..total_tiles).step_by(grid.0 as usize) {
            let (bid_m, bid_n) = if SWIZZLE != 0i32 {
                swizzled_bid(tile_id, num_bid_m, num_bid_n, group_size_m)
            } else {
                linear_bid(tile_id, num_bid_n)
            };
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for i in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, i]);
                let tile_y = part_y.load([i, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            unsafe { part_z.store(tile_z, [bid_m, bid_n]) };
        }
    }

    #[cutile::entry(unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    #[allow(unreachable_code, unused_mut, unused_variables)]
    unsafe fn gemm_persistent_raw_asserts<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const SWIZZLE: i32,
    >(
        z_ptr: *mut T,
        z_dim_0: i32,
        z_dim_1: i32,
        z_stride_0: i32,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        m: i32,
        n: i32,
        k: i32,
        group_size_m: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        let num_bid_m = ceil_div_i32(m, BM);
        let num_bid_n = ceil_div_i32(n, BN);
        let total_tiles = num_bid_m * num_bid_n;
        let k_tiles = ceil_div_i32(k, BK);

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let mut z = get_tensor_2d(z_ptr, z_dim_0, z_dim_1, z_stride_0);
        let mut part_z = z.partition_mut(const_shape![BM, BN]);

        for tile_id in (pid.0..total_tiles).step_by(grid.0 as usize) {
            cuda_tile_assert!(num_bid_m > 0i32, "raw asserts requires num_bid_m > 0");
            cuda_tile_assert!(num_bid_n > 0i32, "raw asserts requires num_bid_n > 0");
            cuda_tile_assert!(tile_id >= 0i32, "raw asserts requires tile_id >= 0");
            cuda_tile_assert!(
                tile_id < total_tiles,
                "raw asserts requires tile_id < total_tiles"
            );
            let (bid_m, bid_n) = if SWIZZLE != 0i32 {
                swizzled_bid(tile_id, num_bid_m, num_bid_n, group_size_m)
            } else {
                linear_bid(tile_id, num_bid_n)
            };
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for i in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, i]);
                let tile_y = part_y.load([i, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            unsafe { part_z.store(tile_z, [bid_m, bid_n]) };
        }
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
            sm_100 = (num_cta_in_cga = 2,),
        )
    )]
    unsafe fn gemm_persistent_mapped_index<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let k_tiles = x.shape()[1] / BK;
        let total_tiles = m * n;

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();

        for tile_id in (pid.0..total_tiles).step_by(grid.0 as usize) {
            let out_idx = z.index(tile_id, m, n);
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            store_view_tko_mapped_mut(
                &mut z,
                tile_z,
                out_idx.coords(),
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
            sm_100 = (num_cta_in_cga = 2,),
        )
    )]
    unsafe fn gemm_persistent_iter_rawloads<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let k_tiles = x.shape()[1] / BK;

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);

        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, out_idx);
        }
    }

    // IR-dumping twin of `gemm`. Selected at runtime via `--dump-ir`.
    #[cutile::entry(print_ir = true,
                    unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    unsafe fn gemm_ir<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry(unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    unsafe fn gemm_full_output<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z_ptr: *mut T,
        z_dim_0: i32,
        z_dim_1: i32,
        z_stride_0: i32,
        x_ptr: *mut T,
        x_dim_0: i32,
        x_dim_1: i32,
        x_stride_0: i32,
        y_ptr: *mut T,
        y_dim_0: i32,
        y_dim_1: i32,
        y_stride_0: i32,
        k: i32,
    ) {
        let mut z = get_tensor_2d(z_ptr, z_dim_0, z_dim_1, z_stride_0);
        let x = get_tensor_2d(x_ptr, x_dim_0, x_dim_1, x_stride_0);
        let y = get_tensor_2d(y_ptr, y_dim_0, y_dim_1, y_stride_0);
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let mut part_z = z.partition_mut(const_shape![BM, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        part_z.store(tile_z, [pid.0, pid.1]);
    }

    #[cutile::entry(print_ir = true,
                    unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    unsafe fn gemm_full_output_ir<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z_ptr: *mut T,
        z_dim_0: i32,
        z_dim_1: i32,
        z_stride_0: i32,
        x_ptr: *mut T,
        x_dim_0: i32,
        x_dim_1: i32,
        x_stride_0: i32,
        y_ptr: *mut T,
        y_dim_0: i32,
        y_dim_1: i32,
        y_stride_0: i32,
        k: i32,
    ) {
        let mut z = get_tensor_2d(z_ptr, z_dim_0, z_dim_1, z_stride_0);
        let x = get_tensor_2d(x_ptr, x_dim_0, x_dim_1, x_stride_0);
        let y = get_tensor_2d(y_ptr, y_dim_0, y_dim_1, y_stride_0);
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let mut part_z = z.partition_mut(const_shape![BM, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        part_z.store(tile_z, [pid.0, pid.1]);
    }

    // Safe variant: same kernel body and optimization hints as `gemm`, but WITHOUT
    // unchecked_accesses — the compiler emits dynamic bounds checks because
    // shapes are runtime-dimensioned. This is the "here's what you pay for
    // safety when the compiler can't prove divisibility" baseline.
    #[cutile::entry(optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    fn gemm_safe<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // IR-dumping twin of `gemm_safe`. Selected via `--safe --dump-ir`.
    #[cutile::entry(print_ir = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    fn gemm_safe_ir<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // Static-shape variant: M, N, K are const generics (not runtime), so the
    // compiler proves divisibility at compile time and bounds checks are
    // eliminated statically. Keep the kernel body aligned with `gemm` so the
    // safety variants do not accidentally benchmark a different algorithm.
    #[cutile::entry(optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    ),
    )]
    fn gemm_static<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
    >(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [M, K] }>,
        y: &Tensor<T, { [K, N] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // Static-shape persistent path. This keeps M/N/K as const generics while
    // using mapped partition indices for safe, compiler-proven disjoint stores.
    #[cutile::entry(optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    ),
    )]
    fn gemm_persistent_static<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [M, K] }>,
        y: &Tensor<T, { [K, N] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let k = Dim::new(K / BK);

        let part_x = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let part_y = y.partition(const_shape![BK, BN]).with_bounds((k, n));

        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in k {
                let tile_x = part_x.load(coord((bid_m, k_tile)));
                let tile_y = part_y.load(coord((k_tile, bid_n)));
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, out_idx);
        }
    }

    // IR-dumping twin of `gemm_static`. Functionally identical, but
    // `print_ir = true` dumps MLIR on first compile so we can confirm no
    // check ops are emitted. Selected at runtime via `--dump-static-ir`.
    #[cutile::entry(print_ir = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                        sm_100 = (num_cta_in_cga = 2,),
                    )
    )]
    fn gemm_static_ir<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
    >(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [M, K] }>,
        y: &Tensor<T, { [K, N] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

const N_WARMUP: u32 = 5;
const ITERS: u32 = 3;
const N_SAMPLES: usize = 10;

#[derive(Clone, Copy)]
enum Variant {
    Gemm,
    GemmIr,
    GemmPersistent,
    GemmPersistentUnchecked,
    GemmPersistentRaw,
    GemmPersistentRawAsserts,
    GemmPersistentMappedIndex,
    GemmPersistentIterRawloads,
    GemmFullOutput,
    GemmFullOutputIr,
    GemmSafe,
    GemmSafeIr,
    GemmPersistentSafe,
    GemmStatic,
    GemmStaticIr,
    GemmPersistentStatic,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            // IR-dumping twins share the label; selected at runtime for MLIR verification.
            Variant::Gemm | Variant::GemmIr => "optimized",
            Variant::GemmPersistent => "persistent_optimized",
            Variant::GemmPersistentUnchecked => "persistent_unchecked",
            Variant::GemmPersistentRaw => "persistent_raw",
            Variant::GemmPersistentRawAsserts => "persistent_raw_asserts",
            Variant::GemmPersistentMappedIndex => "persistent_mapped_index",
            Variant::GemmPersistentIterRawloads => "persistent_iter_rawloads",
            Variant::GemmFullOutput | Variant::GemmFullOutputIr => "full_output",
            Variant::GemmSafe | Variant::GemmSafeIr => "safe",
            Variant::GemmPersistentSafe => "persistent_safe",
            Variant::GemmStatic | Variant::GemmStaticIr => "static",
            Variant::GemmPersistentStatic => "persistent_static",
        }
    }
}

#[derive(Clone)]
struct Sample {
    config: &'static str,
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
    group_size_m: i32,
    num_cta_in_cga: String,
    occupancy: String,
    swizzle: String,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    median_s: f64,
    min_s: f64,
    max_s: f64,
    stdev_s: f64,
    tflops: f64,
    iters: u32,
}

#[derive(Clone)]
struct JitSample {
    config: &'static str,
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
    total_ms: f64,
    cached: bool,
}

#[derive(Clone, Copy, Default)]
struct HintConfig {
    num_cta_in_cga: Option<i32>,
    occupancy: Option<i32>,
    max_divisibility: Option<i32>,
}

impl HintConfig {
    fn from_args(args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            num_cta_in_cga: parse_optional_i32_arg(args, "--cta")?,
            occupancy: parse_optional_i32_arg(args, "--occupancy")?,
            max_divisibility: parse_optional_i32_arg(args, "--max-divisibility")?,
        })
    }

    fn compile_options(self) -> CompileOptions {
        let mut options = CompileOptions::default();
        if let Some(v) = self.num_cta_in_cga {
            options = options.num_cta_in_cga(v);
        }
        if let Some(v) = self.occupancy {
            options = options.occupancy(v);
        }
        if let Some(v) = self.max_divisibility {
            options = options.max_divisibility(v);
        }
        options
    }

    fn num_cta_label(self) -> String {
        option_label(self.num_cta_in_cga)
    }

    fn occupancy_label(self) -> String {
        option_label(self.occupancy)
    }

    fn max_divisibility_label(self) -> String {
        option_label(self.max_divisibility)
    }
}

fn option_label(value: Option<i32>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "default".to_string())
}

fn device_sm_count(device: &Arc<Device>) -> Result<i32, cuda_core::DriverError> {
    let mut result = 0i32;
    unsafe {
        cuda_core::sys::cuDeviceGetAttribute(
            &mut result as *mut i32,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device.cu_device(),
        )
        .result()?;
    }
    Ok(result)
}

fn device_sm_arch(device: &Arc<Device>) -> Result<i32, cuda_core::DriverError> {
    let mut major = 0i32;
    let mut minor = 0i32;
    unsafe {
        cuda_core::sys::cuDeviceGetAttribute(
            &mut major as *mut i32,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device.cu_device(),
        )
        .result()?;
        cuda_core::sys::cuDeviceGetAttribute(
            &mut minor as *mut i32,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device.cu_device(),
        )
        .result()?;
    }
    Ok(major * 10 + minor)
}

fn ceil_div_usize(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn persistent_num_programs(
    m: usize,
    n: usize,
    bm: i32,
    bn: i32,
    num_sms: i32,
    num_cta_in_cga: i32,
    occupancy: i32,
) -> u32 {
    let tiles_m = ceil_div_usize(m, bm as usize);
    let tiles_n = ceil_div_usize(n, bn as usize);
    let total_tiles = (tiles_m * tiles_n).max(1) as i32;
    let sm_programs = (num_sms / num_cta_in_cga.max(1)).max(1);
    let programs = sm_programs.min(total_tiles) * occupancy.max(1);
    programs.max(1) as u32
}

#[derive(Clone, Copy)]
struct PersistentConfig {
    bm: i32,
    bn: i32,
    bk: i32,
    num_cta_in_cga: i32,
    occupancy: i32,
    group_size_m: i32,
    num_programs: Option<u32>,
    swizzle: bool,
}

impl PersistentConfig {
    fn tile(self) -> (i32, i32, i32) {
        (self.bm, self.bn, self.bk)
    }

    fn launch_grid(self, m: usize, n: usize, num_sms: i32) -> (u32, u32, u32) {
        let grid_x = self.num_programs.unwrap_or_else(|| {
            persistent_num_programs(
                m,
                n,
                self.bm,
                self.bn,
                num_sms,
                self.num_cta_in_cga,
                self.occupancy,
            )
        });
        (grid_x, 1, 1)
    }
}

fn persistent_default_tile(size: usize) -> (i32, i32, i32) {
    if size <= 1024 {
        (128, 512, 64)
    } else {
        (256, 256, 64)
    }
}

fn persistent_default_cta(size: usize) -> i32 {
    if size <= 1024 {
        4
    } else {
        2
    }
}

fn persistent_default_config(size: usize, _sm_arch: i32, _num_sms: i32) -> PersistentConfig {
    let (bm, bn, bk) = persistent_default_tile(size);
    let num_cta_in_cga = persistent_default_cta(size);
    PersistentConfig {
        bm,
        bn,
        bk,
        num_cta_in_cga,
        occupancy: 1,
        group_size_m: 8,
        num_programs: None,
        swizzle: true,
    }
}

fn cli_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn parse_arg<T>(args: &[String], flag: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match cli_value(args, flag) {
        Some(value) => value.parse::<T>().map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid value for {flag}: {value} ({err})"),
            )
            .into()
        }),
        None => Ok(default),
    }
}

fn parse_optional_i32_arg(
    args: &[String],
    flag: &str,
) -> Result<Option<i32>, Box<dyn std::error::Error>> {
    match cli_value(args, flag) {
        Some(value) => Ok(Some(value.parse::<i32>().map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid value for {flag}: {value} ({err})"),
            )
        })?)),
        None => Ok(None),
    }
}

fn validate_gemm_config(
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    if bm <= 0 || bn <= 0 || bk <= 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "tile dimensions must be positive",
        )
        .into());
    }
    let bm = bm as usize;
    let bn = bn as usize;
    let bk = bk as usize;
    if m % bm != 0 || n % bn != 0 || k % bk != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("shape ({m},{n},{k}) must be divisible by tile ({bm},{bn},{bk})"),
        )
        .into());
    }
    Ok(())
}

fn open_append_csv(path: &str, header: &str) -> Result<File, Box<dyn std::error::Error>> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let needs_header = std::fs::metadata(path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if needs_header {
        writeln!(file, "{header}")?;
    }
    Ok(file)
}

fn output_path(file_name: &str) -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("RESULTS_DIR") {
        let path = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&path).expect("create RESULTS_DIR");
        path.join(file_name)
    } else {
        std::path::PathBuf::from(file_name)
    }
}

fn sync_stream(stream: &Arc<Stream>, context: &str) {
    stream
        .device()
        .bind_to_thread()
        .expect("bind stream device");
    unsafe { stream.synchronize() }.expect(context);
}

fn check_gemm_output(
    z: Tensor<f16>,
    stream: &Arc<Stream>,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) {
    const MAX_CHECK_ELEMENTS: usize = 16 * 1024 * 1024;
    let total = m * n;
    if total > MAX_CHECK_ELEMENTS {
        println!(
            "  check skipped for {} M=N=K={} ({} elements)",
            label, m, total
        );
        return;
    }

    let host: Vec<f16> = z.to_host_vec().sync_on(stream).expect("copy z to host");
    let expected = k as f32;
    let mut positions = vec![
        0usize,
        n.saturating_sub(1),
        (m / 2) * n + (n / 2),
        m.saturating_sub(1) * n,
        total.saturating_sub(1),
    ];
    positions.sort_unstable();
    positions.dedup();

    for idx in positions.iter().copied() {
        let actual = host[idx].to_f32();
        assert!(
            (actual - expected).abs() <= 0.5,
            "{} check failed at flat idx {}: expected {}, got {}",
            label,
            idx,
            expected,
            actual
        );
    }
    println!(
        "  check ok for {} M=N=K={} ({} sampled values)",
        label,
        m,
        positions.len()
    );
}

fn bench(
    variant: Variant,
    stream: &Arc<Stream>,
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
    compile_options: CompileOptions,
    hints: HintConfig,
    group_size_m: i32,
    launch_grid: Option<(u32, u32, u32)>,
    swizzle: bool,
    check: bool,
) -> Sample {
    let map_m = if swizzle { group_size_m.max(1) } else { 1 };
    let map_n = 1i32;
    // Generics:
    //   Gemm/GemmIr:              [T, BM, BN, BK]  (+ runtime k)
    //   GemmSafe/GemmSafeIr:      [T, BM, BN, BK, K]  (const K so checks elide)
    //   GemmStatic/GemmStaticIr:  [T, BM, BN, BK, M, N, K]  (const shapes + K)
    //   Mapped persistent:        [T, BM, BN, BK, MAP_M, MAP_N]
    let mut generics = vec![
        "f16".to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
    ];
    if matches!(
        variant,
        Variant::GemmStatic | Variant::GemmStaticIr | Variant::GemmPersistentStatic
    ) {
        generics.push(m.to_string());
        generics.push(n.to_string());
        generics.push(k.to_string());
    }
    if matches!(
        variant,
        Variant::GemmPersistent
            | Variant::GemmPersistentUnchecked
            | Variant::GemmPersistentSafe
            | Variant::GemmPersistentMappedIndex
            | Variant::GemmPersistentIterRawloads
            | Variant::GemmPersistentStatic
    ) {
        generics.push(map_m.to_string());
        generics.push(map_n.to_string());
    } else if matches!(
        variant,
        Variant::GemmPersistentRaw | Variant::GemmPersistentRawAsserts
    ) {
        generics.push(if swizzle { "1" } else { "0" }.to_string());
    }

    let x = api::ones(&[m, k]).sync_on(stream).expect("alloc x");
    let y = api::ones(&[k, n]).sync_on(stream).expect("alloc y");

    let mut z = api::zeros::<f16>(&[m, n]).sync_on(stream).expect("alloc z");
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();
    let z_ptr = z.device_pointer();
    let grid = (&mut z)
        .partition([bm as usize, bn as usize])
        .grid()
        .expect("grid");
    let launch_grid = launch_grid.unwrap_or(grid);
    let is_persistent_variant = matches!(
        variant,
        Variant::GemmPersistent
            | Variant::GemmPersistentUnchecked
            | Variant::GemmPersistentSafe
            | Variant::GemmPersistentStatic
            | Variant::GemmPersistentRaw
            | Variant::GemmPersistentRawAsserts
            | Variant::GemmPersistentMappedIndex
            | Variant::GemmPersistentIterRawloads
    );
    let effective_num_cta_in_cga = if hints.num_cta_in_cga.is_some() || is_persistent_variant {
        hints.num_cta_label()
    } else {
        "entry".to_string()
    };
    let effective_occupancy = if hints.occupancy.is_some() || is_persistent_variant {
        hints.occupancy_label()
    } else {
        "entry".to_string()
    };

    macro_rules! launch {
        ($z:expr) => {{
            unsafe {
                match variant {
                    Variant::Gemm => {
                        gemm($z.partition([bm as usize, bn as usize]), &x, &y, k as i32)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmIr => {
                        gemm_ir($z.partition([bm as usize, bn as usize]), &x, &y, k as i32)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistent => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent(mapped_z, &x, &y)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistentUnchecked => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent_unchecked(mapped_z, &x, &y)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistentRaw => {
                        gemm_persistent_raw(
                            z_ptr,
                            m as i32,
                            n as i32,
                            n as i32,
                            &x,
                            &y,
                            m as i32,
                            n as i32,
                            k as i32,
                            group_size_m,
                        )
                        .grid(launch_grid)
                        .generics(generics.clone())
                        .compile_options(compile_options.clone())
                        .async_on(stream)
                        .expect("launch");
                    }
                    Variant::GemmPersistentRawAsserts => {
                        gemm_persistent_raw_asserts(
                            z_ptr,
                            m as i32,
                            n as i32,
                            n as i32,
                            &x,
                            &y,
                            m as i32,
                            n as i32,
                            k as i32,
                            group_size_m,
                        )
                        .grid(launch_grid)
                        .generics(generics.clone())
                        .compile_options(compile_options.clone())
                        .async_on(stream)
                        .expect("launch");
                    }
                    Variant::GemmPersistentMappedIndex => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent_mapped_index(mapped_z, &x, &y)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistentIterRawloads => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent_iter_rawloads(mapped_z, &x, &y)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmFullOutput => {
                        gemm_full_output(
                            z_ptr, m as i32, n as i32, n as i32, x_ptr, m as i32, k as i32,
                            k as i32, y_ptr, k as i32, n as i32, n as i32, k as i32,
                        )
                        .grid(grid)
                        .generics(generics.clone())
                        .compile_options(compile_options.clone())
                        .async_on(stream)
                        .expect("launch");
                    }
                    Variant::GemmFullOutputIr => {
                        gemm_full_output_ir(
                            z_ptr, m as i32, n as i32, n as i32, x_ptr, m as i32, k as i32,
                            k as i32, y_ptr, k as i32, n as i32, n as i32, k as i32,
                        )
                        .grid(grid)
                        .generics(generics.clone())
                        .compile_options(compile_options.clone())
                        .async_on(stream)
                        .expect("launch");
                    }
                    Variant::GemmSafe => {
                        gemm_safe($z.partition([bm as usize, bn as usize]), &x, &y, k as i32)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmSafeIr => {
                        gemm_safe_ir($z.partition([bm as usize, bn as usize]), &x, &y, k as i32)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistentSafe => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent(mapped_z, &x, &y)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmStatic => {
                        gemm_static($z.partition([bm as usize, bn as usize]), &x, &y)
                            .const_grid(grid)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmPersistentStatic => {
                        let mapped_z = $z
                            .partition([bm as usize, bn as usize])
                            .map([map_m as usize, map_n as usize], launch_grid.0);
                        gemm_persistent_static(mapped_z, &x, &y)
                            .const_grid(launch_grid)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                    Variant::GemmStaticIr => {
                        gemm_static_ir($z.partition([bm as usize, bn as usize]), &x, &y)
                            .const_grid(grid)
                            .generics(generics.clone())
                            .compile_options(compile_options.clone())
                            .async_on(stream)
                            .expect("launch");
                    }
                }
            }
        }};
    }

    // Warmup: fixed count, not wall-clock (async launches would queue up).
    for _ in 0..N_WARMUP {
        launch!(&mut z);
    }
    sync_stream(stream, "sync");

    // N_SAMPLES samples, each wrapping fixed ITERS launches. Keep this fixed
    // to match the paper-facing Python GEMM harness and evaluation text.
    let iters = ITERS;
    let mut per_launch_s = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        sync_stream(stream, "sync");
        let s = Instant::now();
        for _ in 0..iters {
            launch!(&mut z);
        }
        sync_stream(stream, "sync");
        per_launch_s.push(s.elapsed().as_secs_f64() / iters as f64);
    }

    per_launch_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_s = per_launch_s[per_launch_s.len() / 2];
    let min_s = per_launch_s[0];
    let max_s = per_launch_s[per_launch_s.len() - 1];
    let mean_s: f64 = per_launch_s.iter().sum::<f64>() / per_launch_s.len() as f64;
    // Sample stdev (N-1 denominator).
    let variance: f64 = per_launch_s
        .iter()
        .map(|v| (v - mean_s).powi(2))
        .sum::<f64>()
        / (per_launch_s.len() as f64 - 1.0).max(1.0);
    let stdev_s = variance.sqrt();
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / median_s / 1e12;

    println!(
        "  {} M=N=K={}: {:.1} TFlops ({:.0} us, min {:.0} / max {:.0} / \u{3c3} {:.1} us, iters={})",
        variant.label(),
        m,
        tflops,
        median_s * 1e6,
        min_s * 1e6,
        max_s * 1e6,
        stdev_s * 1e6,
        iters,
    );
    if is_persistent_variant {
        println!(
            "    persistent schedule: {}",
            if swizzle { "swizzled" } else { "linear" },
        );
    }

    if check {
        check_gemm_output(z, stream, m, n, k, variant.label());
    }

    Sample {
        config: variant.label(),
        m,
        n,
        k,
        bm,
        bn,
        bk,
        group_size_m,
        num_cta_in_cga: effective_num_cta_in_cga,
        occupancy: effective_occupancy,
        swizzle: if is_persistent_variant {
            if swizzle {
                "on".to_string()
            } else {
                "off".to_string()
            }
        } else {
            "n/a".to_string()
        },
        grid_x: launch_grid.0,
        grid_y: launch_grid.1,
        grid_z: launch_grid.2,
        median_s,
        min_s,
        max_s,
        stdev_s,
        tflops,
        iters,
    }
}

/// Measure wall-clock of first async_on call per unique
/// monomorphization. Static: each (M,N,K) is distinct -> fresh JIT per
/// shape. Optimized/safe: monomorphization depends on (T,BM,BN,BK); in
/// this sweep that means shape 1024 (BM=128) and shapes >=2048
/// (BM=256) each JIT once and the rest are cached.
fn jit_pass(
    variant: Variant,
    stream: &Arc<Stream>,
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
    already_jitted: &mut std::collections::HashSet<String>,
) -> JitSample {
    let mut generics = vec![
        "f16".to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
    ];
    if matches!(variant, Variant::GemmStatic) {
        generics.push(m.to_string());
        generics.push(n.to_string());
        generics.push(k.to_string());
    }
    let cache_key = format!("{}:{}", variant.label(), generics.join(","));
    let pre_cached = already_jitted.contains(&cache_key);

    let x = api::ones::<f16>(&[m, k]).sync_on(stream).expect("alloc x");
    let y = api::ones::<f16>(&[k, n]).sync_on(stream).expect("alloc y");
    let z = api::zeros::<f16>(&[m, n])
        .partition([bm as usize, bn as usize])
        .sync_on(stream)
        .expect("alloc z");
    let grid = z.grid().expect("grid");

    // Flush prior work so the measurement doesn't absorb the previous
    // variant's kernel time. Do NOT sync afterwards: JIT is synchronous
    // inside async_on (compile + module load + launch submit), so
    // async_on's return already marks JIT complete. Post-sync would
    // add this kernel's execution time to the number.
    sync_stream(stream, "sync before timing");
    let t0 = Instant::now();
    unsafe {
        match variant {
            Variant::Gemm => {
                gemm(z, &x, &y, k as i32)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            Variant::GemmSafe => {
                gemm_safe(z, &x, &y, k as i32)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            Variant::GemmStatic => {
                gemm_static(z, &x, &y)
                    .const_grid(grid)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            _ => unreachable!("jit_pass only for non-IR variants"),
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // Flush this kernel before returning so it doesn't contaminate the
    // next measurement's pre-sync.
    sync_stream(stream, "sync after");

    if !pre_cached {
        already_jitted.insert(cache_key);
    }
    println!(
        "  M=N=K={:>5} {:>9}: {} {:>8.2} ms",
        m,
        variant.label(),
        if pre_cached { "cached " } else { "jit    " },
        total_ms,
    );
    JitSample {
        config: variant.label(),
        m,
        n,
        k,
        bm,
        bn,
        bk,
        total_ms,
        cached: pre_cached,
    }
}

fn dump_ir_pass(
    variant: Variant,
    stream: &Arc<Stream>,
    m: usize,
    n: usize,
    k: usize,
    bm: i32,
    bn: i32,
    bk: i32,
) {
    let mut generics = vec![
        "f16".to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
    ];
    if matches!(variant, Variant::GemmStaticIr) {
        generics.push(m.to_string());
        generics.push(n.to_string());
        generics.push(k.to_string());
    }

    let x = api::ones::<f16>(&[m, k]).sync_on(stream).expect("alloc x");
    let y = api::ones::<f16>(&[k, n]).sync_on(stream).expect("alloc y");
    let mut z = api::zeros::<f16>(&[m, n]).sync_on(stream).expect("alloc z");
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();
    let z_ptr = z.device_pointer();
    let grid = (&mut z)
        .partition([bm as usize, bn as usize])
        .grid()
        .expect("grid");

    // Single launch triggers JIT compile and, with print_ir = true on the
    // kernel entry, dumps MLIR to stderr. No warmup, no measurement.
    unsafe {
        match variant {
            Variant::GemmIr => {
                gemm_ir(
                    (&mut z).partition([bm as usize, bn as usize]),
                    &x,
                    &y,
                    k as i32,
                )
                .generics(generics)
                .async_on(stream)
                .expect("launch");
            }
            Variant::GemmFullOutputIr => {
                gemm_full_output_ir(
                    z_ptr, m as i32, n as i32, n as i32, x_ptr, m as i32, k as i32, k as i32,
                    y_ptr, k as i32, n as i32, n as i32, k as i32,
                )
                .grid(grid)
                .generics(generics)
                .async_on(stream)
                .expect("launch");
            }
            Variant::GemmSafeIr => {
                gemm_safe_ir(
                    (&mut z).partition([bm as usize, bn as usize]),
                    &x,
                    &y,
                    k as i32,
                )
                .generics(generics)
                .async_on(stream)
                .expect("launch");
            }
            Variant::GemmStaticIr => {
                gemm_static_ir((&mut z).partition([bm as usize, bn as usize]), &x, &y)
                    .const_grid(grid)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            _ => unreachable!("dump_ir_pass only for *_ir variants"),
        }
    }
    sync_stream(stream, "sync");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CLI: optimized is the default. Mutually exclusive mode flags:
    //   --safe    bench/dump the safe (checks-emitted) variant
    //   --static  bench/dump the static (checks-folded) variant
    //   --full-output bench/dump the pointer/full-output diagnostic variant
    //   --persistent --raw-pointer bench the old unchecked persistent pointer path
    //   --persistent --unchecked-mapped bench mapped persistent with unchecked_accesses
    //   --dump-ir dump MLIR (compile-only) for the selected variant
    let args: Vec<String> = std::env::args().collect();
    let is_safe = args.iter().any(|a| a == "--safe");
    let is_static = args.iter().any(|a| a == "--static");
    let is_full_output = args.iter().any(|a| a == "--full-output");
    let is_persistent = args.iter().any(|a| a == "--persistent");
    let is_raw_pointer = args.iter().any(|a| a == "--raw-pointer");
    let is_raw_asserts = args.iter().any(|a| a == "--raw-asserts");
    let is_mapped_index = args.iter().any(|a| a == "--mapped-index");
    let is_iter_rawloads = args.iter().any(|a| a == "--iter-rawloads");
    let is_unchecked_mapped = args.iter().any(|a| a == "--unchecked-mapped");
    let dump_ir = args.iter().any(|a| a == "--dump-ir");
    let jit_mode = args.iter().any(|a| a == "--jit");
    let tune_one = args.iter().any(|a| a == "--tune-one");
    let check = args.iter().any(|a| a == "--check");
    let force_swizzle = args.iter().any(|a| a == "--swizzle");
    let force_no_swizzle = args.iter().any(|a| a == "--no-swizzle");
    let swizzle_override = if force_swizzle {
        Some(true)
    } else if force_no_swizzle {
        Some(false)
    } else {
        None
    };
    let mode_count = [is_safe, is_static, is_full_output, is_unchecked_mapped]
        .iter()
        .filter(|&&x| x)
        .count();
    if mode_count > 1 {
        eprintln!(
            "error: --safe, --static, --full-output, and --unchecked-mapped are mutually exclusive"
        );
        std::process::exit(2);
    }
    if is_persistent && is_full_output {
        eprintln!("error: --persistent and --full-output are mutually exclusive");
        std::process::exit(2);
    }
    if is_raw_pointer && !is_persistent {
        eprintln!("error: --raw-pointer only applies with --persistent");
        std::process::exit(2);
    }
    let persistent_impl_count = [
        is_raw_pointer,
        is_raw_asserts,
        is_mapped_index,
        is_iter_rawloads,
    ]
    .iter()
    .filter(|&&x| x)
    .count();
    if persistent_impl_count > 1 {
        eprintln!(
            "error: --raw-pointer, --raw-asserts, --mapped-index, and --iter-rawloads are mutually exclusive"
        );
        std::process::exit(2);
    }
    if (is_raw_asserts || is_mapped_index || is_iter_rawloads) && !is_persistent {
        eprintln!("error: persistent diagnostic variants only apply with --persistent");
        std::process::exit(2);
    }
    if persistent_impl_count > 0 && (is_static || is_safe || is_unchecked_mapped) {
        eprintln!(
            "error: persistent diagnostic variants are already explicit; do not combine them with --safe/--static/--unchecked-mapped"
        );
        std::process::exit(2);
    }
    if is_unchecked_mapped && !is_persistent {
        eprintln!("error: --unchecked-mapped only applies with --persistent");
        std::process::exit(2);
    }
    if is_unchecked_mapped && is_raw_pointer {
        eprintln!("error: --unchecked-mapped compares the mapped path; do not combine it with --raw-pointer");
        std::process::exit(2);
    }
    if force_swizzle && force_no_swizzle {
        eprintln!("error: --swizzle and --no-swizzle are mutually exclusive");
        std::process::exit(2);
    }
    if !is_persistent && swizzle_override.is_some() {
        eprintln!("error: --swizzle/--no-swizzle only apply with --persistent");
        std::process::exit(2);
    }
    if dump_ir && is_persistent {
        eprintln!("error: --persistent does not currently have an IR-dumping twin");
        std::process::exit(2);
    }
    if jit_mode && (is_safe || is_static || is_full_output || is_persistent || dump_ir) {
        eprintln!(
            "error: --jit runs all variants; --safe/--static/--full-output/--persistent/--dump-ir not applicable"
        );
        std::process::exit(2);
    }
    if tune_one && (jit_mode || dump_ir) {
        eprintln!("error: --tune-one is mutually exclusive with --jit/--dump-ir");
        std::process::exit(2);
    }
    let label = if is_persistent && is_raw_pointer {
        "persistent_raw"
    } else if is_persistent && is_raw_asserts {
        "persistent_raw_asserts"
    } else if is_persistent && is_mapped_index {
        "persistent_mapped_index"
    } else if is_persistent && is_iter_rawloads {
        "persistent_iter_rawloads"
    } else if is_persistent && is_unchecked_mapped {
        "persistent_unchecked"
    } else if is_persistent && is_safe {
        "persistent_safe"
    } else if is_persistent && is_static {
        "persistent_static"
    } else if is_persistent {
        "persistent_optimized"
    } else if is_safe {
        "safe"
    } else if is_static {
        "static"
    } else if is_full_output {
        "full_output"
    } else {
        "optimized"
    };

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");
    let num_sms = device_sm_count(&device).unwrap_or(148);
    let sm_arch = device_sm_arch(&device).unwrap_or(0);

    let shapes: Vec<(usize, usize, usize)> = (10..16).map(|i| (1 << i, 1 << i, 1 << i)).collect();
    let hyper_params: Vec<(i32, i32, i32)> = if is_persistent {
        shapes
            .iter()
            .map(|&(m, _, _)| persistent_default_config(m, sm_arch, num_sms).tile())
            .collect()
    } else {
        vec![
            (128, 128, 64),
            (256, 256, 128),
            (128, 256, 128),
            (128, 256, 128),
            (256, 256, 128),
            (128, 256, 128),
        ]
    };

    if tune_one {
        let default_size = shapes.last().expect("non-empty").0;
        let size = parse_arg(&args, "--size", default_size)?;
        let m = parse_arg(&args, "--m", size)?;
        let n = parse_arg(&args, "--n", size)?;
        let k = parse_arg(&args, "--k", size)?;
        let persistent_config = persistent_default_config(size, sm_arch, num_sms);
        let (default_bm, default_bn, default_bk) = if is_persistent {
            persistent_config.tile()
        } else {
            (128, 256, 128)
        };
        let bm = parse_arg(&args, "--bm", default_bm)?;
        let bn = parse_arg(&args, "--bn", default_bn)?;
        let bk = parse_arg(&args, "--bk", default_bk)?;
        validate_gemm_config(m, n, k, bm, bn, bk)?;

        let mut hints = HintConfig::from_args(&args)?;
        if is_persistent {
            if hints.num_cta_in_cga.is_none() {
                hints.num_cta_in_cga = Some(persistent_config.num_cta_in_cga);
            }
            if hints.occupancy.is_none() {
                hints.occupancy = Some(persistent_config.occupancy);
            }
        }
        let compile_options = hints.compile_options();
        let group_size_m = parse_arg(
            &args,
            "--group-size-m",
            if is_persistent {
                persistent_config.group_size_m
            } else {
                8i32
            },
        )?;
        let num_programs_override = parse_optional_i32_arg(&args, "--num-programs")?;
        let launch_grid = if is_persistent {
            let num_programs = num_programs_override
                .map(|v| v.max(1) as u32)
                .unwrap_or_else(|| {
                    persistent_config.num_programs.unwrap_or_else(|| {
                        persistent_num_programs(
                            m,
                            n,
                            bm,
                            bn,
                            num_sms,
                            hints.num_cta_in_cga.unwrap_or(1),
                            hints.occupancy.unwrap_or(1),
                        )
                    })
                });
            Some((num_programs, 1, 1))
        } else {
            None
        };
        let swizzle_enabled = swizzle_override.unwrap_or_else(|| {
            if is_persistent {
                persistent_config.swizzle
            } else {
                true
            }
        });
        let variant = if is_persistent && is_raw_pointer {
            Variant::GemmPersistentRaw
        } else if is_persistent && is_raw_asserts {
            Variant::GemmPersistentRawAsserts
        } else if is_persistent && is_mapped_index {
            Variant::GemmPersistentMappedIndex
        } else if is_persistent && is_iter_rawloads {
            Variant::GemmPersistentIterRawloads
        } else if is_persistent && is_unchecked_mapped {
            Variant::GemmPersistentUnchecked
        } else if is_persistent && is_safe {
            Variant::GemmPersistentSafe
        } else if is_persistent && is_static {
            Variant::GemmPersistentStatic
        } else if is_persistent {
            Variant::GemmPersistent
        } else if is_safe {
            Variant::GemmSafe
        } else if is_static {
            Variant::GemmStatic
        } else if is_full_output {
            Variant::GemmFullOutput
        } else {
            Variant::Gemm
        };
        println!("=== GEMM Tuning Run: cuTile Rust ===");
        println!(
            "--- {} M={} N={} K={} tile=({}, {}, {}) cta={} occupancy={} max_divisibility={} swizzle={} ---",
            variant.label(),
            m,
            n,
            k,
            bm,
            bn,
            bk,
            hints.num_cta_label(),
            hints.occupancy_label(),
            hints.max_divisibility_label(),
            if swizzle_enabled { "on" } else { "off" },
        );
        if let Some(grid) = launch_grid {
            println!(
                "--- persistent grid=({}, {}, {}) group_size_m={} num_sms={} ---",
                grid.0, grid.1, grid.2, group_size_m, num_sms
            );
        }
        let sample = bench(
            variant,
            &stream,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            compile_options,
            hints,
            group_size_m,
            launch_grid,
            swizzle_enabled,
            check,
        );
        let csv_path = cli_value(&args, "--csv").unwrap_or("gemm_rust_tune_results.csv");
        let mut f = open_append_csv(
            csv_path,
            "config,M,N,K,tm,tn,tk,group_size_m,num_cta_in_cga,occupancy,swizzle,max_divisibility,grid_x,iters,median_s,min_s,max_s,stdev_s,tflops",
        )?;
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.2}",
            sample.config,
            sample.m,
            sample.n,
            sample.k,
            sample.bm,
            sample.bn,
            sample.bk,
            sample.group_size_m,
            sample.num_cta_in_cga,
            sample.occupancy,
            sample.swizzle,
            hints.max_divisibility_label(),
            sample.grid_x,
            sample.iters,
            sample.median_s,
            sample.min_s,
            sample.max_s,
            sample.stdev_s,
            sample.tflops
        )?;
        println!("\nTuning result appended to {}", csv_path);
        return Ok(());
    }

    if dump_ir {
        // IR-dump mode: enable CUTILE_DUMP=ir so the cutile-compiler emits
        // its cutile-ir Module (MLIR-like text) to stderr on first compile.
        // Scope the filter to just the selected kernel so we skip the
        // api::ones / api::zeros `@creation` module.
        if std::env::var_os("CUTILE_DUMP").is_none() {
            // Safe: main is single-threaded at this point.
            unsafe {
                std::env::set_var("CUTILE_DUMP", "ir");
            }
        }
        let (variant, filter) = if is_safe {
            (Variant::GemmSafeIr, "gemm_safe_ir")
        } else if is_static {
            (Variant::GemmStaticIr, "gemm_static_ir")
        } else if is_full_output {
            (Variant::GemmFullOutputIr, "gemm_full_output_ir")
        } else {
            (Variant::GemmIr, "gemm_ir")
        };
        if std::env::var_os("CUTILE_DUMP_FILTER").is_none() {
            unsafe {
                std::env::set_var("CUTILE_DUMP_FILTER", filter);
            }
        }
        let last = shapes.len() - 1;
        let (m, n, k) = shapes[last];
        let (bm, bn, bk) = hyper_params[last];
        eprintln!("=== IR dump: {} at M=N=K={} ===", label, m);
        dump_ir_pass(variant, &stream, m, n, k, bm, bn, bk);
        return Ok(());
    }

    if jit_mode {
        println!("=== GEMM JIT Cost (cuTile Rust) ===");
        println!("Measures total JIT wall time per unique monomorphization.");
        println!("Static: fresh JIT at every (M,N,K). Optimized/safe: keyed on (BM,BN,BK).");
        // JIT pass uses its own shape set: multiple sizes at the same tile
        // shapes so the "hidden cost of static" (one JIT per (M,N,K)) remains
        // visible while matching the runtime sweep's per-size tile schedule.
        let jit_shapes: Vec<(usize, usize, usize, i32, i32, i32)> = vec![
            // BM=128: small shapes that are naturally tiled at BM=128.
            (128, 128, 128, 128, 128, 64),
            (256, 256, 256, 128, 128, 64),
            (512, 512, 512, 128, 128, 64),
            (1024, 1024, 1024, 128, 128, 64),
            (2048, 2048, 2048, 256, 256, 128),
            (4096, 4096, 4096, 128, 256, 128),
            (8192, 8192, 8192, 128, 256, 128),
            (16384, 16384, 16384, 256, 256, 128),
            (32768, 32768, 32768, 128, 256, 128),
        ];
        let mut already = std::collections::HashSet::<String>::new();
        let mut all: Vec<JitSample> = Vec::new();
        for &(m, n, k, bm, bn, bk) in &jit_shapes {
            for v in [Variant::Gemm, Variant::GemmSafe, Variant::GemmStatic] {
                all.push(jit_pass(v, &stream, m, n, k, bm, bn, bk, &mut already));
            }
        }
        let csv_path = output_path("gemm_rust_jit_results.csv");
        let mut f = File::create(&csv_path)?;
        writeln!(f, "config,M,N,K,tm,tn,tk,total_ms,cached")?;
        for s in &all {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.3},{}",
                s.config, s.m, s.n, s.k, s.bm, s.bn, s.bk, s.total_ms, s.cached
            )?;
        }
        println!("\nJIT results written to {}", csv_path.display());
        return Ok(());
    }

    let variant = if is_persistent && is_raw_pointer {
        Variant::GemmPersistentRaw
    } else if is_persistent && is_raw_asserts {
        Variant::GemmPersistentRawAsserts
    } else if is_persistent && is_mapped_index {
        Variant::GemmPersistentMappedIndex
    } else if is_persistent && is_iter_rawloads {
        Variant::GemmPersistentIterRawloads
    } else if is_persistent && is_unchecked_mapped {
        Variant::GemmPersistentUnchecked
    } else if is_persistent && is_safe {
        Variant::GemmPersistentSafe
    } else if is_persistent && is_static {
        Variant::GemmPersistentStatic
    } else if is_persistent {
        Variant::GemmPersistent
    } else if is_safe {
        Variant::GemmSafe
    } else if is_static {
        Variant::GemmStatic
    } else if is_full_output {
        Variant::GemmFullOutput
    } else {
        Variant::Gemm
    };

    println!("=== GEMM Benchmark: cuTile Rust ===");
    println!("--- {} ---", variant.label());

    let mut all: Vec<Sample> = Vec::new();
    for (&(m, n, k), &(bm, bn, bk)) in shapes.iter().zip(hyper_params.iter()) {
        let mut compile_options = CompileOptions::default();
        let mut hints = HintConfig::default();
        let mut launch_grid = None;
        let persistent_config = persistent_default_config(m, sm_arch, num_sms);
        let group_size_m = if is_persistent {
            persistent_config.group_size_m
        } else {
            8i32
        };
        if is_persistent {
            hints.num_cta_in_cga = Some(persistent_config.num_cta_in_cga);
            hints.occupancy = Some(persistent_config.occupancy);
            compile_options = compile_options
                .num_cta_in_cga(persistent_config.num_cta_in_cga)
                .occupancy(persistent_config.occupancy);
            launch_grid = Some(persistent_config.launch_grid(m, n, num_sms));
        }
        let swizzle_enabled = swizzle_override.unwrap_or_else(|| {
            if is_persistent {
                persistent_config.swizzle
            } else {
                true
            }
        });
        all.push(bench(
            variant,
            &stream,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            compile_options,
            hints,
            group_size_m,
            launch_grid,
            swizzle_enabled,
            check,
        ));
    }

    println!("\n============================================================");
    println!("  SUMMARY ({}, f16)", variant.label());
    println!("============================================================");
    println!("  {:>8}  {:>11}", "M=N=K", variant.label());
    for s in &all {
        println!("  {:>8}  {:>9.1} TF", s.m, s.tflops);
    }

    // Per-variant CSV so runs don't clobber each other.
    let csv_path = output_path(&format!("gemm_rust_{}_results.csv", label));
    let mut f = File::create(&csv_path)?;
    writeln!(
        f,
        "config,M,N,K,tm,tn,tk,group_size_m,num_cta_in_cga,occupancy,swizzle,grid_x,grid_y,grid_z,iters,median_s,min_s,max_s,stdev_s,tflops"
    )?;
    for s in &all {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.2}",
            s.config,
            s.m,
            s.n,
            s.k,
            s.bm,
            s.bn,
            s.bk,
            s.group_size_m,
            s.num_cta_in_cga,
            s.occupancy,
            s.swizzle,
            s.grid_x,
            s.grid_y,
            s.grid_z,
            s.iters,
            s.median_s,
            s.min_s,
            s.max_s,
            s.stdev_s,
            s.tflops
        )?;
    }
    println!("\nResults written to {}", csv_path.display());

    Ok(())
}

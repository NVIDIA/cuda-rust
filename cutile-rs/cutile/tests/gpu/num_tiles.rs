/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU-execution correctness tests for `num_tiles(view, axis)`.
//!
//! The kernel stores `num_tiles(input_partition, axis)` as a scalar broadcast
//! into an output tile; the host side compares against manually-computed
//! `cdiv(tensor_dim_i, tile_dim_i)`.
//!
//! `axis` is pinned at compile time per-kernel (the JIT requires it to be a
//! known constant), so there's one kernel per axis.

use cutile::api;
use cutile::prelude::PartitionMut;
use cutile::tile_kernel::{DeviceOp, ToHostVecOp};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub(crate) fn num_tiles_axis0<const BM: i32, const BN: i32>(
        out: &mut Tensor<i32, { [BM, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part = input.partition(shape![BM, BN]);
        let n: i32 = num_tiles(&part, 0);
        let tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(n, shape![BM, BN]);
        out.store(tile);
    }

    #[cutile::entry()]
    pub(crate) fn num_tiles_axis1<const BM: i32, const BN: i32>(
        out: &mut Tensor<i32, { [BM, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part = input.partition(shape![BM, BN]);
        let n: i32 = num_tiles(&part, 1);
        let tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(n, shape![BM, BN]);
        out.store(tile);
    }
}

fn cdiv(n: usize, d: usize) -> usize {
    n.div_ceil(d)
}

/// Launches the axis-0 kernel on an input of shape `[m, n]` with tile `[bm, bn]`,
/// returns the scalar that the kernel wrote (all output cells equal it).
fn run_axis(m: usize, n: usize, bm: usize, bn: usize, axis: usize) -> i32 {
    let input = api::zeros::<f32>(&[m, n]).sync().expect("alloc input");
    let mut out = api::zeros::<i32>(&[bm, bn]).sync().expect("alloc out");
    match axis {
        0 => {
            kernels::num_tiles_axis0((&mut out).partition([bm, bn]), &input)
                .sync()
                .expect("axis0 kernel");
        }
        1 => {
            kernels::num_tiles_axis1((&mut out).partition([bm, bn]), &input)
                .sync()
                .expect("axis1 kernel");
        }
        _ => unreachable!("only axes 0 and 1 covered in this test"),
    }
    let got: Vec<i32> = out.dup().to_host_vec().sync().expect("to_host");
    // Every cell holds the same broadcast scalar.
    assert!(
        got.windows(2).all(|w| w[0] == w[1]),
        "output should be uniform"
    );
    got[0]
}

#[test]
fn axis0_32x32_8x8() {
    let got = run_axis(32, 32, 8, 8, 0);
    assert_eq!(got, cdiv(32, 8) as i32, "axis 0, 32x32 with 8x8 tile");
}

#[test]
fn axis1_32x32_8x8() {
    let got = run_axis(32, 32, 8, 8, 1);
    assert_eq!(got, cdiv(32, 8) as i32, "axis 1, 32x32 with 8x8 tile");
}

#[test]
fn axis0_64x128_16x32() {
    let got = run_axis(64, 128, 16, 32, 0);
    assert_eq!(got, cdiv(64, 16) as i32, "axis 0, 64x128 with 16x32 tile");
}

#[test]
fn axis1_64x128_16x32() {
    let got = run_axis(64, 128, 16, 32, 1);
    assert_eq!(got, cdiv(128, 32) as i32, "axis 1, 64x128 with 16x32 tile");
}

#[test]
fn axis0_ceiling_division() {
    // Non-divisible: cdiv(100, 32) = 4 (100/32 = 3.125 → 4).
    let got = run_axis(100, 128, 32, 32, 0);
    assert_eq!(got, cdiv(100, 32) as i32, "ceiling division along axis 0");
}

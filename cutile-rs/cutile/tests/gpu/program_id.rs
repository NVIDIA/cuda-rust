/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `program_id(axis)` / `num_programs(axis)`: the per-axis tile-program
//! index and grid extent, following Triton's `tl.program_id` /
//! `tl.num_programs`. Pins both against `get_tile_block_id()` /
//! `get_num_tile_blocks()` on a rank-3 grid, and the rank-1 idiom
//! `tensor.partition(shape).load([program_id(0)])`.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod program_id_module {
    use cutile::core::*;

    /// Each program encodes its three `program_id` axes and the three
    /// `num_programs` extents into one i32 and stores it into its own
    /// `[1, 1, 1]` slab, which sits at its block coordinates — so the
    /// host can compare every axis against the block coordinates it
    /// launched.
    #[cutile::entry()]
    fn write_program_axes(out: &mut Tensor<i32, { [1, 1, 1] }>) {
        let encoded = program_id(0)
            + 10 * program_id(1)
            + 100 * program_id(2)
            + 1_000 * num_programs(0)
            + 10_000 * num_programs(1)
            + 100_000 * num_programs(2);
        let id: Tile<i32, { [] }> = scalar_to_tile(encoded);
        out.store(id.reshape(shape![1, 1, 1]));
    }

    /// The rank-1 idiom: each program copies exactly the sub-tensor it
    /// owns.
    #[cutile::entry()]
    fn copy_by_program_id(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let tile = x.partition(shape![4]).load([program_id(0)]);
        z.store(tile);
    }
}

use program_id_module::{copy_by_program_id, write_program_axes};

#[test]
fn program_id_and_num_programs_match_the_block_coordinates_per_axis() {
    common::with_test_stack(|| {
        let (nx, ny, nz) = (4usize, 3usize, 2usize);
        let result = write_program_axes(api::zeros::<i32>(&[nx, ny, nz]).partition([1, 1, 1]))
            .grid((nx as u32, ny as u32, nz as u32))
            .sync()
            .expect("write_program_axes kernel");
        let host = result
            .0
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("to_host");
        // Row-major host layout: out[x][y][z] at x*ny*nz + y*nz + z.
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let got = host[x * ny * nz + y * nz + z];
                    let expected =
                        (x + 10 * y + 100 * z + 1_000 * nx + 10_000 * ny + 100_000 * nz) as i32;
                    assert_eq!(got, expected, "block ({x}, {y}, {z})");
                }
            }
        }
    });
}

#[test]
fn rank1_partition_load_by_program_id_copies_the_owned_subtensor() {
    common::with_test_stack(|| {
        let len = 32usize;
        let z_host = copy_by_program_id(api::zeros(&[len]).partition([4]), api::arange::<f32>(len))
            .grid(((len / 4) as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("copy_by_program_id kernel");
        for (i, v) in z_host.iter().enumerate() {
            assert_eq!(*v, i as f32, "index {i}");
        }
    });
}

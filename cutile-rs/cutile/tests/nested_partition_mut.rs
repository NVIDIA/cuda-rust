/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression repro for nested mutable partition stores.
//!
//! The reported pattern is:
//! - caller passes an already partitioned mutable output tile, e.g. `[1, N]`
//! - kernel repartitions that `&mut Tensor` with `partition_mut([1, BLOCK])`
//! - kernel stores subtiles via local indices like `[0, j]`
//!
//! Direct stores to CTA-owned `[1, BLOCK]` outputs are used as the control.

use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::{DeviceOp, TileKernel};

mod common;

#[cutile::module]
mod nested_partition_mut_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn direct_cta_owned_copy<const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, BLOCK] }>,
        input: &Tensor<f32, { [-1, N] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let block = pid.1;

        let input_part: Partition<f32, { [1, BLOCK] }> = input.partition(shape![1, BLOCK]);
        let tile: Tile<f32, { [1, BLOCK] }> = input_part.load([row, block]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn nested_partition_mut_copy<const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, N] }>,
        input: &Tensor<f32, { [-1, N] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let input_part: Partition<f32, { [1, BLOCK] }> = input.partition(shape![1, BLOCK]);
        let mut out_part: PartitionMut<f32, { [1, BLOCK] }> = out.partition_mut(shape![1, BLOCK]);

        for j in 0i32..((N + BLOCK - 1) / BLOCK) {
            let tile: Tile<f32, { [1, BLOCK] }> = input_part.load([row, j]);
            out_part.store(tile, [0i32, j]);
        }
    }

    #[cutile::entry()]
    fn nested_partition_mut_copy_3d<const HEADS: i32, const N: i32, const BLOCK: i32>(
        out: &mut Tensor<f32, { [1, 1, N] }>,
        input: &Tensor<f32, { [-1, HEADS, N] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let head = pid.1;

        let input_part: Partition<f32, { [1, 1, BLOCK] }> = input.partition(shape![1, 1, BLOCK]);
        let mut out_part: PartitionMut<f32, { [1, 1, BLOCK] }> =
            out.partition_mut(shape![1, 1, BLOCK]);

        for j in 0i32..((N + BLOCK - 1) / BLOCK) {
            let tile: Tile<f32, { [1, 1, BLOCK] }> = input_part.load([row, head, j]);
            out_part.store(tile, [0i32, 0i32, j]);
        }
    }
}

fn assert_matrix_eq(actual: &[f32], expected: &[f32], rows: usize, cols: usize, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            assert!(
                (actual[idx] - expected[idx]).abs() < 1e-5,
                "{label}: element [{row}, {col}] expected {}, got {}\nactual: {actual:?}",
                expected[idx],
                actual[idx]
            );
        }
    }
}

#[test]
fn nested_partition_mut_store_uses_cta_local_indices() {
    common::with_test_stack(|| {
        const ROWS: usize = 4;
        const N: usize = 8;
        const BLOCK: usize = 2;

        let expected: Vec<f32> = (0..(ROWS * N)).map(|x| x as f32).collect();
        let input = api::copy_host_vec_to_device(&Arc::new(expected.clone()))
            .sync()
            .expect("input alloc");
        let input_2d = input.view(&[ROWS, N]).expect("input view");

        let direct_out = api::zeros::<f32>(&[ROWS, N]).sync().expect("direct alloc");
        let (direct_result, _) = nested_partition_mut_module::direct_cta_owned_copy(
            direct_out.partition([1, BLOCK]),
            &input_2d,
        )
        .generics(vec![N.to_string(), BLOCK.to_string()])
        .sync()
        .expect("direct copy");
        let direct_host: Vec<f32> = direct_result
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("direct to_host");
        assert_matrix_eq(&direct_host, &expected, ROWS, N, "direct control");

        let nested_out = api::zeros::<f32>(&[ROWS, N]).sync().expect("nested alloc");
        let (nested_result, _) = nested_partition_mut_module::nested_partition_mut_copy(
            nested_out.partition([1, N]),
            &input_2d,
        )
        .generics(vec![N.to_string(), BLOCK.to_string()])
        .sync()
        .expect("nested copy");
        let nested_host: Vec<f32> = nested_result
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("nested to_host");

        assert_matrix_eq(
            &nested_host,
            &expected,
            ROWS,
            N,
            "nested partition_mut repro",
        );
    });
}

#[test]
fn nested_partition_mut_3d_store_uses_cta_local_indices() {
    common::with_test_stack(|| {
        const ROWS: usize = 2;
        const HEADS: usize = 3;
        const N: usize = 10;
        const BLOCK: usize = 4;

        let expected: Vec<f32> = (0..(ROWS * HEADS * N)).map(|x| x as f32).collect();
        let input = api::copy_host_vec_to_device(&Arc::new(expected.clone()))
            .sync()
            .expect("input alloc");
        let input_3d = input.view(&[ROWS, HEADS, N]).expect("input view");

        let nested_out = api::zeros::<f32>(&[ROWS, HEADS, N])
            .sync()
            .expect("nested alloc");
        let (nested_result, _) = nested_partition_mut_module::nested_partition_mut_copy_3d(
            nested_out.partition([1, 1, N]),
            &input_3d,
        )
        .generics(vec![HEADS.to_string(), N.to_string(), BLOCK.to_string()])
        .sync()
        .expect("nested 3d copy");
        let nested_host: Vec<f32> = nested_result
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("nested to_host");

        assert_eq!(nested_host.len(), expected.len());
        for (idx, (&actual, &expected)) in nested_host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "nested partition_mut 3d repro: element {idx} expected {expected}, got {actual}\nactual: {nested_host:?}"
            );
        }
    });
}

#[test]
fn nested_partition_mut_allows_nested_tile_overhang() {
    common::with_test_stack(|| {
        const ROWS: usize = 3;
        const N: usize = 10;
        const BLOCK: usize = 4;

        let expected: Vec<f32> = (0..(ROWS * N)).map(|x| x as f32).collect();
        let input = api::copy_host_vec_to_device(&Arc::new(expected.clone()))
            .sync()
            .expect("input alloc");
        let input_2d = input.view(&[ROWS, N]).expect("input view");
        let nested_out = api::zeros::<f32>(&[ROWS, N]).sync().expect("nested alloc");

        let (nested_result, _) = nested_partition_mut_module::nested_partition_mut_copy(
            nested_out.partition([1, N]),
            &input_2d,
        )
        .generics(vec![N.to_string(), BLOCK.to_string()])
        .sync()
        .expect("nested overhang copy");
        let nested_host: Vec<f32> = nested_result
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("nested to_host");

        assert_matrix_eq(
            &nested_host,
            &expected,
            ROWS,
            N,
            "nested partition_mut overhang",
        );
    });
}

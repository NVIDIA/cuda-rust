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
//!
//! A `-1` extent on the mutable parameter must keep the per-CTA slab
//! semantics of a static one, so the static kernel is the oracle.

use std::process::Command;
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

    #[cutile::entry()]
    fn static_slab_cell_copy<const SLAB_ROWS: i32, const SLAB_COLS: i32>(
        out: &mut Tensor<f32, { [SLAB_ROWS, SLAB_COLS] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        local_row: i32,
        local_col: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let input_part: Partition<f32, { [1, 1] }> = input.partition(shape![1, 1]);
        let tile: Tile<f32, { [1, 1] }> = input_part.load([pid.0 * SLAB_ROWS, pid.1 * SLAB_COLS]);
        let mut out_part: PartitionMut<f32, { [1, 1] }> = out.partition_mut(shape![1, 1]);
        out_part.store(tile, [local_row, local_col]);
    }

    #[cutile::entry()]
    fn dynamic_slab_cell_copy<const SLAB_ROWS: i32, const SLAB_COLS: i32>(
        out: &mut Tensor<f32, { [-1, -1] }>,
        input: &Tensor<f32, { [-1, -1] }>,
        local_row: i32,
        local_col: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let input_part: Partition<f32, { [1, 1] }> = input.partition(shape![1, 1]);
        let tile: Tile<f32, { [1, 1] }> = input_part.load([pid.0 * SLAB_ROWS, pid.1 * SLAB_COLS]);
        let mut out_part: PartitionMut<f32, { [1, 1] }> = out.partition_mut(shape![1, 1]);
        out_part.store(tile, [local_row, local_col]);
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

#[derive(Clone, Copy, Debug)]
enum SlabExtent {
    Static,
    Dynamic,
}

/// Each CTA stores the input cell at its slab origin to `local` slab coordinates.
#[derive(Clone, Copy)]
struct CellCopy {
    root: [usize; 2],
    slab: [usize; 2],
    local: [i32; 2],
}

const OUT_OF_SLAB_STORE: CellCopy = CellCopy {
    root: [4, 8],
    slab: [2, 8],
    local: [2, 0],
};

fn copy_cell(extent: SlabExtent, case: CellCopy) -> Result<Vec<f32>, String> {
    let [rows, cols] = case.root;
    let [local_row, local_col] = case.local;
    let input_host: Vec<f32> = (0..rows * cols).map(|x| x as f32 + 1.0).collect();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host))
        .sync()
        .expect("input alloc");
    let input_2d = input.view(&[rows, cols]).expect("input view");
    let out = api::zeros::<f32>(&[rows, cols]).sync().expect("out alloc");
    let out = out.partition(case.slab);
    let generics = case.slab.iter().map(|dim| dim.to_string()).collect();

    let launched = match extent {
        SlabExtent::Static => {
            nested_partition_mut_module::static_slab_cell_copy(out, &input_2d, local_row, local_col)
                .generics(generics)
                .sync()
        }
        SlabExtent::Dynamic => nested_partition_mut_module::dynamic_slab_cell_copy(
            out, &input_2d, local_row, local_col,
        )
        .generics(generics)
        .sync(),
    };
    let (result, ..) = launched.map_err(|error| format!("{error:?}"))?;
    Ok(result.unpartition().to_host_vec().sync().expect("to_host"))
}

fn assert_dynamic_matches_static(case: CellCopy, label: &'static str) {
    common::with_test_stack(move || {
        let expected = copy_cell(SlabExtent::Static, case).expect("static slab copy");
        let actual = copy_cell(SlabExtent::Dynamic, case).expect("dynamic slab copy");
        assert_matrix_eq(&actual, &expected, case.root[0], case.root[1], label);
    });
}

#[test]
fn dynamic_extent_single_slab_matches_static() {
    let case = CellCopy {
        root: [4, 8],
        slab: [4, 8],
        local: [3, 0],
    };
    assert_dynamic_matches_static(case, "single slab");
}

#[test]
fn dynamic_extent_row_slabs_match_static() {
    let case = CellCopy {
        root: [4, 8],
        slab: [2, 8],
        local: [1, 0],
    };
    assert_dynamic_matches_static(case, "row slabs");
}

#[test]
fn dynamic_extent_2d_slabs_match_static() {
    let case = CellCopy {
        root: [4, 8],
        slab: [2, 4],
        local: [1, 3],
    };
    assert_dynamic_matches_static(case, "2-D slabs");
}

#[test]
fn dynamic_extent_ragged_slabs_match_static() {
    let case = CellCopy {
        root: [5, 8],
        slab: [2, 8],
        local: [0, 5],
    };
    assert_dynamic_matches_static(case, "ragged slabs");
}

/// Subprocess entry point: an out-of-slab store may trap, which poisons the
/// CUDA context, so it runs alone and reports one `OUTCOME:` line.
#[test]
#[ignore]
fn out_of_slab_store_runner() {
    let extent = match std::env::var("CUTILE_SLAB_EXTENT").as_deref() {
        Ok("static") => SlabExtent::Static,
        Ok("dynamic") => SlabExtent::Dynamic,
        other => panic!("CUTILE_SLAB_EXTENT must be static or dynamic, got {other:?}"),
    };
    common::with_test_stack(move || match copy_cell(extent, OUT_OF_SLAB_STORE) {
        Ok(out) => println!("OUTCOME:WROTE:{out:?}"),
        Err(error) => println!("OUTCOME:STOP:{error}"),
    });
}

fn out_of_slab_store_outcome(extent: SlabExtent) -> String {
    let value = format!("{extent:?}").to_lowercase();
    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args([
            "--exact",
            "out_of_slab_store_runner",
            "--ignored",
            "--nocapture",
        ])
        .env("CUTILE_SLAB_EXTENT", value)
        .output()
        .expect("spawn out-of-slab runner");
    let stdout = String::from_utf8_lossy(&out.stdout);
    match stdout
        .lines()
        .find_map(|line| line.strip_prefix("OUTCOME:"))
    {
        Some(outcome) => outcome.to_string(),
        None => format!(
            "ABORTED:{}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .find(|line| line.contains("panicked"))
                .unwrap_or("no panic line")
        ),
    }
}

#[test]
fn dynamic_extent_out_of_slab_store_stops_like_static() {
    assert_eq!(
        out_of_slab_store_outcome(SlabExtent::Dynamic),
        out_of_slab_store_outcome(SlabExtent::Static),
        "a store one row past the slab must be refused the way the static slab refuses it"
    );
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

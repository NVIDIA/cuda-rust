/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device value-correctness tests for mapped partition schedules beyond
//! rank 2: rank-3 and rank-1 `iter_indices()` and shared index streams via
//! `iter_indices_with()`. Each kernel routes patterned input tiles through
//! the schedule under test, so a misrouted or duplicated `PartitionIndex`
//! scrambles the output and fails the elementwise comparison. Rank-2 grouped
//! schedules are covered by the `persistent_gemm` example.

use crate::common;
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod mapped_value_kernels {
    use cutile::core::*;

    #[cutile::entry]
    fn rank3_copy<T: ElementType, const BS: i32, const BD: i32, const MAP_SHAPE: [i32; 3]>(
        mut z: MappedPartitionMut<T, { [1, BS, BD] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1, -1] }>,
    ) {
        let part_x = x.partition(shape![1, BS, BD]);
        for index in z.iter_indices() {
            let (bid_b, bid_s, bid_d) = index.components();
            let tile = part_x.load([bid_b, bid_s, bid_d]);
            z.store(tile, index);
        }
    }

    #[cutile::entry]
    fn rank1_copy<T: ElementType, const BN: i32, const MAP_SHAPE: [i32; 1]>(
        mut z: MappedPartitionMut<T, { [BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1] }>,
    ) {
        let part_x = x.partition(shape![BN]);
        for index in z.iter_indices() {
            let coords = index.coords();
            let tile = part_x.load([coords[0]]);
            z.store(tile, index);
        }
    }

    #[cutile::entry]
    fn shared_map_copy_and_double<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut doubled_out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in out.iter_indices_with(&doubled_out) {
            let (bid_m, bid_n) = index.components();
            let tile = part_x.load([bid_m, bid_n]);
            out.store(tile, index);
            doubled_out.store(tile + tile, index);
        }
    }

    /// KV-cache-shaped: copy input tiles into a sub-range of rows only,
    /// with runtime start/len (the dynpos codepath).
    #[cutile::entry]
    fn subrange_copy<T: ElementType, const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in z.iter_indices_within([(start_tile, n_tiles), (0i32, -1i32)]) {
            let (bid_m, bid_n) = index.components();
            let tile = part_x.load([bid_m, bid_n]);
            z.store(tile, index);
        }
    }

    /// Sub-range composed with a shared index stream: dual outputs written
    /// only inside the row range.
    #[cutile::entry]
    fn subrange_shared_copy_and_double<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut doubled_out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in
            out.iter_indices_within_with([(start_tile, n_tiles), (0i32, -1i32)], &doubled_out)
        {
            let (bid_m, bid_n) = index.components();
            let tile = part_x.load([bid_m, bid_n]);
            out.store(tile, index);
            doubled_out.store(tile + tile, index);
        }
    }

    /// Pipelined safe load: the latency hint must not change values.
    #[cutile::entry]
    fn pipelined_copy<T: ElementType, const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let tile = part_x.load_pipelined::<4>([bid_m, bid_n]);
            z.store(tile, index);
        }
    }
}

/// Launch helper for the mismatched-ntb probe in the schedule-matrix tests:
/// returns the launch result instead of unwrapping.
pub fn shared_map_kernel_for_ntb_probe(
    out: cutile::tensor::MappedLaunchPartition<Partition<Tensor<f32>>>,
    doubled_out: cutile::tensor::MappedLaunchPartition<Partition<Tensor<f32>>>,
    x: Arc<Tensor<f32>>,
    generics: Vec<String>,
) -> Result<(), cuda_async::error::DeviceError> {
    let _ = mapped_value_kernels::shared_map_copy_and_double(out, doubled_out, x)
        .generics(generics)
        .sync()?;
    Ok(())
}

fn patterned(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.5 + ((i % 7) as f32) * 0.25)
        .collect()
}

/// Outputs are poison-initialized, never zero-initialized: a spurious write
/// of zero into a zero-filled buffer is invisible, so untouched-region and
/// full-coverage assertions only mean something when the sentinel is a value
/// no kernel under test can produce. Outside `patterned`'s range.
const POISON: f32 = -333.25;

fn assert_elementwise(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "{label}: element {i} is {a}, expected {e}"
        );
    }
}

fn run_rank3_copy(map_shape: [usize; 3]) {
    const B: usize = 4;
    const S: usize = 8;
    const D: usize = 16;
    let x_host = patterned(B * S * D);
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
        .reshape(&[B, S, D])
        .sync()
        .unwrap()
        .into();
    // Tile [1, 4, 8] over [4, 8, 16] → a (4, 2, 2) logical grid walked by 4
    // persistent tile blocks (4 indices each).
    let z = api::full::<f32>(POISON, &[B, S, D])
        .sync()
        .unwrap()
        .partition([1, 4, 8])
        .map(map_shape, 4);
    let generics = vec![
        "f32".to_string(),
        "4".to_string(),
        "8".to_string(),
        map_shape[0].to_string(),
        map_shape[1].to_string(),
        map_shape[2].to_string(),
    ];
    let (z, _x) = mapped_value_kernels::rank3_copy(z, x)
        .generics(generics)
        .sync()
        .expect("rank-3 mapped kernel should run");
    let z_host = z.unpartition().to_host_vec().sync().unwrap();
    assert_elementwise(&z_host, &x_host, &format!("rank3 map {map_shape:?}"));
}

#[test]
fn rank3_mapped_store_routes_tiles_correctly() {
    common::with_test_stack(|| run_rank3_copy([1, 1, 1]));
}

#[test]
fn rank3_mapped_store_routes_tiles_correctly_grouped() {
    common::with_test_stack(|| run_rank3_copy([1, 2, 1]));
}

#[test]
fn rank1_mapped_store_routes_tiles_correctly() {
    common::with_test_stack(|| {
        const N: usize = 64;
        let x_host = patterned(N);
        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
            .reshape(&[N])
            .sync()
            .unwrap()
            .into();
        // Tile [16] over [64] → 4 logical tiles walked by 2 persistent blocks.
        let z = api::full::<f32>(POISON, &[N])
            .sync()
            .unwrap()
            .partition([16])
            .map([1], 2);
        let generics = vec!["f32".to_string(), "16".to_string(), "1".to_string()];
        let (z, _x) = mapped_value_kernels::rank1_copy(z, x)
            .generics(generics)
            .sync()
            .expect("rank-1 mapped kernel should run");
        let z_host = z.unpartition().to_host_vec().sync().unwrap();
        assert_elementwise(&z_host, &x_host, "rank1");
    });
}

fn run_shared_map(map_shape: [usize; 2]) {
    const M: usize = 64;
    const N: usize = 64;
    let x_host = patterned(M * N);
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
        .reshape(&[M, N])
        .sync()
        .unwrap()
        .into();
    // Tile [16, 16] over [64, 64] → a (4, 4) logical grid walked by 4
    // persistent tile blocks through one shared index stream.
    let out = api::full::<f32>(POISON, &[M, N])
        .sync()
        .unwrap()
        .partition([16, 16])
        .map(map_shape, 4);
    let doubled_out = api::full::<f32>(POISON, &[M, N])
        .sync()
        .unwrap()
        .partition([16, 16])
        .map(map_shape, 4);
    let generics = vec![
        "f32".to_string(),
        "16".to_string(),
        "16".to_string(),
        map_shape[0].to_string(),
        map_shape[1].to_string(),
    ];
    let (out, doubled_out, _x) =
        mapped_value_kernels::shared_map_copy_and_double(out, doubled_out, x)
            .generics(generics)
            .sync()
            .expect("shared-map kernel should run");
    let out_host = out.unpartition().to_host_vec().sync().unwrap();
    let doubled_host = doubled_out.unpartition().to_host_vec().sync().unwrap();
    let doubled_expected: Vec<f32> = x_host.iter().map(|v| v * 2.0).collect();
    assert_elementwise(&out_host, &x_host, &format!("shared out map {map_shape:?}"));
    assert_elementwise(
        &doubled_host,
        &doubled_expected,
        &format!("shared doubled map {map_shape:?}"),
    );
}

#[test]
fn shared_map_stores_route_to_both_partitions() {
    common::with_test_stack(|| run_shared_map([1, 1]));
}

#[test]
fn shared_map_stores_route_to_both_partitions_grouped() {
    common::with_test_stack(|| run_shared_map([2, 1]));
}

const SUB_M: usize = 64;
const SUB_N: usize = 64;
const SUB_TILE: usize = 16;
const SUB_START_TILE: i32 = 1;
const SUB_N_TILES: i32 = 2;

/// Expected output when rows `[16, 48)` (tiles 1..3 of 4) are written from
/// `x` (optionally scaled) and everything else keeps its poison fill.
fn subrange_expected(x_host: &[f32], scale: f32) -> Vec<f32> {
    let row_lo = SUB_START_TILE as usize * SUB_TILE;
    let row_hi = row_lo + SUB_N_TILES as usize * SUB_TILE;
    (0..SUB_M * SUB_N)
        .map(|i| {
            let row = i / SUB_N;
            if row >= row_lo && row < row_hi {
                x_host[i] * scale
            } else {
                POISON
            }
        })
        .collect()
}

#[test]
fn subrange_mapped_store_writes_only_the_range() {
    common::with_test_stack(|| {
        let x_host = patterned(SUB_M * SUB_N);
        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
            .reshape(&[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .into();
        let z = api::full::<f32>(POISON, &[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .partition([SUB_TILE, SUB_TILE])
            .map([1, 1], 4);
        let generics = vec![
            "f32".to_string(),
            SUB_TILE.to_string(),
            SUB_TILE.to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let (z, _x, _start, _n) =
            mapped_value_kernels::subrange_copy(z, x, SUB_START_TILE, SUB_N_TILES)
                .generics(generics)
                .sync()
                .expect("sub-range mapped kernel should run");
        let z_host = z.unpartition().to_host_vec().sync().unwrap();
        assert_elementwise(&z_host, &subrange_expected(&x_host, 1.0), "subrange");
    });
}

#[test]
fn subrange_shared_map_writes_both_outputs_in_range_only() {
    common::with_test_stack(|| {
        let x_host = patterned(SUB_M * SUB_N);
        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
            .reshape(&[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .into();
        let out = api::full::<f32>(POISON, &[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .partition([SUB_TILE, SUB_TILE])
            .map([1, 1], 4);
        let doubled_out = api::full::<f32>(POISON, &[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .partition([SUB_TILE, SUB_TILE])
            .map([1, 1], 4);
        let generics = vec![
            "f32".to_string(),
            SUB_TILE.to_string(),
            SUB_TILE.to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let (out, doubled_out, _x, _start, _n) =
            mapped_value_kernels::subrange_shared_copy_and_double(
                out,
                doubled_out,
                x,
                SUB_START_TILE,
                SUB_N_TILES,
            )
            .generics(generics)
            .sync()
            .expect("shared sub-range kernel should run");
        let out_host = out.unpartition().to_host_vec().sync().unwrap();
        let doubled_host = doubled_out.unpartition().to_host_vec().sync().unwrap();
        assert_elementwise(
            &out_host,
            &subrange_expected(&x_host, 1.0),
            "subrange shared out",
        );
        assert_elementwise(
            &doubled_host,
            &subrange_expected(&x_host, 2.0),
            "subrange shared doubled",
        );
    });
}

#[test]
fn pipelined_safe_load_preserves_values() {
    common::with_test_stack(|| {
        let x_host = patterned(SUB_M * SUB_N);
        let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
            .reshape(&[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .into();
        let z = api::full::<f32>(POISON, &[SUB_M, SUB_N])
            .sync()
            .unwrap()
            .partition([SUB_TILE, SUB_TILE])
            .map([1, 1], 4);
        let generics = vec![
            "f32".to_string(),
            SUB_TILE.to_string(),
            SUB_TILE.to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let (z, _x) = mapped_value_kernels::pipelined_copy(z, x)
            .generics(generics)
            .sync()
            .expect("pipelined copy kernel should run");
        let z_host = z.unpartition().to_host_vec().sync().unwrap();
        assert_elementwise(&z_host, &x_host, "pipelined copy");
    });
}

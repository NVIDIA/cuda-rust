/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exhaustive coverage-and-disjointness matrix for mapped partition
//! schedules, plus sub-range and overhang edge cases.
//!
//! The matrix kernels write each tile's flat logical index as the tile
//! value into a poison-filled output, so after the launch every element
//! must equal its own tile's index: a swizzle off-by-one shows up as a
//! wrong index somewhere, a dropped tile as surviving poison, and a
//! misrouted duplicate as a wrong index in the tile it clobbered (with the
//! tile it came from left poisoned — tile count equals index count, so
//! duplication implies omission). Ranks 1-3 are crossed with grouped
//! trailing-pair maps, persistent tile-block counts of 1 / a non-divisor /
//! the full grid, and logical grids that do not divide evenly by the map
//! shape (exercising the remainder-band `min` path).

use crate::common;
use cutile::prelude::*;
use std::sync::Arc;

const POISON_I32: i32 = -0x5EAD;
const POISON_F32: f32 = -333.25;

#[cutile::module]
mod schedule_matrix_kernels {
    use cutile::core::*;

    #[cutile::entry]
    fn write_flat_index_rank1<const BN: i32, const MAP_SHAPE: [i32; 1]>(
        mut z: MappedPartitionMut<i32, { [BN] }, MAP_SHAPE>,
    ) {
        for index in z.iter_indices() {
            let coords = index.coords();
            let tile: Tile<i32, { [BN] }> = broadcast_scalar(coords[0], shape![BN]);
            z.store(tile, index);
        }
    }

    #[cutile::entry]
    fn write_flat_index_rank2<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<i32, { [BM, BN] }, MAP_SHAPE>,
    ) {
        let g1 = num_tiles(&z, 1);
        for index in z.iter_indices() {
            let (c0, c1) = index.components();
            let flat: i32 = c0 * g1 + c1;
            let tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(flat, shape![BM, BN]);
            z.store(tile, index);
        }
    }

    #[cutile::entry]
    fn write_flat_index_rank3<
        const B0: i32,
        const B1: i32,
        const B2: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        mut z: MappedPartitionMut<i32, { [B0, B1, B2] }, MAP_SHAPE>,
    ) {
        let g1 = num_tiles(&z, 1);
        let g2 = num_tiles(&z, 2);
        for index in z.iter_indices() {
            let (c0, c1, c2) = index.components();
            let flat: i32 = (c0 * g1 + c1) * g2 + c2;
            let tile: Tile<i32, { [B0, B1, B2] }> = broadcast_scalar(flat, shape![B0, B1, B2]);
            z.store(tile, index);
        }
    }

    /// Sub-range flat-index writer: only tiles inside the axis-0 range may
    /// be written, everything else must keep its poison fill.
    #[cutile::entry]
    fn write_flat_index_subrange<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<i32, { [BM, BN] }, MAP_SHAPE>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        let g1 = num_tiles(&z, 1);
        for index in z.iter_indices_within([(start_tile, n_tiles), (0i32, -1i32)]) {
            let (c0, c1) = index.components();
            let flat: i32 = c0 * g1 + c1;
            let tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(flat, shape![BM, BN]);
            z.store(tile, index);
        }
    }

    /// Remainder spelling: runtime start with the literal `-1` length
    /// ("rest of the axis"), which must resolve at compile time.
    #[cutile::entry]
    fn write_flat_index_subrange_rest<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<i32, { [BM, BN] }, MAP_SHAPE>,
        start_tile: i32,
    ) {
        let g1 = num_tiles(&z, 1);
        for index in z.iter_indices_within([(start_tile, -1i32), (0i32, -1i32)]) {
            let (c0, c1) = index.components();
            let flat: i32 = c0 * g1 + c1;
            let tile: Tile<i32, { [BM, BN] }> = broadcast_scalar(flat, shape![BM, BN]);
            z.store(tile, index);
        }
    }

    /// The grout KV shape: rank-3 shared maps with the seq-axis sub-range
    /// start read from device memory (dynpos).
    #[cutile::entry]
    fn dynpos_shared_rank3<const BS: i32, const BD: i32, const MAP_SHAPE: [i32; 3]>(
        mut k_cache: MappedPartitionMut<f32, { [1, BS, BD] }, MAP_SHAPE>,
        mut v_cache: MappedPartitionMut<f32, { [1, BS, BD] }, MAP_SHAPE>,
        new_k: &Tensor<f32, { [-1, -1, -1] }>,
        pos: &Tensor<i32, { [1] }>,
        n_tiles: i32,
    ) {
        let pos_tile: Tile<i32, { [1] }> = pos.partition(shape![1]).load([0i32]);
        let pos_scalar: i32 = tile_to_scalar(pos_tile.reshape(shape![]));
        let part_k = new_k.partition(shape![1, BS, BD]);
        for index in k_cache.iter_indices_within_with(
            [(0i32, -1i32), (pos_scalar, n_tiles), (0i32, -1i32)],
            &v_cache,
        ) {
            let (h, s, d) = index.components();
            // Source row for cache row s is s - pos.
            let tile = part_k.load([h, s - pos_scalar, d]);
            k_cache.store(tile, index);
            v_cache.store(tile + tile, index);
        }
    }

    /// The grout prefill shape: the cache's sequence axis is much larger than
    /// the source, but the mapped stream visits only `[0, seq_len)`. The
    /// projected `s` coordinate therefore proves a cache store, not a
    /// full-cache-axis bound for the `new_k` load.
    #[cutile::entry]
    fn prefill_shared_rank3<const BS: i32, const BD: i32, const MAP_SHAPE: [i32; 3]>(
        mut k_cache: MappedPartitionMut<f32, { [1, BS, BD] }, MAP_SHAPE>,
        mut v_cache: MappedPartitionMut<f32, { [1, BS, BD] }, MAP_SHAPE>,
        new_k: &Tensor<f32, { [-1, -1, -1] }>,
        seq_len: i32,
    ) {
        let part_k = new_k.partition(shape![1, BS, BD]);
        for index in k_cache
            .iter_indices_within_with([(0i32, -1i32), (0i32, seq_len), (0i32, -1i32)], &v_cache)
        {
            let [head, s, d] = index.coords();
            let tile = part_k.load([s, head, d]);
            k_cache.store(tile, index);
            v_cache.store(tile + tile, index);
        }
    }

    /// Overhang: tile shape overhangs the tensor edge (BN > columns).
    /// Checked loads must pad the out-of-bounds lanes and mapped stores
    /// must drop the overhang instead of corrupting adjacent memory.
    #[cutile::entry]
    fn overhang_copy<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in z.iter_indices() {
            let (c0, c1) = index.components();
            let tile = part_x.load([c0, c1]);
            z.store(tile, index);
        }
    }

    /// Overhang combined with pipelined loads (the risky combo: pipelining
    /// hints on masked boundary tiles).
    #[cutile::entry]
    fn overhang_copy_pipelined<
        const BM: i32,
        const BN: i32,
        const L: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BN]);
        for index in z.iter_indices() {
            let (c0, c1) = index.components();
            let tile = part_x.load_pipelined::<L>([c0, c1]);
            z.store(tile, index);
        }
    }
}

fn to_host_i32(z: Partition<Tensor<i32>>) -> Vec<i32> {
    z.unpartition().to_host_vec().sync().unwrap()
}

/// Every element must equal its tile's flat logical index.
fn assert_flat_index_coverage(actual: &[i32], dims: &[usize], tiles: &[usize], label: &str) {
    assert_eq!(actual.len(), dims.iter().product::<usize>(), "{label}: len");
    let rank = dims.len();
    let grid: Vec<usize> = (0..rank).map(|a| dims[a].div_ceil(tiles[a])).collect();
    for (i, &value) in actual.iter().enumerate() {
        // Element index → coordinates → owning tile → flat tile index.
        let mut rem = i;
        let mut flat = 0usize;
        for axis in 0..rank {
            let inner: usize = dims[axis + 1..].iter().product();
            let coord = rem / inner;
            rem %= inner;
            flat = flat * grid[axis] + coord / tiles[axis];
        }
        assert_eq!(
            value, flat as i32,
            "{label}: element {i} belongs to tile {flat} but holds {value} \
             (poison = {POISON_I32}; poison here means the tile was never written)"
        );
    }
}

#[test]
fn rank1_schedule_matrix_covers_every_tile_exactly() {
    common::with_test_stack(|| {
        for ntb in [1u32, 3, 4] {
            let z = api::full::<i32>(POISON_I32, &[64])
                .sync()
                .unwrap()
                .partition([16])
                .map([1], ntb);
            let generics = vec!["16".to_string(), "1".to_string()];
            let (z,) = schedule_matrix_kernels::write_flat_index_rank1(z)
                .generics(generics)
                .sync()
                .unwrap_or_else(|e| panic!("rank1 ntb={ntb}: {e}"));
            assert_flat_index_coverage(&to_host_i32(z), &[64], &[16], &format!("rank1 ntb={ntb}"));
        }
    });
}

#[test]
fn rank2_schedule_matrix_covers_every_tile_exactly() {
    common::with_test_stack(|| {
        // Grids (4, 4) and (3, 5): the latter does not divide evenly by any
        // grouped map below, exercising the remainder-band min path.
        for (rows, cols) in [(4usize, 4usize), (3, 5)] {
            let dims = [rows * 8, cols * 8];
            let total = (rows * cols) as u32;
            for map in [[1usize, 1], [2, 1], [2, 2], [3, 2]] {
                for ntb in [1u32, 5.min(total), total] {
                    let z = api::full::<i32>(POISON_I32, &dims)
                        .sync()
                        .unwrap()
                        .partition([8, 8])
                        .map(map, ntb);
                    let generics = vec![
                        "8".to_string(),
                        "8".to_string(),
                        map[0].to_string(),
                        map[1].to_string(),
                    ];
                    let label = format!("rank2 grid=({rows},{cols}) map={map:?} ntb={ntb}");
                    let (z,) = schedule_matrix_kernels::write_flat_index_rank2(z)
                        .generics(generics)
                        .sync()
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_flat_index_coverage(&to_host_i32(z), &dims, &[8, 8], &label);
                }
            }
        }
    });
}

#[test]
fn rank3_schedule_matrix_covers_every_tile_exactly() {
    common::with_test_stack(|| {
        // Logical grid (2, 3, 2) = 12 tiles; (1, 2, 2) does not divide the
        // (3, 2) trailing pair evenly.
        let dims = [2usize, 3 * 4, 2 * 8];
        for map in [[1usize, 1, 1], [1, 2, 1], [1, 2, 2], [1, 3, 2]] {
            for ntb in [1u32, 5, 12] {
                let z = api::full::<i32>(POISON_I32, &dims)
                    .sync()
                    .unwrap()
                    .partition([1, 4, 8])
                    .map(map, ntb);
                let generics = vec![
                    "1".to_string(),
                    "4".to_string(),
                    "8".to_string(),
                    map[0].to_string(),
                    map[1].to_string(),
                    map[2].to_string(),
                ];
                let label = format!("rank3 map={map:?} ntb={ntb}");
                let (z,) = schedule_matrix_kernels::write_flat_index_rank3(z)
                    .generics(generics)
                    .sync()
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_flat_index_coverage(&to_host_i32(z), &dims, &[1, 4, 8], &label);
            }
        }
    });
}

/// Expected sub-range output: flat tile index inside the axis-0 tile range,
/// poison elsewhere.
fn subrange_flat_expected(dims: [usize; 2], tiles: [usize; 2], lo: usize, hi: usize) -> Vec<i32> {
    let grid1 = dims[1].div_ceil(tiles[1]);
    (0..dims[0] * dims[1])
        .map(|i| {
            let (r, c) = (i / dims[1], i % dims[1]);
            let (t0, t1) = (r / tiles[0], c / tiles[1]);
            if t0 >= lo && t0 < hi {
                (t0 * grid1 + t1) as i32
            } else {
                POISON_I32
            }
        })
        .collect()
}

fn run_subrange_case(map: [usize; 2], start: i32, len: i32, expect_lo: usize, expect_hi: usize) {
    let dims = [64usize, 64];
    let z = api::full::<i32>(POISON_I32, &dims)
        .sync()
        .unwrap()
        .partition([16, 16])
        .map(map, 4);
    let generics = vec![
        "16".to_string(),
        "16".to_string(),
        map[0].to_string(),
        map[1].to_string(),
    ];
    let label = format!("subrange map={map:?} start={start} len={len}");
    // The `-1` rest-of-axis spelling must be a source literal, so the
    // remainder cases go through a kernel that hardcodes it; passing -1
    // through a runtime scalar is a negative length by contract.
    let z = if len == -1 {
        let (z, _s) = schedule_matrix_kernels::write_flat_index_subrange_rest(z, start)
            .generics(generics)
            .sync()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        z
    } else {
        let (z, _s, _n) = schedule_matrix_kernels::write_flat_index_subrange(z, start, len)
            .generics(generics)
            .sync()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        z
    };
    let expected = subrange_flat_expected(dims, [16, 16], expect_lo, expect_hi);
    assert_eq!(to_host_i32(z), expected, "{label}");
}

#[test]
fn subrange_edge_cases_write_exactly_the_range() {
    common::with_test_stack(|| {
        // Empty range: the loop body must not run at all.
        run_subrange_case([1, 1], 1, 0, 0, 0);
        // Start at the last tile.
        run_subrange_case([1, 1], 3, 1, 3, 4);
        // (start, -1): the rest of the axis from a nonzero start.
        run_subrange_case([1, 1], 2, -1, 2, 4);
        // (0, -1) covers the whole axis, equivalent to plain iter_indices.
        run_subrange_case([1, 1], 0, -1, 0, 4);
        // Sub-range on a grouped axis: the swizzle applies within the range.
        run_subrange_case([2, 1], 1, 2, 1, 3);
        run_subrange_case([2, 2], 0, 3, 0, 3);
    });
}

#[test]
fn dynpos_shared_rank3_updates_only_the_device_read_range() {
    common::with_test_stack(|| {
        // Cache [heads=2, max_seq=8, D=16], tiles [1, 1, 16] → grid (2, 8, 1).
        // Position 3 (device-resident), seq_len 2 → rows 3..5 written.
        const H: usize = 2;
        const MAX_SEQ: usize = 8;
        const D: usize = 16;
        let src: Vec<f32> = (0..H * 2 * D).map(|i| i as f32 * 0.5 + 1.0).collect();
        let new_k: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(src.clone()))
            .reshape(&[H, 2, D])
            .sync()
            .unwrap()
            .into();
        let pos: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(vec![3i32]))
            .reshape(&[1])
            .sync()
            .unwrap()
            .into();
        let k_cache = api::full::<f32>(POISON_F32, &[H, MAX_SEQ, D])
            .sync()
            .unwrap()
            .partition([1, 1, D])
            .map([1, 1, 1], 4);
        let v_cache = api::full::<f32>(POISON_F32, &[H, MAX_SEQ, D])
            .sync()
            .unwrap()
            .partition([1, 1, D])
            .map([1, 1, 1], 4);
        let generics = vec![
            "1".to_string(),
            D.to_string(),
            "1".to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let (k_cache, v_cache, _k, _p, _n) =
            schedule_matrix_kernels::dynpos_shared_rank3(k_cache, v_cache, new_k, pos, 2i32)
                .generics(generics)
                .sync()
                .expect("dynpos shared rank-3 kernel should run");
        let k_host = k_cache.unpartition().to_host_vec().sync().unwrap();
        let v_host = v_cache.unpartition().to_host_vec().sync().unwrap();
        for h in 0..H {
            for s in 0..MAX_SEQ {
                for d in 0..D {
                    let i = (h * MAX_SEQ + s) * D + d;
                    let (k_expect, v_expect) = if (3..5).contains(&s) {
                        let src_value = src[(h * 2 + (s - 3)) * D + d];
                        (src_value, src_value * 2.0)
                    } else {
                        (POISON_F32, POISON_F32)
                    };
                    assert_eq!(k_host[i], k_expect, "k_cache[{h},{s},{d}]");
                    assert_eq!(v_host[i], v_expect, "v_cache[{h},{s},{d}]");
                }
            }
        }
    });
}

#[test]
fn prefill_shared_rank3_bounds_the_source_by_the_subrange_not_the_cache() {
    common::with_test_stack(|| {
        // Grout's layout: cache [heads, max_seq, D], source [seq, heads, D].
        // MAX_SEQ deliberately exceeds SEQ; only cache rows 0..SEQ are visited.
        const H: usize = 2;
        const SEQ: usize = 2;
        const MAX_SEQ: usize = 8;
        const D: usize = 16;
        let src: Vec<f32> = (0..SEQ * H * D).map(|i| i as f32 * 0.25 + 1.0).collect();
        let new_k: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(src.clone()))
            .reshape(&[SEQ, H, D])
            .sync()
            .unwrap()
            .into();
        let k_cache = api::full::<f32>(POISON_F32, &[H, MAX_SEQ, D])
            .sync()
            .unwrap()
            .partition([1, 1, D])
            .map([1, 1, 1], (H * SEQ) as u32);
        let v_cache = api::full::<f32>(POISON_F32, &[H, MAX_SEQ, D])
            .sync()
            .unwrap()
            .partition([1, 1, D])
            .map([1, 1, 1], (H * SEQ) as u32);
        let generics = vec![
            "1".to_string(),
            D.to_string(),
            "1".to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let (k_cache, v_cache, _k, _n) =
            schedule_matrix_kernels::prefill_shared_rank3(k_cache, v_cache, new_k, SEQ as i32)
                .generics(generics)
                .sync()
                .expect("a source covering the mapped sub-range should run");
        let k_host = k_cache.unpartition().to_host_vec().sync().unwrap();
        let v_host = v_cache.unpartition().to_host_vec().sync().unwrap();
        for h in 0..H {
            for s in 0..MAX_SEQ {
                for d in 0..D {
                    let cache_i = (h * MAX_SEQ + s) * D + d;
                    let (k_expect, v_expect) = if s < SEQ {
                        let value = src[(s * H + h) * D + d];
                        (value, value * 2.0)
                    } else {
                        (POISON_F32, POISON_F32)
                    };
                    assert_eq!(k_host[cache_i], k_expect, "k_cache[{h},{s},{d}]");
                    assert_eq!(v_host[cache_i], v_expect, "v_cache[{h},{s},{d}]");
                }
            }
        }
    });
}

fn run_overhang_case(pipelined: bool) {
    // Tensor [40, 48], tiles [16, 32]: grid (3, 2) with overhang on both
    // axes (rows 40 → last row tile is 8 deep, cols 48 → last col tile is
    // 16 wide). Checked loads pad, mapped stores drop the overhang.
    let dims = [40usize, 48];
    let x_host: Vec<f32> = (0..dims[0] * dims[1])
        .map(|i| ((i % 11) as f32 - 5.0) * 0.75)
        .collect();
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
        .reshape(&dims)
        .sync()
        .unwrap()
        .into();
    let z = api::full::<f32>(POISON_F32, &dims)
        .sync()
        .unwrap()
        .partition([16, 32])
        .map([1, 1], 3);
    let generics: Vec<String> = if pipelined {
        vec!["16".into(), "32".into(), "2".into(), "1".into(), "1".into()]
    } else {
        vec!["16".into(), "32".into(), "1".into(), "1".into()]
    };
    let label = format!("overhang pipelined={pipelined}");
    let z_host = if pipelined {
        let (z, _x) = schedule_matrix_kernels::overhang_copy_pipelined(z, x)
            .generics(generics)
            .sync()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        z.unpartition().to_host_vec().sync().unwrap()
    } else {
        let (z, _x) = schedule_matrix_kernels::overhang_copy(z, x)
            .generics(generics)
            .sync()
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        z.unpartition().to_host_vec().sync().unwrap()
    };
    for (i, (actual, expected)) in z_host.iter().zip(x_host.iter()).enumerate() {
        assert_eq!(actual, expected, "{label}: element {i}");
    }
}

#[test]
fn overhang_tiles_load_padded_and_store_clipped() {
    common::with_test_stack(|| run_overhang_case(false));
}

#[test]
fn overhang_tiles_with_pipelined_loads_match() {
    common::with_test_stack(|| run_overhang_case(true));
}

#[test]
fn pipelined_loads_are_value_identical_across_latencies() {
    // Same copy at L ∈ {1, 2, 4, 8} inside a persistent iter_indices loop
    // over masked boundary tiles: the pipelining hint must never change
    // values, only scheduling.
    common::with_test_stack(|| {
        for latency in [1i32, 2, 4, 8] {
            let dims = [40usize, 48];
            let x_host: Vec<f32> = (0..dims[0] * dims[1])
                .map(|i| ((i % 11) as f32 - 5.0) * 0.75)
                .collect();
            let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
                .reshape(&dims)
                .sync()
                .unwrap()
                .into();
            let z = api::full::<f32>(POISON_F32, &dims)
                .sync()
                .unwrap()
                .partition([16, 32])
                .map([1, 1], 3);
            let generics: Vec<String> = vec![
                "16".into(),
                "32".into(),
                latency.to_string(),
                "1".into(),
                "1".into(),
            ];
            let (z, _x) = schedule_matrix_kernels::overhang_copy_pipelined(z, x)
                .generics(generics)
                .sync()
                .unwrap_or_else(|e| panic!("pipelined L={latency}: {e}"));
            let z_host = z.unpartition().to_host_vec().sync().unwrap();
            assert_eq!(
                z_host, x_host,
                "pipelined L={latency} must match plain copy"
            );
        }
    });
}

#[test]
fn shared_maps_with_mismatched_tile_block_counts_error_at_launch() {
    common::with_test_stack(|| {
        // Two mapped outputs requesting different physical grids (4 vs 8)
        // is an inconsistent launch: it must fail, not silently pick one.
        let dims = [64usize, 64];
        let x: Arc<Tensor<f32>> = api::full::<f32>(1.0, &dims).sync().unwrap().into();
        let out = api::full::<f32>(POISON_F32, &dims)
            .sync()
            .unwrap()
            .partition([16, 16])
            .map([1, 1], 4);
        let doubled_out = api::full::<f32>(POISON_F32, &dims)
            .sync()
            .unwrap()
            .partition([16, 16])
            .map([1, 1], 8);
        let generics = vec![
            "f32".to_string(),
            "16".to_string(),
            "16".to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let result = crate::mapped_partition_values::shared_map_kernel_for_ntb_probe(
            out,
            doubled_out,
            x,
            generics,
        );
        assert!(
            result.is_err(),
            "mismatched num_tile_blocks on shared mapped outputs must fail at launch, got Ok"
        );
    });
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU execution tests for reduce, scan, cat, broadcast, iota, and permute ops.
//!
//! Each test compiles a kernel, runs it on the GPU, copies the result back to
//! the host, and verifies correctness.

use cutile::tile_kernel::DeviceOp;
use cutile::{api, tensor::*};
use std::sync::Arc;

mod common;

// ---------------------------------------------------------------------------
// Kernel module
// ---------------------------------------------------------------------------

#[cutile::module]
mod gpu_exec_module {
    use cutile::core::*;

    // 1. Reduce sum: sum elements into a 1-element output.
    #[cutile::entry()]
    fn reduce_sum_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        let tile: Tile<f32, S> = load_tile_mut(input);
        let sum_scalar = reduce(tile, 0i32, 0.0f32, |acc, x| acc + x);
        let sum_tile: Tile<f32, { [1] }> = sum_scalar.reshape(shape![1]);
        result.store(sum_tile);
    }

    // 2. Reduce max: find max element.
    #[cutile::entry()]
    fn reduce_max_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        let tile: Tile<f32, S> = load_tile_mut(input);
        let max_scalar = reduce(tile, 0i32, f32::NEG_INFINITY, |acc, x| max(acc, x));
        let max_tile: Tile<f32, { [1] }> = max_scalar.reshape(shape![1]);
        result.store(max_tile);
    }

    // 3. Scan (prefix sum): cumulative sum, same shape out.
    #[cutile::entry()]
    fn scan_prefix_sum_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = load_tile_mut(output);
        let prefix: Tile<f32, S> = scan_sum(tile, 0i32, reverse::Forward, 0.0f32);
        output.store(prefix);
    }

    // 4. Reduce-and-accumulate: sums a tile and adds to a counter (single block).
    #[cutile::entry()]
    fn reduce_accumulate_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        counter: &mut Tensor<f32, { [1] }>,
    ) {
        let tile: Tile<f32, S> = load_tile_mut(input);
        let sum_scalar = reduce(tile, 0i32, 0.0f32, |acc, x| acc + x);
        let sum_tile: Tile<f32, { [1] }> = sum_scalar.reshape(shape![1]);
        let current: Tile<f32, { [1] }> = load_tile_mut(counter);
        let updated: Tile<f32, { [1] }> = current + sum_tile;
        counter.store(updated);
    }

    // 5. Cat: concatenate two 64-element tiles into a 128-element result.
    #[cutile::entry()]
    fn cat_kernel(
        output: &mut Tensor<f32, { [128] }>,
        a: &mut Tensor<f32, { [64] }>,
        b: &mut Tensor<f32, { [64] }>,
    ) {
        let tile_a: Tile<f32, { [64] }> = load_tile_mut(a);
        let tile_b: Tile<f32, { [64] }> = load_tile_mut(b);
        let combined: Tile<f32, { [128] }> = cat(tile_a, tile_b, 0i32);
        output.store(combined);
    }

    // 6. Broadcast + arithmetic: broadcast a constant scalar and add to input.
    #[cutile::entry()]
    fn broadcast_add_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &mut Tensor<f32, S>,
    ) {
        let tile: Tile<f32, S> = load_tile_mut(input);
        let scalar_tile: Tile<f32, { [1] }> = constant(10.0f32, shape![1]);
        let broadcasted: Tile<f32, S> = scalar_tile.broadcast(output.shape());
        let result: Tile<f32, S> = tile + broadcasted;
        output.store(result);
    }

    // 7. Iota: generate [0, 1, 2, ..., N-1] as i32 tile.
    #[cutile::entry()]
    fn iota_kernel<const S: [i32; 1]>(output: &mut Tensor<i32, S>) {
        let seq: Tile<i32, S> = iota(output.shape());
        output.store(seq);
    }

    // 8. Permute: transpose a 2D tile [8, 16] -> [16, 8].
    #[cutile::entry()]
    fn permute_kernel(output: &mut Tensor<f32, { [16, 8] }>, input: &mut Tensor<f32, { [8, 16] }>) {
        let tile: Tile<f32, { [8, 16] }> = load_tile_mut(input);
        let perm: Array<{ [1, 0] }> = const_array![1, 0];
        let transposed: Tile<f32, { [16, 8] }> = permute(tile, perm);
        output.store(transposed);
    }

    // 9. 2D scans: prefix sums along each axis of an [8, 16] tile, plus a
    // reverse (suffix-sum) scan. Value tests, not just shape tests: the
    // scan dim was once hardcoded to 0 in lowering, which scans preserve
    // shape through, so only executed values can catch a wrong axis.
    #[cutile::entry()]
    fn scan_sum_2d_kernel(
        input: &mut Tensor<i32, { [8, 16] }>,
        out_dim0: &mut Tensor<i32, { [8, 16] }>,
        out_dim1: &mut Tensor<i32, { [8, 16] }>,
        out_dim1_rev: &mut Tensor<i32, { [8, 16] }>,
    ) {
        let tile: Tile<i32, { [8, 16] }> = load_tile_mut(input);
        let d0: Tile<i32, { [8, 16] }> = scan_sum(tile, 0i32, reverse::Forward, 0i32);
        let d1: Tile<i32, { [8, 16] }> = scan_sum(tile, 1i32, reverse::Forward, 0i32);
        let d1r: Tile<i32, { [8, 16] }> = scan_sum(tile, 1i32, reverse::Reverse, 0i32);
        out_dim0.store(d0);
        out_dim1.store(d1);
        out_dim1_rev.store(d1r);
    }

    // 10. 2D reductions along each axis of an [8, 16] tile (the reduce_sum
    // intrinsic drops the reduced axis from the result shape).
    #[cutile::entry()]
    fn reduce_sum_2d_kernel(
        input: &mut Tensor<i32, { [8, 16] }>,
        out_cols: &mut Tensor<i32, { [16] }>,
        out_rows: &mut Tensor<i32, { [8] }>,
    ) {
        let tile: Tile<i32, { [8, 16] }> = load_tile_mut(input);
        let cols: Tile<i32, { [16] }> = reduce_sum(tile, 0i32);
        let rows: Tile<i32, { [8] }> = reduce_sum(tile, 1i32);
        out_cols.store(cols);
        out_rows.store(rows);
    }
}

use gpu_exec_module::{
    broadcast_add_kernel, cat_kernel, iota_kernel, permute_kernel, reduce_accumulate_kernel,
    reduce_max_kernel, reduce_sum_2d_kernel, reduce_sum_kernel, scan_prefix_sum_kernel,
    scan_sum_2d_kernel,
};

// ---------------------------------------------------------------------------
// 1. Reduce sum
// ---------------------------------------------------------------------------

#[test]
fn execute_reduce_sum() {
    common::with_test_stack(|| {
        let n = 128usize;
        // Input: all 1.0 => sum = 128.0
        let input = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; n]))
            .sync()
            .expect("alloc input");
        let result_tensor = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; 1]))
            .sync()
            .expect("alloc result");

        let (_, result) = reduce_sum_kernel(input.partition([n]), result_tensor.partition([1]))
            .sync()
            .expect("reduce_sum_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1);
        assert!(
            (host[0] - n as f32).abs() < 1e-3,
            "reduce_sum: expected {}, got {}",
            n as f32,
            host[0]
        );
    });
}

// ---------------------------------------------------------------------------
// 2. Reduce max
// ---------------------------------------------------------------------------

#[test]
fn execute_reduce_max() {
    common::with_test_stack(|| {
        let n = 128usize;
        // Input: [0, 1, 2, ..., 127] as f32 => max = 127.0
        let input_host = Arc::new((0..n).map(|i| i as f32).collect::<Vec<_>>());
        let input = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc input");
        let result_tensor = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; 1]))
            .sync()
            .expect("alloc result");

        let (_, result) = reduce_max_kernel(input.partition([n]), result_tensor.partition([1]))
            .sync()
            .expect("reduce_max_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1);
        assert!(
            (host[0] - 127.0).abs() < 1e-3,
            "reduce_max: expected 127.0, got {}",
            host[0]
        );
    });
}

// ---------------------------------------------------------------------------
// 3. Scan (prefix sum)
// ---------------------------------------------------------------------------

#[test]
fn execute_scan_prefix_sum() {
    common::with_test_stack(|| {
        let n = 128usize;
        // Input: all 1.0 => prefix sum = [1, 2, 3, ..., 128]
        let data = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; n]))
            .sync()
            .expect("alloc");

        let (result,) = scan_prefix_sum_kernel(data.partition([n]))
            .sync()
            .expect("scan_prefix_sum_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for i in 0..n {
            let expected = (i + 1) as f32;
            assert!(
                (host[i] - expected).abs() < 1e-3,
                "scan prefix sum: element {} expected {}, got {}",
                i,
                expected,
                host[i]
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 4. Reduce-and-accumulate (single-block atomic-like add)
// ---------------------------------------------------------------------------

#[test]
fn execute_reduce_accumulate() {
    common::with_test_stack(|| {
        let n = 128usize;
        // Input: all 1.0, counter starts at 0.
        // Single block sums 128 ones and adds to counter => counter = 128.0
        let input = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; n]))
            .sync()
            .expect("alloc input");
        let counter = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; 1]))
            .sync()
            .expect("alloc counter");

        let (_, result) = reduce_accumulate_kernel(input.partition([n]), counter.partition([1]))
            .sync()
            .expect("reduce_accumulate_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1);
        assert!(
            (host[0] - 128.0).abs() < 1e-3,
            "reduce accumulate: expected 128.0, got {}",
            host[0]
        );
    });
}

// ---------------------------------------------------------------------------
// 5. Cat (concatenate)
// ---------------------------------------------------------------------------

#[test]
fn execute_cat() {
    common::with_test_stack(|| {
        // a = [1.0; 64], b = [2.0; 64] => output = [1.0 x64, 2.0 x64]
        let a = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 64]))
            .sync()
            .expect("alloc a");
        let b = api::copy_host_vec_to_device(&Arc::new(vec![2.0f32; 64]))
            .sync()
            .expect("alloc b");
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; 128]))
            .sync()
            .expect("alloc output");

        let (result, _, _) = cat_kernel(
            output.partition([128]),
            a.partition([64]),
            b.partition([64]),
        )
        .sync()
        .expect("cat_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 128);
        for i in 0..64 {
            assert!(
                (host[i] - 1.0).abs() < 1e-5,
                "cat: element {} expected 1.0, got {}",
                i,
                host[i]
            );
        }
        for i in 64..128 {
            assert!(
                (host[i] - 2.0).abs() < 1e-5,
                "cat: element {} expected 2.0, got {}",
                i,
                host[i]
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 6. Broadcast + arithmetic
// ---------------------------------------------------------------------------

#[test]
fn execute_broadcast_add() {
    common::with_test_stack(|| {
        let n = 128usize;
        // Input: [0, 1, 2, ..., 127] as f32. Kernel adds 10.0 to each.
        let input_host = Arc::new((0..n).map(|i| i as f32).collect::<Vec<_>>());
        let input = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc input");
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; n]))
            .sync()
            .expect("alloc output");

        let (result, _) = broadcast_add_kernel(output.partition([n]), input.partition([n]))
            .sync()
            .expect("broadcast_add_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for i in 0..n {
            let expected = i as f32 + 10.0;
            assert!(
                (host[i] - expected).abs() < 1e-3,
                "broadcast add: element {} expected {}, got {}",
                i,
                expected,
                host[i]
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 7. Iota
// ---------------------------------------------------------------------------

#[test]
fn execute_iota() {
    common::with_test_stack(|| {
        let n = 128usize;
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0i32; n]))
            .sync()
            .expect("alloc output");

        let (result,) = iota_kernel(output.partition([n]))
            .sync()
            .expect("iota_kernel failed");

        let host: Vec<i32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for i in 0..n {
            assert_eq!(
                host[i], i as i32,
                "iota: element {} expected {}, got {}",
                i, i, host[i]
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 8. Permute (transpose)
// ---------------------------------------------------------------------------

#[test]
fn execute_permute() {
    common::with_test_stack(|| {
        let (rows, cols) = (8usize, 16usize);
        // Input: row-major [8, 16] where element [r][c] = r * 16 + c
        let input_host = Arc::new((0..rows * cols).map(|i| i as f32).collect::<Vec<_>>());
        let input = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc input")
            .reshape(&[rows, cols])
            .expect("reshape input");
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32; rows * cols]))
            .sync()
            .expect("alloc output")
            .reshape(&[cols, rows])
            .expect("reshape output");

        let (result, _) = permute_kernel(
            output.partition([cols, rows]),
            input.partition([rows, cols]),
        )
        .sync()
        .expect("permute_kernel failed");

        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        // After transpose [8,16] -> [16,8]:
        // output[c][r] = input[r][c] = r * 16 + c
        for c in 0..cols {
            for r in 0..rows {
                let out_idx = c * rows + r;
                let expected = (r * cols + c) as f32;
                assert!(
                    (host[out_idx] - expected).abs() < 1e-5,
                    "permute: output[{}][{}] (flat {}) expected {}, got {}",
                    c,
                    r,
                    out_idx,
                    expected,
                    host[out_idx]
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 9. 2D scan values along both axes, forward and reverse
// ---------------------------------------------------------------------------

#[test]
fn execute_scan_sum_2d_both_dims() {
    common::with_test_stack(|| {
        let rows = 8usize;
        let cols = 16usize;
        // Distinct values so a wrong-axis scan cannot alias the right one.
        let input_host: Vec<i32> = (0..rows * cols).map(|i| i as i32 % 7 + 1).collect();

        let up = |values: Vec<i32>, shape: &[usize]| {
            api::copy_host_vec_to_device(&Arc::new(values))
                .sync()
                .expect("alloc")
                .reshape(shape)
                .expect("reshape")
        };
        let input = up(input_host.clone(), &[rows, cols]);
        let d0 = up(vec![0i32; rows * cols], &[rows, cols]);
        let d1 = up(vec![0i32; rows * cols], &[rows, cols]);
        let d1r = up(vec![0i32; rows * cols], &[rows, cols]);

        let (_, d0, d1, d1r) = scan_sum_2d_kernel(
            input.partition([rows, cols]),
            d0.partition([rows, cols]),
            d1.partition([rows, cols]),
            d1r.partition([rows, cols]),
        )
        .sync()
        .expect("scan_sum_2d_kernel failed");

        let read =
            |t: Partition<Tensor<i32>>| t.unpartition().to_host_vec().sync().expect("to_host");
        let (d0, d1, d1r) = (read(d0), read(d1), read(d1r));

        for r in 0..rows {
            for c in 0..cols {
                let at = |rr: usize, cc: usize| input_host[rr * cols + cc];
                let idx = r * cols + c;
                let want_d0: i32 = (0..=r).map(|rr| at(rr, c)).sum();
                let want_d1: i32 = (0..=c).map(|cc| at(r, cc)).sum();
                let want_d1r: i32 = (c..cols).map(|cc| at(r, cc)).sum();
                assert_eq!(d0[idx], want_d0, "dim-0 scan at [{r}][{c}]");
                assert_eq!(d1[idx], want_d1, "dim-1 scan at [{r}][{c}]");
                assert_eq!(d1r[idx], want_d1r, "dim-1 reverse scan at [{r}][{c}]");
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 10. 2D reduction values along both axes
// ---------------------------------------------------------------------------

#[test]
fn execute_reduce_sum_2d_both_dims() {
    common::with_test_stack(|| {
        let rows = 8usize;
        let cols = 16usize;
        let input_host: Vec<i32> = (0..rows * cols).map(|i| i as i32 % 5 + 1).collect();

        let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
            .sync()
            .expect("alloc input")
            .reshape(&[rows, cols])
            .expect("reshape input");
        let out_cols = api::copy_host_vec_to_device(&Arc::new(vec![0i32; cols]))
            .sync()
            .expect("alloc cols");
        let out_rows = api::copy_host_vec_to_device(&Arc::new(vec![0i32; rows]))
            .sync()
            .expect("alloc rows");

        let (_, out_cols, out_rows) = reduce_sum_2d_kernel(
            input.partition([rows, cols]),
            out_cols.partition([cols]),
            out_rows.partition([rows]),
        )
        .sync()
        .expect("reduce_sum_2d_kernel failed");

        let cols_host: Vec<i32> = out_cols
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("to_host cols");
        let rows_host: Vec<i32> = out_rows
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("to_host rows");

        for c in 0..cols {
            let want: i32 = (0..rows).map(|r| input_host[r * cols + c]).sum();
            assert_eq!(cols_host[c], want, "dim-0 reduction, column {c}");
        }
        for r in 0..rows {
            let want: i32 = (0..cols).map(|c| input_host[r * cols + c]).sum();
            assert_eq!(rows_host[r], want, "dim-1 reduction, row {r}");
        }
    });
}

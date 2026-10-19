/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests that kernels produce correct results on tensors with
//! dimensions NOT divisible by tile/block size, and on sliced views
//! with byte offsets.

use cutile::api;
use cutile::tensor::{IntoPartition, PartitionMut, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::{DeviceOp, TileKernel, ToHostVecOp};
use std::sync::Arc;

mod common;

#[cutile::module]
mod test_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn scale<const B: i32>(out: &mut Tensor<f32, { [B] }>, a: &Tensor<f32, { [-1] }>, scalar: f32) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_s: Tile<f32, { [B] }> = scalar.broadcast(out.shape());
        out.store(tile_a * tile_s);
    }

    /// 2D copy: out[pid_m, pid_n] = a[pid_m, pid_n].
    #[cutile::entry(print_ir = true)]
    fn copy_2d<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_a = a.load_tile(shape![BM, BN], [pid.0, pid.1]);
        out.store(tile_a);
    }

    /// Load via explicit partition (triggers check_partition_access).
    /// dim 0 is dynamic (-1) → exercises dynamic runtime bounds check.
    /// dim 1 is static (N) but indexed by pid (no compile-time bounds).
    #[cutile::entry()]
    fn partition_load_dynamic<const BM: i32, const BN: i32, const N: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f32, { [-1, N] }>,
    ) {
        let part_a = a.partition(shape![BM, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_a = part_a.load([pid.0, pid.1]);
        out.store(tile_a);
    }

    /// Accumulate tiles via 1D partition loop (triggers check_partition_access).
    /// The loop `for i in 0..NBLOCKS` gives i compile-time bounds [0, NBLOCKS-1].
    /// a's shape is N (static const generic), so the check is fully static:
    /// it verifies (NBLOCKS-1) < ceil(N/B) at compile time.
    #[cutile::entry()]
    fn partition_load_static<const B: i32, const N: i32, const NBLOCKS: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [N] }>,
    ) {
        let mut acc = load_tile_mut(out);
        let part_a = a.partition(shape![B]);
        for i in 0i32..NBLOCKS {
            let tile = part_a.load([i]);
            acc = acc + tile;
        }
        out.store(acc);
    }

    /// GEMM: z = x @ y. M and N can be non-divisible by BM/BN.
    /// K must be divisible by BK (loop bound).
    #[cutile::entry(print_ir = true)]
    fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

use test_kernels::{add, copy_2d, gemm, partition_load_dynamic, partition_load_static, scale};

// ── Non-divisible sizes (no slicing) ────────────────────────────────────────

#[test]
fn add_non_divisible_size() {
    // 1000 elements, block=128. 1000 % 128 != 0.
    common::with_test_stack(|| {
        let n = 1000;
        let block = 128;

        let a = api::ones::<f32>(&[n]).sync().expect("alloc a");
        let b = api::ones::<f32>(&[n]).sync().expect("alloc b");
        let mut out = api::zeros::<f32>(&[n]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a, &b)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0f32).abs() < 1e-5,
                "add: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_non_divisible_just_over() {
    // 129 elements, block=128. 129 % 128 = 1. Just over one tile.
    common::with_test_stack(|| {
        let n = 129;
        let block = 128;

        let a = api::ones::<f32>(&[n]).sync().expect("alloc a");
        let b = api::ones::<f32>(&[n]).sync().expect("alloc b");
        let mut out = api::zeros::<f32>(&[n]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a, &b)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0f32).abs() < 1e-5,
                "add n={n}: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_non_divisible_just_under() {
    // 127 elements, block=128. Tensor smaller than one tile.
    common::with_test_stack(|| {
        let n = 127;
        let block = 128;

        let a = api::ones::<f32>(&[n]).sync().expect("alloc a");
        let b = api::ones::<f32>(&[n]).sync().expect("alloc b");
        let mut out = api::zeros::<f32>(&[n]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a, &b)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0f32).abs() < 1e-5,
                "add n={n}: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_non_divisible_prime() {
    // 251 elements (prime), block=128. 251 % 128 = 123.
    common::with_test_stack(|| {
        let n = 251;
        let block = 128;

        let a = api::ones::<f32>(&[n]).sync().expect("alloc a");
        let b = api::ones::<f32>(&[n]).sync().expect("alloc b");
        let mut out = api::zeros::<f32>(&[n]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a, &b)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0f32).abs() < 1e-5,
                "add n={n}: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn scale_non_divisible_size() {
    // 500 elements, block=128. 500 % 128 != 0.
    common::with_test_stack(|| {
        let n = 500;
        let block = 128;

        let a = api::ones::<f32>(&[n]).sync().expect("alloc a");
        let mut out = api::zeros::<f32>(&[n]).sync().expect("alloc out");

        scale((&mut out).partition([block]), &a, 3.0)
            .sync()
            .expect("scale failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 3.0f32).abs() < 1e-5,
                "scale: element {i} = {v}, expected 3.0"
            );
        }
    });
}

// ── 2D non-divisible sizes ──────────────────────────────────────────────────
//
// Isolates whether 2D partition handling works with non-divisible dimensions,
// without involving MMA. Mirrors cutile-python test_tiled_view_copy_2d with
// shapes like (192, 134) and tile (128, 128).

#[test]
fn copy_2d_non_divisible_both_dims() {
    // Shape (192, 134), tile (128, 128).
    // 192 % 128 = 64, 134 % 128 = 6. Both dimensions non-divisible.
    common::with_test_stack(|| {
        let (m, n) = (192, 134);
        let (bm, bn) = (128, 128);

        let input_host = Arc::new((0..m * n).map(|i| i as f32).collect::<Vec<_>>());
        let a: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc a")
            .reshape(&[m, n])
            .expect("reshape a")
            .into();
        let mut out = api::zeros::<f32>(&[m, n])
            .sync()
            .expect("alloc out")
            .reshape(&[m, n])
            .expect("reshape out");

        copy_2d((&mut out).partition([bm, bn]), &*a)
            .sync()
            .expect("copy_2d failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), m * n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-5,
                "copy_2d (192,134): element {i} = {v}, expected {}",
                i as f32
            );
        }
    });
}

#[test]
fn copy_2d_non_divisible_one_dim() {
    // Shape (128, 100), tile (64, 64).
    // 128 % 64 = 0 (divisible), 100 % 64 = 36 (non-divisible).
    common::with_test_stack(|| {
        let (m, n) = (128, 100);
        let (bm, bn) = (64, 64);

        let input_host = Arc::new((0..m * n).map(|i| i as f32).collect::<Vec<_>>());
        let a: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc a")
            .reshape(&[m, n])
            .expect("reshape a")
            .into();
        let mut out = api::zeros::<f32>(&[m, n])
            .sync()
            .expect("alloc out")
            .reshape(&[m, n])
            .expect("reshape out");

        copy_2d((&mut out).partition([bm, bn]), &*a)
            .sync()
            .expect("copy_2d failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), m * n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-5,
                "copy_2d (128,100): element {i} = {v}, expected {}",
                i as f32
            );
        }
    });
}

#[test]
fn copy_2d_one_short_of_tile() {
    // Shape (63, 63), tile (64, 64).
    // Mirrors cutile-python test_array_copy_2d_with_padding shape.
    // Tests the "one element short" boundary in both dimensions.
    common::with_test_stack(|| {
        let (m, n) = (63, 63);
        let (bm, bn) = (64, 64);

        let input_host = Arc::new((0..m * n).map(|i| i as f32).collect::<Vec<_>>());
        let a: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc a")
            .reshape(&[m, n])
            .expect("reshape a")
            .into();
        let mut out = api::zeros::<f32>(&[m, n])
            .sync()
            .expect("alloc out")
            .reshape(&[m, n])
            .expect("reshape out");

        copy_2d((&mut out).partition([bm, bn]), &*a)
            .sync()
            .expect("copy_2d failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), m * n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-5,
                "copy_2d (63,63): element {i} = {v}, expected {}",
                i as f32
            );
        }
    });
}

// ── Partition bounds checks with non-divisible dims ────────────────────────
//
// These use partition().load() which triggers check_partition_access.
// check_partition_access has two paths:
//   Dynamic: shape dim is -1 (unknown at compile time), emits a runtime assert.
//   Static:  shape dim AND index bounds are known at compile time, checked
//            during JIT compilation (no MLIR emitted if valid).
//
// The dynamic path previously had a false-positive bug: TileIR misoptimized
// the ceil_div when the dividend carried `assume div_by` hints (positive_inf
// rounding was simplified to floor div). Fixed by using (shape+tile-1)/tile
// with floor division instead.

#[test]
fn partition_load_bounds_check_dynamic() {
    // Dynamic check: a's dim 0 is -1, so the bounds check for pid.0
    // emits a runtime assert: pid.0 < (M + BM - 1) / BM.
    // M=100, BM=64 → ceil(100/64) = 2. Blocks 0 and 1 are valid.
    // This test fails without the ceil_div fix (false-positive assert).
    common::with_test_stack(|| {
        let (m, n) = (100, 64);
        let (bm, bn) = (64, 64);
        let generics = vec![bm.to_string(), bn.to_string(), n.to_string()];

        let input_host = Arc::new((0..m * n).map(|i| i as f32).collect::<Vec<_>>());
        let a: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&input_host)
            .sync()
            .expect("alloc a")
            .reshape(&[m, n])
            .expect("reshape a")
            .into();
        let mut out = api::zeros::<f32>(&[m, n])
            .sync()
            .expect("alloc out")
            .reshape(&[m, n])
            .expect("reshape out");

        partition_load_dynamic((&mut out).partition([bm, bn]), &*a)
            .generics(generics)
            .sync()
            .expect("partition_load_dynamic: runtime bounds check should not fire");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), m * n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < 1e-5,
                "dynamic bounds: element {i} = {v}, expected {}",
                i as f32
            );
        }
    });
}

#[test]
fn partition_load_bounds_check_static() {
    // Static check: a has shape [N] where N=1000 (const generic), tile B=128.
    // The loop `for i in 0..NBLOCKS` gives i compile-time bounds [0, NBLOCKS-1].
    // With NBLOCKS = ceil(1000/128) = 8, the static check verifies:
    //   0 <= 0 && 7 < ceil(1000/128) = 8  → passes.
    // If the static check used floor div instead of ceil, it would compute
    // floor(1000/128) = 7 and reject index 7 at compile time.
    //
    // The kernel sums all tile blocks: 7 full blocks of 128 ones + 1 partial
    // block of 104 ones + 24 padded zeros. Each output element = 1000.
    common::with_test_stack(|| {
        let n: usize = 1000;
        let b: usize = 128;
        let nblocks = n.div_ceil(b); // = 8
        let generics = vec![b.to_string(), n.to_string(), nblocks.to_string()];

        let a: Arc<Tensor<f32>> = api::ones::<f32>(&[n]).sync().expect("alloc a").into();
        let mut out = api::zeros::<f32>(&[b]).sync().expect("alloc out");

        partition_load_static((&mut out).partition([b]), &*a)
            .generics(generics)
            .sync()
            .expect("partition_load_static: static bounds check should accept NBLOCKS");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), b);
        let remainder = n % b; // = 104
        for (i, &v) in host.iter().enumerate() {
            // Each output element accumulates one value from each tile block.
            // Elements 0..103: receive 1.0 from all 8 blocks → 8.0
            // Elements 104..127: receive 1.0 from 7 full blocks + 0.0 from
            // the padded tail of block 7 → 7.0
            let expected = if i < remainder {
                nblocks as f32
            } else {
                (nblocks - 1) as f32
            };
            assert!(
                (v - expected).abs() < 1e-3,
                "static bounds: element {i} = {v}, expected {expected}"
            );
        }
    });
}

// ── GEMM non-divisible ──────────────────────────────────────────────────────

#[test]
fn gemm_non_divisible_m_and_n() {
    // M=100, N=100, K=64. BM=16, BN=16, BK=8.
    // M % BM = 100 % 16 = 4 (non-divisible).
    // N % BN = 100 % 16 = 4 (non-divisible).
    // K % BK = 64 % 8 = 0 (must be divisible for the loop).
    // z = ones(100,64) @ ones(64,100) → every element = 64.0
    common::with_test_stack(|| {
        let (m, n, k) = (100, 100, 64);
        let (bm, bn, bk): (usize, usize, usize) = (16, 16, 8);
        let generics = vec![
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
            k.to_string(),
        ];

        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let z = api::zeros::<f32>(&[m, n])
            .sync_on(&stream)
            .unwrap()
            .partition([bm, bn]);
        let x: Arc<Tensor<f32>> = api::ones::<f32>(&[m, k]).sync_on(&stream).unwrap().into();
        let y: Arc<Tensor<f32>> = api::ones::<f32>(&[k, n]).sync_on(&stream).unwrap().into();

        let (z, _x, _y) = gemm(z, x, y).generics(generics).sync_on(&stream).unwrap();

        let host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream).unwrap();
        assert_eq!(host.len(), m * n);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - k as f32).abs() < 1e-3,
                "gemm: element {i} = {v}, expected {k}"
            );
        }
    });
}

// ── Sliced views with offset (divisible size) ───────────────────────────────
//
// These isolate the offset handling from the non-divisible size handling.

#[test]
fn add_sliced_divisible() {
    // arange(1024), slice [128..384] → length 256, offset 128 elements.
    // 256 / 128 = 2 blocks. Tests offset with multi-block partition.
    common::with_test_stack(|| {
        let block = 128;

        let a = api::arange::<f32>(1024).sync().expect("alloc a");
        let b = api::ones::<f32>(&[1024]).sync().expect("alloc b");

        let a_slice = a.slice(&[128..384]).expect("slice a");
        let b_slice = b.slice(&[128..384]).expect("slice b");

        assert_eq!(a_slice.shape(), &[256]);

        let mut out = api::zeros::<f32>(&[256]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a_slice, &b_slice)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 256);
        for (i, &v) in host.iter().enumerate() {
            let expected = (i + 128) as f32 + 1.0;
            assert!(
                (v - expected).abs() < 1e-3,
                "element {i} = {v}, expected {expected}"
            );
        }
    });
}

// ── Sliced views with offset + non-divisible size ───────────────────────────

#[test]
fn add_sliced_non_divisible() {
    // arange(1024), slice [24..1024] → length 1000, offset 24.
    // 1000 % 128 != 0. Tests offset + non-divisible together.
    common::with_test_stack(|| {
        let block = 128;

        let a = api::arange::<f32>(1024).sync().expect("alloc a");
        let b = api::ones::<f32>(&[1024]).sync().expect("alloc b");

        let a_slice = a.slice(&[24..1024]).expect("slice a");
        let b_slice = b.slice(&[24..1024]).expect("slice b");

        let mut out = api::zeros::<f32>(&[1000]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a_slice, &b_slice)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1000);
        for (i, &v) in host.iter().enumerate() {
            let expected = (i + 24) as f32 + 1.0;
            assert!(
                (v - expected).abs() < 1e-3,
                "element {i} = {v}, expected {expected}"
            );
        }
    });
}

#[test]
fn scale_sliced_non_divisible() {
    // arange(512), slice [12..512] → length 500, offset 12.
    common::with_test_stack(|| {
        let block = 128;

        let a = api::arange::<f32>(512).sync().expect("alloc a");
        let a_slice = a.slice(&[12..512]).expect("slice a");

        let mut out = api::zeros::<f32>(&[500]).sync().expect("alloc out");

        scale((&mut out).partition([block]), &a_slice, 2.0)
            .sync()
            .expect("scale failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 500);
        for (i, &v) in host.iter().enumerate() {
            let expected = (i + 12) as f32 * 2.0;
            assert!(
                (v - expected).abs() < 1e-3,
                "element {i} = {v}, expected {expected}"
            );
        }
    });
}

// ── Sliced views with non-divisible offset alignment ────────────────────────

#[test]
fn add_sliced_odd_offset() {
    // arange(300), slice [1..300] → length 299, offset 1.
    // Offset 1 is not aligned to any power-of-2. 299 % 128 = 43.
    // Tests both unaligned offset and non-divisible length.
    common::with_test_stack(|| {
        let block = 128;

        let a = api::arange::<f32>(300).sync().expect("alloc a");
        let b = api::ones::<f32>(&[300]).sync().expect("alloc b");

        let a_slice = a.slice(&[1..300]).expect("slice a");
        let b_slice = b.slice(&[1..300]).expect("slice b");

        let mut out = api::zeros::<f32>(&[299]).sync().expect("alloc out");

        add((&mut out).partition([block]), &a_slice, &b_slice)
            .sync()
            .expect("add failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 299);
        for (i, &v) in host.iter().enumerate() {
            let expected = (i + 1) as f32 + 1.0;
            assert!(
                (v - expected).abs() < 1e-3,
                "element {i} = {v}, expected {expected}"
            );
        }
    });
}

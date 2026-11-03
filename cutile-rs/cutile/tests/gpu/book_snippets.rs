/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke tests for every tutorial snippet in the cuTile Rust Book. Each
//! `#[test]` mirrors one tutorial so a single broken pattern shows up as a
//! single failing test (the previous monolithic example failed opaquely).

#![allow(unused_variables)]

use cuda_async::device_operation::*;
use cutile::api::{arange, ones, randn, zeros};
use cutile::prelude::*;
use cutile::tile_kernel::PartitionOp;

use crate::common;

// ---------------------------------------------------------------------------
// Tutorial 1: Hello World
// ---------------------------------------------------------------------------

#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn hello_world_kernel() {
        let pid0 = program_id(0);
        let pid1 = program_id(1);
        let pid2 = program_id(2);
        let n0 = num_programs(0);
        let n1 = num_programs(1);
        let n2 = num_programs(2);
        cuda_tile_print!(
            "Hello from program <{}, {}, {}> in a grid of <{}, {}, {}> programs!\n",
            pid0,
            pid1,
            pid2,
            n0,
            n1,
            n2
        );
    }
}

#[test]
fn book_hello_world() {
    common::with_test_stack(|| {
        hello_world_module::hello_world_kernel()
            .grid((2, 2, 1))
            .sync()
            .expect("hello_world_kernel");
    });
}

// ---------------------------------------------------------------------------
// Tutorial 2: Vector Addition
// ---------------------------------------------------------------------------

#[cutile::module]
mod vector_add_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

#[test]
fn book_vector_addition() {
    common::with_test_stack(|| {
        let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync().expect("ones").into();
        let y: Arc<Tensor<f32>> = ones(&[32, 32]).sync().expect("ones").into();
        let z = zeros(&[32, 32]).sync().expect("zeros").partition([4, 4]);
        let z_host = vector_add_module::add(z, x, y)
            .unzip()
            .0
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("add kernel");
        assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    });
}

// ---------------------------------------------------------------------------
// Tutorial 3: SAXPY
// ---------------------------------------------------------------------------

#[cutile::module]
mod saxpy_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(a: f32, x: &Tensor<f32, { [-1, -1] }>, y: &mut Tensor<f32, S>) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

#[test]
fn book_saxpy() {
    common::with_test_stack(|| {
        let a = 2.0f32;
        let input: Arc<Tensor<f32>> = arange(32usize).sync().expect("arange").into();
        let x: Arc<Tensor<f32>> = input
            .dup()
            .sync()
            .expect("dup")
            .reshape(&[4, 8])
            .expect("reshape")
            .into();
        let y = input
            .dup()
            .sync()
            .expect("dup")
            .reshape(&[4, 8])
            .expect("reshape")
            .partition([2, 2]);

        let (a_out, _x, y) = saxpy_module::saxpy(a, x, y).sync().expect("saxpy");
        let y_host = y.unpartition().to_host_vec().sync().expect("y to_host");
        let input_host = input.to_host_vec().sync().expect("input to_host");

        for (i, &got) in y_host.iter().enumerate() {
            let expected = a_out * input_host[i] + input_host[i];
            assert!((expected - got).abs() < 1e-5, "saxpy mismatch at {}", i);
        }
    });
}

// ---------------------------------------------------------------------------
// Tutorial 4: GEMM
// ---------------------------------------------------------------------------

#[cutile::module]
mod gemm_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([i, pid_n]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

#[test]
fn book_gemm() {
    use cutile::DType;
    common::with_test_stack(|| {
        let (bm, bn, bk) = (16, 16, 8);
        let (m, n, k) = (64usize, 64usize, 64usize);
        let generics = vec![
            f32::DTYPE.as_str().to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
            k.to_string(),
        ];
        let z = api::zeros(&[m, n])
            .sync()
            .expect("zeros")
            .partition([bm, bn]);
        let x: Arc<Tensor<f32>> = api::ones(&[m, k]).sync().expect("ones").into();
        let y: Arc<Tensor<f32>> = api::ones(&[k, n]).sync().expect("ones").into();
        let (z, _, _) = gemm_module::gemm(z, x, y)
            .generics(generics)
            .sync()
            .expect("gemm");
        let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
        assert!((z_host[0] - k as f32).abs() < 1e-3);
    });
}

// ---------------------------------------------------------------------------
// Tutorial 5: Fused Softmax
// ---------------------------------------------------------------------------

#[cutile::module]
mod softmax_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f32, { [-1, -1] }>,
        y: &mut Tensor<f32, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = x.load_like(y);
        let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f32, { [BM, BN] }> =
            tile_x_max.reshape(shape![BM, 1]).broadcast(y.shape());
        let num: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1);
        let denom = denom.reshape(shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }
}

#[test]
fn book_softmax() {
    common::with_test_stack(|| {
        let (m, n) = (4usize, 8usize);
        let (bm, bn) = (2, n);

        let input: Arc<Tensor<f32>> = arange(m * n).sync().expect("arange").into();
        let x: Arc<Tensor<f32>> = input
            .dup()
            .sync()
            .expect("dup")
            .reshape(&[m, n])
            .expect("reshape")
            .into();
        let y = input
            .dup()
            .sync()
            .expect("dup")
            .reshape(&[m, n])
            .expect("reshape")
            .partition([bm, bn]);

        let (_x, y) = softmax_module::softmax(x, y).sync().expect("softmax");
        let y_host = y.unpartition().to_host_vec().sync().expect("y to_host");
        for i in (0..y_host.len()).step_by(n) {
            let row_sum: f32 = y_host[i..i + n].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "softmax row {} sum != 1",
                i / n
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Tutorial 6: Fused Multihead Attention
// ---------------------------------------------------------------------------

#[cutile::module]
mod fmha_module {
    use cutile::core::*;

    #[cutile::entry(print_ir = false)]
    fn fmha<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f32, { [-1, -1, -1, -1] }>,
        k: &Tensor<f32, { [-1, -1, -1, -1] }>,
        v: &Tensor<f32, { [-1, -1, -1, -1] }>,
        out: &mut Tensor<f32, { [1, BM, D] }>,
        qk_scale: f32,
    ) {
        let pid0 = program_id(0);
        let h = q.shape()[1];
        let batch_idx = pid0 / h;
        let head_idx = pid0 % h;
        let q_m_idx = program_id(1);

        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(shape![BM, BN]);

        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        let q_part: Partition<f32, { [1, 1, BM, D] }> = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [1, 1, BM, D] }> = q_part.load([batch_idx, head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = tq.reshape(shape![BM, D]);

        let n: i32 = k.shape()[2];
        let num_tiles: i32 = ceil_div(n, BN);

        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, D]);

        for j in 0i32..num_tiles {
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);

            l_i = l_i * alpha + l_ij;
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f32, { [1, 1, BN, D] }> = v_part.load([batch_idx, head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = v_tile.reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        acc = true_div(acc, l_i.broadcast(shape![BM, D]));
        let acc = acc.reshape(shape![1, BM, D]);
        out.store(acc);
    }
}

#[test]
fn book_flash_attention() {
    common::with_test_stack(|| {
        let (batch, heads, seq_len, head_dim) = (1usize, 2usize, 32usize, 16usize);
        let (bm, bn) = (16, 16);

        let seed = 42u64;
        let q: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed))
            .sync()
            .expect("q randn")
            .into();
        let k: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 1))
            .sync()
            .expect("k randn")
            .into();
        let v: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 2))
            .sync()
            .expect("v randn")
            .into();

        let out = zeros(&[batch * heads, seq_len, head_dim])
            .sync()
            .expect("zeros")
            .partition([1, bm, head_dim]);

        let qk_scale = 1.0 / f32::sqrt(head_dim as f32);
        let generics = vec![bm.to_string(), bn.to_string(), head_dim.to_string()];

        let (_, _, _, out, _) = fmha_module::fmha(q, k, v, out, qk_scale)
            .generics(generics)
            .sync()
            .expect("fmha");
        let out_host = out.unpartition().to_host_vec().sync().expect("out to_host");

        let expected_len = batch * heads * seq_len * head_dim;
        assert_eq!(out_host.len(), expected_len);
        assert!(out_host.iter().all(|v| v.is_finite()));
    });
}

// ---------------------------------------------------------------------------
// Tutorial 7: Lazy DeviceOp composition
// ---------------------------------------------------------------------------

#[test]
fn book_async_composition() {
    common::with_test_stack(|| {
        let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync().expect("ones").into();
        let y: Arc<Tensor<f32>> = ones(&[32, 32]).sync().expect("ones").into();

        let z_host = vector_add_module::add(
            zeros(&[32, 32]).sync().expect("zeros").partition([4, 4]),
            x,
            y,
        )
        .first()
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add kernel");
        assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));

        let z_host2 = vector_add_module::add(
            zeros(&[32, 32]).partition([4, 4]),
            ones(&[32, 32]).map(|t: Tensor<f32>| -> Arc<Tensor<f32>> { Arc::new(t) }),
            ones(&[32, 32]).map(|t: Tensor<f32>| -> Arc<Tensor<f32>> { Arc::new(t) }),
        )
        .first()
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("lazy graph kernel");
        assert!(z_host2.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    });
}

// ---------------------------------------------------------------------------
// Tutorial 8: Data Parallel MLP
// ---------------------------------------------------------------------------

#[cutile::module]
mod mlp_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn mlp_gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([i, pid_n]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub fn mlp_matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid = program_id(0);
        let mut tile_z = z.load().reshape(shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid, i]);
            let tile_y = part_y.load([i]).reshape(shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z.reshape(shape![BM]));
    }

    #[cutile::entry()]
    fn mlp_relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

#[test]
fn book_data_parallel_mlp() {
    use cutile::DType;
    common::with_test_stack(|| {
        let dim = 16usize;
        let block_dim = 4usize;
        let gemm_generics = vec![
            f32::DTYPE.as_str().to_string(),
            block_dim.to_string(),
            block_dim.to_string(),
            block_dim.to_string(),
            dim.to_string(),
        ];
        let mv_generics = vec![
            block_dim.to_string(),
            block_dim.to_string(),
            dim.to_string(),
        ];

        let w0: Arc<Tensor<f32>> = api::ones(&[dim, dim]).sync().expect("w0").into();
        let w1: Arc<Tensor<f32>> = api::ones(&[dim]).sync().expect("w1").into();
        let data: Arc<Tensor<f32>> = api::ones(&[dim, dim]).sync().expect("data").into();

        let out0 = api::zeros(&[dim, dim])
            .sync()
            .expect("out0 zeros")
            .partition([block_dim, block_dim]);
        let (out0, _, _) = mlp_module::mlp_gemm(out0, data, w0)
            .generics(gemm_generics)
            .sync()
            .expect("mlp_gemm");
        let out0_arc: Arc<Tensor<f32>> = out0.unpartition().into();

        let out1 = api::zeros(&[dim])
            .sync()
            .expect("out1 zeros")
            .partition([block_dim]);
        let (out1, _, _) = mlp_module::mlp_matvec(out1, out0_arc, w1)
            .generics(mv_generics)
            .sync()
            .expect("mlp_matvec");

        let (out1,) = mlp_module::mlp_relu(out1).sync().expect("mlp_relu");

        let out_host = out1.unpartition().to_host_vec().sync().expect("to_host");
        let expected = (dim * dim) as f32;
        assert!((out_host[0] - expected).abs() < 1e-1);
        assert!(out_host.iter().all(|&v| v >= 0.0));
    });
}

// ---------------------------------------------------------------------------
// Tutorial 9: Pointer Addition
// ---------------------------------------------------------------------------

#[cutile::module]
mod pointer_add_module {
    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = get_tensor(z_ptr, len);
        let x_tensor: Tensor<T, { [-1] }> = get_tensor(x_ptr, len);
        let y_tensor: Tensor<T, { [-1] }> = get_tensor(y_ptr, len);
        let pid = program_id(0);
        let tile_shape = shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid]);
        let tile_y = y_tensor.partition(tile_shape).load([pid]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid]);
    }
}

#[test]
fn book_pointer_addition() {
    common::with_test_stack(|| {
        let len = 32usize;
        let tile_size = 4usize;

        let z: Tensor<f32> = zeros(&[len]).sync().expect("z zeros");
        let x: Tensor<f32> = ones(&[len]).sync().expect("x ones");
        let y: Tensor<f32> = ones(&[len]).sync().expect("y ones");

        let z_ptr = z.device_pointer();
        let x_ptr = x.device_pointer();
        let y_ptr = y.device_pointer();

        unsafe { pointer_add_module::add_ptr(z_ptr, x_ptr, y_ptr, len as i32) }
            .grid(((len / tile_size) as u32, 1, 1))
            .sync()
            .expect("add_ptr");

        let z_host = z.to_host_vec().sync().expect("z to_host");
        assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    });
}

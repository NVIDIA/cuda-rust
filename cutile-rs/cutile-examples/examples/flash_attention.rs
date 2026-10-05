/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
extern crate core;

use cuda_async::device_operation::{DeviceOp, Unzippable6};
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=false,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=1, max_divisibility=16,),
                       ))]
    fn fmha<
        const BM: i32, // Q sequence length partition size.
        const BN: i32, // KV Sequence length partition size.
        const D: i32,  // Hidden size (weights).
    >(
        out: &mut Tensor<f32, { [1, BM, D] }>, // (b*h, m, d)
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, h, m, d)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d) where n == m
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d) where n == m
        qk_scale: f32,
        query_group_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b*h, m/bm, 1)
        let h = q.shape()[1];
        let batch_idx = pid.0 / h; // \in  [0, b)
        let q_head_idx = pid.0 % h; // \in [0, h)
        let q_m_idx = pid.1; // \in [0, m/bm)
        let kv_head_idx = q_head_idx / query_group_size;

        // This lets us use exp2 vs exp.
        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(shape![BM, BN]);

        // mask us needed for causal only.
        // let mask_true: Tile<f32, {[BM, BN]}> = constant(0.0f32, shape![BM, BN]);
        // let mask_false: Tile<f32, {[BM, BN]}> = constant(f32::NEG_INFINITY, shape![BM, BN]);

        // offset is needed for causal only.
        // let offs_n_tile: Tile<i32, {[BN]}> = iota(shape![BN]);
        // let offs_n_tile: Tile<i32, {[BM, BN]}> = offs_n_tile.reshape(shape![1, BN])
        //     .broadcast(shape![BM, BN]);
        // let offs_m: i32 = q_m_idx * BM;
        // let offs_m: Tile<i32, {[BM]}> = offs_m.broadcast(shape![BM]);
        // let m_arange: Tile<i32, {[BM]}> = iota(shape![BM]);
        // let offs_m: Tile<i32, {[BM]}> = offs_m + m_arange;
        // let offs_m: Tile<i32, {[BM, BN]}> = offs_m
        //     .reshape(shape![BM, 1])
        //     .broadcast(shape![BM, BN]);

        // m and l are for softmax.
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        // This is the output tile.
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        // We load just one query block per process.
        let q_part: Partition<f32, { [1, 1, BM, D] }> = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [1, 1, BM, D] }> = q_part.load([batch_idx, q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = tq.reshape(shape![BM, D]);

        let n: i32 = k.shape()[2];
        let num_tiles: i32 = ceil_div(n, BN);
        // let mask_start: i32 = n / BN;

        let k_part = k.partition(shape![1, 1, BN, D]); // permuted after loading.
        let v_part = v.partition(shape![1, 1, BN, D]);

        // j corresponds to tile index along key / value seq len dim.
        for j in 0i32..num_tiles {
            // cuda_tile_print!("batch_idx={}, kv_head_idx={}, q_m_idx={}, j={}\n",
            //             batch_idx, kv_head_idx, q_m_idx, j);
            // Compute q @ k^T.
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, kv_head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);

            // Apply mask(q @ k^T).
            // if j >= mask_start {
            //     let mask: Tile<bool, {[BM, BN]}>  = constant(true, shape![BM, BN]);
            //     let offs_n: i32 = j * BN;
            //     let offs_n: Tile<i32, {[BM, BN]}> = offs_n.broadcast(shape![BM, BN]);
            //     let offs_n: Tile<i32, {[BM, BN]}> = offs_n + offs_n_tile;
            //     let mask: Tile<bool, {[BM, BN]}> = mask & ge_tile(offs_m, offs_n); // Causal only.
            //     let mask: Tile<f32, {[BM, BN]}> = select(mask, mask_true, mask_false);
            //     qk = qk + mask;
            // }

            // Apply scale(mask(q @ k^T)).
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            // Recenter before softmax.
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM, BN]);

            // Apply softmax(mask(scale(q @ k^T))).
            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(shape![BM, D]);
            acc = acc * alpha;

            // Compute softmax(mask(scale(q @ k^T))) @ v.
            let v_tile: Tile<f32, { [1, 1, BN, D] }> =
                v_part.load([batch_idx, kv_head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = v_tile.reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        acc = true_div(acc, l_i.broadcast(shape![BM, D]));
        let acc = acc.reshape(shape![1, BM, D]);
        out.store(acc);
    }
}

use cutile_examples::fmha_ref_exec;
use my_module::fmha as fmha_kernel;

#[allow(clippy::too_many_arguments)]
fn fmha(
    b: usize,   // batch size.
    h: usize,   // number of heads (query).
    hkv: usize, // number of heads (key/value).
    m: usize,   // sequence length.
    d: usize,   // hidden size.
    bm: usize,  // q seq len part size.
    bn: usize,  // k, v seq len part size.
    bbh: usize, // batch * num_heads part size.
) -> Result<(), Error> {
    // Create a context. Device 0 is associated with the context.
    let device = Device::new(0)?;
    // Create a new stream on which we run CUDA operations.
    let stream = device.new_stream()?;

    let seed = 123;
    let q: Arc<Tensor<f32>> = randn(0f32, 1., [b, h, m, d], Some(seed))
        .sync_on(&stream)?
        .into();
    let k: Arc<Tensor<f32>> = randn(0f32, 1., [b, hkv, m, d], Some(seed))
        .sync_on(&stream)?
        .into();
    let v: Arc<Tensor<f32>> = randn(0f32, 1., [b, hkv, m, d], Some(seed))
        .sync_on(&stream)?
        .into();
    // launch grid = (b*h, m/bm, 1)
    let out: Partition<Tensor<f32>> = zeros(&[b * h, m, d])
        .sync_on(&stream)?
        .partition([bbh, bm, d]);
    assert_eq!(out.grid()?, ((b * h) as u32, (m / bm) as u32, 1));

    let qk_scale = 1.0 / f32::sqrt(q.shape()[3] as f32);

    // This is always 1.
    let num_heads = q.shape()[1];
    let kv_num_heads = k.shape()[1];
    assert_eq!(num_heads % kv_num_heads, 0);
    let query_group_size = num_heads / kv_num_heads;

    let generics = vec![bm.to_string(), bn.to_string(), d.to_string()];
    let out_vec: Vec<f32> = fmha_kernel(
        out,
        q.clone(),
        k.clone(),
        v.clone(),
        qk_scale,
        query_group_size,
    )
    .generics(generics)
    .first()
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;
    let q_host: Vec<f32> = q.to_host_vec().sync_on(&stream)?;
    let k_host: Vec<f32> = k.to_host_vec().sync_on(&stream)?;
    let v_host: Vec<f32> = v.to_host_vec().sync_on(&stream)?;
    let answer_host = fmha_ref_exec(
        &q_host,
        &[b, h, m, d],
        &k_host,
        &[b, hkv, m, d],
        &v_host,
        &[b, hkv, m, d],
        qk_scale,
    )
    .reshape(((), m, d))
    .expect("Failed to reshape.");
    let out_candle =
        candle_core::Tensor::from_slice(&out_vec, &[b * h, m, d], &candle_core::Device::Cpu)
            .unwrap();
    println!("out shape = {:?}", out_candle.shape());
    for i in 0..(b * h) {
        let answer_mat = answer_host
            .get_on_dim(0, i)
            .expect("Failed to get {i} on dim 0.");
        let out_mat = out_candle
            .get_on_dim(0, i)
            .expect("Failed to get {i} on dim 0.");
        let near_zero = (&answer_mat - &out_mat)
            .unwrap()
            .abs()
            .unwrap()
            .reshape((m * d,))
            .unwrap();
        let vec = near_zero.to_vec1::<f32>().unwrap();
        let check = vec.iter().all(|x: &f32| x.abs() <= 1e-4);
        // Looking at out_host[i, 0, :] (1d slice of last dim d, the predicted token at position 0).
        let sample_dim = 0;
        let sample_idx = 0;
        let out_sample = out_mat
            .get_on_dim(sample_dim, sample_idx)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let answer_sample = answer_mat
            .get_on_dim(sample_dim, sample_idx)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        println!("diff near zero? {check}: {:?}", vec[0]);
        assert!(
            check,
            "output check failed. \noutput={:?} \nanswer={:?}",
            out_sample, answer_sample
        );
    }
    Ok(())
}

const BATCH: usize = 4;
const N_HEADS: usize = 32;
const HEAD_DIM: usize = 64; // or 128
const N_CTX: usize = 1024; // or some multiple of 1024

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("flash_attention", 0)? {
        return Ok(());
    }

    let b = BATCH; // = batch size.
    let h = N_HEADS; // = number of heads (query).
    let hkv = N_HEADS; // = number of heads (key/value).
    let m = N_CTX; // = sequence length.
    let d = HEAD_DIM; // = hidden size.
    let (bm, bn) = (128, 64); // (q seq len part size, k and v seq len part size)
    let bbh = 1; // batch * num_heads part size.
    fmha(b, h, hkv, m, d, bm, bn, bbh)?;
    Ok(())
}

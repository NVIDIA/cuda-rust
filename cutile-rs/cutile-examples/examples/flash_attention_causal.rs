/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
extern crate core;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
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
        const BM: i32, // Query sequence tile size.
        const BN: i32, // KV sequence tile size.
        const D: i32,  // Head dimension.
        const H: i32,  // Number of query heads.
        const CAUSAL: i32,
        const EVEN_K: i32,
    >(
        out: &mut Tensor<f32, { [1, BM, D] }>, // (b*h, m, d)
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, h, m, d)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (b, hkv, n, d)
        qk_scale: f32,
        input_pos: i32,
        query_group_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b*h, m/BM, 1)
        let batch_idx = pid.0 / H;
        let q_head_idx = pid.0 % H;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: Tile<f32, { [BM, BN] }> = broadcast_scalar(qk_scale / log2, shape![BM, BN]);

        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        let q_part: Partition<f32, { [1, 1, BM, D] }> = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [BM, D] }> = q_part
            .load([batch_idx, q_head_idx, q_m_idx, 0i32])
            .reshape(shape![BM, D]);

        let k_seqlen: i32 = k.shape()[2];
        let m_end: i32 = input_pos + (q_m_idx + 1i32) * BM;
        let mut mask_start: i32 = k_seqlen / BN;
        let mut tc: i32 = ceil_div(k_seqlen, BN);
        if CAUSAL == 1i32 {
            mask_start = (input_pos + q_m_idx * BM) / BN;
            let k_seqlen_tiles = k_seqlen / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, k_seqlen), BN);
        }

        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, D]);

        let offs_n_tile: Tile<i32, { [BN] }> = iota(shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> =
            offs_n_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);

        let offs_m_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let offs_m_iota = offs_m_iota.reshape(shape![BM, 1]);
        let offs_m: Tile<i32, { [BM, 1] }> =
            broadcast_scalar(q_m_idx * BM + input_pos, shape![BM, 1]) + offs_m_iota;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m.broadcast(shape![BM, BN]);
        let k_seqlen_tile: Tile<i32, { [BM, BN] }> = k_seqlen.broadcast(shape![BM, BN]);
        let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);

        for j in 0i32..tc {
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, kv_head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            qk = mma(tq, k_tile_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, k_seqlen_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                qk = qk + select(mask, mask_true, mask_false);
            }

            qk = qk * qk_scale;
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(shape![BM, D]);

            let v_tile: Tile<f32, { [BN, D] }> = v_part
                .load([batch_idx, kv_head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        out.store(true_div(acc, l_i.broadcast(shape![BM, D])).reshape(shape![1, BM, D]));
    }
}

use my_module::fmha;

#[allow(clippy::too_many_arguments)]
fn idx4(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    _bsz: usize,
    hsz: usize,
    m: usize,
    dsz: usize,
) -> usize {
    (((a * hsz + b) * m + c) * dsz) + d
}

#[allow(clippy::too_many_arguments)]
fn fmha_ref_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bsz: usize,
    hsz: usize,
    hkv: usize,
    m: usize,
    dsz: usize,
    qk_scale: f32,
    input_pos: usize,
    query_group_size: usize,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; bsz * hsz * m * dsz];
    let mut scores = vec![0.0f32; m];

    for b in 0..bsz {
        for h in 0..hsz {
            let kv_h = h / query_group_size;
            debug_assert!(kv_h < hkv);
            for i in 0..m {
                let mut max_logit = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate().take(m) {
                    if causal && j > i + input_pos {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for dd in 0..dsz {
                        dot += q[idx4(b, h, i, dd, bsz, hsz, m, dsz)]
                            * k[idx4(b, kv_h, j, dd, bsz, hkv, m, dsz)];
                    }
                    *score = dot * qk_scale;
                    max_logit = max_logit.max(*score);
                }

                let mut denom = 0.0f32;
                for score in scores.iter_mut().take(m) {
                    *score = (*score - max_logit).exp();
                    denom += *score;
                }

                for dd in 0..dsz {
                    let mut val = 0.0f32;
                    for (j, score) in scores.iter().enumerate().take(m) {
                        let p = *score / denom;
                        val += p * v[idx4(b, kv_h, j, dd, bsz, hkv, m, dsz)];
                    }
                    out[idx4(b, h, i, dd, bsz, hsz, m, dsz)] = val;
                }
            }
        }
    }

    out
}

fn run_attention_fmha(causal: bool) -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let (batch, heads, heads_kv, seq_len, head_dim) = (2usize, 8usize, 8usize, 64usize, 32usize);
    let (bm, bn) = (32, 32);
    let bbh = 1usize;
    let input_pos = if causal { 5i32 } else { 0i32 };
    let even_k = seq_len as i32 % bn == 0;
    let qk_scale = 1.0 / f32::sqrt(head_dim as f32);
    let query_group_size = (heads / heads_kv) as i32;

    let q: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads, seq_len, head_dim], Some(7))
        .sync_on(&stream)?
        .into();
    let k: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads_kv, seq_len, head_dim], Some(11))
        .sync_on(&stream)?
        .into();
    let v: Arc<Tensor<f32>> = randn(0.0, 1.0, [batch, heads_kv, seq_len, head_dim], Some(13))
        .sync_on(&stream)?
        .into();
    let out: Partition<Tensor<f32>> = zeros(&[batch * heads, seq_len, head_dim])
        .sync_on(&stream)?
        .partition([bbh, bm, head_dim]);

    let generics = vec![
        bm.to_string(),
        bn.to_string(),
        head_dim.to_string(),
        heads.to_string(),
        (causal as i32).to_string(),
        (even_k as i32).to_string(),
    ];

    let (out, _, _, _, _, _, _) = fmha(
        out,
        q.clone(),
        k.clone(),
        v.clone(),
        qk_scale,
        input_pos,
        query_group_size,
    )
    .generics(generics)
    .sync_on(&stream)?;

    let out_host: Vec<f32> = out.unpartition().to_host_vec().sync_on(&stream)?;
    let q_host: Vec<f32> = q.to_host_vec().sync_on(&stream)?;
    let k_host: Vec<f32> = k.to_host_vec().sync_on(&stream)?;
    let v_host: Vec<f32> = v.to_host_vec().sync_on(&stream)?;
    let ref_host: Vec<f32> = fmha_ref_cpu(
        &q_host,
        &k_host,
        &v_host,
        batch,
        heads,
        heads_kv,
        seq_len,
        head_dim,
        qk_scale,
        input_pos as usize,
        query_group_size as usize,
        causal,
    );

    let atol = 2e-3f32;
    let rtol = 2e-3f32;
    for iterand in out_host.iter().zip(ref_host.iter()).enumerate() {
        let (i, (got, exp)): (usize, (&f32, &f32)) = iterand;
        let ok = (got - exp).abs() <= atol + rtol * exp.abs();
        if !ok {
            println!("FMHA causal mismatch at idx={i}: got={got}, exp={exp}");
        }
    }

    println!(
        "attention_fmha ({}) check passed: shape=[{}, {}, {}, {}]",
        if causal { "causal" } else { "non-causal" },
        batch,
        heads,
        seq_len,
        head_dim
    );
    Ok(())
}

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("flash_attention_causal", 0)? {
        return Ok(());
    }

    run_attention_fmha(false)?;
    run_attention_fmha(true)?;
    Ok(())
}

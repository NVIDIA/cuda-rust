/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{randn_f16, zeros};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use kernels::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1,),
                       ))]
    // TODO (hme): Make D static, and pass static stride if dim is static.
    unsafe fn fmha<
        const BM: i32, // Q sequence length partition size.
        const BN: i32, // KV Sequence length partition size.
        const D: i32,  // Hidden size (weights).
    >(
        q: &Tensor<f16, { [-1, -1, -1, -1] }>, // (b, h, m, d)
        k: &Tensor<f16, { [-1, -1, -1, -1] }>, // (b, hkv, n, d) where n == m
        v: &Tensor<f16, { [-1, -1, -1, -1] }>, // (b, hkv, n, d) where n == m
        out: &mut Tensor<f16, { [1, BM, D] }>, // (b*h, m, d)
        qk_scale: f16,
        query_group_size: i32,
    ) {
        // TODO (hme): If precision issues occur, use f32 accumulator.
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b*h, m/bm, 1)
        let h = get_shape_dim(q.shape(), 1i32);
        let batch_idx = pid.0 / h; // \in  [0, b)
        let q_head_idx = pid.0 % h; // \in [0, h)
        let q_m_idx = pid.1; // \in [0, m/bm)
        let kv_head_idx = q_head_idx / query_group_size;

        // This lets us use exp2 vs exp.
        let two: Tile<f16, { [] }> =
            constant(f16::ONE, const_shape![]) + constant(f16::ONE, const_shape![]);
        let log2: f16 = tile_to_scalar(log(two));
        let qk_scale: f16 = qk_scale / log2;
        let qk_scale: Tile<f16, { [BM, BN] }> = broadcast_scalar(qk_scale, const_shape![BM, BN]);

        // mask is needed for causal only.
        // let mask_true: Tile<f16, {[BM, BN]}> = constant(0.0f16, const_shape![BM, BN]);
        // let mask_false: Tile<f16, {[BM, BN]}> = constant(f16::NEG_INFINITY, const_shape![BM, BN]);

        // offset is needed for causal only.
        // let offs_n_tile: Tile<i32, {[BN]}> = iota(const_shape![BN]);
        // let offs_n_tile: Tile<i32, {[BM, BN]}> = offs_n_tile.reshape(const_shape![1, BN])
        //     .broadcast(const_shape![BM, BN]);
        // let offs_m: i32 = q_m_idx * BM;
        // let offs_m: Tile<i32, {[BM]}> = offs_m.broadcast(const_shape![BM]);
        // let m_arange: Tile<i32, {[BM]}> = iota(const_shape![BM]);
        // let offs_m: Tile<i32, {[BM]}> = offs_m + m_arange;
        // let offs_m: Tile<i32, {[BM, BN]}> = offs_m
        //     .reshape(const_shape![BM, 1])
        //     .broadcast(const_shape![BM, BN]);

        // m and l are for softmax.
        let mut m_i: Tile<f16, { [BM, 1] }> = constant(f16::NEG_INFINITY, const_shape![BM, 1]);
        let mut l_i: Tile<f16, { [BM, 1] }> = constant(f16::ZERO, const_shape![BM, 1]);
        // This is the output tile.
        let mut acc: Tile<f16, { [BM, D] }> = constant(f16::ZERO, const_shape![BM, D]);

        // We load just one query block per process.
        let q_part: Partition<f16, { [1, 1, BM, D] }> = q.partition(const_shape![1, 1, BM, D]);
        let tq: Tile<f16, { [1, 1, BM, D] }> = q_part.load([batch_idx, q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f16, { [BM, D] }> = tq.reshape(const_shape![BM, D]);

        let n: i32 = get_shape_dim(k.shape(), 2i32);
        let num_tiles: i32 = ceil_div(n, BN);
        // let mask_start: i32 = n / BN;

        let k_part = k.partition_permuted(const_shape![1, 1, D, BN], const_array![0, 1, 3, 2]);
        let v_part = v.partition(const_shape![1, 1, BN, D]);

        // j corresponds to tile index along key / value seq len dim.
        for j in 0i32..num_tiles {
            // cuda_tile_print!("batch_idx={}, kv_head_idx={}, q_m_idx={}, j={}\n",
            //             batch_idx, kv_head_idx, q_m_idx, j);
            // Compute q @ k^T.
            let k_tile_trans: Tile<f16, { [D, BN] }> = k_part
                .load([batch_idx, kv_head_idx, 0i32, j])
                .reshape(const_shape![D, BN]);
            let qk: Tile<f16, { [BM, BN] }> = constant(f16::ZERO, const_shape![BM, BN]);
            let qk: Tile<f16, { [BM, BN] }> = mma(tq, k_tile_trans, qk);

            // Apply mask(q @ k^T).
            // if j >= mask_start {
            //     let mask: Tile<bool, {[BM, BN]}>  = constant(true, const_shape![BM, BN]);
            //     let offs_n: i32 = j * BN;
            //     let offs_n: Tile<i32, {[BM, BN]}> = offs_n.broadcast(const_shape![BM, BN]);
            //     let offs_n: Tile<i32, {[BM, BN]}> = offs_n + offs_n_tile;
            //     let mask: Tile<bool, {[BM, BN]}> = mask & ge_tile(offs_m, offs_n); // Causal only.
            //     let mask: Tile<f16, {[BM, BN]}> = select(mask, mask_true, mask_false);
            //     qk = qk + mask;
            // }

            // Apply scale(mask(q @ k^T)).
            let qk: Tile<f16, { [BM, BN] }> = qk * qk_scale;

            // Recenter before softmax.
            let qk_max: Tile<f16, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f16, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f16, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(const_shape![BM, BN]);

            // Apply softmax(mask(scale(q @ k^T))).
            let p: Tile<f16, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f16, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f16, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f16, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f16, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            // Compute softmax(mask(scale(q @ k^T))) @ v.
            // let v_tile: Tile<f16, {[1, 1, BN, D]}> = v_part.load([batch_idx, kv_head_idx, j, 0i32]);
            // TODO (hme): Separate this into safe/unsafe unchecked versions.
            let v_tile: Tile<f16, { [1, 1, BN, D] }> = load_view_tko(
                &v_part,
                [batch_idx, kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(4),
                tma::Enabled,
            );
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }
        acc = acc / l_i.broadcast(const_shape![BM, D]);
        let acc = acc.reshape(const_shape![1, BM, D]);
        out.store(acc);
    }
}

fn ocean_fmha(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmha");
    if cfg!(feature = "smoke-test") {
        group
            .warm_up_time(Duration::from_millis(1))
            .sample_size(10)
            .measurement_time(Duration::from_millis(1));
    } else {
        group
            .warm_up_time(Duration::from_millis(500))
            .sample_size(20)
            .measurement_time(Duration::from_millis(2000));
    }

    let device = Device::new(0).expect("Failed to get device.");
    let stream = device.new_stream().expect("Failed to get stream.");

    // This is what the ocean benchmark uses.
    let context_lengths = (0..6).map(|i| (2usize.pow(10 + i),)).collect::<Vec<_>>();
    const TILE_SHAPE: (usize, usize) = (128, 64);
    let hyper_params = [
        [TILE_SHAPE],
        [TILE_SHAPE],
        [TILE_SHAPE],
        [TILE_SHAPE],
        [TILE_SHAPE],
        [TILE_SHAPE],
        [TILE_SHAPE],
    ];
    for (ctx_len, tile) in context_lengths.iter().zip(hyper_params.iter()) {
        // Using calculation from https://tridao.me/publications/flash3/flash3.pdf
        // 1. Seq len from 512-16k.
        // 2. Batch size set so total tokens is 16k (e.g. @ 16k seq len batch size is 1).
        // 3. "Hidden dim" is fixed to 2048, head dim is 64, 128 or 256.
        // The calculation without causal masking is: 4 * seqlen^2 * head dimension * number of heads.
        let (max_ctx_len,) = context_lengths.last().unwrap();
        let (seq_len,) = *ctx_len;
        let batch_size = max_ctx_len / seq_len;
        let num_heads = 8;
        let head_dim = 512 / num_heads;
        println!("seq_len: {seq_len}, batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}");
        println!("total_tokens: {}", seq_len * batch_size);
        let num_ops = (4 * seq_len * seq_len * head_dim * num_heads * batch_size) as f64;

        let b = batch_size; // = batch size.
        let h = num_heads; // = number of heads (query).
        let hkv = num_heads; // = number of heads (key/value).
        let m = seq_len; // = sequence length.
        let d = head_dim; // = hidden size / head_dim.
        let bbh = 1; // batch * num_heads partition size.
        assert!(tile.len() == 1);
        let (bm, bn) = tile[0];

        let seed = 123;
        let q: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [b, h, m, d], Some(seed))
            .sync_on(&stream)
            .expect("Failed.")
            .into();
        let k: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [b, hkv, m, d], Some(seed))
            .sync_on(&stream)
            .expect("Failed.")
            .into();
        let v: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [b, hkv, m, d], Some(seed))
            .sync_on(&stream)
            .expect("Failed.")
            .into();

        let qk_scale = f16::from_f32(1.0 / f32::sqrt(q.shape()[3] as f32));

        // This is always 1.
        let num_heads = q.shape()[1];
        let kv_num_heads = k.shape()[1];
        assert_eq!(num_heads % kv_num_heads, 0);
        let query_group_size = num_heads / kv_num_heads;
        let generics = vec![bm.to_string(), bn.to_string(), d.to_string()];
        let context_length_label = m.to_string();

        // This should report flops.
        group.throughput(Throughput::Elements(num_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("N_CTX", &context_length_label),
            &context_length_label,
            |bencher, _size_mb| {
                bencher.iter_custom(|iters| {
                    // launch grid = (b*h, m/bm, 1)
                    let mut out: Partition<Tensor<f16>> = zeros(&[b * h, m, d])
                        .sync_on(&stream)
                        .expect("Failed.")
                        .partition([bbh, bm, d]);
                    assert_eq!(
                        out.grid().expect("Invalid grid."),
                        ((b * h) as u32, (m / bm) as u32, 1)
                    );
                    unsafe { stream.synchronize() }.expect("Failed to synchronize.");
                    let start = Instant::now();
                    for _i in 0..iters {
                        let (_, _, _, out_local, _, _) = unsafe {
                            fmha(
                                q.clone(),
                                k.clone(),
                                v.clone(),
                                out,
                                qk_scale,
                                query_group_size,
                            )
                            .generics(generics.clone())
                            .async_on(&stream)
                            .expect("Failed.")
                        };
                        out = out_local;
                    }
                    unsafe { stream.synchronize() }.expect("Failed to synchronize.");
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

fn bench_config() -> Criterion {
    if cfg!(feature = "smoke-test") {
        Criterion::default()
            .without_plots()
            .save_baseline("smoke-discard".to_string())
    } else {
        Criterion::default()
    }
}
criterion_group!(name = benches; config = bench_config(); targets = ocean_fmha);
criterion_main!(benches);

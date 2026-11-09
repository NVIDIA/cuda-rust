/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::cuda_core::Stream;
use cutile::half::f16;
use cutile::prelude::*;
use cutile_kernels::experimental::{
    fmha_prefill_gqa_lpt_split, prefill_splitk_reduce_merge, rms_norm_persistent_f16,
};

fn on_test_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(32_000_000)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

fn upload(values: &[f16], shape: &[usize], stream: &Arc<Stream>) -> Tensor<f16> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(shape)
        .sync_on(stream)
        .unwrap()
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn split_prefill_matches_cpu_with_tail_rows_and_empty_splits() {
    on_test_thread(|| {
        const BM: usize = 16;
        const BN: usize = 32;
        const D: usize = 64;
        const GROUP: usize = 2;
        const HEAD_GROUPS: usize = 3;
        const Q_HEADS: usize = GROUP * HEAD_GROUPS;
        const Q_LEN: usize = 33;
        const SPLITS: usize = 4;
        const M_EFF: usize = BM * GROUP;
        const NS_M: usize = SPLITS * M_EFF;
        const BLOCKS: usize = Q_LEN.div_ceil(BM);
        const TILES: usize = BLOCKS * HEAD_GROUPS;
        const SWIZZLE: usize = 2;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let scale = 1.0 / (D as f32).sqrt();
        let values = |len, salt| {
            (0..len)
                .map(|i| f16::from_f32(((i * 17 + salt) % 41) as f32 / 41.0 - 0.5))
                .collect::<Vec<_>>()
        };
        let q_host = values(Q_LEN * Q_HEADS * D, 3);
        let q = upload(&q_host, &[Q_LEN, Q_HEADS, D], &stream);
        // Nonzero query_start covers cached-prefix prefill. The two KV lengths
        // exercise both masked tail loads and the EVEN_K specialization.
        for (kv_len, query_start) in [(40usize, 7usize), (64, 31)] {
            let k_host = values(HEAD_GROUPS * kv_len * D, 9);
            let v_host = values(HEAD_GROUPS * kv_len * D, 21);
            let k = upload(&k_host, &[HEAD_GROUPS, kv_len, D], &stream);
            let v = upload(&v_host, &[HEAD_GROUPS, kv_len, D], &stream);
            let mut reference = vec![0.0f32; q_host.len()];
            for row in 0..Q_LEN {
                for head in 0..Q_HEADS {
                    let kv_head = head / GROUP;
                    let visible = (query_start + row + 1).min(kv_len);
                    let scores = (0..visible)
                        .map(|key| {
                            (0..D)
                                .map(|d| {
                                    q_host[(row * Q_HEADS + head) * D + d].to_f32()
                                        * k_host[(kv_head * kv_len + key) * D + d].to_f32()
                                })
                                .sum::<f32>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let sum: f32 = weights.iter().sum();
                    for d in 0..D {
                        reference[(row * Q_HEADS + head) * D + d] = weights
                            .iter()
                            .enumerate()
                            .map(|(key, w)| w * v_host[(kv_head * kv_len + key) * D + d].to_f32())
                            .sum::<f32>()
                            / sum;
                    }
                }
            }
            for schedule in 0..=3 {
                let partial = api::zeros::<f16>(&[TILES, NS_M, D])
                    .sync_on(&stream)
                    .unwrap();
                let lse = api::zeros::<f32>(&[TILES, NS_M]).sync_on(&stream).unwrap();
                let out = api::full(f16::NAN, &[Q_LEN, Q_HEADS, D])
                    .sync_on(&stream)
                    .unwrap();
                // SAFETY: CUDA allocations satisfy the alignment assumptions;
                // all shapes, strides, specialization constants and scratch sizes
                // match the entry contracts. Buffers remain alive and distinct,
                // and synchronous same-stream launches order producer and merge.
                unsafe {
                    fmha_prefill_gqa_lpt_split(
                        q.device_pointer(),
                        k.device_pointer(),
                        v.device_pointer(),
                        partial.device_pointer(),
                        lse.device_pointer(),
                        value(scale),
                        value(GROUP as i32),
                        value(Q_LEN as i32),
                        value(kv_len as i32),
                        value(query_start as i32),
                        value(BLOCKS as i32),
                        value(HEAD_GROUPS as i32),
                        value(SWIZZLE as i32),
                        value((HEAD_GROUPS / SWIZZLE) as i32),
                        value((HEAD_GROUPS % SWIZZLE) as i32),
                    )
                }
                .generics(
                    [
                        BM,
                        BN,
                        D,
                        GROUP,
                        M_EFF,
                        usize::from(kv_len % BN == 0),
                        SPLITS,
                        NS_M,
                        1,
                        schedule,
                    ]
                    .map(|v| v.to_string())
                    .to_vec(),
                )
                // One extra CTA verifies the guard replacing Grout's early return.
                .grid(((TILES + 1) as u32, SPLITS as u32, 1))
                .sync_on(&stream)
                .unwrap();
                // SAFETY: the matching producer completed above; merge writes
                // each output element once, with two nonoverlapping D chunks.
                unsafe {
                    prefill_splitk_reduce_merge(
                        partial.device_pointer(),
                        lse.device_pointer(),
                        out.device_pointer(),
                        value(Q_LEN as i32),
                        value(BLOCKS as i32),
                        value(HEAD_GROUPS as i32),
                        value(SWIZZLE as i32),
                        value((HEAD_GROUPS / SWIZZLE) as i32),
                        value((HEAD_GROUPS % SWIZZLE) as i32),
                    )
                }
                .generics(
                    [BM, GROUP, D, M_EFF, 32, SPLITS, NS_M, schedule, 1]
                        .map(|v| v.to_string())
                        .to_vec(),
                )
                .grid(((TILES + 1) as u32, (D / 32) as u32, 1))
                .sync_on(&stream)
                .unwrap();
                let actual = out.to_host_vec().sync_on(&stream).unwrap();
                for (index, (actual, expected)) in actual.iter().zip(&reference).enumerate() {
                    let actual = actual.to_f32();
                    assert!(actual.is_finite() && (actual - expected).abs() < 0.002,
                        "schedule={schedule}, kv_len={kv_len}, index={index}: {actual} != {expected}");
                }
            }
        }
    });
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; run via run_gpu_tests.sh"]
fn persistent_rmsnorm_matches_cpu_across_grid_strides() {
    on_test_thread(|| {
        const N: usize = 64;
        const ROWS: usize = 7;
        let context = Context::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let x_host = (0..ROWS * N)
            .map(|i| f16::from_f32((i % 19) as f32 / 8.0 - 1.0))
            .collect::<Vec<_>>();
        let w_host = (0..N)
            .map(|i| f16::from_f32(0.5 + (i % 7) as f32 / 16.0))
            .collect::<Vec<_>>();
        let x = upload(&x_host, &[ROWS, N], &stream);
        let w = upload(&w_host, &[N], &stream);
        for blocks in [1, 3, 10] {
            let out = api::full(f16::NAN, &[ROWS, N]).sync_on(&stream).unwrap();
            // SAFETY: allocations are aligned and disjoint, strides are multiples
            // of 16, N is divisible by 32, and all buffers outlive the launch.
            unsafe {
                rms_norm_persistent_f16(
                    x.device_pointer(),
                    value(ROWS as i32),
                    value(N as i32),
                    w.device_pointer(),
                    out.device_pointer(),
                    value(N as i32),
                    value(1e-5f32),
                )
            }
            .generics(vec![N.to_string(), "32".into()])
            .grid((blocks, 1, 1))
            .sync_on(&stream)
            .unwrap();
            let actual = out.to_host_vec().sync_on(&stream).unwrap();
            for row in 0..ROWS {
                let rms = (x_host[row * N..(row + 1) * N]
                    .iter()
                    .map(|x| x.to_f32().powi(2))
                    .sum::<f32>()
                    / N as f32
                    + 1e-5)
                    .sqrt();
                for d in 0..N {
                    let expected = x_host[row * N + d].to_f32() / rms * w_host[d].to_f32();
                    let actual = actual[row * N + d].to_f32();
                    assert!(
                        actual.is_finite() && (actual - expected).abs() < 0.002,
                        "blocks={blocks}, row={row}, d={d}: {actual} != {expected}"
                    );
                }
            }
        }
    });
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A/B timing of the two rank-by-key cores (#1658): the one-hot `[N, K]`
//! scan (`rank_by_key_probe`) against one `[N]` scan per key value
//! (`rank_by_key_loop`). The issue measured the per-key-scan form ~40%
//! faster inside a full radix sort on sm_86; this isolates the primitive.
//!
//! Opt-in and device-only; not part of `run_gpu_tests.sh`. Run with
//! `cargo test -p cutile-kernels --test rank_sort_bench_gpu -- --ignored --nocapture`.

use std::sync::Arc;

use cutile::bench::{do_bench_paired, BenchOptions};
use cutile::cuda_core::Stream;
use cutile::prelude::*;
use cutile_kernels::sort::{rank_by_key_loop, rank_by_key_probe};

fn on_test_thread(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(32_000_000)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

fn pattern(n: usize, k: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + 3) % k) as u32).collect()
}

fn up_u32(values: &[u32], shape: &[usize], stream: &Arc<Stream>) -> Tensor<u32> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(shape)
        .sync_on(stream)
        .unwrap()
}

#[test]
#[ignore = "requires a CUDA GPU and Tile IR assembler; opt-in benchmark"]
fn bench_rank_by_key_cores() {
    on_test_thread(|| {
        let ctx = Context::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        // 1024 x 32 is absent: the one-hot form's [N, K] working set at that
        // shape (128 KiB of i32 before the scan) fails Tile IR assembly on
        // sm_86, so 512 x 32 keeps the K=32 comparison compilable.
        for (n, k) in [(1024usize, 8usize), (1024, 16), (512, 32)] {
            let digit = pattern(n, k);
            let d_t = up_u32(&digit, &[n], &stream);
            let zeros_n = vec![0i32; n];
            let zeros_k = vec![0i32; k];
            let mut rank_a = api::copy_host_vec_to_device(&Arc::new(zeros_n.clone()))
                .reshape(&[n])
                .sync_on(&stream)
                .unwrap();
            let mut counts_a = api::copy_host_vec_to_device(&Arc::new(zeros_k.clone()))
                .reshape(&[k])
                .sync_on(&stream)
                .unwrap();
            let mut rank_b = api::copy_host_vec_to_device(&Arc::new(zeros_n))
                .reshape(&[n])
                .sync_on(&stream)
                .unwrap();
            let mut counts_b = api::copy_host_vec_to_device(&Arc::new(zeros_k))
                .reshape(&[k])
                .sync_on(&stream)
                .unwrap();
            let generics = vec![n.to_string(), k.to_string()];

            let (one_hot, per_key) = do_bench_paired(
                &stream,
                &BenchOptions::default(),
                |s| {
                    unsafe {
                        rank_by_key_probe(
                            &d_t,
                            (&mut rank_a).partition([n]),
                            (&mut counts_a).partition([k]),
                        )
                    }
                    .generics(generics.clone())
                    .sync_on(s)
                    .map(|_| ())
                    .map_err(Into::into)
                },
                |s| {
                    unsafe {
                        rank_by_key_loop(
                            &d_t,
                            (&mut rank_b).partition([n]),
                            (&mut counts_b).partition([k]),
                        )
                    }
                    .generics(generics.clone())
                    .sync_on(s)
                    .map(|_| ())
                    .map_err(Into::into)
                },
            )
            .unwrap();

            println!(
                "N={n} K={k}: one-hot median {:.3} ms (min {:.3}), per-key-scan median {:.3} ms \
                 (min {:.3}), one-hot/per-key = {:.2}x",
                one_hot.median_ms(),
                one_hot.min_ms(),
                per_key.median_ms(),
                per_key.min_ms(),
                one_hot.median_ms() / per_key.median_ms(),
            );
        }
    });
}

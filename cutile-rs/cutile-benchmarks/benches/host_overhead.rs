/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side cost of enqueueing work, isolated from GPU time.
//!
//! Times only the enqueue (`async_on`), never the wait: a batch of launches
//! is timed, then the stream is drained outside the timer so the launch
//! queue never fills. Two shapes that matter for decode-style serving:
//!
//! - `launch`: one tiny 4-tensor kernel launch (argument retention, launch
//!   validation, `cuLaunchKernel`).
//! - `graph_replay`: one replay of a graph recording 32 such launches over
//!   8 buffers (per-replay resource reacquisition plus `cuGraphLaunch`).

use criterion::{criterion_group, criterion_main, Criterion};
use cuda_async::cuda_graph::CudaGraph;
use cuda_core::Device;
use cutile::prelude::*;
use std::future::IntoFuture;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add3(
        z: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
    ) {
        let ta: Tile<f32, { [128] }> = a.load_like(z);
        let tb: Tile<f32, { [128] }> = b.load_like(z);
        let tc: Tile<f32, { [128] }> = c.load_like(z);
        z.store(ta + tb + tc);
    }

    /// Same kernel with a scalar argument: a scalar's divisibility hint is
    /// part of the launch-site probe, so a value that changes every launch
    /// defeats the site cache and takes the key-rebuild path each time.
    #[cutile::entry()]
    fn add3_scaled(
        z: &mut Tensor<f32, { [128] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
        k: i32,
    ) {
        let ta: Tile<f32, { [128] }> = a.load_like(z);
        let tb: Tile<f32, { [128] }> = b.load_like(z);
        let tc: Tile<f32, { [128] }> = c.load_like(z);
        let kf: f32 = convert_scalar(k);
        let scale: Tile<f32, { [128] }> = broadcast_scalar(kf, shape![128]);
        z.store((ta + tb + tc) * scale);
    }
}

const N: usize = 128;
/// Launches (or replays) per drain of the stream.
const BATCH: u64 = 32;

fn host_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("host_overhead");
    if cfg!(feature = "smoke-test") {
        group
            .warm_up_time(Duration::from_millis(1))
            .sample_size(10)
            .measurement_time(Duration::from_millis(1));
    } else {
        group
            .warm_up_time(Duration::from_millis(500))
            .sample_size(100)
            .measurement_time(Duration::from_millis(2000));
    }

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");

    let a: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("a").into();
    let b: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("b").into();
    let c3: Arc<Tensor<f32>> = api::ones::<f32>(&[N]).sync_on(&stream).expect("c").into();
    let fresh_z = || -> Partition<Tensor<f32>> {
        api::zeros::<f32>(&[N])
            .sync_on(&stream)
            .expect("z")
            .partition([N])
    };

    // JIT + steady-state warmup.
    let mut z = fresh_z();
    for _ in 0..200 {
        let (local_z, _, _, _) = kernels::add3(z, a.clone(), b.clone(), c3.clone())
            .sync_on(&stream)
            .expect("warmup");
        z = local_z;
    }
    drop(z);
    let mut z = fresh_z();

    group.bench_function("launch", |bench| {
        bench.iter_custom(|iters| {
            let mut z = fresh_z();
            let mut elapsed = Duration::ZERO;
            let mut done = 0;
            while done < iters {
                let batch = BATCH.min(iters - done);
                let start = Instant::now();
                for _ in 0..batch {
                    let (local_z, _, _, _) = unsafe {
                        kernels::add3(z, a.clone(), b.clone(), c3.clone())
                            .async_on(&stream)
                            .expect("launch")
                    };
                    z = local_z;
                }
                elapsed += start.elapsed();
                unsafe { stream.synchronize() }.expect("drain");
                done += batch;
            }
            elapsed
        });
    });

    // Launch-site miss on every launch: the scalar changes each time.
    for _ in 0..50 {
        let (local_z, _, _, _, _) = kernels::add3_scaled(z, a.clone(), b.clone(), c3.clone(), 3)
            .sync_on(&stream)
            .expect("warmup scaled");
        z = local_z;
    }
    group.bench_function("launch_site_miss", |bench| {
        bench.iter_custom(|iters| {
            let mut z = fresh_z();
            let mut elapsed = Duration::ZERO;
            let mut done = 0u64;
            let mut k: i32 = 1;
            while done < iters {
                let batch = BATCH.min(iters - done);
                let start = Instant::now();
                for _ in 0..batch {
                    k = k % 97 + 1;
                    let (local_z, _, _, _, _) = unsafe {
                        kernels::add3_scaled(z, a.clone(), b.clone(), c3.clone(), k)
                            .async_on(&stream)
                            .expect("launch")
                    };
                    z = local_z;
                }
                elapsed += start.elapsed();
                unsafe { stream.synchronize() }.expect("drain");
                done += batch;
            }
            elapsed
        });
    });

    // Each launch awaited individually: the future path (poll, inline spin
    // or reactor registration, completion, release of the submission).
    group.bench_function("launch_awaited", |bench| {
        bench.iter_custom(|iters| {
            let mut z = fresh_z();
            let start = Instant::now();
            for _ in 0..iters {
                let (local_z, _, _, _) = futures::executor::block_on(
                    kernels::add3(z, a.clone(), b.clone(), c3.clone()).into_future(),
                )
                .expect("awaited launch");
                z = local_z;
            }
            start.elapsed()
        });
    });

    // A graph of 32 launches over 8 buffers, replayed.
    let mut bufs: Vec<Tensor<f32>> = (0..8)
        .map(|_| api::zeros::<f32>(&[N]).sync_on(&stream).expect("buf"))
        .collect();
    let graph = CudaGraph::scope(&stream, |s| {
        for i in 0..32usize {
            let (out, rest) = bufs.split_at_mut(1);
            let out = &mut out[0];
            let src = &rest[i % 7];
            s.record(kernels::add3((&mut *out).partition([N]), src, &a, &b))?;
        }
        Ok(())
    })
    .expect("capture");
    for _ in 0..50 {
        graph.launch().sync_on(&stream).expect("graph warmup");
    }

    group.bench_function("graph_replay", |bench| {
        bench.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;
            let mut done = 0;
            while done < iters {
                let batch = 8u64.min(iters - done);
                let start = Instant::now();
                for _ in 0..batch {
                    unsafe { graph.launch().async_on(&stream) }.expect("replay");
                }
                elapsed += start.elapsed();
                unsafe { stream.synchronize() }.expect("drain");
                done += batch;
            }
            elapsed
        });
    });

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
criterion_group!(name = benches; config = bench_config(); targets = host_overhead);
criterion_main!(benches);

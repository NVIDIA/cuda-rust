/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Measures host-side launch overhead across execution modes.
//!
//! Compares three ways to execute the same kernel:
//!   1. `sync_on`  — execute + stream synchronize (no callbacks)
//!   2. `await`    — execute + cuLaunchHostFunc callback + async poll
//!   3. `async_on` — execute only, single stream synchronize at end
//!
//! Uses a small GEMM (1024x1024) so kernel time is short and host overhead
//! is a measurable fraction of total time.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = z.load();
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

const WARMUP_ITERS: u64 = 500;

fn launch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("launch_overhead");
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

    let device = Device::new(0).expect("Failed to get device.");
    let stream = device.new_stream().expect("Failed to get stream.");

    let n: usize = 1024;
    let (bm, bn, bk) = (128, 128, 64);
    let generics = vec![
        "f16".to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
    ];

    let x: Arc<Tensor<f16>> = api::ones::<f16>(&[n, n])
        .sync_on(&stream)
        .expect("Failed.")
        .into();
    let y: Arc<Tensor<f16>> = api::ones::<f16>(&[n, n])
        .sync_on(&stream)
        .expect("Failed.")
        .into();

    // JIT warmup + steady-state warmup for sync_on path.
    {
        let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
            .sync_on(&stream)
            .expect("Failed.")
            .partition([bm, bn]);
        for _ in 0..WARMUP_ITERS {
            let (local_z, _, _, _) = unsafe {
                kernels::gemm(z, x.clone(), y.clone(), n as i32)
                    .generics(generics.clone())
                    .sync_on(&stream)
                    .expect("Failed.")
            };
            z = local_z;
        }
    }
    unsafe { stream.synchronize() }.expect("Failed.");
    std::thread::sleep(Duration::from_millis(200));

    // 1. sync_on: execute + unsafe { stream.synchronize() }, no callbacks.
    group.bench_function(BenchmarkId::new("mode", "sync_on"), |b| {
        b.iter_custom(|iters| {
            let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
                .sync_on(&stream)
                .expect("Failed.")
                .partition([bm, bn]);
            unsafe { stream.synchronize() }.expect("Failed.");
            let start = Instant::now();
            for _ in 0..iters {
                let (local_z, _, _, _) = unsafe {
                    kernels::gemm(z, x.clone(), y.clone(), n as i32)
                        .generics(generics.clone())
                        .sync_on(&stream)
                        .expect("Failed.")
                };
                z = local_z;
            }
            start.elapsed()
        });
    });

    // Steady-state warmup for await path.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime.");
    {
        rt.block_on(async {
            let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
                .sync_on(&stream)
                .expect("Failed.")
                .partition([bm, bn]);
            for _ in 0..WARMUP_ITERS {
                let (local_z, _, _, _) = unsafe {
                    kernels::gemm(z, x.clone(), y.clone(), n as i32).generics(generics.clone())
                }
                .await
                .expect("Failed.");
                z = local_z;
            }
        });
    }
    unsafe { stream.synchronize() }.expect("Failed.");
    std::thread::sleep(Duration::from_millis(200));

    // 2. await: execute on first poll + cuLaunchHostFunc callback.
    group.bench_function(BenchmarkId::new("mode", "await"), |b| {
        b.iter_custom(|iters| {
            rt.block_on(async {
                let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
                    .sync_on(&stream)
                    .expect("Failed.")
                    .partition([bm, bn]);
                unsafe { stream.synchronize() }.expect("Failed.");
                let start = Instant::now();
                for _ in 0..iters {
                    let (local_z, _, _, _) = unsafe {
                        kernels::gemm(z, x.clone(), y.clone(), n as i32).generics(generics.clone())
                    }
                    .await
                    .expect("Failed.");
                    z = local_z;
                }
                start.elapsed()
            })
        });
    });

    // Steady-state warmup for async_on path.
    {
        let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
            .sync_on(&stream)
            .expect("Failed.")
            .partition([bm, bn]);
        for _ in 0..WARMUP_ITERS {
            unsafe {
                let (local_z, _, _, _) = kernels::gemm(z, x.clone(), y.clone(), n as i32)
                    .generics(generics.clone())
                    .async_on(&stream)
                    .expect("Failed.");
                z = local_z;
            }
        }
    }
    unsafe { stream.synchronize() }.expect("Failed.");
    std::thread::sleep(Duration::from_millis(200));

    // 3. async_on: execute only, single synchronize at end.
    group.bench_function(BenchmarkId::new("mode", "async_on"), |b| {
        b.iter_custom(|iters| {
            let mut z: Partition<Tensor<f16>> = api::zeros::<f16>(&[n, n])
                .sync_on(&stream)
                .expect("Failed.")
                .partition([bm, bn]);
            unsafe { stream.synchronize() }.expect("Failed.");
            let start = Instant::now();
            for _ in 0..iters {
                unsafe {
                    let (local_z, _, _, _) = kernels::gemm(z, x.clone(), y.clone(), n as i32)
                        .generics(generics.clone())
                        .async_on(&stream)
                        .expect("Failed.");
                    z = local_z;
                }
            }
            unsafe { stream.synchronize() }.expect("Failed.");
            start.elapsed()
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
criterion_group!(name = benches; config = bench_config(); targets = launch_overhead);
criterion_main!(benches);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Benchmark: thread-local execution lock overhead.
//!
//! Compares the cost of `sync_on` (with lock) vs `async_on` + manual
//! synchronize (without lock) to measure the overhead of the thread-local
//! execution guard.

use criterion::{criterion_group, criterion_main, Criterion};
use cuda_async::device_operation::{value, DeviceOp};
use std::time::Duration;

fn has_gpu() -> bool {
    cuda_core::Device::device_count()
        .map(|n| n > 0)
        .unwrap_or(false)
}

fn bench_execution_lock(c: &mut Criterion) {
    if !has_gpu() {
        eprintln!("No GPU available, skipping execution_lock benchmark.");
        return;
    }

    // Run on a fresh thread to get a clean CUDA context.
    let (stream,) = std::thread::spawn(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        (stream,)
    })
    .join()
    .unwrap();

    let mut group = c.benchmark_group("execution_lock");
    if cfg!(feature = "smoke-test") {
        group.warm_up_time(Duration::from_millis(1));
        group.sample_size(10);
        group.measurement_time(Duration::from_millis(1));
    } else {
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
    }

    // Baseline: value(42).sync_on(&stream) — includes lock acquire/release.
    group.bench_function("sync_on_with_lock", |b| {
        b.iter(|| {
            value(42).sync_on(&stream).unwrap();
        });
    });

    // Comparison: unsafe async_on + manual synchronize — no lock.
    group.bench_function("async_on_no_lock", |b| {
        b.iter(|| {
            unsafe { value(42).async_on(&stream).unwrap() };
            unsafe { stream.synchronize() }.unwrap();
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
criterion_group!(name = benches; config = bench_config(); targets = bench_execution_lock);
criterion_main!(benches);

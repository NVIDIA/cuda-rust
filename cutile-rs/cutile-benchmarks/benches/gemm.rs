/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_async::device_operation::{value, DeviceOp};
use cuda_core::Device;
use cutile::api;
use cutile::core::f16;
use cutile::tile_kernel::{PartitionOp, TileKernel};
use kernels::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {

    use cutile::core::*;

    #[cutile::entry(unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2,),
                       )
    )]
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
    #[cutile::entry()]
    unsafe fn gemm_no_opt<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
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
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemm_no_opt_unchecked<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
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

fn ocean_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
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
    let shapes = (0..6)
        .map(|i| {
            let n = 2usize.pow(10 + i);
            (n, n, n)
        })
        .collect::<Vec<_>>();
    let hyper_params = [
        ((128, 128, 64), 2),
        ((256, 256, 64), 2),
        ((256, 256, 64), 2),
        ((256, 256, 64), 2),
        ((256, 256, 64), 2),
        ((256, 256, 64), 2),
    ];
    for (&(m, n, k), &((bm, bn, bk), _group_size_m)) in shapes.iter().zip(hyper_params.iter()) {
        let generics = vec![
            "f16".to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
        ];
        let x = api::ones(&[m, k])
            .then(|t| value(Arc::new(t)))
            .sync_on(&stream)
            .expect("Failed.");
        let y = api::ones(&[k, n])
            .then(|t| value(Arc::new(t)))
            .sync_on(&stream)
            .expect("Failed.");

        let num_ops = (2 * m * n * k) as f64;
        let label = n.to_string();
        // This should report flops.
        group.throughput(Throughput::Elements(num_ops as u64));
        group.bench_with_input(BenchmarkId::new("N", &label), &label, |b, _size_mb| {
            b.iter_custom(|iters| {
                let mut z = api::zeros::<f16>(&[m, n])
                    .partition([bm, bn])
                    .sync_on(&stream)
                    .expect("Failed.");
                unsafe { stream.synchronize() }.expect("Failed to synchronize.");
                let start = Instant::now();
                for _i in 0..iters {
                    unsafe {
                        let (local_z, _, _, _) = gemm(z, x.clone(), y.clone(), k as i32)
                            .generics(generics.clone())
                            .async_on(&stream)
                            .expect("Failed.");
                        z = local_z;
                    }
                }
                unsafe { stream.synchronize() }.expect("Failed to synchronize.");
                start.elapsed()
            });
        });
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
criterion_group!(name = benches; config = bench_config(); targets = ocean_gemm);
criterion_main!(benches);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Execution-mode overhead benchmark — paper §5.2.
//
// Measures host-visible latency for an N-kernel pipeline under the sync,
// async, and graph execution modes. Sync is split into two measured schedules:
//   1. sync-individual : N separate .sync_on()  — host blocks between each kernel
//   2. sync-chained    : .sync() on a dependent chain of N launches
//   3. async           : one .await on a dependent chain of N launches
//   4. graph           : captured once via CudaGraph::capture, replayed in hot loop
//
// The pipeline is N dependent k_scale launches: each launch rewrites the
// single buffer the previous launch produced (z = x * g), so the launches
// must execute in order. Every mode scales this same buffer N times, varying
// only in how the launches are driven. We report total per-iteration wall
// time (not per-launch) so the scaling story is visible: sync-individual
// grows linearly with N, the others amortize.
//
// Methodology:
//   - N_WARMUP fixed-count warmup iterations per configuration
//   - N_SAMPLES samples; each sample times ITERS full N-kernel pipelines
//   - median per-iteration wall-clock time → CSV (plus min/p25/p75/max)
//
// CLI:
//   --mode <sync-individual|sync-chained|async|graph>   required
//   --n <ops-per-pipeline>                              default 3
//   --d <tile size>                                     default 2048
//   --csv <path>                                        default results_part_a.csv

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{BoxedDeviceOp, DeviceOp, Unzippable3};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use kernels::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn k_scale<T: ElementType, const D: i32>(
        y: &mut Tensor<T, { [D] }>,
        x: &Tensor<T, { [-1] }>,
        g: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, y);
        let tile_g = load_tile_like(g, y);
        y.store(tile_x * tile_g);
    }
}

const D: i32 = 2048;
const DEFAULT_N_WARMUP: u32 = 200;
const DEFAULT_ITERS: u32 = 100;
const DEFAULT_N_SAMPLES: usize = 50;

#[derive(Clone, Copy)]
enum Mode {
    SyncIndividual,
    SyncChained,
    Async,
    Graph,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "sync-individual" => Some(Mode::SyncIndividual),
            "sync-chained" => Some(Mode::SyncChained),
            "async" => Some(Mode::Async),
            "graph" => Some(Mode::Graph),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::SyncIndividual => "sync-individual",
            Mode::SyncChained => "sync-chained",
            Mode::Async => "async",
            Mode::Graph => "graph",
        }
    }
}

fn usage() -> ! {
    eprintln!("usage: launch_overhead --mode <sync-individual|sync-chained|async|graph> [--n <ops>] [--d <size>] [--csv <path>]");
    std::process::exit(2);
}

type TensorF16 = Arc<Tensor<f16>>;
type Part = Partition<Tensor<f16>>;

fn alloc_inputs(stream: &Arc<Stream>, d: usize) -> (TensorF16, TensorF16) {
    let x: TensorF16 = api::ones::<f16>(&[d])
        .sync_on(stream)
        .expect("alloc x")
        .into();
    let g: TensorF16 = api::ones::<f16>(&[d])
        .sync_on(stream)
        .expect("alloc g")
        .into();
    (x, g)
}

fn fresh_partition(stream: &Arc<Stream>, d: usize) -> Part {
    api::zeros::<f16>(&[d])
        .sync_on(stream)
        .expect("alloc partition")
        .partition([d])
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn summarize(samples: &mut Vec<f64>) -> (f64, f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        samples[0],
        quantile(samples, 0.25),
        samples[samples.len() / 2],
        quantile(samples, 0.75),
        samples[samples.len() - 1],
    )
}

// ---------- mode implementations ----------

fn bench_sync_individual(
    stream: &Arc<Stream>,
    generics: &[String],
    x: TensorF16,
    g: TensorF16,
    d: usize,
    n: usize,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    let mut part: Part = fresh_partition(stream, d);

    // Each op is built and launched eagerly: k_scale(...).sync_on(stream)
    // constructs the kernel launcher and issues it, blocking the host between
    // launches. That per-kernel launcher construction is the launch overhead
    // under test, so (unlike the chained modes' chain-assembly boxing) it stays
    // inside the timed region -- it is the same per-kernel construction the
    // chained modes pay inside their execute walk. The single buffer is
    // threaded through all N launches, matching the chained/graph modes.
    let run = |mut p: Part| -> Part {
        for _ in 0..n {
            let (q, _, _) = unsafe {
                k_scale(p, &*x, &*g)
                    .generics(generics.to_vec())
                    .sync_on(stream)
                    .expect("k_scale")
            };
            p = q;
        }
        p
    };

    for _ in 0..n_warmup {
        part = run(part);
    }
    unsafe { stream.synchronize() }.expect("sync");

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        unsafe { stream.synchronize() }.expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            part = run(part);
        }
        samples.push(t0.elapsed().as_secs_f64() / iters as f64);
    }
    samples
}

fn bench_sync_chained(
    stream: &Arc<Stream>,
    generics: &[String],
    x: TensorF16,
    g: TensorF16,
    d: usize,
    n: usize,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    let mut part: Part = fresh_partition(stream, d);

    // Warm up by building and executing the chain. This is the same chain the
    // async mode drives (build_chain), but executed with .sync() instead of
    // .await.
    for _ in 0..n_warmup {
        let chain = build_chain(part, n, &x, &g, generics);
        part = chain.sync().expect("sync chain");
    }
    unsafe { stream.synchronize() }.expect("sync");

    // Time chain *execution* only, not construction. build_chain boxes one op
    // per launch; that boxing is an artifact of assembling the chain
    // dynamically in a loop, so its O(N) host cost is built outside the timer.
    // Each launcher is still constructed inside the chain's execute walk, so
    // the per-launch cost remains in the measured region.
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        unsafe { stream.synchronize() }.expect("sync");
        let mut elapsed = std::time::Duration::ZERO;
        for _ in 0..iters {
            let chain = build_chain(part, n, &x, &g, generics);
            let t0 = Instant::now();
            part = chain.sync().expect("sync chain");
            elapsed += t0.elapsed();
        }
        samples.push(elapsed.as_secs_f64() / iters as f64);
    }
    samples
}

/// Fold N dependent k_scale launches into a single boxed DeviceOp. Each launch
/// writes the partition the previous launch produced -- the same buffer
/// rewritten N times, a write-after-write dependency -- so the launches must
/// execute in order. The dependency is expressed by dataflow rather than
/// `.then()`: the running `chain` op is passed straight in as the next launch's
/// output buffer (a kernel launcher accepts any `IntoDeviceOp` argument, so it
/// waits on `chain` to produce the buffer before launching), then `.first()`
/// recovers the output partition and the result replaces `chain`. The pipeline
/// collapses to one `BoxedDeviceOp<Part>` -- the op the graph mode captures and
/// replays. Each `.boxed()` is one heap allocation, so construction is O(N);
/// every caller builds the chain outside its timer.
fn build_chain(
    part: Part,
    n: usize,
    x: &TensorF16,
    g: &TensorF16,
    generics: &[String],
) -> BoxedDeviceOp<Part> {
    let mut chain: BoxedDeviceOp<Part> =
        unsafe { k_scale(part, x.clone(), g.clone()).generics(generics.to_vec()) }
            .first()
            .boxed();

    for _ in 1..n {
        chain = unsafe { k_scale(chain, x.clone(), g.clone()).generics(generics.to_vec()) }
            .first()
            .boxed();
    }
    chain
}

fn bench_async(
    stream: &Arc<Stream>,
    generics: &[String],
    x: TensorF16,
    g: TensorF16,
    d: usize,
    n: usize,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async {
        let mut part: Part = fresh_partition(stream, d);

        // Warm up: build the chain and await it, recovering the buffer for the
        // next build.
        for _ in 0..n_warmup {
            let chain = build_chain(part, n, &x, &g, generics);
            part = chain.await.expect("chain");
        }
        unsafe { stream.synchronize() }.expect("sync");

        // Time chain execution (.await) only; build_chain is built outside the
        // timer for the same reason as the sync-chained mode (its per-launch
        // boxing is a dynamic-construction artifact, not part of the execution
        // path), while per-launch cost stays in the execute walk.
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            unsafe { stream.synchronize() }.expect("sync");
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iters {
                let chain = build_chain(part, n, &x, &g, generics);
                let t0 = Instant::now();
                part = chain.await.expect("chain");
                elapsed += t0.elapsed();
            }
            samples.push(elapsed.as_secs_f64() / iters as f64);
        }
        samples
    })
}

fn bench_graph(
    stream: &Arc<Stream>,
    generics: &[String],
    x: TensorF16,
    g: TensorF16,
    d: usize,
    n: usize,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    // Capture one replayable N-launch chain; the hot loop pays only the
    // graph replay + stream sync, not per-launch construction.
    let part = fresh_partition(stream, d);
    let chain = build_chain(part, n, &x, &g, generics);

    let graph = CudaGraph::capture(stream.clone(), chain).expect("graph capture");

    for _ in 0..n_warmup {
        graph.launch().sync_on(stream).expect("graph warmup");
    }
    unsafe { stream.synchronize() }.expect("sync");

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        unsafe { stream.synchronize() }.expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            graph.launch().sync_on(stream).expect("graph replay");
        }
        samples.push(t0.elapsed().as_secs_f64() / iters as f64);
    }
    samples
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mode = arg_value(&args, "--mode")
        .and_then(Mode::parse)
        .unwrap_or_else(|| usage());
    let n: usize = arg_value(&args, "--n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let csv_path = arg_value(&args, "--csv")
        .unwrap_or("results_part_a.csv")
        .to_string();
    let n_warmup: u32 = arg_value(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_WARMUP);
    let iters: u32 = arg_value(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    let n_samples: usize = arg_value(&args, "--samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_SAMPLES);

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");

    let d = D as usize;
    let generics = vec!["f16".to_string(), D.to_string()];

    let (x, g) = alloc_inputs(&stream, d);

    let mut samples = match mode {
        Mode::SyncIndividual => {
            bench_sync_individual(&stream, &generics, x, g, d, n, n_warmup, iters, n_samples)
        }
        Mode::SyncChained => {
            bench_sync_chained(&stream, &generics, x, g, d, n, n_warmup, iters, n_samples)
        }
        Mode::Async => bench_async(&stream, &generics, x, g, d, n, n_warmup, iters, n_samples),
        Mode::Graph => bench_graph(&stream, &generics, x, g, d, n, n_warmup, iters, n_samples),
    };

    let (min_s, p25_s, med_s, p75_s, max_s) = summarize(&mut samples);
    let med_us = med_s * 1e6;
    let min_us = min_s * 1e6;
    let max_us = max_s * 1e6;
    let p25_us = p25_s * 1e6;
    let p75_us = p75_s * 1e6;

    println!(
        "mode={} d={} n_ops={} samples={} us min={:.2} p25={:.2} med={:.2} p75={:.2} max={:.2}",
        mode.label(),
        D,
        n,
        samples.len(),
        min_us,
        p25_us,
        med_us,
        p75_us,
        max_us,
    );

    let need_header = !std::path::Path::new(&csv_path).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)?;
    if need_header {
        writeln!(
            f,
            "mode,d,n_ops,median_us,min_us,p25_us,p75_us,max_us,n_samples"
        )?;
    }
    writeln!(
        f,
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        mode.label(),
        D,
        n,
        med_us,
        min_us,
        p25_us,
        p75_us,
        max_us,
        samples.len()
    )?;

    Ok(())
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// §5.2 Part B — async/sync throughput under configurable host work.
//
// Measures iteration rate of a pipeline consisting of (a) a graph-replayed
// N-kernel chain on the GPU and (b) a host-side busy-spin of duration W.
//
//   sync  mode: graph.launch().sync_on(stream); spin(W);  // serial
//   async mode: tokio::join!(gpu_future, async { spin(W) })   // overlap
//
// Both modes run on the same explicit CUDA stream (we bypass the default
// scheduling policy by constructing DeviceFuture directly with an
// ExecutionContext), and both recover the pipeline outputs every
// iteration, so the only difference is whether the host thread blocks on
// the GPU (sync) or yields to the spin during the await (async).
//
// The pipeline is graph-replay, not a chain-of-`.then()`, because at the
// scales we care about (N in the hundreds) chained host submission
// itself dominates the pipeline's wall time — the async path can only
// overlap *after* the last submit returns, leaving very little room for
// W to hide behind. A captured graph collapses N submits into one
// driver call and exposes the full GPU execution time as overlappable
// work. This matches how today's decode engines (e.g.\ Grout) submit.
//
// CLI:
//   --mode <sync|async>         required
//   --w   <host-us>             default 0
//   --n   <ops per graph>       default 300
//   --iters <per-sample>        default 200
//   --warmup <warmup iters>     default 50
//   --samples <#>               default 20
//   --csv   <path>              default results_part_b.csv

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{BoxedDeviceOp, DeviceOp, ExecutionContext};
use cuda_core::{Device, Stream};
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use kernels::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Tiny elementwise kernel. The pipeline is N of these chained via
    // .then() and then captured as one graph, so the host sees a single
    // replay op while the GPU runs N kernels in stream order.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn k_scale<T: ElementType, const D: i32>(
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
const DEFAULT_N: usize = 300;
const DEFAULT_N_WARMUP: u32 = 50;
const DEFAULT_ITERS: u32 = 200;
const DEFAULT_N_SAMPLES: usize = 20;

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

fn fresh_partitions(stream: &Arc<Stream>, d: usize, n: usize) -> Vec<Part> {
    (0..n)
        .map(|_| {
            api::zeros::<f16>(&[d])
                .sync_on(stream)
                .expect("alloc partition")
                .partition([d])
        })
        .collect()
}

/// Fold N k_scale launches into a single DeviceOp. Output is the vec of
/// recovered partitions; for graph capture we only need the op itself
/// (output is thrown away on replay since the graph reuses partitions).
fn build_chain(
    parts: Vec<Part>,
    x: &TensorF16,
    g: &TensorF16,
    generics: &[String],
) -> BoxedDeviceOp<Vec<Part>> {
    assert!(!parts.is_empty());
    let mut it = parts.into_iter();

    let p0 = it.next().unwrap();
    let x0 = x.clone();
    let g0 = g.clone();
    let gen0 = generics.to_vec();
    let mut chain: BoxedDeviceOp<Vec<Part>> = unsafe { k_scale(p0, x0, g0).generics(gen0) }
        .map(move |(q, _, _)| vec![q])
        .boxed();

    for p in it {
        let xi = x.clone();
        let gi = g.clone();
        let gen_i = generics.to_vec();
        chain = chain
            .then(move |mut acc: Vec<Part>| {
                unsafe { k_scale(p, xi, gi).generics(gen_i) }.map(move |(q, _, _)| {
                    acc.push(q);
                    acc
                })
            })
            .boxed();
    }
    chain
}

/// Busy-spin for `w` microseconds on the current thread. Uses
/// spin_loop() to keep the CPU pegged (so we actually occupy the host
/// thread rather than yielding to OS scheduling).
#[inline(never)]
fn spin_us(w: u64) {
    if w == 0 {
        return;
    }
    let target = Duration::from_micros(w);
    let start = Instant::now();
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
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

#[derive(Clone, Copy)]
enum Mode {
    Sync,
    Async,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "sync" => Some(Mode::Sync),
            "async" => Some(Mode::Async),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Sync => "sync",
            Mode::Async => "async",
        }
    }
}

fn bench_sync(
    stream: &Arc<Stream>,
    graph: &CudaGraph<Vec<Part>>,
    w_us: u64,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    for _ in 0..n_warmup {
        graph.launch().sync_on(stream).expect("graph replay");
        spin_us(w_us);
    }
    unsafe { stream.synchronize() }.expect("sync");

    let mut throughput = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        unsafe { stream.synchronize() }.expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            graph.launch().sync_on(stream).expect("graph replay");
            spin_us(w_us);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        throughput.push(iters as f64 / elapsed);
    }
    throughput
}

fn bench_async(
    stream: &Arc<Stream>,
    graph: &CudaGraph<Vec<Part>>,
    w_us: u64,
    n_warmup: u32,
    iters: u32,
    n_samples: usize,
) -> Vec<f64> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // tokio::join! polls branches in the order given. We poll the GPU
    // future first so it submits the graph launch and registers the
    // completion callback, then the CPU future starts the spin on the
    // same thread. When the spin ends, the GPU future is re-polled; if
    // the callback already fired, it returns immediately. Overall wall
    // time per iter ≈ max(GPU_total, W + submit_time).
    let stream_clone = stream.clone();

    rt.block_on(async move {
        macro_rules! one_iter {
            () => {{
                let ctx = ExecutionContext::new(stream_clone.clone());
                let fut = DeviceFuture::scheduled(graph.launch(), ctx);
                let (gpu_res, _) = tokio::join!(fut, async {
                    spin_us(w_us);
                });
                gpu_res.expect("graph replay");
            }};
        }

        for _ in 0..n_warmup {
            one_iter!();
        }
        unsafe { stream_clone.synchronize() }.expect("sync");

        let mut throughput = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            unsafe { stream_clone.synchronize() }.expect("sync");
            let t0 = Instant::now();
            for _ in 0..iters {
                one_iter!();
            }
            let elapsed = t0.elapsed().as_secs_f64();
            throughput.push(iters as f64 / elapsed);
        }
        throughput
    })
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn usage() -> ! {
    eprintln!(
        "usage: async_throughput --mode <sync|async> [--w <us>] [--n <ops>] \
         [--iters <n>] [--warmup <n>] [--samples <n>] [--csv <path>]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mode = arg_value(&args, "--mode")
        .and_then(Mode::parse)
        .unwrap_or_else(|| usage());
    let w_us: u64 = arg_value(&args, "--w")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let n: usize = arg_value(&args, "--n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N);
    let n_warmup: u32 = arg_value(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_WARMUP);
    let iters: u32 = arg_value(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    let n_samples: usize = arg_value(&args, "--samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_SAMPLES);
    let csv_path = arg_value(&args, "--csv")
        .unwrap_or("results_part_b.csv")
        .to_string();

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");

    let d = D as usize;
    let generics = vec!["f16".to_string(), D.to_string()];
    let (x, g) = alloc_inputs(&stream, d);

    // Capture once at setup; hot loop replays only.
    let parts = fresh_partitions(&stream, d, n);
    let chain = build_chain(parts, &x, &g, &generics);
    let graph = CudaGraph::capture(stream.clone(), chain).expect("graph capture");

    let mut samples = match mode {
        Mode::Sync => bench_sync(&stream, &graph, w_us, n_warmup, iters, n_samples),
        Mode::Async => bench_async(&stream, &graph, w_us, n_warmup, iters, n_samples),
    };

    let (min_tp, p25_tp, med_tp, p75_tp, max_tp) = summarize(&mut samples);
    println!(
        "mode={} n={} w_us={} samples={} tp(iter/s) min={:.1} p25={:.1} med={:.1} p75={:.1} max={:.1}",
        mode.label(), n, w_us, samples.len(),
        min_tp, p25_tp, med_tp, p75_tp, max_tp,
    );

    let need_header = !std::path::Path::new(&csv_path).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)?;
    if need_header {
        writeln!(
            f,
            "mode,n_ops,w_us,tp_median,tp_min,tp_p25,tp_p75,tp_max,n_samples"
        )?;
    }
    writeln!(
        f,
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        mode.label(),
        n,
        w_us,
        med_tp,
        min_tp,
        p25_tp,
        p75_tp,
        max_tp,
        samples.len()
    )?;

    Ok(())
}

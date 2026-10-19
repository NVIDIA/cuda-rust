/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// §5.2 Part B2 — GPU work-distribution benchmark (bimodal GEMMs).
//
// Three modes demonstrating what the async primitive buys for
// heterogeneous GPU work distribution:
//
//   A. serial    — one host thread, one stream. Drains a bimodal-size
//                  GEMM queue strictly in order.
//   B. threaded  — S host threads, one stream each. Each thread pulls
//                  work units from a shared Mutex<VecDeque> queue and
//                  issues them via .sync_on() on its own stream.
//   C. async     — one host thread (tokio current-thread), S tasks, one
//                  stream each. Each task pulls from a (thread-local)
//                  RefCell<VecDeque> queue and issues its kernel via
//                  DeviceFuture::scheduled(op, ExecutionContext(stream))
//                  bound to its own stream.
//
// Workload: a sequence of GEMM work units, each Small (M=N=K=256) or
// Large (M=N=K=2048), with a configurable mix ratio. Each worker has
// pre-allocated buffers sized for both classes so the measurement is
// pure dispatch + kernel time, no allocation.
//
// Metric: total work units / second to drain the queue. Run K times
// per config and report median.
//
// CLI:
//   --mode <serial|threaded|async>   required
//   --streams <S>                    default 4   (ignored for serial)
//   --n-work <N>                     default 2000
//   --small-ratio <r>                default 0.8
//   --samples <K>                    default 5
//   --warmup                         run one warmup pass before timing
//   --csv <path>                     default results_bimodal.csv

use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_core::sys::CUctx_flags_enum_CU_CTX_SCHED_SPIN;
use cuda_core::{sys, Device, IntoResult, Stream};
use cutile::api;
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use kernels::*;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Unchecked GEMM: partitioned output Z over tile block (BM, BN),
    // K-dimension runtime-valued (x and y have dynamic shape). One kernel
    // shape handles all GEMMs by passing the concrete k per launch.
    #[cutile::entry(unchecked_accesses = true,
                    optimization_hints = (
                        sm_120 = (num_cta_in_cga = 2,),
                    )
    )]
    pub unsafe fn gemm<T: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        z: &mut Tensor<T, { [BM, BN] }>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
        k: i32,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        for i in 0i32..(k / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

// ---------- workload & buffer shape ----------

const DEFAULT_SMALL_M: i32 = 256; // default small GEMM dimension (M=N=K)
const DEFAULT_LARGE_M: i32 = 2048; // default large GEMM dimension (M=N=K)
const BM: i32 = 128; // shared tile block M and N
const BN: i32 = 128;
const BK: i32 = 64;

#[derive(Clone, Copy, Debug)]
enum Size {
    Small,
    Large,
}

/// Bimodal-workload size configuration: the actual M=N=K dimensions
/// for the small and large work units. Both must be a multiple of BM.
#[derive(Clone, Copy, Debug)]
struct SizeSpec {
    small_m: i32,
    large_m: i32,
}

impl SizeSpec {
    fn m(&self, size: Size) -> i32 {
        match size {
            Size::Small => self.small_m,
            Size::Large => self.large_m,
        }
    }
}

type TensorF16 = Arc<Tensor<f16>>;
type Part = Partition<Tensor<f16>>;

/// Per-size, per-worker GEMM buffers. x and y are shared Arc<Tensor>;
/// z rotates in and out of the worker's ownership on each launch via
/// `Option<Partition>` (take + replace).
struct GemmBufs {
    x: TensorF16,
    y: TensorF16,
    z: Option<Part>,
    generics: Vec<String>,
    k: i32,
}

/// Busy-spin for `w` microseconds on the calling thread. Used by
/// serial and threaded modes to model per-task host-side compute
/// that runs serially with each GEMM.
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

/// Async-friendly "wait" for `w` microseconds: loops `yield_now`
/// against a wall-clock deadline. Used by async mode so tasks
/// release the executor at yield points and multiple per-task
/// waits overlap on the same tokio thread. We cannot use
/// `tokio::time::sleep` here because tokio's timer wheel has 1 ms
/// resolution, so `sleep(10us)` actually takes ~1 ms — that would
/// swamp our measurement at microsecond-scale W.
#[inline(never)]
async fn async_wait_us(w: u64) {
    if w == 0 {
        return;
    }
    let target = Duration::from_micros(w);
    let start = Instant::now();
    while start.elapsed() < target {
        tokio::task::yield_now().await;
    }
}

fn alloc_bufs(stream: &Arc<Stream>, m_dim: i32) -> GemmBufs {
    let m = m_dim as usize;
    let x: TensorF16 = api::ones::<f16>(&[m, m])
        .sync_on(stream)
        .expect("alloc x")
        .into();
    let y: TensorF16 = api::ones::<f16>(&[m, m])
        .sync_on(stream)
        .expect("alloc y")
        .into();
    let z = api::zeros::<f16>(&[m, m])
        .sync_on(stream)
        .expect("alloc z")
        .partition([BM as usize, BN as usize]);
    GemmBufs {
        x,
        y,
        z: Some(z),
        generics: vec![
            "f16".to_string(),
            BM.to_string(),
            BN.to_string(),
            BK.to_string(),
        ],
        k: m_dim,
    }
}

/// Worker-local state: one stream + buffers for both workload sizes.
struct Worker {
    stream: Arc<Stream>,
    small: GemmBufs,
    large: GemmBufs,
}

impl Worker {
    fn new(stream: Arc<Stream>, spec: SizeSpec) -> Self {
        let small = alloc_bufs(&stream, spec.m(Size::Small));
        let large = alloc_bufs(&stream, spec.m(Size::Large));
        Self {
            stream,
            small,
            large,
        }
    }
    fn bufs_mut(&mut self, size: Size) -> &mut GemmBufs {
        match size {
            Size::Small => &mut self.small,
            Size::Large => &mut self.large,
        }
    }
}

/// Deterministic bimodal workload: interleaves Small and Large so a
/// random pattern isn't timing-sensitive. `small_ratio` controls the
/// fraction of Small units.
fn make_workload(n: usize, small_ratio: f64) -> Vec<Size> {
    let n_small = (n as f64 * small_ratio).round() as usize;
    let n_large = n - n_small;
    let mut out = Vec::with_capacity(n);
    // Interleave via a simple ratio counter so the mix is evenly spread
    // rather than bursty.
    let (mut small_left, mut large_left) = (n_small, n_large);
    let (mut s_num, mut l_num) = (0u64, 0u64);
    while small_left > 0 || large_left > 0 {
        if (small_left > 0 && s_num * n_large as u64 <= l_num * n_small as u64) || large_left == 0 {
            out.push(Size::Small);
            small_left -= 1;
            s_num += 1;
        } else {
            out.push(Size::Large);
            large_left -= 1;
            l_num += 1;
        }
    }
    out
}

// ---------- (A) serial: 1 thread, 1 stream ----------

fn run_serial_samples(
    workload: &[Size],
    device: &Arc<Device>,
    samples: usize,
    warmup: bool,
    spec: SizeSpec,
    w_host_us: u64,
) -> Vec<f64> {
    let stream = device.new_stream().expect("stream");
    let mut worker = Worker::new(stream.clone(), spec);
    unsafe { stream.synchronize() }.expect("sync");

    let run_once = |worker: &mut Worker| -> f64 {
        let t0 = Instant::now();
        for &size in workload {
            let stream = worker.stream.clone();
            let buf = worker.bufs_mut(size);
            let z_part = buf.z.take().unwrap();
            let (z_back, _, _, _) = unsafe {
                gemm(z_part, buf.x.clone(), buf.y.clone(), buf.k)
                    .generics(buf.generics.clone())
                    .sync_on(&stream)
                    .expect("gemm")
            };
            buf.z = Some(z_back);
            spin_us(w_host_us);
        }
        workload.len() as f64 / t0.elapsed().as_secs_f64()
    };

    if warmup {
        run_once(&mut worker);
    }
    (0..samples).map(|_| run_once(&mut worker)).collect()
}

// ---------- (B) threaded: S persistent threads, one stream each ----------
//
// We spawn the S worker threads ONCE and keep them alive across samples.
// Without this, each sample would pay the per-thread kernel-JIT cost
// (`DEVICE_CONTEXTS` is `thread_local!`, so every fresh thread starts
// with an empty kernel cache). With persistent threads, each worker
// JITs on its first launch and amortizes across all subsequent samples.
//
// Workers receive batch commands over per-thread mpsc channels and
// signal completion on a shared done channel.

enum Cmd {
    Run(Arc<Mutex<VecDeque<Size>>>),
    Quit,
}

fn run_threaded_samples(
    workload: &[Size],
    device: &Arc<Device>,
    stream_count: usize,
    samples: usize,
    warmup: bool,
    spec: SizeSpec,
    w_host_us: u64,
) -> Vec<f64> {
    use std::sync::mpsc;
    let streams: Vec<Arc<Stream>> = (0..stream_count)
        .map(|_| device.new_stream().expect("stream"))
        .collect();
    let workers: Vec<Worker> = streams.into_iter().map(|s| Worker::new(s, spec)).collect();
    for w in &workers {
        unsafe { w.stream.synchronize() }.expect("sync");
    }

    let (done_tx, done_rx) = mpsc::channel::<u64>();
    let mut cmd_senders: Vec<mpsc::Sender<Cmd>> = Vec::with_capacity(stream_count);
    let mut handles = Vec::with_capacity(stream_count);

    for mut worker in workers {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        cmd_senders.push(cmd_tx);
        let done_tx = done_tx.clone();
        let h = std::thread::spawn(move || {
            worker.stream.device().bind_to_thread().expect("bind");
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    Cmd::Quit => break,
                    Cmd::Run(queue) => {
                        let mut count: u64 = 0;
                        loop {
                            let size = match queue.lock().unwrap().pop_front() {
                                Some(s) => s,
                                None => break,
                            };
                            let stream = worker.stream.clone();
                            let buf = worker.bufs_mut(size);
                            let z_part = buf.z.take().unwrap();
                            let (z_back, _, _, _) = unsafe {
                                gemm(z_part, buf.x.clone(), buf.y.clone(), buf.k)
                                    .generics(buf.generics.clone())
                                    .sync_on(&stream)
                                    .expect("gemm")
                            };
                            buf.z = Some(z_back);
                            spin_us(w_host_us);
                            count += 1;
                        }
                        done_tx.send(count).unwrap();
                    }
                }
            }
        });
        handles.push(h);
    }

    let fire = |queue: Arc<Mutex<VecDeque<Size>>>| -> u64 {
        for tx in &cmd_senders {
            tx.send(Cmd::Run(queue.clone())).unwrap();
        }
        let mut total = 0u64;
        for _ in 0..stream_count {
            total += done_rx.recv().unwrap();
        }
        total
    };

    if warmup {
        let q = Arc::new(Mutex::new(workload.iter().copied().collect()));
        let _ = fire(q);
    }

    let mut throughputs = Vec::with_capacity(samples);
    for _ in 0..samples {
        let q: Arc<Mutex<VecDeque<Size>>> =
            Arc::new(Mutex::new(workload.iter().copied().collect()));
        let t0 = Instant::now();
        let total = fire(q);
        let elapsed = t0.elapsed().as_secs_f64();
        throughputs.push(total as f64 / elapsed);
    }

    for tx in &cmd_senders {
        tx.send(Cmd::Quit).unwrap();
    }
    for h in handles {
        h.join().unwrap();
    }
    throughputs
}

// ---------- (C) async: T tokio threads, S persistent tasks, one stream each ----------

fn run_async_samples(
    workload: &[Size],
    device: &Arc<Device>,
    stream_count: usize,
    samples: usize,
    warmup: bool,
    tokio_threads: usize,
    spec: SizeSpec,
    w_host_us: u64,
) -> Vec<f64> {
    // Tokio runtime is `current_thread` (T=1) or `multi_thread` (T>1).
    // Multi-thread uses Send futures and work-stealing; it distributes
    // tokio's poll/wake work across T OS threads. Tasks still submit
    // to their own CUDA streams, so the concurrency story is unchanged
    // --- only the executor's own thread footprint grows.
    let rt = if tokio_threads <= 1 {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(tokio_threads)
            .enable_all()
            .build()
            .expect("tokio runtime")
    };

    let workload_vec = workload.to_vec();
    let n = workload.len();
    let device_outer = device.clone();

    rt.block_on(async move {
        use tokio::sync::mpsc;
        let mut cmd_senders: Vec<mpsc::UnboundedSender<Option<Arc<Mutex<VecDeque<Size>>>>>> =
            Vec::with_capacity(stream_count);
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<u64>();
        let mut handles = Vec::with_capacity(stream_count);

        for _ in 0..stream_count {
            let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
            cmd_senders.push(cmd_tx);
            let done_tx = done_tx.clone();
            let device = device_outer.clone();
            let h = tokio::spawn(async move {
                let stream = device.new_stream().expect("stream");
                let mut worker = Worker::new(stream, spec);
                unsafe { worker.stream.synchronize() }.expect("sync");
                while let Some(maybe_queue) = cmd_rx.recv().await {
                    let queue = match maybe_queue {
                        Some(q) => q,
                        None => break, // terminate
                    };
                    let mut count: u64 = 0;
                    loop {
                        let size = match queue.lock().unwrap().pop_front() {
                            Some(s) => s,
                            None => break,
                        };
                        let stream = worker.stream.clone();
                        let buf = worker.bufs_mut(size);
                        let z_part = buf.z.take().unwrap();
                        let op = unsafe {
                            gemm(z_part, buf.x.clone(), buf.y.clone(), buf.k)
                                .generics(buf.generics.clone())
                        };
                        let ctx = ExecutionContext::new(stream);
                        let fut = DeviceFuture::scheduled(op, ctx);
                        let (z_back, _, _, _) = fut.await.expect("gemm");
                        buf.z = Some(z_back);
                        async_wait_us(w_host_us).await;
                        count += 1;
                    }
                    done_tx.send(count).unwrap();
                }
            });
            handles.push(h);
        }

        macro_rules! fire {
            ($queue:expr) => {{
                for tx in &cmd_senders {
                    tx.send(Some($queue.clone())).unwrap();
                }
                let mut total = 0u64;
                for _ in 0..stream_count {
                    total += done_rx.recv().await.unwrap();
                }
                total
            }};
        }

        if warmup {
            let q: Arc<Mutex<VecDeque<Size>>> =
                Arc::new(Mutex::new(workload_vec.iter().copied().collect()));
            let _ = fire!(q);
        }

        let mut throughputs = Vec::with_capacity(samples);
        for _ in 0..samples {
            let q: Arc<Mutex<VecDeque<Size>>> =
                Arc::new(Mutex::new(workload_vec.iter().copied().collect()));
            let t0 = Instant::now();
            let total = fire!(q);
            let elapsed = t0.elapsed().as_secs_f64();
            throughputs.push(total as f64 / elapsed);
            assert_eq!(total as usize, n);
        }

        for tx in &cmd_senders {
            tx.send(None).unwrap();
        }
        for h in handles {
            h.await.unwrap();
        }
        throughputs
    })
}

// ---------- driver ----------

#[derive(Clone, Copy)]
enum Mode {
    Serial,
    Threaded,
    Async,
}

impl Mode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "serial" => Some(Mode::Serial),
            "threaded" => Some(Mode::Threaded),
            "async" => Some(Mode::Async),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Serial => "serial",
            Mode::Threaded => "threaded",
            Mode::Async => "async",
        }
    }
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn usage() -> ! {
    eprintln!(
        "usage: bimodal_gemms --mode <serial|threaded|async> \
         [--streams <S>] [--n-work <N>] [--small-ratio <r>] \
         [--samples <K>] [--warmup] [--csv <path>] \
         [--tokio-threads <T>]"
    );
    std::process::exit(2);
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mode = arg_value(&args, "--mode")
        .and_then(Mode::parse)
        .unwrap_or_else(|| usage());
    let stream_count: usize = arg_value(&args, "--streams")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_work: usize = arg_value(&args, "--n-work")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let small_ratio: f64 = arg_value(&args, "--small-ratio")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8);
    let samples: usize = arg_value(&args, "--samples")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let warmup = args.iter().any(|a| a == "--warmup");
    let csv_path = arg_value(&args, "--csv")
        .unwrap_or("results_bimodal.csv")
        .to_string();
    // T=0 means "don't care / not applicable"; we log T=1 for serial and
    // threaded (they're fixed by construction) and the actual T for async.
    let tokio_threads: usize = arg_value(&args, "--tokio-threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let small_m: i32 = arg_value(&args, "--small-m")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SMALL_M);
    let large_m: i32 = arg_value(&args, "--large-m")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LARGE_M);
    // Per-task host-side busy-spin in microseconds, run after every
    // GEMM completion. Default 0 = pure GPU work (current behavior).
    let w_host_us: u64 = arg_value(&args, "--w-host-us")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if small_m % BM != 0 || large_m % BM != 0 {
        eprintln!(
            "error: --small-m and --large-m must both be multiples of BM={}",
            BM
        );
        std::process::exit(2);
    }
    let spec = SizeSpec { small_m, large_m };

    let device = Device::new(0).expect("device");

    // Pin the context to CU_CTX_SCHED_SPIN for a fair cross-mode
    // comparison. The CUDA default CU_CTX_SCHED_AUTO adapts sync
    // behavior based on thread count (spin vs yield), which made the
    // threaded mode's S=2 point nondeterministic. Fixing the policy
    // removes that confound — all three modes use the same sync
    // semantics.
    unsafe { sys::cuCtxSetFlags(CUctx_flags_enum_CU_CTX_SCHED_SPIN).result() }
        .expect("sched flags");

    let workload = make_workload(n_work, small_ratio);

    let throughputs: Vec<f64> = match mode {
        Mode::Serial => run_serial_samples(&workload, &device, samples, warmup, spec, w_host_us),
        Mode::Threaded => run_threaded_samples(
            &workload,
            &device,
            stream_count,
            samples,
            warmup,
            spec,
            w_host_us,
        ),
        Mode::Async => run_async_samples(
            &workload,
            &device,
            stream_count,
            samples,
            warmup,
            tokio_threads,
            spec,
            w_host_us,
        ),
    };
    let min_tp = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tp = throughputs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let med_tp = median(throughputs.clone());

    let effective_streams = match mode {
        Mode::Serial => 1,
        _ => stream_count,
    };
    // effective_tokio: number of OS threads that async mode actually
    // uses for the tokio runtime. For serial and threaded, this is 1
    // (there is no tokio). For async T=1 we use current_thread (1 OS
    // thread); for async T>=2 we use multi_thread with that many
    // worker threads.
    let effective_tokio: usize = match mode {
        Mode::Async => tokio_threads.max(1),
        _ => 1,
    };

    println!(
        "mode={} streams={} tokio_threads={} small_m={} large_m={} w_host_us={} \
         n_work={} small_ratio={:.2} samples={} \
         tp(wu/s) min={:.1} med={:.1} max={:.1}",
        mode.label(),
        effective_streams,
        effective_tokio,
        spec.small_m,
        spec.large_m,
        w_host_us,
        n_work,
        small_ratio,
        samples,
        min_tp,
        med_tp,
        max_tp,
    );

    let need_header = !std::path::Path::new(&csv_path).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)?;
    if need_header {
        writeln!(f, "mode,streams,tokio_threads,small_m,large_m,w_host_us,n_work,small_ratio,tp_median,tp_min,tp_max,n_samples")?;
    }
    writeln!(
        f,
        "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{}",
        mode.label(),
        effective_streams,
        effective_tokio,
        spec.small_m,
        spec.large_m,
        w_host_us,
        n_work,
        small_ratio,
        med_tp,
        min_tp,
        max_tp,
        samples
    )?;

    Ok(())
}

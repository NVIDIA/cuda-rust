/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Element-wise add benchmark for cuTile Rust -- paper §5.1 (companion
// to rust/gemm/). Measures z = x + y for f16 tensors across three
// bounds-check configurations:
//   optimized  unchecked_accesses=true, no checks in IR
//   safe       dynamic-shape runtime checks emitted
//   static     const-generic shapes, checks elided at compile time
//
// Modes (CLI):
//   (default)  Isolated runtime bench: for each N in the sweep,
//              measure the selected variant(s) with their own allocations,
//              warmup, cache-rotation chunks, and fixed timed windows. Output:
//              elemwise_rust_runtime_results.csv (variant column).
//              Use --variant optimized|safe|static|all to restrict the run.
//   --jit      Per-unique-shape JIT cost bench: for each variant × N,
//              measure wall time of the first async_on call on a fresh
//              monomorphization (includes the full compile pipeline
//              and module load; launch+kernel time is sub-µs next to
//              ~100 ms JIT and rounds to zero). For static every N is
//              a fresh monomorphization; for optimized/safe only the
//              first N triggers JIT. Output:
//              elemwise_rust_jit_results.csv.
//   --dump-ir  Compile-only, dump MLIR. Mutually exclusive with other
//              modes; combine with --safe or --static to pick which
//              variant's IR to dump.
//
// Runtime methodology:
//   - CUDA driver events for GPU-side timing.
//     Removes host launch/sync jitter that dominated the earlier
//     wall-clock measurement at small N.
//   - Timed windows rotate across multiple independent x/y/z chunks
//     when N is small. This keeps the aggregate footprint above the
//     GPU cache so logical GB/s reflects DRAM movement instead of
//     repeatedly measuring cache-hot buffers.
//   - N_WARMUP = 20 launches per (variant, N) pair before timing.
//   - N_SAMPLES = 25 timed windows, each with fixed ITERS launches;
//     median per-launch time reported.
//
// Lock clocks for paper-final runs: `sudo nvidia-smi -lgc 2400,2400`.

use cuda_async::device_operation::DeviceOp;
use cuda_core::{sys, Device, IntoResult, Stream};
use cutile::api;
use cutile::core::f16;
use cutile::tile_kernel::{CompileOptions, PartitionOp, TileKernel};
use kernels::*;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // Unchecked: dynamic input shapes with unchecked_accesses. Checks
    // are NOT emitted in the IR. Requires `unsafe fn`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn add<T: ElementType, const B: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true, unchecked_accesses = true)]
    unsafe fn add_ir<T: ElementType, const B: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    // Safe: dynamic input shapes, runtime bounds checks at each
    // partition load. No `unsafe`.
    #[cutile::entry()]
    fn add_safe<T: ElementType, const B: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true)]
    fn add_safe_ir<T: ElementType, const B: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    // Static: N as const generic. Front-end proves divisibility and
    // elides bounds checks at compile time; no `unsafe` needed.
    #[cutile::entry()]
    fn add_static<T: ElementType, const B: i32, const N: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [N] }>,
        y: &Tensor<T, { [N] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(print_ir = true)]
    fn add_static_ir<T: ElementType, const B: i32, const N: i32>(
        z: &mut Tensor<T, { [B] }>,
        x: &Tensor<T, { [N] }>,
        y: &Tensor<T, { [N] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

const N_WARMUP: u32 = 20;
const N_SAMPLES: usize = 25;
const DEFAULT_ITERS: u32 = 10;
const DEFAULT_CACHE_SWEEP_MIB: usize = 1024;
const MAX_CACHE_CHUNKS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Variant {
    Add,
    AddSafe,
    AddStatic,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::Add => "optimized",
            Variant::AddSafe => "safe",
            Variant::AddStatic => "static",
        }
    }
    fn all() -> &'static [Variant] {
        &[Variant::Add, Variant::AddSafe, Variant::AddStatic]
    }
}

#[derive(Clone)]
struct Sample {
    config: &'static str,
    n: usize,
    b: i32,
    cache_chunks: usize,
    iters: u32,
    per_launch_us_median: f64,
    per_launch_us_min: f64,
    per_launch_us_max: f64,
    per_launch_us_p25: f64,
    per_launch_us_p75: f64,
    per_launch_us_stdev: f64,
    gb_per_s: f64,
}

#[derive(Clone)]
struct JitSample {
    config: &'static str,
    n: usize,
    b: i32,
    total_ms: f64,
    cached: bool,
}

#[derive(Clone, Copy, Default)]
struct HintConfig {
    num_cta_in_cga: Option<i32>,
    occupancy: Option<i32>,
    max_divisibility: Option<i32>,
}

impl HintConfig {
    fn from_args(args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            num_cta_in_cga: parse_optional_i32_arg(args, "--cta")?,
            occupancy: parse_optional_i32_arg(args, "--occupancy")?,
            max_divisibility: parse_optional_i32_arg(args, "--max-divisibility")?,
        })
    }

    fn compile_options(self) -> CompileOptions {
        let mut options = CompileOptions::default();
        if let Some(v) = self.num_cta_in_cga {
            options = options.num_cta_in_cga(v);
        }
        if let Some(v) = self.occupancy {
            options = options.occupancy(v);
        }
        if let Some(v) = self.max_divisibility {
            options = options.max_divisibility(v);
        }
        options
    }

    fn num_cta_label(self) -> String {
        option_label(self.num_cta_in_cga)
    }

    fn occupancy_label(self) -> String {
        option_label(self.occupancy)
    }

    fn max_divisibility_label(self) -> String {
        option_label(self.max_divisibility)
    }
}

fn option_label(value: Option<i32>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "default".to_string())
}

fn cli_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn parse_arg<T>(args: &[String], flag: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match cli_value(args, flag) {
        Some(value) => value.parse::<T>().map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid value for {flag}: {value} ({err})"),
            )
            .into()
        }),
        None => Ok(default),
    }
}

fn parse_optional_i32_arg(
    args: &[String],
    flag: &str,
) -> Result<Option<i32>, Box<dyn std::error::Error>> {
    match cli_value(args, flag) {
        Some(value) => Ok(Some(value.parse::<i32>().map_err(|err| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid value for {flag}: {value} ({err})"),
            )
        })?)),
        None => Ok(None),
    }
}

fn selected_tune_variants(
    args: &[String],
    is_safe: bool,
    is_static: bool,
) -> Result<Vec<Variant>, Box<dyn std::error::Error>> {
    if let Some(value) = cli_value(args, "--variant") {
        return match value {
            "optimized" | "add" => Ok(vec![Variant::Add]),
            "safe" => Ok(vec![Variant::AddSafe]),
            "static" => Ok(vec![Variant::AddStatic]),
            "all" => Ok(Variant::all().to_vec()),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown --variant value: {value}"),
            )
            .into()),
        };
    }
    if is_safe {
        Ok(vec![Variant::AddSafe])
    } else if is_static {
        Ok(vec![Variant::AddStatic])
    } else {
        Ok(vec![Variant::Add])
    }
}

fn validate_elemwise_config(n: usize, b: i32) -> Result<(), Box<dyn std::error::Error>> {
    if b <= 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "tile size B must be positive",
        )
        .into());
    }
    if n % b as usize != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("N={n} must be divisible by B={b}"),
        )
        .into());
    }
    Ok(())
}

fn cache_chunks_for_n(n: usize, cache_sweep_mib: usize) -> usize {
    if cache_sweep_mib == 0 {
        return 1;
    }
    let target_bytes = cache_sweep_mib as f64 * 1024.0 * 1024.0;
    let logical_bytes_per_launch = 3.0 * n as f64 * 2.0;
    ((target_bytes / logical_bytes_per_launch).ceil() as usize)
        .max(1)
        .min(MAX_CACHE_CHUNKS)
}

fn open_append_csv(path: &str, header: &str) -> Result<File, Box<dyn std::error::Error>> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let needs_header = std::fs::metadata(path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if needs_header {
        writeln!(file, "{header}")?;
    }
    Ok(file)
}

fn output_path(file_name: &str) -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("RESULTS_DIR") {
        let path = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&path).expect("create RESULTS_DIR");
        path.join(file_name)
    } else {
        std::path::PathBuf::from(file_name)
    }
}

fn sync_stream(stream: &Arc<Stream>, context: &str) {
    stream
        .device()
        .bind_to_thread()
        .expect("bind stream device");
    unsafe { stream.synchronize() }.expect(context);
}

struct TimingEvent(sys::CUevent);

impl TimingEvent {
    fn new(stream: &Arc<Stream>) -> Self {
        stream
            .device()
            .bind_to_thread()
            .expect("bind stream device");
        let mut event = std::mem::MaybeUninit::uninit();
        unsafe {
            sys::cuEventCreate(
                event.as_mut_ptr(),
                sys::CUevent_flags_enum_CU_EVENT_DEFAULT as u32,
            )
            .result()
            .expect("create event");
            Self(event.assume_init())
        }
    }

    fn record(&self, stream: &Arc<Stream>) {
        stream
            .device()
            .bind_to_thread()
            .expect("bind stream device");
        unsafe { sys::cuEventRecord(self.0, stream.cu_stream()) }
            .result()
            .expect("record event");
    }

    fn synchronize(&self, stream: &Arc<Stream>) {
        stream
            .device()
            .bind_to_thread()
            .expect("bind stream device");
        unsafe { sys::cuEventSynchronize(self.0) }
            .result()
            .expect("sync event");
    }

    fn elapsed_us(&self, end: &Self, stream: &Arc<Stream>) -> f64 {
        stream
            .device()
            .bind_to_thread()
            .expect("bind stream device");
        let mut ms: f32 = 0.0;
        unsafe { sys::cuEventElapsedTime_v2((&mut ms) as *mut _, self.0, end.0) }
            .result()
            .expect("elapsed event");
        ms as f64 * 1000.0
    }
}

impl Drop for TimingEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuEventDestroy_v2(self.0);
        }
    }
}

// Single launch of one variant. Consumes the partition, returns the
// recovered one. x/y are shared across variants at a given N, passed
// by &Tensor to avoid per-launch Arc refcount bumps in the hot path.
macro_rules! launch_one {
    ($variant:expr, $stream:expr, $z:expr, $x:expr, $y:expr, $generics:expr, $grid:expr, $compile_options:expr) => {{
        unsafe {
            match $variant {
                Variant::Add => {
                    let (z, ..) = add($z, $x, $y)
                        .generics($generics.clone())
                        .compile_options($compile_options.clone())
                        .async_on($stream)
                        .expect("launch add");
                    z
                }
                Variant::AddSafe => {
                    let (z, ..) = add_safe($z, $x, $y)
                        .generics($generics.clone())
                        .compile_options($compile_options.clone())
                        .async_on($stream)
                        .expect("launch add_safe");
                    z
                }
                Variant::AddStatic => {
                    let (z, ..) = add_static($z, $x, $y)
                        .const_grid($grid.expect("grid for static"))
                        .generics($generics.clone())
                        .compile_options($compile_options.clone())
                        .async_on($stream)
                        .expect("launch add_static");
                    z
                }
            }
        }
    }};
}

/// Returns elapsed microseconds between two timing-enabled stream events.
fn event_time_us(stream: &Arc<Stream>, f: impl FnOnce()) -> f64 {
    let start = TimingEvent::new(stream);
    let end = TimingEvent::new(stream);
    start.record(stream);
    f();
    end.record(stream);
    end.synchronize(stream);
    start.elapsed_us(&end, stream)
}

fn summarize_sample(
    variant: Variant,
    n: usize,
    b: i32,
    cache_chunks: usize,
    iters: u32,
    mut per_launch_us: Vec<f64>,
) -> Sample {
    per_launch_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_launch_us[per_launch_us.len() / 2];
    let min = per_launch_us[0];
    let max = per_launch_us[per_launch_us.len() - 1];
    // Percentile helper: nearest-rank method, clamped to valid indices.
    let pct = |q: f64| -> f64 {
        let idx = ((per_launch_us.len() as f64 - 1.0) * q).round() as usize;
        per_launch_us[idx.min(per_launch_us.len() - 1)]
    };
    let p25 = pct(0.25);
    let p75 = pct(0.75);
    let mean = per_launch_us.iter().sum::<f64>() / per_launch_us.len() as f64;
    let var = per_launch_us
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / (per_launch_us.len() as f64 - 1.0).max(1.0);
    let stdev = var.sqrt();
    // Elementwise add: read x + read y + write z = 3 × N × 2 B (f16).
    let bytes = 3.0 * n as f64 * 2.0;
    let gb_per_s = bytes / (median / 1e6) / 1e9;
    println!(
        "  N={:>10} {:>10}: iters={:>4} {:>7.2} GB/s (median {:>7.2} us, min {:>7.2} / max {:>7.2} / σ {:>6.2} us)",
        n,
        variant.label(),
        iters,
        gb_per_s,
        median,
        min,
        max,
        stdev,
    );
    Sample {
        config: variant.label(),
        n,
        b,
        cache_chunks,
        iters,
        per_launch_us_median: median,
        per_launch_us_min: min,
        per_launch_us_max: max,
        per_launch_us_p25: p25,
        per_launch_us_p75: p75,
        per_launch_us_stdev: stdev,
        gb_per_s,
    }
}

/// Bench one variant at a single N as an isolated experiment.
///
/// Cache avoidance comes from rotating through independent x/y/z chunks
/// inside this variant's timed windows, not from running different
/// variants back-to-back. This keeps fixed variant order out of the
/// memory-cache methodology.
fn bench_variant_n(
    stream: &Arc<Stream>,
    n: usize,
    b: i32,
    variant: Variant,
    compile_options: CompileOptions,
    cache_sweep_mib: usize,
    iters: u32,
) -> Sample {
    let generics_dyn = vec!["f16".to_string(), b.to_string()];
    let generics_static = vec!["f16".to_string(), b.to_string(), n.to_string()];
    let generics = if variant == Variant::AddStatic {
        &generics_static
    } else {
        &generics_dyn
    };
    let cache_chunks = cache_chunks_for_n(n, cache_sweep_mib);
    let logical_mib = cache_chunks as f64 * 3.0 * n as f64 * 2.0 / (1024.0 * 1024.0);

    println!(
        "--- N={} {} cache_chunks={} logical_window_footprint={:.1} MiB ---",
        n,
        variant.label(),
        cache_chunks,
        logical_mib
    );

    let x_by_chunk: Vec<_> = (0..cache_chunks)
        .map(|_| api::ones::<f16>(&[n]).sync_on(stream).expect("alloc x"))
        .collect();
    let y_by_chunk: Vec<_> = (0..cache_chunks)
        .map(|_| api::ones::<f16>(&[n]).sync_on(stream).expect("alloc y"))
        .collect();

    // One z partition per cache-rotation chunk, held in Option so we
    // can take() ownership in and out of the slot without fake
    // allocations in the hot timing path.
    let mut z_chunks: Vec<Option<_>> = (0..cache_chunks)
        .map(|_| {
            Some(
                api::zeros::<f16>(&[n])
                    .partition([b as usize])
                    .sync_on(stream)
                    .expect("alloc z"),
            )
        })
        .collect();
    let grid = z_chunks[0].as_ref().unwrap().grid().expect("grid");
    let mut launch_counter = 0usize;

    // Warmup this variant before any timed sample.
    for _ in 0..N_WARMUP {
        let chunk = launch_counter % cache_chunks;
        launch_counter += 1;
        let z = z_chunks[chunk].take().unwrap();
        z_chunks[chunk] = Some(launch_one!(
            variant,
            stream,
            z,
            &x_by_chunk[chunk],
            &y_by_chunk[chunk],
            generics,
            Some(grid),
            compile_options
        ));
    }
    sync_stream(stream, "sync after warmup");

    // Time N_SAMPLES windows for this variant only. The chunk counter
    // continues across samples instead of restarting at chunk zero.
    let mut per_launch_us: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        sync_stream(stream, "sync before sample");
        let us = event_time_us(stream, || {
            for _ in 0..iters {
                let chunk = launch_counter % cache_chunks;
                launch_counter += 1;
                let z = z_chunks[chunk].take().unwrap();
                z_chunks[chunk] = Some(launch_one!(
                    variant,
                    stream,
                    z,
                    &x_by_chunk[chunk],
                    &y_by_chunk[chunk],
                    generics,
                    Some(grid),
                    compile_options
                ));
            }
        });
        per_launch_us.push(us / iters as f64);
    }

    summarize_sample(variant, n, b, cache_chunks, iters, per_launch_us)
}

fn bench_n(
    stream: &Arc<Stream>,
    n: usize,
    b: i32,
    variants: &[Variant],
    compile_options: CompileOptions,
    cache_sweep_mib: usize,
    iters: u32,
) -> Vec<Sample> {
    variants
        .iter()
        .map(|&variant| {
            bench_variant_n(
                stream,
                n,
                b,
                variant,
                compile_options.clone(),
                cache_sweep_mib,
                iters,
            )
        })
        .collect()
}

/// Measure wall-clock of the first async_on call on a monomorphization.
/// For static, every N is a fresh monomorphization; for optimized/safe,
/// only the first N triggers JIT and the rest report as `cached=true`
/// (essentially zero JIT). We do NOT try to invalidate the module
/// cache — the comparison across variants is the point.
fn jit_n(
    stream: &Arc<Stream>,
    n: usize,
    b: i32,
    compile_options: CompileOptions,
    already_jitted: &mut std::collections::HashSet<&'static str>,
) -> Vec<JitSample> {
    let generics_dyn = vec!["f16".to_string(), b.to_string()];
    let generics_static = vec!["f16".to_string(), b.to_string(), n.to_string()];

    let x = api::ones::<f16>(&[n]).sync_on(stream).expect("alloc x");
    let y = api::ones::<f16>(&[n]).sync_on(stream).expect("alloc y");

    let mut out = Vec::with_capacity(3);
    for v in Variant::all() {
        let generics = if *v == Variant::AddStatic {
            &generics_static
        } else {
            &generics_dyn
        };
        // Static never gets cached across N (unique const N per shape).
        // Others get cached after first N.
        let pre_cached = match v {
            Variant::AddStatic => false,
            _ => already_jitted.contains(v.label()),
        };

        let mut z = api::zeros::<f16>(&[n])
            .partition([b as usize])
            .sync_on(stream)
            .expect("alloc z");
        let grid = z.grid().expect("grid");

        // Fresh-monomorphization first launch: includes the full JIT
        // pipeline (bytecode emission, tileiras backend, module load)
        // which runs synchronously inside async_on before the launch
        // is queued on the stream. We sync BEFORE the timed call to
        // flush pending work, then don't sync after — that would
        // wait for this kernel to finish and contaminate the JIT
        // number with kernel execution time.
        sync_stream(stream, "sync before timing");
        let t0 = Instant::now();
        z = launch_one!(*v, stream, z, &x, &y, generics, Some(grid), compile_options);
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        sync_stream(stream, "sync after");

        // Drop z so later iterations don't accumulate device memory.
        drop(z);

        if !pre_cached {
            already_jitted.insert(v.label());
        }
        println!(
            "  N={:>10} {:>9}: {} {:>8.2} ms",
            n,
            v.label(),
            if pre_cached { "cached " } else { "jit    " },
            total_ms,
        );
        out.push(JitSample {
            config: v.label(),
            n,
            b,
            total_ms,
            cached: pre_cached,
        });
    }
    out
}

fn dump_ir_pass(variant: Variant, stream: &Arc<Stream>, n: usize, b: i32) {
    let generics = match variant {
        Variant::AddStatic => vec!["f16".to_string(), b.to_string(), n.to_string()],
        _ => vec!["f16".to_string(), b.to_string()],
    };

    let x = api::ones::<f16>(&[n]).sync_on(stream).expect("alloc x");
    let y = api::ones::<f16>(&[n]).sync_on(stream).expect("alloc y");
    let z = api::zeros::<f16>(&[n])
        .partition([b as usize])
        .sync_on(stream)
        .expect("alloc z");
    let grid = z.grid().expect("grid");

    unsafe {
        match variant {
            Variant::Add => {
                add_ir(z, &x, &y)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            Variant::AddSafe => {
                add_safe_ir(z, &x, &y)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
            Variant::AddStatic => {
                add_static_ir(z, &x, &y)
                    .const_grid(grid)
                    .generics(generics)
                    .async_on(stream)
                    .expect("launch");
            }
        }
    }
    sync_stream(stream, "sync");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let is_safe = args.iter().any(|a| a == "--safe");
    let is_static = args.iter().any(|a| a == "--static");
    let dump_ir = args.iter().any(|a| a == "--dump-ir");
    let jit_mode = args.iter().any(|a| a == "--jit");
    let tune_one = args.iter().any(|a| a == "--tune-one");
    if is_safe && is_static {
        eprintln!("error: --safe and --static are mutually exclusive");
        std::process::exit(2);
    }
    if jit_mode && (is_safe || is_static) {
        eprintln!("error: --jit runs all variants; --safe/--static not applicable");
        std::process::exit(2);
    }
    if tune_one && (jit_mode || dump_ir) {
        eprintln!("error: --tune-one is mutually exclusive with --jit/--dump-ir");
        std::process::exit(2);
    }

    let device = Device::new(0).expect("device");
    let stream = device.new_stream().expect("stream");

    // Sweep N from 2^20 (1M, ~2 MB per f16 tensor) to 2^28 (256M,
    // ~512 MB). Three tensors at 2^28 ≈ 1.5 GB working set, past L2.
    let ns: Vec<usize> = (20..=28).map(|i| 1usize << i).collect();
    let b: i32 = parse_arg(&args, "--b", 128)?;
    for &n in &ns {
        validate_elemwise_config(n, b)?;
    }
    let cache_sweep_mib: usize = parse_arg(&args, "--cache-sweep-mib", DEFAULT_CACHE_SWEEP_MIB)?;
    let default_iters = std::env::var("ELEM_ITERS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_ITERS);
    let iters: u32 = parse_arg(&args, "--iters", default_iters)?;
    if iters == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--iters must be positive",
        )
        .into());
    }
    let hints = HintConfig::from_args(&args)?;
    let compile_options = hints.compile_options();

    if tune_one {
        let n = parse_arg(&args, "--n", *ns.last().expect("non-empty"))?;
        validate_elemwise_config(n, b)?;
        let variants = selected_tune_variants(&args, is_safe, is_static)?;
        println!("=== Element-wise Add Tuning Run: cuTile Rust ===");
        println!(
            "--- N={} B={} variants={} cta={} occupancy={} max_divisibility={} iters={} ---",
            n,
            b,
            variants
                .iter()
                .map(|v| v.label())
                .collect::<Vec<_>>()
                .join("/"),
            hints.num_cta_label(),
            hints.occupancy_label(),
            hints.max_divisibility_label(),
            iters,
        );
        let samples = bench_n(
            &stream,
            n,
            b,
            &variants,
            compile_options,
            cache_sweep_mib,
            iters,
        );
        let csv_path = cli_value(&args, "--csv").unwrap_or("elemwise_rust_tune_results.csv");
        let mut f = open_append_csv(
            csv_path,
            "config,N,B,cache_chunks,num_cta_in_cga,occupancy,max_divisibility,iters,median_us,min_us,max_us,p25_us,p75_us,stdev_us,gb_per_s",
        )?;
        for s in &samples {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2}",
                s.config,
                s.n,
                s.b,
                s.cache_chunks,
                hints.num_cta_label(),
                hints.occupancy_label(),
                hints.max_divisibility_label(),
                s.iters,
                s.per_launch_us_median,
                s.per_launch_us_min,
                s.per_launch_us_max,
                s.per_launch_us_p25,
                s.per_launch_us_p75,
                s.per_launch_us_stdev,
                s.gb_per_s,
            )?;
        }
        println!("\nTuning result appended to {}", csv_path);
        return Ok(());
    }

    if dump_ir {
        if std::env::var_os("CUTILE_DUMP").is_none() {
            unsafe {
                std::env::set_var("CUTILE_DUMP", "ir");
            }
        }
        let variants = selected_tune_variants(&args, is_safe, is_static)?;
        if variants.len() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--dump-ir requires exactly one variant",
            )
            .into());
        }
        let variant = variants[0];
        let (filter, label) = match variant {
            Variant::Add => ("add_ir", "optimized"),
            Variant::AddSafe => ("add_safe_ir", "safe"),
            Variant::AddStatic => ("add_static_ir", "static"),
        };
        if std::env::var_os("CUTILE_DUMP_FILTER").is_none() {
            unsafe {
                std::env::set_var("CUTILE_DUMP_FILTER", filter);
            }
        }
        let n = *ns.last().expect("non-empty");
        eprintln!("=== IR dump: {} at N={} ===", label, n);
        dump_ir_pass(variant, &stream, n, b);
        return Ok(());
    }

    if jit_mode {
        println!("=== Element-wise Add JIT Cost (cuTile Rust) ===");
        println!(
            "B={} cta={} occupancy={} max_divisibility={}",
            b,
            hints.num_cta_label(),
            hints.occupancy_label(),
            hints.max_divisibility_label(),
        );
        println!("Measures total JIT wall time per unique monomorphization.");
        println!("Static: fresh JIT at every N. Optimized/safe: JIT once, cached after.");
        let mut already = std::collections::HashSet::<&'static str>::new();
        let mut all: Vec<JitSample> = Vec::new();
        for &n in &ns {
            all.extend(jit_n(&stream, n, b, compile_options.clone(), &mut already));
        }

        let csv_path = output_path("elemwise_rust_jit_results.csv");
        let mut f = File::create(&csv_path)?;
        writeln!(f, "config,N,B,total_ms,cached")?;
        for s in &all {
            writeln!(
                f,
                "{},{},{},{:.3},{}",
                s.config, s.n, s.b, s.total_ms, s.cached
            )?;
        }
        println!("\nJIT results written to {}", csv_path.display());
        return Ok(());
    }

    // Default: isolated runtime bench across selected variants.
    let variants = selected_tune_variants(&args, is_safe, is_static)?;
    let append_csv = args.iter().any(|a| a == "--append-csv");
    println!("=== Element-wise Add Benchmark: cuTile Rust ===");
    println!(
        "Isolated variants per N. Timing via CUDA events, warmup={} samples={} iters={}.",
        N_WARMUP, N_SAMPLES, iters
    );
    println!(
        "variants={}",
        variants
            .iter()
            .map(|v| v.label())
            .collect::<Vec<_>>()
            .join("/")
    );
    println!(
        "B={} cta={} occupancy={} max_divisibility={}",
        b,
        hints.num_cta_label(),
        hints.occupancy_label(),
        hints.max_divisibility_label(),
    );
    println!(
        "cache_sweep_mib={} max_cache_chunks={}",
        cache_sweep_mib, MAX_CACHE_CHUNKS
    );

    let mut all: Vec<Sample> = Vec::new();
    for &n in &ns {
        all.extend(bench_n(
            &stream,
            n,
            b,
            &variants,
            compile_options.clone(),
            cache_sweep_mib,
            iters,
        ));
    }

    println!("\n============================================================");
    println!("  SUMMARY (f16, isolated, CUDA events)");
    println!("============================================================");
    println!("  {:>10}  {:>9}  {:>11}", "N", "variant", "GB/s");
    for s in &all {
        println!("  {:>10}  {:>9}  {:>8.1}", s.n, s.config, s.gb_per_s);
    }

    let csv_path = output_path("elemwise_rust_runtime_results.csv");
    let header =
        "config,N,B,cache_chunks,iters,median_us,min_us,max_us,p25_us,p75_us,stdev_us,gb_per_s";
    let mut f = if append_csv {
        open_append_csv(csv_path.to_str().expect("utf-8 csv path"), header)?
    } else {
        let mut f = File::create(&csv_path)?;
        writeln!(f, "{header}")?;
        f
    };
    for s in &all {
        writeln!(
            f,
            "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2}",
            s.config,
            s.n,
            s.b,
            s.cache_chunks,
            s.iters,
            s.per_launch_us_median,
            s.per_launch_us_min,
            s.per_launch_us_max,
            s.per_launch_us_p25,
            s.per_launch_us_p75,
            s.per_launch_us_stdev,
            s.gb_per_s,
        )?;
    }
    println!("\nRuntime results written to {}", csv_path.display());

    Ok(())
}

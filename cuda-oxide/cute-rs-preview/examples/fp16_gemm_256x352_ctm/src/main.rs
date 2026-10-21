/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host input preparation, validation, and timing for the two-CTA 256x352 FP16 GEMM.

#![feature(f16)]

mod kernel;
mod physical;

use ctm_rs::host::tma::{EncodeOptions, encode_2d};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use serde_json::json;
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Host-side result type carrying CUDA, file, and argument-validation errors.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Choose per-launch event timing or repeated launches captured in a CUDA graph.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimingMode {
    /// Time individual kernel launches between CUDA events.
    Direct,
    /// Time replayed CUDA graphs and report latency per captured kernel launch.
    Graph,
}

impl TimingMode {
    /// Stable timing-mode label included in output reports.
    fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Graph => "graph",
        }
    }

    /// Report the CUDA timing mechanism used for the selected mode.
    fn tool(self) -> &'static str {
        match self {
            Self::Direct => "cuda-events-single-kernel",
            Self::Graph => "cuda-graph-events-per-kernel",
        }
    }
}

/// Validated matrix shape, timing controls, and input/output paths for the benchmark.
struct Args {
    /// Number of output rows; must be a positive multiple of 256.
    m: u32,
    /// Number of output columns; must be positive and divisible by eight.
    n: u32,
    /// Reduction extent in FP16 elements; must be positive and divisible by eight.
    k: u32,
    /// Add one FP16 bias value per output row when enabled.
    has_bias: bool,
    /// Untimed warmup iterations before collecting timing samples.
    warmup: usize,
    /// Number of timed iterations used for the latency report.
    iters: usize,
    /// Kernel submissions captured into each timed graph replay.
    graph_launches: usize,
    /// Whether timings measure direct launches or graph replay.
    timing_mode: TimingMode,
    /// Optional persistent two-CTA cluster count; otherwise chosen from occupancy.
    clusters: Option<u32>,
    /// Optional directory containing little-endian FP16 input files.
    input_dir: Option<PathBuf>,
    /// Optional destination for the little-endian FP16 result matrix.
    output: Option<PathBuf>,
    /// Optional destination for the machine-readable benchmark report.
    json: Option<PathBuf>,
    /// Use externally verified input/output files instead of running the CPU reference.
    skip_cpu_check: bool,
}

impl Args {
    /// Parse command-line options and validate shape, launch limits, timing, and oracle-mode requirements.
    fn parse() -> Result<Self> {
        let mut args = Self {
            m: 256,
            n: 352,
            k: 64,
            has_bias: false,
            warmup: 25,
            iters: 100,
            graph_launches: 10,
            timing_mode: TimingMode::Graph,
            clusters: None,
            input_dir: None,
            output: None,
            json: None,
            skip_cpu_check: false,
        };
        let mut cli = std::env::args().skip(1);
        while let Some(key) = cli.next() {
            match key.as_str() {
                "--help" | "-h" => {
                    println!(
                        "fp16_gemm_256x352_ctm [--mnk M,N,K] [--has-bias] [--clusters COUNT]\n  [--warmup 25] [--iters 100] [--timing-mode direct|graph] [--graph-launches 10]\n  [--json RESULT.json] [--input-dir DIR] [--output C.f16]\n  [--skip-cpu-check] (requires external input/output oracle)\nctm-rs GEMM with CPU verification. M must be a positive multiple\nof 256, N and K of 8. SM100 required. Graph timing is the default.\nDirect --warmup 0 --iters 1 launches exactly once."
                    );
                    std::process::exit(0);
                }
                "--has-bias" | "--has_bias" => args.has_bias = true,
                "--skip-cpu-check" => args.skip_cpu_check = true,
                "--rust-only" => {} // Compatibility with the shared validation protocol.
                _ => {
                    let value = cli
                        .next()
                        .ok_or_else(|| format!("missing value for {key}"))?;
                    match key.as_str() {
                        "--mnk" => {
                            let shape = value
                                .split(',')
                                .map(str::parse::<u32>)
                                .collect::<std::result::Result<Vec<_>, _>>()?;
                            if shape.len() != 3 {
                                return Err("--mnk requires M,N,K".into());
                            }
                            (args.m, args.n, args.k) = (shape[0], shape[1], shape[2]);
                        }
                        "--warmup" => args.warmup = value.parse()?,
                        "--iters" => args.iters = value.parse()?,
                        "--graph-launches" => {
                            args.graph_launches = value.parse()?;
                        }
                        "--timing-mode" => {
                            args.timing_mode = match value.as_str() {
                                "direct" => TimingMode::Direct,
                                "graph" => TimingMode::Graph,
                                _ => return Err("--timing-mode requires direct or graph".into()),
                            };
                        }
                        "--clusters" => args.clusters = Some(value.parse()?),
                        "--input-dir" => args.input_dir = Some(value.into()),
                        "--output" => args.output = Some(value.into()),
                        "--json" => args.json = Some(value.into()),
                        _ => return Err(format!("unknown argument {key}").into()),
                    }
                }
            }
        }
        if args.m == 0
            || args.n == 0
            || args.k == 0
            || !args.m.is_multiple_of(256)
            || !args.n.is_multiple_of(8)
            || !args.k.is_multiple_of(8)
        {
            return Err("require positive M divisible by 256, N and K by 8".into());
        }
        if args.m > i32::MAX as u32 || args.n > i32::MAX as u32 || args.k > i32::MAX as u32 {
            return Err("TMA coordinates must fit i32".into());
        }
        // The last N tile issues all eleven stores even when some are
        // out-of-bounds. Their starting coordinates must remain positive
        // after the kernel casts to i32.
        let ntiles = args.n.div_ceil(352);
        let last_store_col = u64::from(ntiles - 1) * 352 + 320;
        if last_store_col > i32::MAX as u64 {
            return Err("padded N-tile TMA coordinates must fit i32".into());
        }
        let tiles = (args.m / 256)
            .checked_mul(ntiles)
            .ok_or("tile count overflow")?;
        // The occupancy probe uses the full problem grid. This bound also
        // prevents tile += tile_stride from wrapping in the persistent loop.
        if tiles > i32::MAX as u32 / 2 {
            return Err("two-CTA problem grid must fit the CUDA grid-x limit".into());
        }
        for count in [
            u64::from(args.m) * u64::from(args.k),
            u64::from(args.n) * u64::from(args.k),
            u64::from(args.m) * u64::from(args.n),
        ] {
            if count > isize::MAX as u64 / 2 {
                return Err("FP16 allocation size exceeds the host addressable limit".into());
            }
        }
        if args.iters == 0 || args.clusters == Some(0) {
            return Err("iters and clusters must be positive".into());
        }
        if args.timing_mode == TimingMode::Graph && args.graph_launches == 0 {
            return Err("graph-launches must be positive in graph timing mode".into());
        }
        if args.skip_cpu_check && (args.input_dir.is_none() || args.output.is_none()) {
            return Err(
                "--skip-cpu-check requires --input-dir and --output for external verification"
                    .into(),
            );
        }
        Ok(args)
    }
}

/// Convert a CUDA driver result into a host error annotated with the failed operation.
fn cuda_status(status: sys::CUresult, operation: &str) -> Result<()> {
    if status == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation}: {}", cuda_core::DriverError(status)).into())
    }
}

// The executable owns graph resources only; main keeps every captured
// module/allocation alive until after this object is dropped.
/// Own an executable CUDA graph; captured modules and allocations remain owned by main.
struct KernelGraph {
    /// Instantiated CUDA graph handle destroyed when this wrapper is dropped.
    executable: sys::CUgraphExec,
    /// Stream and context on which the graph was captured and is replayed.
    stream: Arc<CudaStream>,
}

impl KernelGraph {
    /// Capture repeated calls to run, instantiate the graph, and clean up partial capture on failure.
    fn capture(
        stream: &Arc<CudaStream>,
        repetitions: usize,
        run: impl Fn() -> Result<()>,
    ) -> Result<Self> {
        stream.context().bind_to_thread()?;
        // All input copies and warmups have completed before capture begins.
        cuda_status(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "begin graph capture",
        )?;
        let captured = (|| {
            for _ in 0..repetitions {
                run()?;
            }
            Ok(())
        })();
        let mut graph = std::ptr::null_mut();
        // Always terminate capture, including when a kernel submission fails.
        let ended = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if let Err(error) = captured {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(error);
        }
        if let Err(error) = cuda_status(ended, "end graph capture") {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(error);
        }
        let mut executable = std::ptr::null_mut();
        let instantiated = unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0) };
        // Instantiation owns its own graph state; the source graph is no
        // longer needed regardless of whether instantiation succeeded.
        unsafe {
            sys::cuGraphDestroy(graph);
        }
        cuda_status(instantiated, "instantiate graph")?;
        Ok(Self {
            executable,
            stream: Arc::clone(stream),
        })
    }

    /// Bind the stream's context and enqueue the captured graph once.
    fn replay(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        cuda_status(
            unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()) },
            "replay graph",
        )
    }
}

impl Drop for KernelGraph {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_ok() {
            unsafe {
                sys::cuGraphExecDestroy(self.executable);
            }
        }
    }
}

/// Read exactly count little-endian FP16 bit patterns, rejecting a mismatched file size.
fn load_f16(path: &Path, count: usize) -> Result<Vec<u16>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != count.checked_mul(2).ok_or("input size overflow")? {
        return Err(format!(
            "{}: expected {count} FP16 values, got {} bytes",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect())
}

/// Write FP16 bit patterns as little-endian bytes and flush the output file.
fn store_f16(path: &Path, data: &[u16]) -> Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

// The standalone run uses reproducible integers in [-2, 2), exact in FP16.
// External-oracle runs may supply shared input bytes to multiple implementations.
/// Generate reproducible integers in [-2, 2) and return their exact FP16 bit patterns.
fn integer_input(count: usize, mut state: u32) -> Vec<u16> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (((state >> 16) % 4) as i32 - 2) as f16
        })
        .map(f16::to_bits)
        .collect()
}

/// Compare GPU output with FP32 CPU accumulation rounded to FP16, including optional row bias.
fn verify(args: &Args, a: &[u16], b: &[u16], bias: &[u16], got: &[u16]) -> Result<()> {
    let (m, n, k) = (args.m as usize, args.n as usize, args.k as usize);
    let mut failures = 0;
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += f16::from_bits(a[row * k + inner]) as f32
                    * f16::from_bits(b[col * k + inner]) as f32;
            }
            if args.has_bias {
                sum += f16::from_bits(bias[row]) as f32;
            }
            let actual = f16::from_bits(got[row * n + col]) as f32;
            // The Python oracle rounds its FP32 accumulation to the output
            // dtype before comparison. Apply the identical acceptance band.
            let expected = (sum as f16) as f32;
            if !actual.is_finite() || (actual - expected).abs() > 0.1 + 1.0e-5 * expected.abs() {
                if failures < 5 {
                    eprintln!("C[{row},{col}]: expected {expected}, got {actual}");
                }
                failures += 1;
            }
        }
    }
    if failures != 0 {
        return Err(format!("CPU verification failed: {failures} / {} values", got.len()).into());
    }
    eprintln!("CPU verification: PASS ({} values)", got.len());
    Ok(())
}

/// Validate inputs, launch the SM100 kernel, verify results, and report direct or graph timing.
fn main() -> Result<()> {
    let args = Args::parse()?;
    let single_launch =
        args.timing_mode == TimingMode::Direct && args.warmup == 0 && args.iters == 1;
    let ctx = CudaContext::new(0)?;
    let capability = ctx.compute_capability()?;
    if capability != (10, 0) {
        return Err(format!(
            "this example requires SM100; found sm_{}{}",
            capability.0, capability.1
        )
        .into());
    }
    // Stream capture cannot start on CUDA's legacy default stream.
    let stream = ctx.new_stream()?;
    let (m, n, k) = (args.m as usize, args.n as usize, args.k as usize);
    let (a, b, bias) = if let Some(dir) = &args.input_dir {
        (
            load_f16(&dir.join("a.f16"), m * k)?,
            load_f16(&dir.join("b.f16"), n * k)?,
            if args.has_bias {
                load_f16(&dir.join("bias.f16"), m)?
            } else {
                vec![0; m]
            },
        )
    } else {
        (
            integer_input(m * k, 1),
            integer_input(n * k, 7),
            (0..m)
                .map(|i| (((i % 17) as f32 - 8.0) / 8.0) as f16)
                .map(f16::to_bits)
                .collect(),
        )
    };
    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let bias_dev = DeviceBuffer::from_host(&stream, &bias)?;
    // Poison the output so missing tiles, including N tails, fail verification.
    let mut c_dev = DeviceBuffer::from_host(&stream, &vec![f16::NAN.to_bits(); m * n])?;
    // These physical format types set both the descriptor's tile shape and swizzle
    // and the kernel's shared-memory layout.
    let a_desc = encode_2d::<f16, kernel::ASmem>(
        a_dev.cu_deviceptr() as *mut core::ffi::c_void,
        u64::from(args.m),
        u64::from(args.k),
        u64::from(args.k),
        EncodeOptions::default(),
    )?;
    let b0_desc = encode_2d::<f16, kernel::B0Smem>(
        b_dev.cu_deviceptr() as *mut core::ffi::c_void,
        u64::from(args.n),
        u64::from(args.k),
        u64::from(args.k),
        EncodeOptions::default(),
    )?;
    let b1_desc = encode_2d::<f16, kernel::B1Smem>(
        b_dev.cu_deviceptr() as *mut core::ffi::c_void,
        u64::from(args.n),
        u64::from(args.k),
        u64::from(args.k),
        EncodeOptions::default(),
    )?;
    let c_desc = encode_2d::<f16, kernel::CSmem>(
        c_dev.cu_deviceptr() as *mut core::ffi::c_void,
        u64::from(args.m),
        u64::from(args.n),
        u64::from(args.n),
        EncodeOptions::default(),
    )?;
    // SAFETY: this executable exclusively owns the module and all its buffers.
    let module = unsafe { kernel::kernels::load(&ctx) }?;
    let tiles = (args.m / 256)
        .checked_mul(args.n.div_ceil(352))
        .ok_or("tile count overflow")?;
    let blocks = tiles.checked_mul(2).ok_or("grid size overflow")?;
    let probe = module.prepare_fp16_gemm_256x352(LaunchConfig1D::new(
        2,
        kernel::THREADS,
        kernel::DYNAMIC_SMEM_BYTES as u32,
    ))?;
    let capacity = probe.function().max_active_clusters(
        (blocks, 1, 1),
        (kernel::THREADS, 1, 1),
        kernel::DYNAMIC_SMEM_BYTES as u32,
        (2, 1, 1),
    )?;
    let clusters = args.clusters.unwrap_or(capacity.min(tiles));
    if clusters == 0 || clusters > capacity || clusters > tiles {
        return Err(format!(
            "requested {clusters} clusters; resident capacity={capacity}, tiles={tiles}"
        )
        .into());
    }
    let registers = probe.function().num_registers()?;
    drop(probe);
    let launch = module.prepare_fp16_gemm_256x352(LaunchConfig1D::new(
        clusters * 2,
        kernel::THREADS,
        kernel::DYNAMIC_SMEM_BYTES as u32,
    ))?;
    let run = || -> Result<()> {
        // SAFETY: each descriptor is copied into read-only kernel parameters.
        // We keep all tensor buffers alive until the stream finishes. The
        // prepared launch checks block, cluster, and shared-memory sizes.
        // Output tiles do not overlap, and the tensor shapes were checked above.
        unsafe {
            module.fp16_gemm_256x352(
                &stream,
                &launch,
                a_desc,
                b0_desc,
                b1_desc,
                c_desc,
                bias_dev.cu_deviceptr() as *const f16,
                u64::from(args.m),
                u64::from(args.n),
                u64::from(args.k),
                u64::from(args.has_bias),
            )?;
        }
        Ok(())
    };
    // A single direct sample verifies its timed output without a preliminary launch.
    let got = if single_launch {
        None
    } else {
        run()?;
        let got = c_dev.to_host_vec(&stream)?;
        if !args.skip_cpu_check {
            verify(&args, &a, &b, &bias, &got)?;
        }
        Some(got)
    };
    for _ in 0..args.warmup {
        run()?;
    }
    stream.synchronize()?;
    let graph = if args.timing_mode == TimingMode::Graph {
        let graph = KernelGraph::capture(&stream, args.graph_launches, run)?;
        // Keep graph setup and the first replay outside timing. Poison the
        // output so an empty or wrong-stream graph cannot preserve a pass.
        c_dev.copy_from_host(&stream, &vec![f16::NAN.to_bits(); m * n])?;
        graph.replay()?;
        if Some(c_dev.to_host_vec(&stream)?) != got {
            return Err("captured graph output differs from the checked direct launch".into());
        }
        for _ in 0..args.warmup {
            graph.replay()?;
        }
        stream.synchronize()?;
        Some(graph)
    } else {
        None
    };
    let launches_per_sample = if graph.is_some() {
        args.graph_launches
    } else {
        1
    };
    let events = (0..args.iters)
        .map(|_| {
            Ok((
                ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (start, end) in &events {
        start.record(&stream)?;
        if let Some(graph) = &graph {
            graph.replay()?;
        } else {
            run()?;
        }
        end.record(&stream)?;
        if args.timing_mode == TimingMode::Direct {
            // Match blockscale_gemm_cute: complete each single-kernel sample
            // before submitting the next one, reusing the same input buffers.
            end.synchronize()?;
        }
    }
    stream.synchronize()?;
    let after_timing = c_dev.to_host_vec(&stream)?;
    if let Some(got) = got {
        if after_timing != got {
            return Err("output changed across identical repeated launches".into());
        }
    } else if !args.skip_cpu_check {
        verify(&args, &a, &b, &bias, &after_timing)?;
    }
    if let Some(path) = &args.output {
        store_f16(path, &after_timing)?;
    }
    let mut times = events
        .iter()
        .map(|(start, end)| {
            Ok(f64::from(start.elapsed_ms(end)?) * 1000.0 / launches_per_sample as f64)
        })
        .collect::<Result<Vec<_>>>()?;
    times.sort_by(f64::total_cmp);
    let mid = times.len() / 2;
    let median = if times.len() % 2 == 0 {
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[mid]
    };
    let result = json!({"implementation": "ctm-rs", "mnk": [m,n,k], "has_bias": args.has_bias,
        "metric_value": median, "metric_unit": "us", "tflops": 2.0*m as f64*n as f64*k as f64/(median*1.0e6),
        "verification": if args.skip_cpu_check { "external-oracle-required" } else { "cpu-pass" },
        "provenance": {"gpu": ctx.device_name()?, "compute_capability": [capability.0,capability.1],
            "tool": args.timing_mode.tool(), "timing_mode": args.timing_mode.name(),
            "warmup": args.warmup, "iters": args.iters, "single_launch": single_launch,
            "graph_launches": launches_per_sample,
            "clusters": clusters, "resident_cluster_capacity": capacity, "threads": kernel::THREADS,
            "shared_memory_bytes": kernel::DYNAMIC_SMEM_BYTES, "registers_per_thread": registers}});
    let rendered = serde_json::to_string_pretty(&result)?;
    if let Some(path) = args.json {
        std::fs::write(path, &rendered)?;
    }
    println!("{rendered}");
    Ok(())
}

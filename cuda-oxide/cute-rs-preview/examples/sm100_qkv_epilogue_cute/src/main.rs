/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host input preparation, CPU FP8 (`e4m3fn`) oracle verification, 1,000× CUDA Graph replay
//! benchmarking, and live comparison against Python `CuTeDSL` on NVIDIA B200 (`sm_100a`).

#![feature(f16)]
#![deny(missing_docs, clippy::missing_docs_in_private_items)]

mod kernel;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::attention::sm100::create_tensor_map_tiled;
use kernel::{DYNAMIC_SMEM_BYTES, THREADS, WLayout, XLayout, kernels};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

/// Host-side result alias carrying CUDA driver, I/O, and verification errors.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Head dimension `D` (`128` elements per Q, K, or V attention head).
/// Matches the 128-column TMEM accumulator width per head on SM100.
const HEAD_DIM: usize = 128;

/// Default token counts `M` swept from `128` to `8,192` tokens.
/// Evaluates both single-wave and multi-wave persistent execution on B200.
const DEFAULT_M_LIST: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];

/// Command-line configuration for the fused QKV + Epilogue benchmark and verifier.
struct Args {
    /// Run only the `cute-rs` kernels without invoking the Python `CuTeDSL` harness.
    rust_only: bool,
    /// Optional explicit Python interpreter with `nvidia-cutlass-dsl==4.7.0` and `torch`.
    python: Option<PathBuf>,
    /// Token counts `M` to verify and benchmark (each must be a positive multiple of `128`).
    m_list: Vec<usize>,
    /// Hidden dimension `K` (must be a positive multiple of `64`, default `4,096`).
    k: usize,
    /// Number of Query heads (`N_q = num_q_heads * 128`, default `32`).
    num_q_heads: usize,
    /// Number of Key/Value heads (`N_k = N_v = num_kv_heads * 128`, default `8`).
    num_kv_heads: usize,
    /// Untimed CUDA Graph replay warmups before collecting timing samples.
    warmup: usize,
    /// Number of back-to-back CUDA Graph replays timed per event pair (`default = 1,000`).
    replays: usize,
    /// Number of independent `replays`-iteration samples averaged per token count `M`.
    samples: usize,
    /// Optional path to write the combined JSON benchmark report.
    json: Option<PathBuf>,
}

impl Args {
    /// Parse CLI flags and validate all tensor dimensions and alignment divisibility rules.
    fn parse() -> Result<Self> {
        let mut args = Self {
            rust_only: false,
            python: None,
            m_list: DEFAULT_M_LIST.to_vec(),
            k: 4096,
            num_q_heads: 32,
            num_kv_heads: 8,
            warmup: 25,
            replays: 1000,
            samples: 5,
            json: None,
        };
        let mut cli = std::env::args().skip(1);
        while let Some(flag) = cli.next() {
            match flag.as_str() {
                "--help" | "-h" => {
                    println!(
                        "sm100_qkv_epilogue_cute [--m-list 128,256,512,1024,2048,4096,8192] [--k 4096]\n  \
                         [--q-heads 32] [--kv-heads 8] [--warmup 25] [--replays 1000] [--samples 5]\n  \
                         [--rust-only] [--python PATH] [--json REPORT.json]"
                    );
                    std::process::exit(0);
                }
                "--rust-only" => args.rust_only = true,
                _ => {
                    let val = cli
                        .next()
                        .ok_or_else(|| format!("missing value for {flag}"))?;
                    match flag.as_str() {
                        "--m-list" | "--m" => {
                            args.m_list = val
                                .split(',')
                                .filter(|s| !s.trim().is_empty())
                                .map(|s| s.trim().parse::<usize>())
                                .collect::<std::result::Result<Vec<_>, _>>()?;
                        }
                        "--k" => args.k = val.parse()?,
                        "--q-heads" => args.num_q_heads = val.parse()?,
                        "--kv-heads" => args.num_kv_heads = val.parse()?,
                        "--warmup" => args.warmup = val.parse()?,
                        "--replays" => args.replays = val.parse()?,
                        "--samples" => args.samples = val.parse()?,
                        "--python" => args.python = Some(val.into()),
                        "--json" => args.json = Some(val.into()),
                        _ => return Err(format!("unknown argument: {flag}").into()),
                    }
                }
            }
        }
        if args.m_list.is_empty()
            || args
                .m_list
                .iter()
                .any(|&m| m == 0 || !m.is_multiple_of(128))
        {
            return Err("all M values must be positive multiples of 128".into());
        }
        if args.k == 0 || !args.k.is_multiple_of(64) {
            return Err("K must be a positive multiple of 64".into());
        }
        let total_heads = args.num_q_heads + 2 * args.num_kv_heads;
        if total_heads == 0 || !total_heads.is_multiple_of(2) {
            return Err(
                "total heads (num_q_heads + 2 * num_kv_heads) must be a positive even integer"
                    .into(),
            );
        }
        if args.replays == 0 || args.samples == 0 {
            return Err("replays and samples must be positive".into());
        }
        Ok(args)
    }
}

/// Convert a CUDA driver status into a descriptive `Result`.
fn cuda_status(status: sys::CUresult, op: &str) -> Result<()> {
    if status == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op}: {}", cuda_core::DriverError(status)).into())
    }
}

/// Own an instantiated CUDA Graph (`CUgraphExec`) for 1,000× back-to-back replay timing.
struct KernelGraph {
    /// Instantiated CUDA graph executable handle destroyed on drop.
    executable: sys::CUgraphExec,
    /// Stream on which the graph was captured and is replayed.
    stream: Arc<CudaStream>,
}

impl KernelGraph {
    /// Capture one invocation of `launch_fn` between `cuStreamBeginCapture` and `cuStreamEndCapture`.
    fn capture(stream: &Arc<CudaStream>, launch_fn: impl FnOnce() -> Result<()>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        cuda_status(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "begin graph capture",
        )?;
        let cap_res = launch_fn();
        let mut graph = std::ptr::null_mut();
        let end_status = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        if let Err(err) = cap_res {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(err);
        }
        cuda_status(end_status, "end graph capture")?;
        let mut executable = std::ptr::null_mut();
        let inst_status = unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0) };
        unsafe {
            sys::cuGraphDestroy(graph);
        }
        cuda_status(inst_status, "instantiate graph")?;
        Ok(Self {
            executable,
            stream: Arc::clone(stream),
        })
    }

    /// Enqueue the captured CUDA graph once on its bound stream.
    fn replay(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        cuda_status(
            unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()) },
            "launch graph",
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

/// Encode one `f32` value into `Float8E4M3FN` (`u8`) with round-to-nearest-even and finite saturation (`±448.0`).
fn f32_to_e4m3fn(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs = x.abs().min(448.0);
    if abs == 0.0 {
        return sign;
    }
    let mut best_code = 0u8;
    let mut best_diff = f32::INFINITY;
    for code in 0u8..127u8 {
        let val = e4m3fn_to_f32(code);
        let diff = (abs - val).abs();
        if diff < best_diff || (diff == best_diff && (code & 1) == 0) {
            best_diff = diff;
            best_code = code;
        }
    }
    sign | best_code
}

/// Decode one `Float8E4M3FN` byte (`u8`) into its exact `f32` value.
fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 15 && mant == 7 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        (mant as f32) * (2.0f32).powi(-9)
    } else {
        (1.0f32 + (mant as f32) * 0.125f32) * (2.0f32).powi((exp as i32) - 7)
    };
    sign * mag
}

/// Generate deterministic integer-valued FP16 bit patterns in `{-1.0, 0.0, 1.0}` for exact verification.
/// Every row/head dot product is an exact integer in FP32 with zero accumulation rounding ambiguity.
fn deterministic_f16(count: usize, mut seed: u32) -> Vec<u16> {
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let v = ((seed >> 16) % 3) as i32 - 1;
            (v as f16).to_bits()
        })
        .collect()
}

/// Verify the GPU `q_out`, `k_cache`, and `v_cache` `FP8 (e4m3fn)` outputs against the CPU reference
/// for sample rows across the first, middle, and last tiles of `M`.
#[allow(clippy::too_many_arguments)]
fn verify_outputs(
    m: usize,
    k: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    x_bits: &[u16],
    w_bits: &[u16],
    gamma_q: &[f32],
    gamma_k: &[f32],
    slots: &[i32],
    q_out: &[u8],
    k_cache: &[u8],
    v_cache: &[u8],
    rms_eps: f32,
    inv_scale: f32,
) -> Result<f32> {
    let check_rows = [0usize, 1, 63, 64, 127, m / 2, m - 1];
    let mut max_err = 0.0f32;

    for &row in &check_rows {
        if row >= m {
            continue;
        }
        let x_row = &x_bits[row * k..(row + 1) * k];
        for head in 0..(num_q_heads + 2 * num_kv_heads) {
            let mut acc = [0.0f32; HEAD_DIM];
            for (d, acc_d) in acc.iter_mut().enumerate() {
                let col = head * HEAD_DIM + d;
                let w_row = &w_bits[col * k..(col + 1) * k];
                let mut sum = 0.0f32;
                for inner in 0..k {
                    sum += (f16::from_bits(x_row[inner]) as f32)
                        * (f16::from_bits(w_row[inner]) as f32);
                }
                *acc_d = sum;
            }
            if head < num_q_heads + num_kv_heads {
                let mean_sq = acc.iter().map(|v| v * v).sum::<f32>() / (HEAD_DIM as f32);
                let inv_rms = 1.0f32 / (mean_sq + rms_eps).sqrt();
                let gamma = if head < num_q_heads { gamma_q } else { gamma_k };
                for (d, (&acc_d, &gamma_d)) in acc.iter().zip(gamma.iter()).enumerate() {
                    let expected_f32 =
                        e4m3fn_to_f32(f32_to_e4m3fn(acc_d * inv_rms * gamma_d * inv_scale));
                    let got_byte = if head < num_q_heads {
                        q_out[(row * num_q_heads + head) * HEAD_DIM + d]
                    } else {
                        let kv_head = head - num_q_heads;
                        let slot = slots[row] as usize;
                        k_cache[(slot * num_kv_heads + kv_head) * HEAD_DIM + d]
                    };
                    let got_f32 = e4m3fn_to_f32(got_byte);
                    let err = (got_f32 - expected_f32).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if err > 0.125 {
                        return Err(format!(
                            "mismatch at M={m}, row={row}, head={head}, d={d}: expected {expected_f32}, got {got_f32}"
                        )
                        .into());
                    }
                }
            } else {
                let kv_head = head - (num_q_heads + num_kv_heads);
                let slot = slots[row] as usize;
                for (d, &acc_d) in acc.iter().enumerate() {
                    let expected_f32 = e4m3fn_to_f32(f32_to_e4m3fn(acc_d * inv_scale));
                    let got_byte = v_cache[(slot * num_kv_heads + kv_head) * HEAD_DIM + d];
                    let got_f32 = e4m3fn_to_f32(got_byte);
                    let err = (got_f32 - expected_f32).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    if err > 0.125 {
                        return Err(format!(
                            "V mismatch at M={m}, row={row}, kv_head={kv_head}, d={d}: expected {expected_f32}, got {got_f32}"
                        )
                        .into());
                    }
                }
            }
        }
    }
    Ok(max_err)
}

/// Locate the Python interpreter with `nvidia-cutlass-dsl` (`--python`, `CUDA_OXIDE_CUTE_PYTHON`, or `.venv`).
fn find_python(manifest_dir: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Some(env_p) = std::env::var_os("CUDA_OXIDE_CUTE_PYTHON") {
        return Ok(PathBuf::from(env_p));
    }
    for candidate in [
        manifest_dir.join(".venv/bin/python"),
        manifest_dir.join("../../.venv/bin/python"),
    ] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Ok(PathBuf::from("python3"))
}

/// Run `reference/benchmark.py` in a child process before initializing the Rust CUDA context.
fn run_cutedsl_reference(args: &Args) -> Result<BTreeMap<usize, f64>> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = find_python(manifest_dir, args.python.as_deref())?;
    let script = manifest_dir.join("reference/benchmark.py");
    let m_csv = args
        .m_list
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let output = Command::new(&python)
        .arg(&script)
        .args(["--m-list", &m_csv])
        .args(["--k", &args.k.to_string()])
        .args(["--q-heads", &args.num_q_heads.to_string()])
        .args(["--kv-heads", &args.num_kv_heads.to_string()])
        .args(["--warmup", &args.warmup.to_string()])
        .args(["--replays", &args.replays.to_string()])
        .args(["--samples", &args.samples.to_string()])
        .output()
        .map_err(|e| {
            format!(
                "failed to launch Python CuTeDSL benchmark ({}): {e}",
                python.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "Python CuTeDSL benchmark failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let parsed: BTreeMap<String, Value> = serde_json::from_slice(&output.stdout)?;
    let mut table = BTreeMap::new();
    for (k_str, val) in parsed {
        let m: usize = k_str.parse()?;
        let ovl_us = val["overlapped_us"]
            .as_f64()
            .ok_or("missing overlapped_us")?;
        table.insert(m, ovl_us);
    }
    Ok(table)
}

/// Time 1,000× CUDA Graph replay of `launch_fn` on `stream` and return mean microseconds per launch.
fn time_graph_replays(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    warmup: usize,
    replays: usize,
    samples: usize,
    launch_fn: impl FnOnce() -> Result<()>,
) -> Result<f64> {
    let graph = KernelGraph::capture(stream, launch_fn)?;
    for _ in 0..warmup {
        graph.replay()?;
    }
    stream.synchronize()?;

    let mut total_ms = 0.0f32;
    for _ in 0..samples {
        let start = ctx.new_event()?;
        let end = ctx.new_event()?;
        start.record(stream)?;
        for _ in 0..replays {
            graph.replay()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        total_ms += start.elapsed_ms(&end)?;
    }
    Ok((total_ms as f64) * 1000.0 / ((replays * samples) as f64))
}

/// Run the Python `CuTeDSL` reference (unless `--rust-only`), then verify and benchmark `cute-rs`
/// across all requested token counts `M` and print the full SOTA parity table.
fn main() -> Result<()> {
    let args = Args::parse()?;
    let cutedsl_table = if !args.rust_only {
        eprintln!("Running Python CuTeDSL 4.7 reference benchmark on B200 (sm_100a)...");
        Some(run_cutedsl_reference(&args)?)
    } else {
        None
    };

    let ctx = CudaContext::new(0)?;
    let cap = ctx.compute_capability()?;
    if cap != (10, 0) {
        return Err(format!("requires SM100 (sm_100a); found sm_{}{}", cap.0, cap.1).into());
    }
    let stream = ctx.new_stream()?;
    let n_total = (args.num_q_heads + 2 * args.num_kv_heads) * HEAD_DIM;
    let max_m = *args.m_list.iter().max().unwrap();
    let total_slots = ((max_m / 16) + 16) * 16;
    let inv_scale = 0.0625f32;
    let rms_eps = 1.0e-6f32;

    let w_host = deterministic_f16(n_total * args.k, 7);
    let gamma_q_host = vec![1.0f32; HEAD_DIM];
    let gamma_k_host = vec![1.0f32; HEAD_DIM];
    let d_w = DeviceBuffer::from_host(&stream, &w_host)?;
    let d_gamma_q = DeviceBuffer::from_host(&stream, &gamma_q_host)?;
    let d_gamma_k = DeviceBuffer::from_host(&stream, &gamma_k_host)?;
    let d_k_cache = DeviceBuffer::from_host(
        &stream,
        &vec![0u8; total_slots * args.num_kv_heads * HEAD_DIM],
    )?;
    let d_v_cache = DeviceBuffer::from_host(
        &stream,
        &vec![0u8; total_slots * args.num_kv_heads * HEAD_DIM],
    )?;
    let w_tma = create_tensor_map_tiled::<WLayout>(
        d_w.cu_deviceptr() as *mut core::ffi::c_void,
        n_total as u64,
        args.k as u64,
        args.k as u64,
    )?;

    // SAFETY: this executable exclusively owns the device module and all its buffers.
    let module = unsafe { kernels::load(&ctx) }?;
    let probe = module.prepare_sm100_qkv_epilogue_cute(LaunchConfig1D::new(
        2,
        THREADS,
        DYNAMIC_SMEM_BYTES,
    ))?;
    let max_clusters = probe.function().max_active_clusters(
        (2 * 74, 1, 1),
        (THREADS, 1, 1),
        DYNAMIC_SMEM_BYTES,
        (2, 1, 1),
    )?;
    drop(probe);

    println!(
        "\nFused Pre-Attention QKV Projection + RMSNorm + FP8 (E4M3) + Q Store + Paged KV Scatter on B200 (sm_100a)"
    );
    println!(
        "Shape: K = {}, Q Heads = {} (N_q = {}), KV Heads = {} (N_k = N_v = {}), N_total = {}, Head Dim = 128",
        args.k,
        args.num_q_heads,
        args.num_q_heads * HEAD_DIM,
        args.num_kv_heads,
        args.num_kv_heads * HEAD_DIM,
        n_total
    );
    println!(
        "Timing: {}x CUDA Graph Replay x {} Samples (Warmup = {}, Max Active 2-CTA Clusters = {})\n",
        args.replays, args.samples, args.warmup, max_clusters
    );

    if cutedsl_table.is_some() {
        println!(
            "{:>6} | {:>18} | {:>18} | {:>19} | {:>13} | {:>9}",
            "Tokens",
            "CuTeDSL (2-TMEM)",
            "cute-rs (2-TMEM)",
            "cute-rs vs. CuTeDSL",
            "cute-rs TFLOP",
            "Max Error"
        );
        println!("{}", "-".repeat(98));
    } else {
        println!(
            "{:>6} | {:>18} | {:>13} | {:>9}",
            "Tokens", "cute-rs (2-TMEM)", "cute-rs TFLOP", "Max Error"
        );
        println!("{}", "-".repeat(55));
    }

    let mut report_rows = Vec::new();
    for &m in &args.m_list {
        let x_host = deterministic_f16(m * args.k, 1);
        let slots_host: Vec<i32> = (0..m as i32).collect();
        let d_x = DeviceBuffer::from_host(&stream, &x_host)?;
        let d_slots = DeviceBuffer::from_host(&stream, &slots_host)?;
        let d_q_out =
            DeviceBuffer::from_host(&stream, &vec![0u8; m * args.num_q_heads * HEAD_DIM])?;
        let x_tma = create_tensor_map_tiled::<XLayout>(
            d_x.cu_deviceptr() as *mut core::ffi::c_void,
            m as u64,
            args.k as u64,
            args.k as u64,
        )?;

        let tiles = ((m / 128) * (n_total / 256)) as u32;
        let clusters = tiles.min(max_clusters);
        let cfg = LaunchConfig1D::new(2 * clusters, THREADS, DYNAMIC_SMEM_BYTES);
        let launch_overlapped = module.prepare_sm100_qkv_epilogue_cute(cfg)?;

        // Launch overlapped cute-rs once and verify against CPU FP8 oracle
        unsafe {
            module.sm100_qkv_epilogue_cute(
                &stream,
                &launch_overlapped,
                x_tma,
                w_tma,
                d_gamma_q.cu_deviceptr() as *const f32,
                d_gamma_k.cu_deviceptr() as *const f32,
                d_slots.cu_deviceptr() as *const i32,
                d_q_out.cu_deviceptr() as *mut u8,
                d_k_cache.cu_deviceptr() as *mut u8,
                d_v_cache.cu_deviceptr() as *mut u8,
                m as u64,
                n_total as u64,
                args.k as u64,
                args.num_q_heads as u64,
                args.num_kv_heads as u64,
                rms_eps.to_bits() as u64,
                inv_scale.to_bits() as u64,
                inv_scale.to_bits() as u64,
                inv_scale.to_bits() as u64,
            )?;
        }
        stream.synchronize()?;

        let q_got = d_q_out.to_host_vec(&stream)?;
        let k_got = d_k_cache.to_host_vec(&stream)?;
        let v_got = d_v_cache.to_host_vec(&stream)?;
        let max_err = verify_outputs(
            m,
            args.k,
            args.num_q_heads,
            args.num_kv_heads,
            &x_host,
            &w_host,
            &gamma_q_host,
            &gamma_k_host,
            &slots_host,
            &q_got,
            &k_got,
            &v_got,
            rms_eps,
            inv_scale,
        )?;

        let rust_overlapped_us = time_graph_replays(
            &ctx,
            &stream,
            args.warmup,
            args.replays,
            args.samples,
            || {
                unsafe {
                    module.sm100_qkv_epilogue_cute(
                        &stream,
                        &launch_overlapped,
                        x_tma,
                        w_tma,
                        d_gamma_q.cu_deviceptr() as *const f32,
                        d_gamma_k.cu_deviceptr() as *const f32,
                        d_slots.cu_deviceptr() as *const i32,
                        d_q_out.cu_deviceptr() as *mut u8,
                        d_k_cache.cu_deviceptr() as *mut u8,
                        d_v_cache.cu_deviceptr() as *mut u8,
                        m as u64,
                        n_total as u64,
                        args.k as u64,
                        args.num_q_heads as u64,
                        args.num_kv_heads as u64,
                        rms_eps.to_bits() as u64,
                        inv_scale.to_bits() as u64,
                        inv_scale.to_bits() as u64,
                        inv_scale.to_bits() as u64,
                    )?;
                }
                Ok(())
            },
        )?;

        let flops = 2.0 * (m as f64) * (n_total as f64) * (args.k as f64);
        let rust_tflops = flops / (rust_overlapped_us * 1.0e6);

        if let Some(ref dsl_map) = cutedsl_table {
            let &dsl_ovl_us = dsl_map
                .get(&m)
                .ok_or_else(|| format!("missing M={m} in CuTeDSL results"))?;
            let parity = dsl_ovl_us / rust_overlapped_us;
            println!(
                "{:6} | {:15.3} us | {:15.3} us | {:11.2}x CuTeDSL | {:11.1} T | {:9.6}",
                m, dsl_ovl_us, rust_overlapped_us, parity, rust_tflops, max_err
            );
            report_rows.push(json!({
                "m": m,
                "k": args.k,
                "n_total": n_total,
                "clusters": clusters,
                "rust_overlapped_us": rust_overlapped_us,
                "cutedsl_overlapped_us": dsl_ovl_us,
                "parity_vs_cutedsl": parity,
                "rust_tflops": rust_tflops,
                "max_error": max_err,
            }));
        } else {
            println!(
                "{:6} | {:15.3} us | {:11.1} T | {:9.6}",
                m, rust_overlapped_us, rust_tflops, max_err
            );
            report_rows.push(json!({
                "m": m,
                "k": args.k,
                "n_total": n_total,
                "clusters": clusters,
                "rust_overlapped_us": rust_overlapped_us,
                "rust_tflops": rust_tflops,
                "max_error": max_err,
            }));
        }
    }

    if let Some(ref json_path) = args.json {
        let report = json!({
            "kernel": "sm100_qkv_epilogue_cute",
            "architecture": "sm_100a",
            "k": args.k,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": HEAD_DIM,
            "warmup": args.warmup,
            "replays": args.replays,
            "samples": args.samples,
            "results": report_rows,
        });
        std::fs::write(json_path, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    Ok(())
}

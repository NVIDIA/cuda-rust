/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CLI dispatch, correctness checks, and timing for SM120a prefill attention.

use crate::{fixture, kernel};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, sys};
use fixture::{HEAD_DIM, Shape, bits_hash, make_inputs, reference, verify};
use serde_json::json;
use std::{error::Error, path::PathBuf};

/// Host-side errors from CUDA, file I/O, validation, or comparison execution.
type Result<T> = std::result::Result<T, Box<dyn Error>>;
/// Threads in each 64-query CTA.
const THREADS: u32 = 128;
/// Shared bytes reserved for the three FP16 Q/K/V tiles.
const SHARED_BYTES: u32 = 24_576;

/// Whether sequence dimensions are runtime arguments or a precompiled shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeMode {
    /// Pass query length, KV length, and head count as kernel arguments.
    Runtime,
    /// Use the precompiled 1024-query, 1024-key, eight-head shape.
    Static,
}

/// Validated prefill workload, comparison, and reporting options.
struct Args {
    /// Run the Rust kernel alone instead of launching the CuTeDSL comparison.
    rust_only: bool,
    /// Compare every supported mode combination for the selected workload.
    all_cases: bool,
    /// Optional Python interpreter for the comparison process.
    python: Option<PathBuf>,
    /// Requested runtime or precompiled sequence dimensions.
    shape_mode: ShapeMode,
    /// Attention dimensions and mask after CLI validation.
    shape: Shape,
    /// Power-of-two multiplier applied to Q and K, leaving V unchanged.
    input_scale: u32,
    /// Use an all-zero Q tensor to exercise uniform softmax.
    zero_query: bool,
    /// CUDA device ordinal after visible-device remapping.
    device: usize,
    /// Untimed invocations before collecting samples.
    warmup: usize,
    /// Number of timed samples; zero requests correctness checks only.
    iters: usize,
    /// Optional path for raw little-endian FP16 output in standalone mode.
    output: Option<PathBuf>,
    /// Optional path for the standalone or comparison JSON report.
    json: Option<PathBuf>,
}

impl Args {
    /// Read and validate options from the process command line.
    fn parse() -> Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    /// Parse prefill options, enforce supported shapes, and handle the help request.
    fn parse_from(mut cli: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = Self {
            rust_only: false,
            all_cases: false,
            python: None,
            shape_mode: ShapeMode::Runtime,
            shape: Shape {
                seq_q: 1024,
                seq_k: 1024,
                heads: 8,
                causal: true,
            },
            input_scale: 1,
            zero_query: false,
            device: 0,
            warmup: 100,
            iters: 31,
            output: None,
            json: None,
        };
        while let Some(key) = cli.next() {
            match key.as_str() {
                "--help" | "-h" => {
                    let mode = "Causal mode defaults to Sq=Sk=1024; --noncausal also permits Sq=1.";
                    println!(
                        "{} [--seq-q {}] [--seq-k 1024] [--heads 8]\n  [--causal | --noncausal] [--shape-mode runtime|static] [--input-scale 1|4|16] [--zero-query]\n  [--device 0] [--warmup 100] [--iters 31] [--json result.json] [--python PATH] [--rust-only | --all-cases]\nDefault: verify and compare CuTeDSL and cute-rs-preview; prints two performance lines.\n--all-cases prints a four-case runtime/static x causal/noncausal comparison table; requires Sq=Sk=1024, H=8.\nIt overrides the selected --shape-mode and mask; cannot be combined with --rust-only.\nLive timings use CUDA events; archived Nsight measurements are a separate metric.\n--json saves the combined report; --rust-only saves standalone JSON and permits --output o.f16.\nPython selection: --python, CUDA_OXIDE_CUTE_PYTHON, example .venv, cute-rs-preview/.venv, then python3.\nB=1, D=64, FP16, softmax scale=0.125, 64x64 tiles, 128 threads.\nEvery output is checked against a CPU FP64 oracle before timing.\n--input-scale scales Q/K only; --zero-query exercises uniform softmax.\n--iters 0 verifies two launches without timing.\n{mode}\nStatic shapes require Sq=Sk=1024 and H=8; runtime is the default.\nKV length must be a positive multiple of 64; causal mode requires Sq=Sk.\nDevice ordinal is after CUDA_VISIBLE_DEVICES remapping.",
                        env!("CARGO_PKG_NAME"),
                        1024,
                    );
                    std::process::exit(0);
                }
                "--rust-only" => args.rust_only = true,
                "--all-cases" => args.all_cases = true,
                "--causal" => args.shape.causal = true,
                "--noncausal" => args.shape.causal = false,
                "--zero-query" => args.zero_query = true,
                _ => {
                    let value = cli
                        .next()
                        .ok_or_else(|| format!("missing value for {key}"))?;
                    match key.as_str() {
                        "--shape-mode" => {
                            args.shape_mode = match value.as_str() {
                                "runtime" => ShapeMode::Runtime,
                                "static" => ShapeMode::Static,
                                _ => return Err("--shape-mode must be runtime or static".into()),
                            }
                        }
                        "--seq-q" => args.shape.seq_q = value.parse()?,
                        "--seq-k" => args.shape.seq_k = value.parse()?,
                        "--heads" => args.shape.heads = value.parse()?,
                        "--input-scale" => args.input_scale = value.parse()?,
                        "--device" => args.device = value.parse()?,
                        "--warmup" => args.warmup = value.parse()?,
                        "--iters" => args.iters = value.parse()?,
                        "--python" => args.python = Some(value.into()),
                        "--output" => args.output = Some(value.into()),
                        "--json" => args.json = Some(value.into()),
                        _ => return Err(format!("unknown argument {key}").into()),
                    }
                }
            }
        }
        crate::comparison::validate_mode(
            args.rust_only,
            args.python.as_ref(),
            args.output.as_ref(),
        )?;
        if args.all_cases && args.rust_only {
            return Err("--all-cases cannot be combined with --rust-only".into());
        }
        if args.all_cases
            && (args.shape.seq_q != 1024 || args.shape.seq_k != 1024 || args.shape.heads != 8)
        {
            return Err("--all-cases requires Sq=Sk=1024 and H=8".into());
        }
        args.shape.validate()?;
        if args.shape_mode == ShapeMode::Static
            && (args.shape.seq_q != 1024 || args.shape.seq_k != 1024 || args.shape.heads != 8)
        {
            return Err("--shape-mode static requires Sq=Sk=1024 and H=8".into());
        }
        if ![1, 4, 16].contains(&args.input_scale) {
            return Err("--input-scale must be 1, 4, or 16".into());
        }
        Ok(args)
    }
}

/// Dispatch the comparison process or a standalone prefill run.
pub fn run() -> Result<()> {
    let args = Args::parse()?;
    if !args.rust_only {
        return crate::comparison::run(
            "prefill",
            args.python.as_ref(),
            args.all_cases,
            args.shape,
            args.input_scale,
            args.zero_query,
            args.device,
            args.warmup,
            args.iters,
            "--shape-mode",
            if args.shape_mode == ShapeMode::Static {
                "static"
            } else {
                "runtime"
            },
            args.json.as_ref(),
        );
    }
    if args.shape.causal {
        run_mask::<true>(args)
    } else {
        run_mask::<false>(args)
    }
}

/// Select runtime or static dimensions for the chosen causal-mask specialization.
fn run_mask<const CAUSAL: bool>(args: Args) -> Result<()> {
    match args.shape_mode {
        ShapeMode::Runtime => run_shape::<CAUSAL, 0, 0, 0>(args),
        ShapeMode::Static => run_shape::<CAUSAL, 1024, 1024, 8>(args),
    }
}

/// Verify the kernel and replay, collect CUDA-event samples, and write the report.
fn run_shape<const CAUSAL: bool, const SQ: u32, const SK: u32, const HEADS: u32>(
    args: Args,
) -> Result<()> {
    let shape = args.shape;
    let (q, k, v) = make_inputs(shape, args.input_scale, args.zero_query);
    let expected = reference(shape, &q, &k, &v);
    let ctx = CudaContext::new(args.device)?;
    let capability = ctx.compute_capability()?;
    if capability.0 < 12 {
        return Err("this Rust specialization requires SM120 or newer".into());
    }
    let stream = ctx.default_stream();
    // All AOT mask/shape specializations live in this package's CUTLASS CUBIN.
    let image = cuda_host::load_embedded_module(&ctx, env!("CARGO_PKG_NAME"))?;
    // SAFETY: this bundle contains the typed kernels and their launch contracts.
    let module = unsafe { kernel::kernels::from_module(image) }?;
    let q_dev = DeviceBuffer::from_host(&stream, &q)?;
    let k_dev = DeviceBuffer::from_host(&stream, &k)?;
    let v_dev = DeviceBuffer::from_host(&stream, &v)?;
    let poison = vec![f16::NAN; shape.q_elements()];
    let mut output = DeviceBuffer::from_host(&stream, &poison)?;
    let blocks = (shape.seq_q.div_ceil(64) * shape.heads) as u32;
    let launch = module.prepare_fmha_prefill::<CAUSAL, SQ, SK, HEADS>(LaunchConfig1D::new(
        blocks,
        THREADS,
        SHARED_BYTES,
    ))?;
    let registers = launch.function().num_registers()?;
    let run = |output: &mut DeviceBuffer<f16>| -> Result<()> {
        // Q/K/V have the checked contiguous BSHD extents; each CTA
        // owns one disjoint 64-row/head output tile. All allocations remain
        // live until synchronization, and the launch reserves all shared data.
        module.fmha_prefill::<CAUSAL, SQ, SK, HEADS>(
            &stream,
            &launch,
            &q_dev,
            &k_dev,
            &v_dev,
            output,
            shape.seq_q as u32,
            shape.seq_k as u32,
            shape.heads as u32,
        )?;
        Ok(())
    };
    run(&mut output)?;
    let checked = output.to_host_vec(&stream)?;
    let validation = verify(&checked, &expected)?;
    let checked_bits: Vec<u16> = checked.iter().map(|x| x.to_bits()).collect();
    // Re-poison before replay so incomplete writes cannot preserve a pass.
    output.copy_from_host(&stream, &poison)?;
    run(&mut output)?;
    let replay = output.to_host_vec(&stream)?;
    verify(&replay, &expected)?;
    if !replay
        .iter()
        .map(|x| x.to_bits())
        .eq(checked_bits.iter().copied())
    {
        return Err("output changed across identical launches".into());
    }
    let warmup = if args.iters == 0 { 0 } else { args.warmup };
    for _ in 0..warmup {
        run(&mut output)?;
    }
    stream.synchronize()?;
    let events = (0..args.iters)
        .map(|_| {
            Ok((
                ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut samples = Vec::with_capacity(args.iters);
    for (start, end) in &events {
        start.record(&stream)?;
        run(&mut output)?;
        end.record(&stream)?;
        end.synchronize()?;
        let elapsed = f64::from(start.elapsed_ms(end)?) * 1000.0;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err(format!("invalid event timing: {elapsed}").into());
        }
        samples.push(elapsed);
    }
    let after = output.to_host_vec(&stream)?;
    verify(&after, &expected)?;
    if !after
        .iter()
        .map(|x| x.to_bits())
        .eq(checked_bits.iter().copied())
    {
        return Err("output changed during timing".into());
    }
    if let Some(path) = args.output {
        let bytes: Vec<u8> = after
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        None
    } else {
        Some((sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5)
    };
    let report = json!({
        "implementation": "cute-rs", "workload": "prefill-fmha", "shape": [1, shape.seq_q, shape.seq_k, shape.heads, HEAD_DIM],
        "causal": shape.causal, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "input_hashes": {"q": format!("{:016x}", bits_hash(&q)), "k": format!("{:016x}", bits_hash(&k)), "v": format!("{:016x}", bits_hash(&v))},
        "output_hash": format!("{:016x}", bits_hash(&after)), "verification": "cpu-pass",
        "validation": {"elements": validation.elements, "max_abs_error": validation.max_abs_error,
            "max_tolerance_ratio": validation.max_tolerance_ratio, "atol": fixture::ATOL, "rtol": fixture::RTOL,
            "oracle": "cpu-fp64-stable-softmax"},
        "replay_bitwise_equal": true, "metric_value": median, "metric_unit": "us", "samples_us": samples,
        "provenance": {"gpu": ctx.device_name()?, "compute_capability": [capability.0, capability.1],
            "device": args.device, "tool": "cuda-events-single-kernel", "timing_mode": "direct",
            "warmup": warmup, "iters": args.iters, "grid": [blocks, 1, 1], "threads": THREADS, "shared_memory_bytes": SHARED_BYTES,
            "dtype": "float16", "registers_per_thread": registers,
            "causal_specialization": "static",
            "shape_specialization": if args.shape_mode == ShapeMode::Static { "static" } else { "runtime" }}
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.json {
        std::fs::write(path, format!("{rendered}\n"))?;
    }
    println!("{rendered}");
    Ok(())
}

/// Tests for prefill CLI modes and exact static-shape dispatch.
#[cfg(test)]
mod tests {
    use super::*;

    /// Parse borrowed CLI words without reading process arguments.
    fn parse(values: &[&str]) -> Result<Args> {
        Args::parse_from(values.iter().map(|s| s.to_string()))
    }

    /// Check that raw output is standalone-only and Python selection is comparison-only.
    #[test]
    fn comparison_cli_requires_standalone_mode_for_output() {
        assert!(parse(&["--python", "/tmp/python"]).is_ok());
        assert!(parse(&["--output", "o.f16"]).is_err());
        assert!(parse(&["--rust-only", "--output", "o.f16"]).is_ok());
        assert!(parse(&["--rust-only", "--python", "/tmp/python"]).is_err());
    }

    /// Keep runtime shape selection as the default, including asymmetric noncausal shapes.
    #[test]
    fn runtime_shapes_remain_the_default() {
        assert_eq!(parse(&[]).unwrap().shape_mode, ShapeMode::Runtime);
        assert!(parse(&["--seq-q", "64", "--seq-k", "128", "--noncausal"]).is_ok());
    }

    /// Accept either mask and shape mode when requesting the full comparison matrix.
    #[test]
    fn all_cases_accepts_both_selected_masks_and_shape_modes() {
        for mode in ["runtime", "static"] {
            for mask in ["--causal", "--noncausal"] {
                assert!(parse(&["--all-cases", "--shape-mode", mode, mask]).is_ok());
            }
        }
    }

    /// Reject standalone-only options in a multi-case comparison.
    #[test]
    fn all_cases_rejects_standalone_execution_and_output() {
        for options in [
            vec!["--all-cases", "--rust-only"],
            vec!["--rust-only", "--all-cases"],
            vec!["--all-cases", "--output", "o.f16"],
        ] {
            assert!(parse(&options).is_err());
        }
    }

    /// Require the exact static shape for the runtime-versus-static comparison matrix.
    #[test]
    fn all_cases_requires_dimensions_supported_by_static_kernels() {
        for (dimension, value) in [("--seq-q", "64"), ("--seq-k", "64"), ("--heads", "3")] {
            assert!(parse(&["--all-cases", "--noncausal", dimension, value]).is_err());
        }
    }

    /// Check static dispatch for both causal and noncausal masks.
    #[test]
    fn static_dispatch_accepts_both_masks_for_its_exact_shape() {
        for mask in ["--causal", "--noncausal"] {
            let args = parse(&["--shape-mode", "static", mask]).unwrap();
            assert_eq!(args.shape_mode, ShapeMode::Static);
        }
    }

    /// Reject static shapes with a mismatched query length, KV length, or head count.
    #[test]
    fn static_dispatch_rejects_every_mismatched_dimension() {
        for (dimension, value) in [("--seq-q", "64"), ("--seq-k", "64"), ("--heads", "3")] {
            assert!(parse(&["--shape-mode", "static", "--noncausal", dimension, value]).is_err());
        }
        assert!(parse(&["--shape-mode", "unknown"]).is_err());
    }
}

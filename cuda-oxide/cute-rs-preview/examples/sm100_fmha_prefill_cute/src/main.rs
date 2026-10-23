/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 FP16 prefill comparison runner and file-based CUDA adapter.
//! Adapter mode validates against an FP32 oracle before collecting CUDA-event timings.

#![feature(f16)]

mod kernel;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::attention::sm100::{SmemLayout, create_tensor_map_tiled};
use serde_json::json;
use std::{error::Error, path::PathBuf};

/// Fallible host operation carrying CLI, file, Python, or CUDA errors.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Validated comparison options and optional file-based adapter inputs.
struct Args {
    /// Batch, query length, key length, query heads, and KV heads; head dimension is 128.
    shape: [u32; 5],
    /// Enable top-left causal masking instead of dense attention.
    causal: bool,
    /// Untimed launches before each timing window.
    warmup: usize,
    /// Timed launches per CUDA-event window; must be nonzero.
    iters: usize,
    /// Number of independent timing windows; must be nonzero.
    windows: usize,
    /// Validate output without collecting timing samples.
    check_only: bool,
    /// Adapter directory containing little-endian q.f16, k.f16, and v.f16 files.
    input_dir: Option<PathBuf>,
    /// Adapter output path for the validated little-endian FP16 tensor.
    output: Option<PathBuf>,
    /// Adapter input path for the little-endian FP32 reference tensor.
    oracle: Option<PathBuf>,
    /// Optional destination for the comparison or adapter report.
    json: Option<PathBuf>,
    /// Absolute error tolerance in [0, 0.1], combined with a fixed 1e-5 relative tolerance.
    atol: f32,
}

impl Args {
    /// Parse CLI options and reject incompatible heads, overflowing shapes, or invalid timing/tolerance values.
    fn parse() -> Result<Self> {
        let mut out = Self {
            shape: [1, 1024, 1024, 8, 8],
            causal: false,
            warmup: 10,
            iters: 100,
            windows: 3,
            check_only: false,
            input_dir: None,
            output: None,
            oracle: None,
            json: None,
            atol: 0.1,
        };
        let mut cli = std::env::args().skip(1);
        while let Some(key) = cli.next() {
            match key.as_str() {
                "--help" | "-h" => {
                    println!(
                        "sm100_fmha_prefill_cute [--shape B,Sq,Sk,Hq,Hkv] [--causal]\n  [--warmup 10] [--iters 100] [--windows 3] [--check-only] [--json FILE]\nDefault: compare the Rust cubin and pinned CuTeDSL on identical FP16 inputs.\nPrints CuTeDSL and CuTe-RS timings; --json saves the full report.\nD=128, top-left causal or dense, MHA/GQA, persistent schedule, SM100 required.\nAdapter interface: --input-dir DIR --oracle oracle.f32 --output o.f16 [--atol 0.1]\nPython: CUDA_OXIDE_CUTE_PYTHON, example .venv, cute-rs-preview/.venv, then python3."
                    );
                    std::process::exit(0);
                }
                "--causal" => out.causal = true,
                "--check-only" => out.check_only = true,
                _ => {
                    let value = cli
                        .next()
                        .ok_or_else(|| format!("missing value for {key}"))?;
                    match key.as_str() {
                        "--shape" => {
                            let values = value
                                .split(',')
                                .map(str::parse::<u32>)
                                .collect::<std::result::Result<Vec<_>, _>>()?;
                            out.shape = values
                                .try_into()
                                .map_err(|_| "--shape requires B,Sq,Sk,Hq,Hkv")?;
                        }
                        "--warmup" => out.warmup = value.parse()?,
                        "--iters" => out.iters = value.parse()?,
                        "--windows" => out.windows = value.parse()?,
                        "--input-dir" => out.input_dir = Some(value.into()),
                        "--output" => out.output = Some(value.into()),
                        "--oracle" => out.oracle = Some(value.into()),
                        "--json" => out.json = Some(value.into()),
                        "--atol" => out.atol = value.parse()?,
                        _ => return Err(format!("unsupported argument {key}").into()),
                    }
                }
            }
        }
        let [b, q, k, h, hk] = out.shape;
        if out
            .shape
            .iter()
            .any(|&n| n == 0 || n >= i32::MAX as u32 - 256)
            || h % hk != 0
        {
            return Err("require positive i32 dimensions, Hq divisible by Hkv, and padded coordinates fitting i32".into());
        }
        let tiles = u64::from(b)
            .checked_mul(u64::from(h))
            .and_then(|value| value.checked_mul(u64::from(q.div_ceil(256))))
            .ok_or("query tile count overflows u64")?;
        if tiles > i32::MAX as u64 / 2 {
            return Err("persistent work grid exceeds supported range".into());
        }
        for dims in [[b, q, h, 128], [b, k, hk, 128]] {
            let count = dims
                .into_iter()
                .try_fold(1u64, |product, value| product.checked_mul(u64::from(value)))
                .ok_or("tensor element count overflow")?;
            if count > isize::MAX as u64 / 4 {
                return Err("tensor byte size exceeds addressable memory".into());
            }
        }
        if out.iters == 0
            || out.windows == 0
            || !out.atol.is_finite()
            || out.atol < 0.0
            || out.atol > 0.1
        {
            return Err("iters/windows must be positive and atol in [0,0.1]".into());
        }
        if out.input_dir.is_some() && (out.output.is_none() || out.oracle.is_none()) {
            return Err("adapter mode requires --output and --oracle; validation precedes every timing interval".into());
        }
        Ok(out)
    }
}

/// Run the pinned Python comparison harness, forwarding shape and timing options.
fn compare(args: &Args) -> Result<()> {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var_os("CUDA_OXIDE_CUTE_PYTHON")
        .map(PathBuf::from)
        .or_else(|| {
            [
                directory.join(".venv/bin/python"),
                directory.join("../../.venv/bin/python"),
            ]
            .into_iter()
            .find(|path| path.is_file())
        })
        .unwrap_or_else(|| PathBuf::from("python3"));
    let shape = args.shape.map(|n| n.to_string()).join(",");
    let mut command = std::process::Command::new(&python);
    command
        .arg(directory.join("compare.py"))
        .arg("--rust-bin")
        .arg(std::env::current_exe()?)
        .args([
            "--shape",
            &shape,
            "--warmup",
            &args.warmup.to_string(),
            "--iters",
            &args.iters.to_string(),
            "--windows",
            &args.windows.to_string(),
        ]);
    if args.causal {
        command.arg("--causal");
    }
    if args.check_only {
        command.arg("--check-only");
    }
    if let Some(path) = &args.json {
        command.arg("--json").arg(path);
    }
    let status = command.status().map_err(|error| {
        format!(
            "could not start comparison with {}: {error}; create cute-rs-preview/.venv as the README shows, or set CUDA_OXIDE_CUTE_PYTHON",
            python.display()
        )
    })?;
    if !status.success() {
        return Err(format!("comparison exited with {status}").into());
    }
    Ok(())
}

/// Read exactly `count` little-endian FP16 elements, preserving their raw bits.
fn load_half(path: &std::path::Path, count: usize) -> Result<Vec<u16>> {
    let data = std::fs::read(path)?;
    if data.len() != count.checked_mul(2).ok_or("FP16 byte count overflow")? {
        return Err(format!("{} has incorrect FP16 byte count", path.display()).into());
    }
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect())
}

/// Pack one 128-byte TMA descriptor per batch/head for a BSHD allocation with D=128.
fn descriptors<L: SmemLayout>(base: u64, batch: u32, sequence: u32, heads: u32) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(batch as usize * heads as usize * 128);
    for b in 0..batch {
        for h in 0..heads {
            let offset =
                (u64::from(b) * u64::from(sequence) * u64::from(heads) + u64::from(h)) * 128 * 2;
            let desc = create_tensor_map_tiled::<L>(
                (base + offset) as *mut core::ffi::c_void,
                u64::from(sequence),
                128,
                u64::from(heads) * 128,
            )?;
            result.extend_from_slice(&desc.bytes);
        }
    }
    Ok(result)
}

/// Check finite FP16 outputs against FP32 values using `atol + 1e-5 * abs(expected)`.
/// Return maximum absolute error and root-mean-square error, or reject any mismatch.
fn verify(output: &[u16], oracle: &[f32], atol: f32) -> Result<(f32, f64)> {
    if output.len() != oracle.len() {
        return Err("oracle and output element counts differ".into());
    }
    let mut maximum = 0.0f32;
    let mut square = 0.0f64;
    let mut failures = 0;
    for (index, (&bits, &expected)) in output.iter().zip(oracle).enumerate() {
        let actual = f16::from_bits(bits) as f32;
        if !actual.is_finite() || !expected.is_finite() {
            return Err(
                format!("nonfinite output/oracle at element {index}; timing forbidden").into(),
            );
        }
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        square += f64::from(error).powi(2);
        if error > atol + 1e-5 * expected.abs() {
            if failures < 5 {
                eprintln!("output[{index}]: actual={actual}, expected={expected}");
            }
            failures += 1;
        }
    }
    if failures != 0 {
        return Err(format!(
            "{failures}/{} outputs fail oracle, maximum error={maximum}; timing forbidden",
            output.len()
        )
        .into());
    }
    Ok((maximum, (square / output.len() as f64).sqrt()))
}

/// Run the comparison harness, or validate, time, and save the SM100 adapter’s output.
fn main() -> Result<()> {
    let args = Args::parse()?;
    let Some(directory) = &args.input_dir else {
        return compare(&args);
    };
    let [b, q, k, h, hk] = args.shape;
    let qcount = b as usize * q as usize * h as usize * 128;
    let kvcount = b as usize * k as usize * hk as usize * 128;
    let qhost = load_half(&directory.join("q.f16"), qcount)?;
    let khost = load_half(&directory.join("k.f16"), kvcount)?;
    let vhost = load_half(&directory.join("v.f16"), kvcount)?;
    let raw_oracle = std::fs::read(args.oracle.as_ref().ok_or("missing oracle")?)?;
    if raw_oracle.len() != qcount * 4 {
        return Err("oracle has incorrect FP32 byte count".into());
    }
    let oracle: Vec<_> = raw_oracle
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let ctx = CudaContext::new(0)?;
    if ctx.compute_capability()? != (10, 0) {
        return Err("this matched FP16 example requires SM100".into());
    }
    let stream = ctx.new_stream()?;
    let qdev = DeviceBuffer::from_host(&stream, &qhost)?;
    let kdev = DeviceBuffer::from_host(&stream, &khost)?;
    let vdev = DeviceBuffer::from_host(&stream, &vhost)?;
    let output = DeviceBuffer::from_host(&stream, &vec![f16::NAN.to_bits(); qcount])?;
    let qmaps = DeviceBuffer::from_host(
        &stream,
        &descriptors::<kernel::QLayout>(qdev.cu_deviceptr(), b, q, h)?,
    )?;
    let kmaps = DeviceBuffer::from_host(
        &stream,
        &descriptors::<kernel::KLayout>(kdev.cu_deviceptr(), b, k, hk)?,
    )?;
    let vmaps = DeviceBuffer::from_host(
        &stream,
        &descriptors::<kernel::VLayout>(vdev.cu_deviceptr(), b, k, hk)?,
    )?;
    let omaps = DeviceBuffer::from_host(
        &stream,
        &descriptors::<kernel::OLayout>(output.cu_deviceptr(), b, q, h)?,
    )?;
    // SAFETY: this process owns all descriptor arrays and their referenced
    // buffers, retained until stream synchronization after the final launch.
    let module = unsafe { kernel::kernels::load(&ctx) }?;
    let total = b * h * q.div_ceil(256);
    let blocks = ctx.multiprocessor_count()?.min(total);
    let prepared = module.prepare_sm100_fmha_prefill_cute_rs(LaunchConfig1D::new(
        blocks,
        kernel::THREADS,
        kernel::DYNAMIC_SMEM_BYTES as u32,
    ))?;
    let registers = prepared.function().num_registers()?;
    let run = || -> Result<()> {
        #[allow(unused_unsafe)] // Required by device-enabled launch expansion.
        unsafe {
            module.sm100_fmha_prefill_cute_rs(
                &stream,
                &prepared,
                qmaps.cu_deviceptr() as *const _,
                kmaps.cu_deviceptr() as *const _,
                vmaps.cu_deviceptr() as *const _,
                omaps.cu_deviceptr() as *const _,
                u64::from(b),
                u64::from(q),
                u64::from(k),
                u64::from(h),
                u64::from(hk),
                u64::from(args.causal),
            )?;
        }
        Ok(())
    };
    run()?;
    let initial = output.to_host_vec(&stream)?;
    let (max_abs, rms) = verify(&initial, &oracle, args.atol)?;
    let mut samples = Vec::new();
    if !args.check_only {
        for _ in 0..args.windows {
            for _ in 0..args.warmup {
                run()?;
            }
            stream.synchronize()?;
            let start = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            let end = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            start.record(&stream)?;
            for _ in 0..args.iters {
                run()?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            samples.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / args.iters as f64);
        }
    }
    let final_output = output.to_host_vec(&stream)?;
    verify(&final_output, &oracle, args.atol)?;
    if initial != final_output {
        return Err("identical repeated launches changed output".into());
    }
    let output_bytes: Vec<_> = final_output.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(
        args.output.as_ref().ok_or("missing output path")?,
        output_bytes,
    )?;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        None
    } else {
        Some((sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5)
    };
    let report = json!({"implementation":"sm100_fmha_prefill_cute_rs", "shape":args.shape,
        "causal":args.causal,"head_dim":128,"validation":{"max_abs":max_abs,"rms":rms,"atol":args.atol,"rtol":1e-5},
        "samples_us":samples,"median_us":median,"timing":"direct-cuda-events","cache":"hot",
        "warmup":args.warmup,"iterations":args.iters,"windows":args.windows,
        "gpu":ctx.device_name()?,"blocks":blocks,"threads":kernel::THREADS,
        "shared_memory_bytes":kernel::DYNAMIC_SMEM_BYTES,"registers_per_thread":registers,
        "compiler_optimization_level":option_env!("CUDA_OXIDE_CUTLASS_OPT_LEVEL")
            .and_then(|value| value.parse::<u32>().ok()).unwrap_or(3),
        "compiler":"CUTLASS 4.7.0", "target":"sm_100a",
        "source_commit":"dcf215af68a2d08d305076c152a06f201728cd53"});
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.json {
        std::fs::write(path, &rendered)?;
    }
    println!("{rendered}");
    Ok(())
}

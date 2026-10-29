/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked host launch and complete partial-plus-merge timing for decode.

use crate::{
    config::{
        Args, KV_CHUNK, LaunchPlan, MERGE_THREADS, PARTIAL_SHARED_BYTES, PARTIAL_THREADS,
        SCRATCH_STRIDE, validate_kernel_shape,
    },
    fixture::{self, HEAD_DIM, bits_hash, make_inputs, reference, verify},
    kernel,
};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, sys};
use serde_json::json;
use std::error::Error;

/// Host-side errors from CUDA, file I/O, validation, or the comparison process.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Parse options and dispatch the comparison or selected standalone kernel shape.
pub fn run() -> Result<()> {
    let Some(args) = Args::parse_from(std::env::args().skip(1))? else {
        println!(
            "{} [--seq-q 1] [--seq-k 1024] [--heads 8] [--noncausal]\n  \
             [--input-scale 1|4|16] [--zero-query] [--device 0]\n  \
             [--specialization auto|runtime|static]\n  \
             [--warmup 100] [--iters 31] [--json result.json] [--python PATH] [--rust-only | --all-cases]\nDefault: verify and compare CuTeDSL and cute-rs-preview; prints two performance lines.\n--all-cases prints runtime and static comparisons at the selected noncausal shape.\nIt overrides --specialization, requires a listed static shape, and cannot be combined with --rust-only.\nLive timings use CUDA events; archived Nsight measurements are a separate metric.\n--json saves the combined report; --rust-only saves standalone JSON and permits --output o.f16.\nPython selection: --python, CUDA_OXIDE_CUTE_PYTHON, example .venv, cute-rs-preview/.venv, then python3.\n\
             B=1, Sq=1, D=64, FP16, softmax scale=0.125, noncausal only.\n\
             One 128-thread CTA per head/64-key split; one 32-thread merge CTA per head.\n\
             KV length may be any positive value. Partial state and merge arithmetic are FP32.\n\
             Auto selects a static kernel for (Sk,H)=(64,8),(65,3),(1024,8),(4097,8);\n\
             other shapes use the runtime kernel. Static mode requires a listed shape.\n\
             Every output is checked against a CPU FP64 oracle before timing.\n\
             --input-scale scales Q/K only; --zero-query exercises uniform softmax.\n\
             --iters 0 verifies two partial-plus-merge invocations without timing.\n\
             CUDA events measure the full partial-plus-merge sequence with reused buffers.\n\
             Device ordinal is after CUDA_VISIBLE_DEVICES remapping.",
            env!("CARGO_PKG_NAME"),
        );
        return Ok(());
    };
    if !args.rust_only {
        return crate::comparison::run(
            "decode",
            args.python.as_ref(),
            args.all_cases,
            args.shape,
            args.input_scale,
            args.zero_query,
            args.device,
            args.warmup,
            args.iters,
            "--specialization",
            args.specialization.as_str(),
            args.json.as_ref(),
        );
    }
    match args.kernel_shape()? {
        (0, 0) => run_shape::<0, 0>(args),
        (64, 8) => run_shape::<64, 8>(args),
        (65, 3) => run_shape::<65, 3>(args),
        (1024, 8) => run_shape::<1024, 8>(args),
        (4097, 8) => run_shape::<4097, 8>(args),
        _ => Err("unsupported kernel specialization dispatch".into()),
    }
}

/// Verify both decode launches, time their combined work, and write the report.
fn run_shape<const SEQ_K: u32, const HEADS: u32>(args: Args) -> Result<()> {
    let shape = args.shape;
    validate_kernel_shape::<SEQ_K, HEADS>(shape)?;
    let plan = LaunchPlan::new(shape)?;
    let (q, k, v) = make_inputs(shape, args.input_scale, args.zero_query);
    let expected = reference(shape, &q, &k, &v);
    let ctx = CudaContext::new(args.device)?;
    let capability = ctx.compute_capability()?;
    if capability.0 < 12 {
        return Err("this Rust specialization requires SM120 or newer".into());
    }
    let stream = ctx.default_stream();
    // All AOT specializations live in this executable's one CUTLASS cubin.
    // The generic cuda_module loader merges PTX bundles and skips cubins, so
    // load the named bundle directly and bind its matching typed launch API.
    let image = cuda_host::load_embedded_module(&ctx, env!("CARGO_PKG_NAME"))?;
    // SAFETY: this package's bundle contains the kernels and launch contracts
    // declared by kernel::kernels, including every dispatched specialization.
    let module = unsafe { kernel::kernels::from_module(image) }?;
    let q_dev = DeviceBuffer::from_host(&stream, &q)?;
    let k_dev = DeviceBuffer::from_host(&stream, &k)?;
    let v_dev = DeviceBuffer::from_host(&stream, &v)?;
    let output_poison = vec![f16::NAN; shape.q_elements()];
    let scratch_poison = vec![f32::NAN; plan.scratch_elements];
    let mut output = DeviceBuffer::from_host(&stream, &output_poison)?;
    let mut scratch = DeviceBuffer::from_host(&stream, &scratch_poison)?;
    let partial = module.prepare_fmha_decode_partial::<SEQ_K, HEADS>(LaunchConfig1D::new(
        plan.partial_blocks,
        PARTIAL_THREADS,
        PARTIAL_SHARED_BYTES,
    ))?;
    let merge = module.prepare_fmha_decode_merge::<SEQ_K, HEADS>(LaunchConfig1D::new(
        shape.heads as u32,
        MERGE_THREADS,
        0,
    ))?;
    let partial_registers = partial.function().num_registers()?;
    let merge_registers = merge.function().num_registers()?;
    let invoke = |scratch: &mut DeviceBuffer<f32>, output: &mut DeviceBuffer<f16>| -> Result<()> {
        // Q/K/V have the checked contiguous BSHD extents. Each partial CTA
        // owns one disjoint scratch slot; each merge CTA owns one head's output.
        // The same stream orders merge reads after all partial writes. Buffers
        // remain live until synchronization, with both launch contracts met.
        module.fmha_decode_partial::<SEQ_K, HEADS>(
            &stream,
            &partial,
            &q_dev,
            &k_dev,
            &v_dev,
            scratch,
            shape.seq_k as u32,
            shape.heads as u32,
        )?;
        module.fmha_decode_merge::<SEQ_K, HEADS>(
            &stream,
            &merge,
            scratch,
            output,
            shape.heads as u32,
            plan.splits,
        )?;
        Ok(())
    };

    invoke(&mut scratch, &mut output)?;
    let checked = output.to_host_vec(&stream)?;
    let validation = verify(&checked, &expected)?;
    let checked_bits: Vec<u16> = checked.iter().map(|x| x.to_bits()).collect();
    // Poison the producer's intermediate state as well as the final output:
    // otherwise missing partial writes could reuse valid data during replay.
    scratch.copy_from_host(&stream, &scratch_poison)?;
    output.copy_from_host(&stream, &output_poison)?;
    invoke(&mut scratch, &mut output)?;
    let replay = output.to_host_vec(&stream)?;
    verify(&replay, &expected)?;
    if !replay
        .iter()
        .map(|x| x.to_bits())
        .eq(checked_bits.iter().copied())
    {
        return Err("output changed across identical partial-plus-merge invocations".into());
    }

    let warmup = if args.iters == 0 { 0 } else { args.warmup };
    for _ in 0..warmup {
        invoke(&mut scratch, &mut output)?;
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
        invoke(&mut scratch, &mut output)?;
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
        "implementation": "cute-rs", "workload": "fmha-decode",
        "shape": [1, shape.seq_q, shape.seq_k, shape.heads, HEAD_DIM],
        "causal": shape.causal, "input_scale": args.input_scale, "zero_query": args.zero_query,
        "input_hashes": {"q": format!("{:016x}", bits_hash(&q)),
            "k": format!("{:016x}", bits_hash(&k)), "v": format!("{:016x}", bits_hash(&v))},
        "output_hash": format!("{:016x}", bits_hash(&after)), "verification": "cpu-pass",
        "validation": {"elements": validation.elements, "max_abs_error": validation.max_abs_error,
            "max_tolerance_ratio": validation.max_tolerance_ratio, "atol": fixture::ATOL, "rtol": fixture::RTOL,
            "oracle": "cpu-fp64-stable-softmax"},
        "replay_bitwise_equal": true, "metric_value": median, "metric_unit": "us", "samples_us": samples,
        "provenance": {
            "gpu": ctx.device_name()?, "compute_capability": [capability.0, capability.1],
            "device": args.device, "tool": "cuda-events-kernel-sequence", "timing_mode": "direct",
            "warmup": warmup, "iters": args.iters, "dtype": "float16", "accumulator_dtype": "float32",
            "algorithm": "single-query-split-kv", "kv_chunk": KV_CHUNK, "split_count": plan.splits,
            "shape_specialization": if SEQ_K == 0 { "runtime" } else { "static" },
            "specialization_request": args.specialization.as_str(),
            "kernel_count": 2, "kernel_sequence": ["fmha_decode_partial", "fmha_decode_merge"],
            "scratch": {"shape": [shape.heads, plan.splits as usize, SCRATCH_STRIDE],
                "dtype": "float32", "bytes": plan.scratch_elements * size_of::<f32>(),
                "layout": "maximum, denominator, 64 unnormalized numerator components"},
            "kernels": [
                {"name": "fmha_decode_partial", "grid": [plan.partial_blocks, 1, 1],
                 "threads": PARTIAL_THREADS, "shared_memory_bytes": PARTIAL_SHARED_BYTES,
                 "registers_per_thread": partial_registers},
                {"name": "fmha_decode_merge", "grid": [shape.heads, 1, 1],
                 "threads": MERGE_THREADS, "shared_memory_bytes": 0,
                 "registers_per_thread": merge_registers}
            ]
        }
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.json {
        std::fs::write(path, format!("{rendered}\n"))?;
    }
    println!("{rendered}");
    Ok(())
}

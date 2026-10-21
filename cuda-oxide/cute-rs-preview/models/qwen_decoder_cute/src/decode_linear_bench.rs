/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Event-timed projection microbenchmarks; excludes allocation and model orchestration.
use crate::{
    ffn_kernel::kernels,
    operator::Result,
    reference_linear::{ReductionPlan, fma_chain_blocks, reference_plan},
    reference_linear_kernel::kernels as reference,
};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::Bf16;
use std::sync::Arc;

/// Measure original, decode-specialized and reference-order projections for one- and two-row shapes.
///
/// Reports per-launch microsecond medians/minima/maxima from CUDA events with
/// warm weights, excluding allocation and upload. Checks the exact constant-input
/// result and drains pending work on enqueue errors. Returns CUDA or mismatch errors.
pub fn run(stream: Arc<CudaStream>) -> Result<()> {
    // SAFETY: this executable embeds the artifacts from the same kernel sources.
    let module = unsafe { kernels::load(stream.context())? };
    let reference_module = unsafe { reference::load(stream.context())? };
    /// Repeated launches in each event-timed sample.
    const ITERATIONS: usize = 128;
    /// Independent timing samples summarized for each implementation and shape.
    const SAMPLES: usize = 7;
    println!(
        "event-timed repeated projection launches; {SAMPLES} samples of {ITERATIONS}; warm weights; no end-to-end claim"
    );
    for (rows, width, outputs) in [
        (1, 1024, 1024),
        (1, 1024, 2048),
        (1, 1024, 3072),
        (1, 3072, 1024),
        (1, 2048, 1024),
        (1, 2048, 2048),
        (1, 2048, 3072),
        (1, 2048, 6144),
        (1, 6144, 2048),
        (2, 2048, 1024),
        (2, 3072, 1024),
        (2, 6144, 2048),
    ] {
        let plan = reference_plan(rows, width, outputs);
        let warp = |blocks: usize| LaunchConfig1D::new(blocks as u32, 32, 0);
        let fma16 =
            reference_module.prepare_fma16_linear_bf16(warp(fma_chain_blocks(16, outputs)))?;
        let fma32 =
            reference_module.prepare_fma32_linear_bf16(warp(fma_chain_blocks(32, outputs)))?;
        let split_k = reference_module.prepare_split_k_linear_bf16(warp(outputs.div_ceil(8)))?;
        let input = DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.25); rows * width])?;
        let weight =
            DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.125); outputs * width])?;
        let mut output =
            DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.0); rows * outputs])?;
        let config = LaunchConfig1D::new(outputs.div_ceil(8) as u32, 32, 0);
        let prepared = module.prepare_decode_linear_bf16(config)?;
        let original = cuda_core::simt::LaunchConfig {
            grid_dim: (outputs.div_ceil(32) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let start = stream
            .context()
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let end = stream
            .context()
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for variant in ["original", "decode", "reference"] {
            if variant == "reference" && plan == ReductionPlan::MmaChain {
                continue;
            }
            let mut enqueue = || -> Result<()> {
                if variant == "reference" {
                    match plan {
                        ReductionPlan::FmaChains { chains: 16 } => reference_module
                            .fma16_linear_bf16(
                                &stream,
                                &fma16,
                                &input,
                                &weight,
                                &mut output,
                                width,
                                outputs,
                                width,
                            )?,
                        ReductionPlan::FmaChains { .. } => reference_module.fma32_linear_bf16(
                            &stream,
                            &fma32,
                            &input,
                            &weight,
                            &mut output,
                            width,
                            outputs,
                            width,
                        )?,
                        ReductionPlan::MmaSplitK { slice_width } => reference_module
                            .split_k_linear_bf16(
                                &stream,
                                &split_k,
                                &input,
                                &weight,
                                &mut output,
                                rows,
                                width,
                                outputs,
                                width,
                                width,
                                slice_width,
                            )?,
                        ReductionPlan::MmaChain => unreachable!("skipped above"),
                    }
                } else if variant == "decode" {
                    module.decode_linear_bf16(
                        &stream,
                        &prepared,
                        &input,
                        &weight,
                        &mut output,
                        rows,
                        width,
                        outputs,
                        width,
                        width,
                    )?;
                } else {
                    // SAFETY: full warps, exact contiguous extents and an
                    // exclusively borrowed output with one owner per tile.
                    unsafe {
                        module.linear_bf16(
                            &stream,
                            original,
                            &input,
                            &weight,
                            &mut output,
                            rows,
                            width,
                            outputs,
                            width,
                            width,
                        )?;
                    }
                }
                Ok(())
            };
            // On any enqueue/event failure, drain the stream before the borrowed
            // buffers or module may drop. Timed batches have no internal waits.
            let measured = (|| -> Result<Vec<f64>> {
                for _ in 0..20 {
                    enqueue()?;
                }
                stream.synchronize()?;
                let mut samples = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    start.record(&stream)?;
                    for _ in 0..ITERATIONS {
                        enqueue()?;
                    }
                    end.record(&stream)?;
                    samples.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / ITERATIONS as f64);
                }
                Ok(samples)
            })();
            let completed = stream.synchronize();
            let mut samples = measured?;
            completed?;
            let actual = output.to_host_vec(&stream)?;
            let expected = Bf16::from_f32(width as f32 / 32.0).to_bits();
            if actual.iter().any(|value| value.to_bits() != expected) {
                return Err(
                    "projection microbenchmark output differs from exact constant-input result"
                        .into(),
                );
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{variant} rows={rows} width={width} outputs={outputs} plan={plan:?} median_us={:.6} min_us={:.6} max_us={:.6}",
                samples[SAMPLES / 2],
                samples[0],
                samples[SAMPLES - 1]
            );
        }
    }
    Ok(())
}

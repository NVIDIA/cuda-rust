/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Width-specialized RMSNorm dispatch and exact original-kernel regression checks.
use crate::{
    norm_kernel::{ROWS_PER_BLOCK, kernels},
    operator::Result,
};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig, sys};
use cute_rs::BFloat16;
use std::sync::Arc;

/// Enqueue normalization after checking its allocation and launch geometry.
///
/// Input/output are dense `[rows, cols]`; weight has at least `cols` elements.
/// Widths 128, 1024 and 2048 select specialized kernels; other widths use the
/// original kernel. Rejects invalid extents, epsilon, grid or buffer contexts
/// and propagates enqueue failures without waiting for completion.
///
/// # Safety
/// The caller must establish input/weight readiness, retain all buffers and the
/// module until stream completion (including errors/unwinding), and prohibit
/// concurrent writes to input/weight or any other access to output until then.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<BFloat16>,
    weight: &DeviceBuffer<BFloat16>,
    output: &mut DeviceBuffer<BFloat16>,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<()> {
    let config = checked_geometry(rows, cols, input.len(), weight.len(), output.len(), epsilon)?;
    for buffer in [input, weight, &*output] {
        if !Arc::ptr_eq(buffer.context(), stream.context()) {
            return Err("normalization buffers must share the stream context".into());
        }
    }
    let Some(config) = config else { return Ok(()) };
    // SAFETY: checked_geometry establishes complete warps, dense extents and
    // exclusive output ownership. The caller provides readiness and retention.
    unsafe {
        match cols {
            128 => module
                .rmsnorm_bf16_128(stream, config, input, weight, output, rows, cols, epsilon)?,
            1024 => module
                .rmsnorm_bf16_1024(stream, config, input, weight, output, rows, cols, epsilon)?,
            2048 => module
                .rmsnorm_bf16_2048(stream, config, input, weight, output, rows, cols, epsilon)?,
            _ => module.rmsnorm_bf16(stream, config, input, weight, output, rows, cols, epsilon)?,
        }
    }
    Ok(())
}

/// Validate dense RMSNorm storage lengths and a positive finite epsilon.
///
/// Lengths are in BF16 elements and may include guard suffixes. Returns one
/// warp per logical row, no launch for zero rows, or extent/grid errors.
fn checked_geometry(
    rows: usize,
    cols: usize,
    input: usize,
    weight: usize,
    output: usize,
    epsilon: f32,
) -> Result<Option<LaunchConfig>> {
    if cols == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("normalization width and finite epsilon must be positive".into());
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or("normalization extent overflow")?;
    if input < elements || output < elements || weight < cols {
        return Err("normalization allocation is shorter than its logical extent".into());
    }
    let blocks = rows.div_ceil(ROWS_PER_BLOCK);
    if blocks > i32::MAX as usize {
        return Err("normalization exceeds CUDA grid-x range".into());
    }
    Ok((rows != 0).then_some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: ((ROWS_PER_BLOCK * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Return a labeled error for unequal lengths or any BF16 storage-bit difference.
fn exact_bits(actual: &[BFloat16], expected: &[BFloat16], label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: normalization output extent mismatch").into());
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "{label}[{index}]: specialized bits {:04x} != original bits {:04x}",
                actual.to_bits(),
                expected.to_bits()
            )
            .into());
        }
    }
    Ok(())
}

/// Input distributions that expose reduction, exponent and rounding regressions.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Bounded deterministic values with variation across rows and columns.
    Regular,
    /// Signed values spanning a wide range of BF16 exponent fields.
    Varied,
    /// Small magnitudes stressing the epsilon-dominated normalization regime.
    Small,
    /// Large-magnitude values stressing FP32 square/reduction range.
    Large,
    /// Positive and negative zeros with varied sign bits.
    Zero,
    /// Quarter-row nonzeros making mean-square division hit an FP32 underflow tie.
    UnderflowTie,
}

/// Compare original and specialized kernels bit-for-bit for one input distribution.
///
/// Checks finite logical outputs and unchanged input/weight/output canaries;
/// returns the logical element count or a CUDA/numerical/guard error.
fn fixture(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
    rows: usize,
    cols: usize,
    epsilon: f32,
    pattern: Pattern,
) -> Result<usize> {
    /// Number of BF16 canaries appended to each fixture allocation.
    const GUARD: usize = 32;
    let poison = BFloat16::from_bits(0x7fc1);
    let len = rows * cols;
    let mut input = vec![poison; len + GUARD];
    let mut weight = vec![poison; cols + GUARD];
    let mut seed = 0x4212_3991u32;
    for (index, value) in input[..len].iter_mut().enumerate() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let sign = (seed >> 16) as u16 & 0x8000;
        let exponent = match pattern {
            Pattern::Varied => 80 + (seed >> 7) as u16 % 90,
            Pattern::Small => 70 + (seed >> 7) as u16 % 12,
            Pattern::Large => 171 + (seed >> 7) as u16 % 12,
            _ => 127,
        };
        *value = match pattern {
            Pattern::Regular => {
                BFloat16::from_f32(((index * 17 + index / cols * 11) % 131) as f32 / 32.0 - 2.0)
            }
            Pattern::Zero => BFloat16::from_bits(sign),
            // For widths divisible by four, sum(x*x)/cols is exactly half
            // the smallest FP32 subnormal. With epsilon=MIN_SUBNORMAL, the
            // original division rounds to zero before adding epsilon. Fusing
            // the scale+epsilon incorrectly rounds to two subnormal units.
            Pattern::UnderflowTie => {
                BFloat16::from_bits(if index % cols < cols / 4 { 53 << 7 } else { 0 })
            }
            _ => BFloat16::from_bits(sign | exponent << 7 | seed as u16 & 127),
        };
    }
    for (index, value) in weight[..cols].iter_mut().enumerate() {
        // Nontrivial weights expose the intermediate BF16 rounding boundary.
        *value = BFloat16::from_f32(((index * 19) % 79) as f32 / 32.0 - 1.125);
    }
    let input_dev = DeviceBuffer::from_host(stream, &input)?;
    let weight_dev = DeviceBuffer::from_host(stream, &weight)?;
    let mut original = DeviceBuffer::from_host(stream, &vec![poison; len + GUARD])?;
    let mut specialized = DeviceBuffer::from_host(stream, &vec![poison; len + GUARD])?;
    stream.context().synchronize()?;
    let launched = (|| -> Result<()> {
        if rows != 0 {
            let config = LaunchConfig {
                grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: full warp ownership, checked initialized extents, private
            // output buffers and completion below on either enqueue result.
            unsafe {
                module.rmsnorm_bf16(
                    stream,
                    config,
                    &input_dev,
                    &weight_dev,
                    &mut original,
                    rows,
                    cols,
                    epsilon,
                )?;
            }
        }
        unsafe {
            enqueue(
                module,
                stream,
                &input_dev,
                &weight_dev,
                &mut specialized,
                rows,
                cols,
                epsilon,
            )
        }
    })();
    let completed = stream.synchronize();
    launched?;
    completed?;
    let label = format!("RMSNorm rows={rows} cols={cols} epsilon={epsilon} {pattern:?}");
    let actual = specialized.to_host_vec(stream)?;
    let expected = original.to_host_vec(stream)?;
    exact_bits(&actual, &expected, &label)?;
    if actual[..len]
        .iter()
        .any(|value| !value.to_f32().is_finite())
    {
        return Err(format!("{label}: nonfinite output").into());
    }
    if actual[len..]
        .iter()
        .any(|value| value.to_bits() != poison.to_bits())
    {
        return Err(format!("{label}: output guard overwritten").into());
    }
    exact_bits(
        &input_dev.to_host_vec(stream)?,
        &input,
        "input/guard modified",
    )?;
    exact_bits(
        &weight_dev.to_host_vec(stream)?,
        &weight,
        "weight/guard modified",
    )?;
    Ok(len)
}

/// Exact old/new comparisons cover real widths, warp/block tails, exponent
/// stress, both model epsilons, and the generic width fallback.
pub fn validate(stream: Arc<CudaStream>) -> Result<()> {
    for (rows, cols, input, weight, output, epsilon) in [
        (1, 0, 128, 128, 128, 1e-6),
        (2, 128, 255, 128, 256, 1e-6),
        (2, 128, 256, 127, 256, 1e-6),
        (2, 128, 256, 128, 255, 1e-6),
        (usize::MAX, 128, usize::MAX, 128, usize::MAX, 1e-6),
        (
            i32::MAX as usize * 4 + 1,
            1,
            usize::MAX,
            1,
            usize::MAX,
            1e-6,
        ),
        (1, 128, 128, 128, 128, 0.0),
        (1, 128, 128, 128, 128, -1e-6),
        (1, 128, 128, 128, 128, f32::NAN),
        (1, 128, 128, 128, 128, f32::INFINITY),
    ] {
        if checked_geometry(rows, cols, input, weight, output, epsilon).is_ok() {
            return Err("normalization accepted invalid geometry or epsilon".into());
        }
    }
    if exact_bits(
        &[BFloat16::from_bits(0x3f80)],
        &[BFloat16::from_bits(0x3f81)],
        "mutation",
    )
    .is_ok()
    {
        return Err("normalization exact comparison accepted a one-ULP error".into());
    }
    stream.context().bind_to_thread()?;
    let module = kernels::load(stream.context())?;
    let mut values = 0;
    let mut cases = 0;
    for cols in [128, 1024, 2048, 17, 129] {
        for rows in [0, 1, 2, 3, 4, 5, 17, 65] {
            for epsilon in [1e-6, 1e-5, 0.03125, f32::from_bits(1)] {
                for pattern in [
                    Pattern::Regular,
                    Pattern::Varied,
                    Pattern::Small,
                    Pattern::Large,
                    Pattern::Zero,
                    Pattern::UnderflowTie,
                ] {
                    values += fixture(&stream, &module, rows, cols, epsilon, pattern)?;
                    cases += 1;
                }
            }
        }
    }
    println!(
        "RMSNorm: {cases} exact original-kernel cases, {values} BF16 values; input/weight/output guards preserved"
    );
    Ok(())
}

/// Event timings include repeated launches and exclude allocation/weight upload.
pub fn benchmark(stream: Arc<CudaStream>) -> Result<()> {
    let module = kernels::load(stream.context())?;
    /// Repeated normalization launches per event sample.
    const ITERATIONS: usize = 128;
    /// Paired samples whose medians determine the reported per-launch speedup.
    const SAMPLES: usize = 7;
    println!(
        "RMSNorm event timings: {SAMPLES} paired samples of {ITERATIONS} launches; warm weights; no end-to-end claim"
    );
    for cols in [128, 1024, 2048] {
        for rows in [1usize, 2, 8, 16, 32, 64] {
            let input =
                DeviceBuffer::from_host(&stream, &vec![BFloat16::from_f32(0.25); rows * cols])?;
            let weight = DeviceBuffer::from_host(&stream, &vec![BFloat16::from_f32(1.125); cols])?;
            let mut original =
                DeviceBuffer::from_host(&stream, &vec![BFloat16::from_bits(0x7fc1); rows * cols])?;
            let mut specialized =
                DeviceBuffer::from_host(&stream, &vec![BFloat16::from_bits(0x7fc1); rows * cols])?;
            let config = LaunchConfig {
                grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let start = stream
                .context()
                .new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            let end = stream
                .context()
                .new_event_with_flags(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            stream.context().synchronize()?;
            let mut submit = |new: bool| -> Result<()> {
                // SAFETY: ready independent buffers, full warp ownership and
                // completion before leaving this scope even on launch failure.
                unsafe {
                    if new {
                        enqueue(
                            &module,
                            &stream,
                            &input,
                            &weight,
                            &mut specialized,
                            rows,
                            cols,
                            1e-6,
                        )
                    } else {
                        module
                            .rmsnorm_bf16(
                                &stream,
                                config,
                                &input,
                                &weight,
                                &mut original,
                                rows,
                                cols,
                                1e-6,
                            )
                            .map_err(Into::into)
                    }
                }
            };
            let measured = (|| -> Result<[Vec<f64>; 2]> {
                for _ in 0..20 {
                    submit(false)?;
                    submit(true)?;
                }
                stream.synchronize()?;
                let mut timings = [Vec::with_capacity(SAMPLES), Vec::with_capacity(SAMPLES)];
                for sample in 0..SAMPLES {
                    // Alternate order to avoid favoring one path during drift.
                    for index in [sample % 2, 1 - sample % 2] {
                        start.record(&stream)?;
                        for _ in 0..ITERATIONS {
                            submit(index == 1)?;
                        }
                        end.record(&stream)?;
                        timings[index]
                            .push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / ITERATIONS as f64);
                    }
                }
                Ok(timings)
            })();
            let completed = stream.synchronize();
            let mut timings = measured?;
            completed?;
            exact_bits(
                &specialized.to_host_vec(&stream)?,
                &original.to_host_vec(&stream)?,
                "RMSNorm benchmark",
            )?;
            for values in &mut timings {
                values.sort_by(f64::total_cmp);
            }
            println!(
                "rows={rows} cols={cols} original_us={:.6} specialized_us={:.6} speedup={:.3}",
                timings[0][SAMPLES / 2],
                timings[1][SAMPLES / 2],
                timings[0][SAMPLES / 2] / timings[1][SAMPLES / 2]
            );
        }
    }
    Ok(())
}

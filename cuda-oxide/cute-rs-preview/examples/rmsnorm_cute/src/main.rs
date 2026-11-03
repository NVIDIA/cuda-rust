/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! RMSNorm validation with FP32 accumulation and explicit BF16 rounding boundaries.

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cute_rs::BFloat16;

/// Shared FP32/BF16 row kernels and bit-preserving BF16 copy kernel.
#[path = "../../common/rmsnorm.rs"]
mod rmsnorm;
use rmsnorm::{ROWS_PER_BLOCK, kernels};

/// Logical dimensions include head/hidden widths and awkward row boundaries.
const WIDTHS: [usize; 9] = [1, 7, 31, 32, 33, 128, 1024, 2048, 2051];
/// Empty and partial final blocks must preserve full-warp participation.
const ROW_COUNTS: [usize; 5] = [0, 1, 3, 4, 5];
/// Epsilon used by this synthetic validation, not a checkpoint configuration.
const EPSILON: f32 = 1e-6;

/// Validate dense `[rows, cols]` input/output and a shared `[cols]` weight vector.
///
/// Lengths are element counts. Epsilon must be finite and positive; zero rows
/// return `None`, while nonempty shapes return the number of four-warp blocks.
fn checked_work(
    rows: usize,
    cols: usize,
    input: usize,
    weight: usize,
    output: usize,
    epsilon: f32,
) -> Result<Option<u32>, &'static str> {
    if cols == 0 {
        return Err("RMSNorm width must be positive");
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("RMSNorm epsilon must be finite and positive");
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or("matrix extent overflows usize")?;
    if input != elements || output != elements || weight != cols {
        return Err("RMSNorm input, weight and output lengths must match the shape");
    }
    if rows == 0 {
        return Ok(None);
    }
    let blocks = rows.div_ceil(ROWS_PER_BLOCK);
    if blocks > i32::MAX as usize {
        return Err("row count exceeds the CUDA grid-x limit");
    }
    Ok(Some(blocks as u32))
}

/// Independent host rounding oracle, separate from the device BF16 implementation.
fn reference_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let retained = (bits >> 16) as u16;
    if value.is_nan() {
        return retained | 0x0040;
    }
    let discarded = bits & 0xffff;
    let increment = discarded > 0x8000 || (discarded == 0x8000 && retained & 1 != 0);
    retained.wrapping_add(u16::from(increment))
}

/// Read a BF16 storage pattern without using the device conversion helper.
fn reference_widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// FP64 square sum and normalization reference on the actual stored inputs.
fn reference_row(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f64> {
    let squares: f64 = input.iter().map(|&value| f64::from(value).powi(2)).sum();
    let inverse_rms = (squares / input.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    input
        .iter()
        .zip(weight)
        .map(|(&value, &scale)| f64::from(value) * inverse_rms * f64::from(scale))
        .collect()
}

/// Require finite results within an absolute-plus-relative error budget.
fn close(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// Match the kernel's four complete warps per block.
fn launch_config(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Check dynamic shape and epsilon before allocating and launching the FP32 kernel.
#[allow(clippy::too_many_arguments)] // Keep every checked launch-boundary argument explicit.
fn run_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(blocks) = checked_work(rows, cols, input.len(), weight.len(), output.len(), epsilon)?
    else {
        return Ok(());
    };
    let input_device = DeviceBuffer::from_host(stream, input)?;
    let weight_device = DeviceBuffer::from_host(stream, weight)?;
    let mut output_device = DeviceBuffer::from_host(stream, output)?;
    // SAFETY: exact allocation extents were checked; each complete warp owns
    // one row and each lane writes distinct columns. Copies synchronize lifetime.
    unsafe {
        module.rmsnorm_f32(
            stream,
            launch_config(blocks),
            &input_device,
            &weight_device,
            &mut output_device,
            rows,
            cols,
            epsilon,
        )
    }?;
    output_device.copy_to_host(stream, output)?;
    Ok(())
}

/// Check dynamic shape and epsilon before allocating and launching the BF16 kernel.
#[allow(clippy::too_many_arguments)] // Mirrors the FP32 boundary and kernel parameters.
fn run_bf16(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[BFloat16],
    weight: &[BFloat16],
    output: &mut [BFloat16],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(blocks) = checked_work(rows, cols, input.len(), weight.len(), output.len(), epsilon)?
    else {
        return Ok(());
    };
    let input_device = DeviceBuffer::from_host(stream, input)?;
    let weight_device = DeviceBuffer::from_host(stream, weight)?;
    let mut output_device = DeviceBuffer::from_host(stream, output)?;
    // SAFETY: same full-warp ownership and checked exact extents as run_f32.
    unsafe {
        module.rmsnorm_bf16(
            stream,
            launch_config(blocks),
            &input_device,
            &weight_device,
            &mut output_device,
            rows,
            cols,
            epsilon,
        )
    }?;
    output_device.copy_to_host(stream, output)?;
    Ok(())
}

/// Generate bounded rows so the FP64 oracle does not hide FP32 overflow behavior.
fn input_values(rows: usize, cols: usize, pattern: usize) -> (Vec<f32>, Vec<f32>) {
    /// Bounded signed row samples; the tiny pattern scales these by `1e-7`.
    const SIGNED: [f32; 9] = [-3.0, 0.25, -0.5, 1.0, -1.25, 2.0, 0.0, 4.0, -0.125];
    let input = (0..rows * cols)
        .map(|index| match pattern {
            0 => 0.0,
            1 => 2.0,
            2 => SIGNED[(index * 7 + index / cols) % SIGNED.len()],
            _ => SIGNED[index % SIGNED.len()] * 1e-7,
        })
        .collect();
    let weight = (0..cols)
        .map(|col| [0.5078125, 1.0, -0.75, 0.0, 1.5][col % 5])
        .collect();
    (input, weight)
}

/// Verify pure BF16 storage copies, including signed zeros, infinities and NaN bits.
fn validate_copy(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut checked = 0;
    for length in [0, 1, 7, 8, 9, 1031] {
        let input: Vec<BFloat16> = (0..length)
            .map(|i| {
                let bits = [
                    0x0000, 0x8000, 0x7f80, 0xff80, 0x7f81, 0xffc5, 0x0001, 0xffff, 0x3f80,
                ][i % 9];
                BFloat16::from_bits(bits)
            })
            .collect();
        let mut output = vec![BFloat16::from_bits(0x5555); length];
        if length != 0 {
            let input_device = DeviceBuffer::from_host(stream, &input)?;
            let mut output_device = DeviceBuffer::from_host(stream, &output)?;
            // SAFETY: exact equal lengths and each thread owns one aligned
            // eight-element vector or scalar tail; allocations remain live.
            unsafe {
                module.copy_bf16(
                    stream,
                    LaunchConfig::for_num_elems(length.div_ceil(8) as u32),
                    &input_device,
                    &mut output_device,
                )
            }?;
            output_device.copy_to_host(stream, &mut output)?;
        }
        for (index, (actual, expected)) in output.iter().zip(&input).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "BF16 copy length={length} index={index}"
            );
        }
        checked += length;
    }
    Ok(checked)
}

/// Validate the standalone operators on device zero; this is not a model benchmark.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut checked = 0;
    let mut max_error_f32 = 0.0_f64;
    let mut max_error_bf16 = 0.0_f64;
    assert!(
        run_f32(
            &module,
            &stream,
            &[1.0],
            &[],
            &mut [f32::NAN],
            1,
            1,
            EPSILON
        )
        .is_err()
    );
    assert!(
        run_bf16(
            &module,
            &stream,
            &[BFloat16::from_bits(0)],
            &[],
            &mut [BFloat16::from_bits(0)],
            1,
            1,
            EPSILON
        )
        .is_err()
    );
    for rows in ROW_COUNTS {
        for cols in WIDTHS {
            for pattern in 0..4 {
                let (input, weight) = input_values(rows, cols, pattern);
                let mut output = vec![f32::NAN; input.len()];
                run_f32(
                    &module,
                    &stream,
                    &input,
                    &weight,
                    &mut output,
                    rows,
                    cols,
                    EPSILON,
                )?;
                for row in 0..rows {
                    let expected =
                        reference_row(&input[row * cols..(row + 1) * cols], &weight, EPSILON);
                    for (col, &expected) in expected.iter().enumerate() {
                        let actual = f64::from(output[row * cols + col]);
                        assert!(
                            close(actual, expected, 1e-6, 2e-5),
                            "FP32 rows={rows} cols={cols} pattern={pattern} row={row} col={col}: expected={expected} actual={actual}"
                        );
                        max_error_f32 = max_error_f32.max((actual - expected).abs());
                    }
                }
                let input: Vec<BFloat16> = input
                    .into_iter()
                    .map(|v| BFloat16::from_bits(reference_bf16(v)))
                    .collect();
                let weight: Vec<BFloat16> = weight
                    .into_iter()
                    .map(|v| BFloat16::from_bits(reference_bf16(v)))
                    .collect();
                let mut output = vec![BFloat16::from_bits(0x7fc0); input.len()];
                run_bf16(
                    &module,
                    &stream,
                    &input,
                    &weight,
                    &mut output,
                    rows,
                    cols,
                    EPSILON,
                )?;
                for row in 0..rows {
                    let stored: Vec<f32> = input[row * cols..(row + 1) * cols]
                        .iter()
                        .map(|v| reference_widen(v.to_bits()))
                        .collect();
                    let normalized = reference_row(&stored, &vec![1.0; cols], EPSILON);
                    for col in 0..cols {
                        let normalized = reference_widen(reference_bf16(normalized[col] as f32));
                        let expected = f64::from(reference_widen(reference_bf16(
                            normalized * reference_widen(weight[col].to_bits()),
                        )));
                        let actual = f64::from(reference_widen(output[row * cols + col].to_bits()));
                        assert!(
                            close(actual, expected, 1e-7, 0.016),
                            "BF16 rows={rows} cols={cols} pattern={pattern} row={row} col={col}: expected={expected} actual={actual}"
                        );
                        max_error_bf16 = max_error_bf16.max((actual - expected).abs());
                    }
                }
                checked += input.len();
            }
        }
    }

    // An exact separate fixture rejects a kernel that casts only once, even
    // though a loose BF16 tolerance alone could accept that semantic error.
    let mut fixture = [BFloat16::from_bits(0); 2];
    run_bf16(
        &module,
        &stream,
        &[BFloat16::from_bits(0x3f80), BFloat16::from_bits(0x4000)],
        &[BFloat16::from_bits(0x3f02); 2],
        &mut fixture,
        1,
        2,
        EPSILON,
    )?;
    assert_eq!(
        fixture[0].to_bits(),
        0x3ea5,
        "cast-before-weight fixture; cast-once would produce 0x3ea4"
    );
    let copied = validate_copy(&module, &stream)?;
    println!(
        "PASS: FP32/BF16 RMSNorm, {} shapes, four patterns, {checked} outputs per dtype",
        WIDTHS.len() * ROW_COUNTS.len()
    );
    println!("FP32 max absolute error={max_error_f32:.8e}; atol=1e-6, rtol=2e-5");
    println!(
        "BF16 max absolute error={max_error_bf16:.8e}; atol=1e-7, rtol=0.016; exact cast-before-weight fixture passed"
    );
    println!(
        "BF16 typed vector/scalar copies preserved {copied} storage patterns; empty rows skipped."
    );
    Ok(())
}

/// Host checks for RMSNorm shapes, rounding boundaries, and reference arithmetic.
#[cfg(test)]
mod tests {
    use super::*;

    /// Reject malformed row storage, nonpositive epsilon, and overflowing matrix extents.
    #[test]
    fn boundary_rejects_shape_epsilon_and_overflow_errors() {
        for (rows, cols, input, weight, output, epsilon) in [
            (1, 0, 0, 0, 0, 1e-6),
            (2, 3, 5, 3, 6, 1e-6),
            (2, 3, 6, 2, 6, 1e-6),
            (2, 3, 6, 3, 5, 1e-6),
            (0, 3, 1, 3, 0, 1e-6),
            (0, 3, 0, 2, 0, 1e-6),
            (1, 1, 1, 1, 1, 0.0),
            (1, 1, 1, 1, 1, -1.0),
            (1, 1, 1, 1, 1, f32::NAN),
            (1, 1, 1, 1, 1, f32::INFINITY),
            (usize::MAX, 2, 0, 2, 0, 1e-6),
        ] {
            assert!(checked_work(rows, cols, input, weight, output, epsilon).is_err());
        }
    }

    /// Check empty launches, four-row block tails, and the signed CUDA grid limit.
    #[test]
    fn boundary_skips_empty_rows_and_covers_partial_blocks() {
        assert_eq!(checked_work(0, 128, 0, 128, 0, 1e-6), Ok(None));
        for (rows, blocks) in [(1, 1), (3, 1), (4, 1), (5, 2)] {
            assert_eq!(
                checked_work(rows, 33, rows * 33, 33, rows * 33, 1e-6),
                Ok(Some(blocks))
            );
        }
        if usize::BITS > 32 {
            let rows = (i32::MAX as usize + 1) * 4;
            assert!(checked_work(rows, 1, rows, 1, rows, 1e-6).is_err());
        }
    }

    /// Verify independent BF16 tie handling, signed zero, and a retained NaN encoding.
    #[test]
    fn rounding_reference_uses_ties_to_even_and_preserves_nan() {
        assert_eq!(reference_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(reference_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(reference_bf16(-0.0), 0x8000);
        assert!(reference_widen(reference_bf16(f32::from_bits(0x7f80_0001))).is_nan());
    }

    /// Check analytic zero/constant-row RMSNorm values against the FP64 reference.
    #[test]
    fn reference_normalizes_zero_and_constant_rows() {
        assert_eq!(reference_row(&[0.0, 0.0], &[1.0, -2.0], 1e-6), [0.0, -0.0]);
        let out = reference_row(&[2.0, 2.0], &[1.0, -2.0], 1e-6);
        assert!((out[0] - 1.0).abs() < 2e-7);
        assert!((out[1] + 2.0).abs() < 3e-7);
    }

    /// Require the intermediate BF16 cast that changes the weighted result by one bit.
    #[test]
    fn bf16_fixture_distinguishes_cast_before_weight_from_cast_once() {
        let normalized = reference_row(&[1.0, 2.0], &[1.0, 1.0], 1e-6)[0] as f32;
        let weight = reference_widen(0x3f02);
        let correct = reference_bf16(reference_widen(reference_bf16(normalized)) * weight);
        let cast_once = reference_bf16(normalized * weight);
        assert_eq!(correct, 0x3ea5);
        assert_eq!(cast_once, 0x3ea4);
        assert_ne!(correct, cast_once);
    }

    /// Ensure finite-value tolerance checks reject NaNs, infinity, and excessive error.
    #[test]
    fn acceptance_rejects_nonfinite_or_incorrect_outputs() {
        assert!(close(1.000_01, 1.0, 1e-6, 2e-5));
        assert!(!close(1.001, 1.0, 1e-6, 2e-5));
        assert!(!close(f64::NAN, 1.0, 1e-6, 2e-5));
        assert!(!close(f64::INFINITY, 1.0, 1e-6, 2e-5));
        assert!(!close(1.0, f64::NAN, 1e-6, 2e-5));
    }
}

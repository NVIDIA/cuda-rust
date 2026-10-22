/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fused SwiGLU validation with typed register tiles and explicit FP32 arithmetic.

#![feature(f16)]

use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

/// Empty inputs, sub-tile inputs, aligned FFN widths, and a partial final tile.
const LENGTHS: [usize; 10] = [0, 1, 3, 4, 7, 8, 9, 3072, 6144, 6147];
/// Four FP32 values make one 16-byte vector copy.
const TILE_F32: usize = 4;
/// Eight FP16 values make one 16-byte vector copy.
const TILE_F16: usize = 8;

/// Both entry points use the same FP32 activation and preserve scalar tails.
#[cuda_module]
mod kernels {
    use super::*;

    /// Evaluate `(gate * sigmoid(gate)) * up` with approximate FP32 math.
    ///
    /// Nonpositive exponential arguments avoid overflow for large finite gates.
    /// Reciprocal arithmetic flushes subnormals; NaN/infinity behavior is outside
    /// this example's finite-input validation contract.
    #[inline(always)]
    fn swiglu(gate: f32, up: f32) -> f32 {
        let sigmoid = if gate >= 0.0 {
            let exponential = ctm_rs::float::ex2_approx_f32(-gate * core::f32::consts::LOG2_E);
            ctm_rs::float::rcp_approx_ftz_f32(1.0 + exponential)
        } else {
            let exponential = ctm_rs::float::ex2_approx_f32(gate * core::f32::consts::LOG2_E);
            exponential * ctm_rs::float::rcp_approx_ftz_f32(1.0 + exponential)
        };
        (gate * sigmoid) * up
    }

    /// Compute FP32 SwiGLU with a vector tile for every complete group of four.
    ///
    /// Launch with matching input/output lengths, 16-byte-aligned allocations,
    /// and one unique linear thread index per tile. Output must not alias either
    /// input and all buffers must remain ready and live through completion.
    /// There are no warp collectives; incomplete tiles use scalar accesses.
    #[kernel]
    pub fn swiglu_f32(gate: &[f32], up: &[f32], mut output: DisjointSlice<f32>) {
        let tid = thread::index_1d().get();
        let gate_tile = cute_rs::Tensor::from_slice(gate)
            .zipped_divide::<TILE_F32>()
            .slice(tid);
        let up_tile = cute_rs::Tensor::from_slice(up)
            .zipped_divide::<TILE_F32>()
            .slice(tid);
        let mut output_tile = cute_rs::TensorMut::from_disjoint_slice(&mut output)
            .zipped_divide::<TILE_F32>()
            .slice(tid);

        if gate_tile.is_full() {
            let gates = unsafe { gate_tile.load() };
            let ups = unsafe { up_tile.load() };
            let result = gates.zip_map(ups, swiglu);
            unsafe { output_tile.store(result) };
        } else {
            let mut index = gate_tile.base();
            while index < gate.len() {
                unsafe { output_tile.store_at(index, swiglu(gate[index], up[index])) };
                index += 1;
            }
        }
    }

    /// Widen FP16 inputs, compute in FP32, and narrow once when storing output.
    ///
    /// Uses eight adjacent values per 16-byte tile and scalar tails. The caller
    /// must satisfy the matching-length, alignment, ownership, and completion
    /// requirements of `swiglu_f32`; no intermediate FP16 rounding occurs.
    #[kernel]
    pub fn swiglu_f16(gate: &[f16], up: &[f16], mut output: DisjointSlice<f16>) {
        let tid = thread::index_1d().get();
        let gate_tile = cute_rs::Tensor::from_slice(gate)
            .zipped_divide::<TILE_F16>()
            .slice(tid);
        let up_tile = cute_rs::Tensor::from_slice(up)
            .zipped_divide::<TILE_F16>()
            .slice(tid);
        let mut output_tile = cute_rs::TensorMut::from_disjoint_slice(&mut output)
            .zipped_divide::<TILE_F16>()
            .slice(tid);

        if gate_tile.is_full() {
            let gates = unsafe { gate_tile.load() }.map(|value| value as f32);
            let ups = unsafe { up_tile.load() }.map(|value| value as f32);
            let result = gates.zip_map(ups, swiglu).map(|value| value as f16);
            unsafe { output_tile.store(result) };
        } else {
            let mut index = gate_tile.base();
            while index < gate.len() {
                let result = swiglu(gate[index] as f32, up[index] as f32) as f16;
                unsafe { output_tile.store_at(index, result) };
                index += 1;
            }
        }
    }
}

/// Validate all dynamic lengths before allocating buffers or launching a kernel.
fn checked_work<const TILE: usize>(
    gate: usize,
    up: usize,
    output: usize,
) -> Result<Option<u32>, &'static str> {
    const { assert!(TILE > 0) };
    if gate != up || gate != output {
        return Err("gate, up and output lengths must match");
    }
    if gate == 0 {
        return Ok(None);
    }
    let threads = u32::try_from(gate.div_ceil(TILE))
        .map_err(|_| "tile count exceeds the CUDA launch integer range")?;
    Ok(Some(threads))
}

/// Stable FP64 reference, independent of the device's approximate base-two math.
fn swiglu_reference(gate: f64, up: f64) -> f64 {
    let sigmoid = if gate >= 0.0 {
        1.0 / (1.0 + (-gate).exp())
    } else {
        let exponential = gate.exp();
        exponential / (1.0 + exponential)
    };
    gate * sigmoid * up
}

/// Require finite output within an absolute-plus-relative tolerance.
fn within_tolerance(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// Signed finite values, including tiny values and large gates in both directions.
fn inputs(length: usize) -> (Vec<f32>, Vec<f32>) {
    /// Gate samples spanning signed zeros, tiny magnitudes, and finite FP16 extremes.
    const GATES: [f32; 20] = [
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.25, -0.5, 8.0, -8.0, 20.0, -20.0, 80.0, -100.0, 65504.0,
        -65504.0, 0.00001, -0.00001, 0.75, -0.75,
    ];
    /// Bounded multipliers varied independently from the gate samples.
    const UPS: [f32; 9] = [0.5, -0.5, 0.0, -0.0, 1.0, -1.0, 0.125, -0.25, 0.75];
    let gate = (0..length).map(|i| GATES[i % GATES.len()]).collect();
    let up = (0..length).map(|i| UPS[(i * 7 + 1) % UPS.len()]).collect();
    (gate, up)
}

/// Check arguments before allocating or launching, and copy back all FP32 outputs.
fn run_f32(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(threads) = checked_work::<TILE_F32>(gate.len(), up.len(), output.len())? else {
        return Ok(());
    };
    let gate_device = DeviceBuffer::from_host(stream, gate)?;
    let up_device = DeviceBuffer::from_host(stream, up)?;
    let mut output_device = DeviceBuffer::from_host(stream, output)?;
    // SAFETY: lengths match, and each thread owns one complete tile or the
    // final scalar tail. The input/output allocations are distinct and live
    // until copy_to_host synchronizes the stream.
    unsafe {
        module.swiglu_f32(
            stream,
            LaunchConfig::for_num_elems(threads),
            &gate_device,
            &up_device,
            &mut output_device,
        )
    }?;
    output_device.copy_to_host(stream, output)?;
    Ok(())
}

/// Check arguments before allocating or launching, and copy back all FP16 outputs.
fn run_f16(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    gate: &[f16],
    up: &[f16],
    output: &mut [f16],
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(threads) = checked_work::<TILE_F16>(gate.len(), up.len(), output.len())? else {
        return Ok(());
    };
    let gate_device = DeviceBuffer::from_host(stream, gate)?;
    let up_device = DeviceBuffer::from_host(stream, up)?;
    let mut output_device = DeviceBuffer::from_host(stream, output)?;
    // SAFETY: as in run_f32, each output tile has one writer, matching inputs,
    // and allocations that remain valid through the synchronized copy back.
    unsafe {
        module.swiglu_f16(
            stream,
            LaunchConfig::for_num_elems(threads),
            &gate_device,
            &up_device,
            &mut output_device,
        )
    }?;
    output_device.copy_to_host(stream, output)?;
    Ok(())
}

/// Validate both stored dtypes against an independent FP64 reference on device 0.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut max_error_f32 = 0.0_f64;
    let mut max_error_f16 = 0.0_f64;
    let mut checked = 0;

    // Exercise both checked wrappers with mismatched input/output lengths.
    assert!(run_f32(&module, &stream, &[1.0], &[], &mut [f32::NAN]).is_err());
    assert!(run_f32(&module, &stream, &[1.0], &[1.0], &mut []).is_err());
    assert!(run_f16(&module, &stream, &[1.0], &[], &mut [f16::NAN]).is_err());
    assert!(run_f16(&module, &stream, &[1.0], &[1.0], &mut []).is_err());

    for length in LENGTHS {
        let (gate, up) = inputs(length);
        let mut output = vec![f32::NAN; length];
        run_f32(&module, &stream, &gate, &up, &mut output)?;
        for i in 0..length {
            let expected = swiglu_reference(gate[i] as f64, up[i] as f64);
            let actual = output[i] as f64;
            assert!(
                within_tolerance(actual, expected, 2e-6, 2e-5),
                "FP32 length={length} index={i}: gate={}, up={}, expected={expected}, actual={actual}",
                gate[i],
                up[i],
            );
            max_error_f32 = max_error_f32.max((actual - expected).abs());
        }

        let gate: Vec<f16> = gate.into_iter().map(|value| value as f16).collect();
        let up: Vec<f16> = up.into_iter().map(|value| value as f16).collect();
        let mut output = vec![f16::NAN; length];
        run_f16(&module, &stream, &gate, &up, &mut output)?;
        for i in 0..length {
            // Reference the actual stored FP16 inputs, not pre-conversion values.
            let expected = swiglu_reference(gate[i] as f64, up[i] as f64);
            let actual = output[i] as f64;
            assert!(
                within_tolerance(actual, expected, 2e-3, 2e-3),
                "FP16 length={length} index={i}: gate={}, up={}, expected={expected}, actual={actual}",
                gate[i] as f32,
                up[i] as f32,
            );
            max_error_f16 = max_error_f16.max((actual - expected).abs());
        }
        checked += length;
    }

    println!(
        "PASS: FP32/FP16 SwiGLU, {} lengths, {checked} outputs per dtype",
        LENGTHS.len()
    );
    println!("FP32: max absolute error={max_error_f32:.8e}, atol=2e-6, rtol=2e-5");
    println!("FP16: max absolute error={max_error_f16:.8e}, atol=2e-3, rtol=2e-3");
    println!("Length mismatches rejected; empty inputs skipped without a launch.");
    Ok(())
}

/// Host checks for launch extents, finite-value acceptance, and the FP64 oracle.
#[cfg(test)]
mod tests {
    use super::*;

    /// A mismatch must be rejected even when the gate is empty.
    #[test]
    fn launch_boundary_rejects_every_length_mismatch() {
        for (gate, up, output) in [(1, 2, 1), (2, 2, 1), (0, 1, 0), (0, 0, 1)] {
            assert!(checked_work::<4>(gate, up, output).is_err());
        }
    }

    /// An empty valid operation has no launch, while tails reserve one final thread.
    #[test]
    fn launch_boundary_skips_empty_inputs_and_covers_tails() {
        assert_eq!(checked_work::<4>(0, 0, 0), Ok(None));
        for (length, threads) in [(1, 1), (3, 1), (4, 1), (7, 2), (8, 2), (9, 3)] {
            assert_eq!(checked_work::<4>(length, length, length), Ok(Some(threads)));
        }
        assert_eq!(checked_work::<8>(9, 9, 9), Ok(Some(2)));
    }

    /// A host length must not silently truncate at the CUDA launch integer boundary.
    #[test]
    fn launch_boundary_rejects_unrepresentable_thread_counts() {
        if usize::BITS > 32 {
            let length = u32::MAX as usize + 1;
            assert!(checked_work::<1>(length, length, length).is_err());
        }
    }

    /// Check analytic SiLU values and extreme gates without exponent overflow.
    #[test]
    fn reference_is_stable_for_signed_and_extreme_inputs() {
        assert_eq!(swiglu_reference(0.0, -3.0), 0.0);
        assert_eq!(swiglu_reference(65504.0, 0.5), 32752.0);
        assert_eq!(swiglu_reference(-65504.0, -0.5), 0.0);
        assert!((swiglu_reference(1.0, 2.0) - 1.462_117_157_260_009_8).abs() < 1e-15);
        assert!((swiglu_reference(-1.0, -2.0) - 0.537_882_842_739_990_2).abs() < 1e-15);
    }

    /// NaN and infinities cannot pass through comparisons with a permissive tolerance.
    #[test]
    fn acceptance_rejects_nonfinite_and_wrong_outputs() {
        assert!(within_tolerance(1.000_01, 1.0, 2e-6, 2e-5));
        assert!(!within_tolerance(1.001, 1.0, 2e-6, 2e-5));
        assert!(!within_tolerance(f64::NAN, 1.0, 2e-6, 2e-5));
        assert!(!within_tolerance(f64::INFINITY, 1.0, 2e-6, 2e-5));
        assert!(!within_tolerance(1.0, f64::NAN, 2e-6, 2e-5));
    }
}

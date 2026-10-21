/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Checked BF16 feed-forward composition and intermediate numerical validation.
/// BF16 matrix projections and tiled SiLU/product kernels.
mod kernel;
/// Checked synchronous FFN execution and owned intermediate storage.
mod operator;
/// Independent host matrix and activation oracles with integer BF16 rounding.
mod reference;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cute_rs::{
    Bf16,
    ffn::{Down, FfnShape, FfnWeights, Gate, ProjectionWeights, RowMajorShape, Up},
};
use operator::{GUARD, Result, Runner, Workspace};

/// Asymmetric bounded dyadic values keep the projection reference well-conditioned.
fn matrix(shape: RowMajorShape, seed: usize) -> Vec<Bf16> {
    let mut data = vec![Bf16::from_bits(GUARD); shape.required_len()];
    for row in 0..shape.rows() {
        for col in 0..shape.cols() {
            let code = ((row * 37 + col * 13 + (row ^ col) * 3 + seed * 11) % 17) as i32 - 8;
            let scale = if seed == 0 { 8.0 } else { 128.0 };
            data[row * shape.row_stride() + col] = reference::round(code as f32 / scale);
        }
    }
    data
}

/// Copy every allocated element and check that no logical write touched the guard.
fn read_output(
    stream: &CudaStream,
    data: &DeviceBuffer<Bf16>,
    logical: usize,
) -> Result<Vec<Bf16>> {
    let mut actual = vec![Bf16::from_bits(0); data.len()];
    data.copy_to_host(stream, &mut actual)?;
    assert!(
        actual[logical..].iter().all(|v| v.to_bits() == GUARD),
        "output guard overwritten"
    );
    actual.truncate(logical);
    Ok(actual)
}

/// Compare independent reference values, allowing one BF16 output quantum plus cancellation error.
fn verify(label: &str, actual: &[Bf16], expected: &[Bf16], atol: f64, rtol: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let a = reference::widen(a);
        let e = reference::widen(e);
        let error = (a - e).abs();
        assert!(
            a.is_finite() && e.is_finite() && error <= atol + rtol * e.abs(),
            "{label} index={i}: actual={a} expected={e} error={error}"
        );
        maximum = maximum.max(error);
    }
    maximum
}

/// Allocate checked weights once and execute all requested batches at the same model widths.
fn validate<const H: usize, const I: usize>(
    runner: &Runner,
    stream: &CudaStream,
    rows: &[usize],
    padding: usize,
) -> Result<usize> {
    let gate_shape = RowMajorShape::new(I, H, H + padding)?;
    let up_shape = RowMajorShape::new(I, H, H + padding + 1)?;
    let down_shape = RowMajorShape::new(H, I, I + padding + 2)?;
    let gate = matrix(gate_shape, 1);
    let up = matrix(up_shape, 2);
    let down = matrix(down_shape, 3);
    let gate_device = DeviceBuffer::from_host(stream, &gate)?;
    let up_device = DeviceBuffer::from_host(stream, &up)?;
    let down_device = DeviceBuffer::from_host(stream, &down)?;
    let weights = FfnWeights::<H, I>::new(
        ProjectionWeights::<Gate, H, I>::new(&gate_device, gate_shape.row_stride())?,
        ProjectionWeights::<Up, H, I>::new(&up_device, up_shape.row_stride())?,
        ProjectionWeights::<Down, I, H>::new(&down_device, down_shape.row_stride())?,
    );
    // A safe checked constructor must reject an overlapping stride before any launch.
    assert!(ProjectionWeights::<Gate, H, I>::new(&gate_device, H - 1).is_err());
    let mut checked = 0;
    for &rows in rows {
        let shape = FfnShape::<H, I>::new(rows, H + padding)?;
        let input = matrix(shape.input(), 0);
        let input_device = DeviceBuffer::from_host(stream, &input)?;
        let mut workspace = Workspace::new(stream, shape)?;
        runner.run(shape, &input_device, &weights, &mut workspace)?;
        let gates = read_output(stream, &workspace.gate, shape.intermediate().required_len())?;
        let ups = read_output(stream, &workspace.up, shape.intermediate().required_len())?;
        let silu = read_output(stream, &workspace.silu, shape.intermediate().required_len())?;
        let product = read_output(
            stream,
            &workspace.product,
            shape.intermediate().required_len(),
        )?;
        let output = read_output(stream, &workspace.output, shape.hidden().required_len())?;
        let expected_gate = reference::linear(
            &input,
            &gate,
            rows,
            H,
            I,
            shape.input().row_stride(),
            gate_shape.row_stride(),
        );
        let expected_up = reference::linear(
            &input,
            &up,
            rows,
            H,
            I,
            shape.input().row_stride(),
            up_shape.row_stride(),
        );
        let (expected_silu, expected_product) = reference::activate(&expected_gate, &expected_up);
        let expected_output = reference::linear(
            &expected_product,
            &down,
            rows,
            I,
            H,
            I,
            down_shape.row_stride(),
        );
        let mut maximum = 0.0_f64;
        for (label, actual, expected) in [
            ("gate", &gates, &expected_gate),
            ("up", &ups, &expected_up),
            ("silu", &silu, &expected_silu),
            ("product", &product, &expected_product),
        ] {
            maximum = maximum.max(verify(label, actual, expected, 1e-6, 0.008));
        }
        // Check down independently on its actual input as well as the complete
        // composition, so propagated activation error cannot hide a bad GEMM.
        let local_down = reference::linear(&product, &down, rows, I, H, I, down_shape.row_stride());
        maximum = maximum.max(verify("down", &output, &local_down, 1e-5, 0.008));
        maximum = maximum.max(verify("end-to-end", &output, &expected_output, 2e-5, 0.02));
        checked += 4 * rows * I + rows * H;
        println!(
            "PASS FFN H={H} I={I} rows={rows} input_stride={} max_abs_error={maximum:.8e}",
            shape.input().row_stride()
        );
    }
    Ok(checked)
}

/// Safe asynchronous buffer writes can precede a call on a different stream.
fn check_pending_input(runner: &Runner, stream: &CudaStream) -> Result<()> {
    use std::sync::mpsc;
    use std::time::Duration;

    let producer = stream.context().new_stream()?;
    let one = [Bf16::from_bits(0x3f80)];
    let mut input = DeviceBuffer::from_host(stream, &one)?;
    let weight = DeviceBuffer::from_host(stream, &one)?;
    let weights = FfnWeights::<1, 1>::new(
        ProjectionWeights::<Gate, 1, 1>::new(&weight, 1)?,
        ProjectionWeights::<Up, 1, 1>::new(&weight, 1)?,
        ProjectionWeights::<Down, 1, 1>::new(&weight, 1)?,
    );
    let shape = FfnShape::new(1, 1)?;
    let mut workspace = Workspace::new(stream, shape)?;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    producer.launch_host_function(move || {
        let _ = entered_tx.send(());
        // No CUDA calls in the callback. Bound the wait even on a test failure.
        let _ = release_rx.recv_timeout(Duration::from_secs(10));
    })?;
    entered_rx.recv_timeout(Duration::from_secs(10))?;
    input.zero_async(&producer)?;
    assert!(!producer.query()?, "producer must still be pending");
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(250));
        let _ = release_tx.send(());
    });
    let result = runner.run(shape, &input, &weights, &mut workspace);
    let completion = producer.synchronize();
    release.join().expect("producer release thread panicked");
    result?;
    completion?;
    let output = read_output(stream, &workspace.output, 1)?;
    assert_eq!(
        output[0].to_bits(),
        0,
        "FFN read input before its producer completed"
    );
    println!("PASS: pending input on a separate stream is ordered before FFN");
    Ok(())
}

/// Validate every FFN stage, guarded output, and producer ordering on CUDA device 0.
fn main() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::new(stream.clone())?;
    check_pending_input(&runner, &stream)?;
    let mut checked = validate::<19, 23>(&runner, &stream, &[0, 1, 17], 5)?;
    checked += validate::<32, 17>(&runner, &stream, &[5], 3)?;
    checked += validate::<1, 1>(&runner, &stream, &[3], 0)?;
    checked += validate::<1024, 3072>(&runner, &stream, &[1, 3], 0)?;
    checked += validate::<2048, 6144>(&runner, &stream, &[1, 3], 0)?;
    runner.check_rounding()?;
    println!(
        "PASS: {checked} BF16 FFN intermediate/output values; guards, tails, empty rows and exact SiLU rounding fixture passed"
    );
    Ok(())
}

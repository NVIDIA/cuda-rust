/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Bounded synchronous groups of existing decode projection kernels.

use super::{Runner, launch_config};
use crate::{
    ffn_kernel::kernels,
    model::Batch,
    operator::Result,
    reference_linear::{ReductionPlan, fma_chain_blocks, reference_plan},
    reference_linear_kernel::kernels as reference,
};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cute_rs::{
    BFloat16,
    decoder::AttentionWeights,
    ffn::{FfnWeights, ProjectionRole, ProjectionWeights, RowMajorShape},
};
use std::sync::Arc;

/// Prepared unbiased projection in the shape's reference reduction order.
enum ProjectionLaunch {
    /// One tensor-core K16 chain over the width.
    Mma(PreparedLaunch<kernels::__decode_linear_bf16_CudaKernel>),
    /// Sixteen interleaved FP32 chains per output.
    Fma16(PreparedLaunch<reference::__fma16_linear_bf16_CudaKernel>),
    /// Thirty-two interleaved FP32 chains per output.
    Fma32(PreparedLaunch<reference::__fma32_linear_bf16_CudaKernel>),
    /// Tensor-core chains per slice of the given width, summed in order.
    SplitK(
        PreparedLaunch<reference::__split_k_linear_bf16_CudaKernel>,
        usize,
    ),
}

/// Drain submitted work on both ordinary returns and unwinding. Callers own
/// their buffers outside this scope, so completion runs before those can drop.
struct Completion<F: FnOnce() -> Result<()>> {
    /// Completion action, taken once by explicit finish or unwinding cleanup.
    finish: Option<F>,
}

impl<F: FnOnce() -> Result<()>> Completion<F> {
    /// Run the completion action exactly once and propagate its result.
    fn finish(mut self) -> Result<()> {
        self.finish.take().expect("completion is consumed once")()
    }
}

impl<F: FnOnce() -> Result<()>> Drop for Completion<F> {
    /// Attempt completion during unwinding or early return, discarding cleanup errors.
    fn drop(&mut self) {
        if let Some(finish) = self.finish.take() {
            let _ = finish();
        }
    }
}

/// Submit a group and attempt completion even after submission failure or panic.
///
/// On ordinary return, submission errors take precedence over completion errors.
/// The completion guard is established before invoking the submission closure.
fn submit_and_complete(
    submit: impl FnOnce() -> Result<()>,
    finish: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let completion = Completion {
        finish: Some(finish),
    };
    let submitted = submit();
    let completed = completion.finish();
    submitted?;
    completed
}

impl Runner {
    /// Validate dense one/two-row input and weight extents/contexts, then prepare its grid.
    ///
    /// Performs no readiness fence or enqueue; callers finish all group preparation
    /// and allocate separate outputs before submitting the first kernel.
    fn prepare_group_projection<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
    ) -> Result<ProjectionLaunch> {
        let config = launch_config(input.rows(), OUT)?;
        RowMajorShape::dense(input.rows(), IN)?.validate_len(input.buffer().len())?;
        RowMajorShape::dense(input.rows(), OUT)?;
        weight.shape().validate_len(weight.buffer().len())?;
        for buffer in [input.buffer(), weight.buffer()] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("grouped decode buffers must share the runner context".into());
            }
        }
        let eight_column_blocks = OUT.div_ceil(8) as u32;
        match reference_plan(input.rows(), IN, OUT) {
            ReductionPlan::MmaChain => Ok(ProjectionLaunch::Mma(
                self.launches.get_or_prepare(eight_column_blocks, || {
                    self.module.prepare_decode_linear_bf16(config)
                })?,
            )),
            ReductionPlan::FmaChains { chains: 16 } => {
                let blocks = fma_chain_blocks(16, OUT) as u32;
                Ok(ProjectionLaunch::Fma16(
                    self.fma16_launches.get_or_prepare(blocks, || {
                        self.reference
                            .prepare_fma16_linear_bf16(LaunchConfig1D::new(blocks, 32, 0))
                    })?,
                ))
            }
            ReductionPlan::FmaChains { chains: 32 } => {
                let blocks = fma_chain_blocks(32, OUT) as u32;
                Ok(ProjectionLaunch::Fma32(
                    self.fma32_launches.get_or_prepare(blocks, || {
                        self.reference
                            .prepare_fma32_linear_bf16(LaunchConfig1D::new(blocks, 32, 0))
                    })?,
                ))
            }
            ReductionPlan::FmaChains { .. } => {
                Err("grouped decode has no entry for this chain count".into())
            }
            ReductionPlan::MmaSplitK { slice_width } => Ok(ProjectionLaunch::SplitK(
                self.split_k_launches
                    .get_or_prepare(eight_column_blocks, || {
                        self.reference.prepare_split_k_linear_bf16(config)
                    })?,
                slice_width,
            )),
        }
    }

    /// Enqueue one previously validated projection, without waiting.
    ///
    /// # Safety
    /// `prepare_group_projection` must have accepted these input/weight views;
    /// output must be a distinct dense allocation with the same row count.
    /// All producers must have completed. Input, weight and output must remain
    /// alive and inaccessible to conflicting work until this stream completes,
    /// including after an enqueue error or panic.
    unsafe fn enqueue_group_projection<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        launch: &ProjectionLaunch,
        input: &Batch<IN>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let stride = weight.shape().row_stride();
        match launch {
            ProjectionLaunch::Mma(launch) => self.module.decode_linear_bf16(
                &self.stream,
                launch,
                input.buffer(),
                weight.buffer(),
                output.buffer_mut(),
                input.rows(),
                IN,
                OUT,
                IN,
                stride,
            )?,
            ProjectionLaunch::Fma16(launch) => self.reference.fma16_linear_bf16(
                &self.stream,
                launch,
                input.buffer(),
                weight.buffer(),
                output.buffer_mut(),
                IN,
                OUT,
                stride,
            )?,
            ProjectionLaunch::Fma32(launch) => self.reference.fma32_linear_bf16(
                &self.stream,
                launch,
                input.buffer(),
                weight.buffer(),
                output.buffer_mut(),
                IN,
                OUT,
                stride,
            )?,
            ProjectionLaunch::SplitK(launch, slice_width) => self.reference.split_k_linear_bf16(
                &self.stream,
                launch,
                input.buffer(),
                weight.buffer(),
                output.buffer_mut(),
                input.rows(),
                IN,
                OUT,
                IN,
                stride,
                *slice_width,
            )?,
        }
        Ok(())
    }

    /// Project Q, K and V in their existing order, returning only after all
    /// submitted work has completed. The borrowed typed roles stay distinct.
    pub(crate) fn project_qkv<const H: usize, const QW: usize, const KW: usize>(
        &self,
        input: &Batch<H>,
        weights: &AttentionWeights<'_, H, QW, KW, 128>,
    ) -> Result<(Batch<QW>, Batch<KW>, Batch<KW>)> {
        let query_launch = self.prepare_group_projection(input, weights.query())?;
        let key_launch = self.prepare_group_projection(input, weights.key())?;
        let value_launch = self.prepare_group_projection(input, weights.value())?;
        let mut query = Batch::allocate_from(&self.pool, &self.stream, input.rows())?;
        let mut key = Batch::allocate_from(&self.pool, &self.stream, input.rows())?;
        let mut value = Batch::allocate_from(&self.pool, &self.stream, input.rows())?;
        // This includes producers on foreign streams. All fallible preparation
        // and allocation precedes the first projection submission.
        self.stream.context().synchronize()?;
        submit_and_complete(
            || {
                // SAFETY: every view/launch was validated, the context fence
                // established readiness, and distinct owned outputs plus all
                // borrowed inputs outlive the unconditional completion guard.
                unsafe {
                    self.enqueue_group_projection(
                        &query_launch,
                        input,
                        weights.query(),
                        &mut query,
                    )?;
                    self.enqueue_group_projection(&key_launch, input, weights.key(), &mut key)?;
                    self.enqueue_group_projection(
                        &value_launch,
                        input,
                        weights.value(),
                        &mut value,
                    )?;
                }
                Ok(())
            },
            || Ok(self.stream.synchronize()?),
        )?;
        Ok((query, key, value))
    }

    /// Project gate then up on one stream, preserving the original kernels and
    /// each projection's independent accumulator and BF16 rounding boundary.
    pub(crate) fn project_gate_up<const H: usize, const I: usize>(
        &self,
        input: &Batch<H>,
        weights: &FfnWeights<'_, H, I>,
    ) -> Result<(Batch<I>, Batch<I>)> {
        let gate_launch = self.prepare_group_projection(input, weights.gate())?;
        let up_launch = self.prepare_group_projection(input, weights.up())?;
        let mut gate = Batch::allocate_from(&self.pool, &self.stream, input.rows())?;
        let mut up = Batch::allocate_from(&self.pool, &self.stream, input.rows())?;
        self.stream.context().synchronize()?;
        submit_and_complete(
            || {
                // SAFETY: same validation, readiness and completion proof as
                // project_qkv; both output allocations are exclusively owned.
                unsafe {
                    self.enqueue_group_projection(&gate_launch, input, weights.gate(), &mut gate)?;
                    self.enqueue_group_projection(&up_launch, input, weights.up(), &mut up)?;
                }
                Ok(())
            },
            || Ok(self.stream.synchronize()?),
        )?;
        Ok((gate, up))
    }
}

/// Upload a strided BF16 fixture with deterministic logical values and NaN padding.
///
/// Dimensions and stride are in elements; shape or extent overflow is an error.
fn matrix(
    stream: &CudaStream,
    rows: usize,
    cols: usize,
    stride: usize,
    seed: &mut u32,
) -> Result<DeviceBuffer<BFloat16>> {
    RowMajorShape::new(rows, cols, stride)?;
    let length = rows.checked_mul(stride).ok_or("fixture extent overflow")?;
    let mut values = vec![BFloat16::from_bits(0x7fc1); length];
    for row in 0..rows {
        for col in 0..cols {
            values[row * stride + col] = super::varied_bf16(seed);
        }
    }
    Ok(DeviceBuffer::from_host(stream, &values)?)
}

/// Require equal-length bit-identical BF16 outputs and return the compared value count.
fn same_values(actual: &[BFloat16], expected: &[BFloat16]) -> Result<usize> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("grouped decode differs from individual projection bits".into());
    }
    Ok(actual.len())
}

/// Compare grouped Q/K/V and gate/up projections with individual launches.
///
/// Uses strided weights and typed roles, and exercises rejection of unsupported
/// rows and foreign-context weights before submission.
fn fixture<const H: usize, const QW: usize, const KW: usize, const I: usize>(
    runner: &Runner,
    rows: usize,
    stride: usize,
) -> Result<usize> {
    use cute_rs::decoder::NormWeights;
    let mut seed = 0x1f29_81a7;
    let input_values: Vec<_> = (0..rows * H)
        .map(|_| super::varied_bf16(&mut seed))
        .collect();
    let input = Batch::<H>::from_host(&runner.stream, rows, &input_values)?;
    let query = matrix(&runner.stream, QW, H, stride, &mut seed)?;
    let key = matrix(&runner.stream, KW, H, stride, &mut seed)?;
    let value = matrix(&runner.stream, KW, H, stride, &mut seed)?;
    let output = matrix(&runner.stream, H, QW, QW, &mut seed)?;
    let norm = DeviceBuffer::from_host(&runner.stream, &[BFloat16::from_f32(1.0); 128])?;
    let attention = AttentionWeights::<H, QW, KW, 128>::new(
        ProjectionWeights::new(&query, stride)?,
        ProjectionWeights::new(&key, stride)?,
        ProjectionWeights::new(&value, stride)?,
        ProjectionWeights::new(&output, QW)?,
        NormWeights::new(&norm)?,
        NormWeights::new(&norm)?,
    );
    let gate = matrix(&runner.stream, I, H, stride, &mut seed)?;
    let up = matrix(&runner.stream, I, H, stride, &mut seed)?;
    let down = matrix(&runner.stream, H, I, I, &mut seed)?;
    let ffn = FfnWeights::<H, I>::new(
        ProjectionWeights::new(&gate, stride)?,
        ProjectionWeights::new(&up, stride)?,
        ProjectionWeights::new(&down, I)?,
    );
    let (q, k, v) = runner.project_qkv(&input, &attention)?;
    let mut checked = same_values(
        &q.read(&runner.stream)?,
        &runner
            .project_buffer(input.buffer(), rows, H, attention.query(), None)?
            .read(&runner.stream)?,
    )?;
    checked += same_values(
        &k.read(&runner.stream)?,
        &runner
            .project_buffer(input.buffer(), rows, H, attention.key(), None)?
            .read(&runner.stream)?,
    )?;
    checked += same_values(
        &v.read(&runner.stream)?,
        &runner
            .project_buffer(input.buffer(), rows, H, attention.value(), None)?
            .read(&runner.stream)?,
    )?;
    let (g, u) = runner.project_gate_up(&input, &ffn)?;
    checked += same_values(
        &g.read(&runner.stream)?,
        &runner
            .project_buffer(input.buffer(), rows, H, ffn.gate(), None)?
            .read(&runner.stream)?,
    )?;
    checked += same_values(
        &u.read(&runner.stream)?,
        &runner
            .project_buffer(input.buffer(), rows, H, ffn.up(), None)?
            .read(&runner.stream)?,
    )?;
    if rows == 1 {
        for invalid_rows in [0, 3] {
            let invalid = Batch::<H>::from_host(
                &runner.stream,
                invalid_rows,
                &vec![BFloat16::from_bits(0); invalid_rows * H],
            )?;
            if runner.project_qkv(&invalid, &attention).is_ok()
                || runner.project_gate_up(&invalid, &ffn).is_ok()
            {
                return Err("grouped decode accepted an unsupported row count".into());
            }
        }
        let foreign_context = cuda_core::CudaContext::new(0)?;
        let foreign_stream = foreign_context.new_stream()?;
        let foreign_value = matrix(&foreign_stream, KW, H, stride, &mut seed)?;
        let mixed_attention = AttentionWeights::<H, QW, KW, 128>::new(
            ProjectionWeights::new(&query, stride)?,
            ProjectionWeights::new(&key, stride)?,
            ProjectionWeights::new(&foreign_value, stride)?,
            ProjectionWeights::new(&output, QW)?,
            NormWeights::new(&norm)?,
            NormWeights::new(&norm)?,
        );
        let foreign_up = matrix(&foreign_stream, I, H, stride, &mut seed)?;
        let mixed_ffn = FfnWeights::<H, I>::new(
            ProjectionWeights::new(&gate, stride)?,
            ProjectionWeights::new(&foreign_up, stride)?,
            ProjectionWeights::new(&down, I)?,
        );
        if runner.project_qkv(&input, &mixed_attention).is_ok()
            || runner.project_gate_up(&input, &mixed_ffn).is_ok()
        {
            return Err("grouped decode accepted a foreign-context weight".into());
        }
        runner.stream.context().bind_to_thread()?;
    }
    Ok(checked)
}

/// Delay an input producer on another stream and verify grouped execution waits for it.
///
/// Warm caches and allocations first so incidental setup cannot hide a missing
/// readiness fence; verify exact output values and completion on a fresh host thread.
fn foreign_producer(runner: &Runner) -> Result<usize> {
    use std::{sync::mpsc, time::Duration};
    let stream = runner.stream.context().new_stream()?;
    let foreign = runner.stream.context().new_stream()?;
    let grouped = Runner::new(Arc::clone(&stream))?;
    let mut input = Batch::<16>::from_host(&stream, 2, &[BFloat16::from_bits(0); 32])?;
    let source = DeviceBuffer::from_host(&stream, &[BFloat16::from_f32(0.25); 32])?;
    let gate = DeviceBuffer::from_host(&stream, &[BFloat16::from_f32(1.0); 8 * 16])?;
    let up = DeviceBuffer::from_host(&stream, &[BFloat16::from_f32(2.0); 8 * 16])?;
    let down = DeviceBuffer::from_host(&stream, &[BFloat16::from_bits(0); 16 * 8])?;
    let weights = FfnWeights::<16, 8>::new(
        ProjectionWeights::new(&gate, 16)?,
        ProjectionWeights::new(&up, 16)?,
        ProjectionWeights::new(&down, 8)?,
    );
    // Warm both cache and output allocations before the delayed producer, so
    // allocation/preparation cannot accidentally stand in for the readiness fence.
    drop(grouped.project_gate_up(&input, &weights)?);
    let (release, wait_release) = mpsc::channel();
    let (started, wait_started) = mpsc::channel();
    foreign.launch_host_function(move || {
        let _ = started.send(());
        let _ = wait_release.recv_timeout(Duration::from_secs(5));
    })?;
    input
        .buffer_mut()
        .copy_from_device_async(&source, &foreign)?;
    wait_started.recv_timeout(Duration::from_secs(5))?;
    let (finished, wait_finished) = mpsc::channel();
    let (early, result) = std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            // The worker starts without a current CUDA context. No output read
            // or pool return is allowed to hide premature group completion.
            let projected = grouped.project_gate_up(&input, &weights);
            let _ = finished.send(());
            projected
                .and_then(|(gate, up)| {
                    let checked =
                        same_values(&gate.read(&stream)?, &[BFloat16::from_f32(4.0); 16])?;
                    Ok(checked + same_values(&up.read(&stream)?, &[BFloat16::from_f32(8.0); 16])?)
                })
                .map_err(|error| error.to_string())
        });
        let early = wait_finished.recv_timeout(Duration::from_millis(100));
        let _ = release.send(());
        (early, worker.join())
    });
    let result = result.map_err(|_| "grouped decode producer worker panicked")?;
    if !matches!(early, Err(mpsc::RecvTimeoutError::Timeout)) {
        return Err("grouped decode completed before its foreign-stream producer".into());
    }
    let checked = result.map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    foreign.synchronize()?;
    Ok(checked)
}

/// Run grouped projection identity, validation, and delayed-producer GPU fixtures.
pub(super) fn validate(runner: &Runner) -> Result<()> {
    let mut checked = fixture::<32, 16, 8, 24>(runner, 1, 34)?;
    checked += fixture::<17, 19, 7, 23>(runner, 2, 21)?;
    checked += fixture::<48, 24, 16, 16>(runner, 2, 50)?;
    checked += foreign_producer(runner)?;
    println!(
        "grouped decode: {checked} exact values, invalid rows/contexts, and delayed foreign-stream producer passed"
    );
    Ok(())
}

/// CPU-only checks that grouped completion covers errors and unwinding.
#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    /// Verify a partially submitted group still completes and preserves its submission error.
    #[test]
    fn completion_runs_after_partial_submission_error() {
        let events = RefCell::new(Vec::new());
        let result = submit_and_complete(
            || {
                events.borrow_mut().push("first submitted");
                Err("second submission failed".into())
            },
            || {
                events.borrow_mut().push("completed");
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err().to_string(), "second submission failed");
        assert_eq!(*events.borrow(), ["first submitted", "completed"]);
    }

    /// Verify completion errors propagate after successful submission.
    #[test]
    fn completion_failure_is_propagated() {
        let result = submit_and_complete(|| Ok(()), || Err("completion failed".into()));
        assert_eq!(result.unwrap_err().to_string(), "completion failed");
    }

    /// Ensure completion cleanup precedes destruction of caller-owned resources during a panic.
    #[test]
    fn unwinding_completes_before_caller_owned_buffers_drop() {
        /// Record destruction order for a stand-in caller-owned device allocation.
        struct Owner(
            /// Shared ordered log receiving the resource-destruction event.
            Rc<RefCell<Vec<&'static str>>>,
        );
        impl Drop for Owner {
            /// Append the resource-destruction event for comparison with completion order.
            fn drop(&mut self) {
                self.0.borrow_mut().push("buffer dropped");
            }
        }
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _owner = Owner(Rc::clone(&events));
            let _ = submit_and_complete(
                || panic!("panic after submission"),
                || {
                    events.borrow_mut().push("completed");
                    Ok(())
                },
            );
        }));
        assert!(result.is_err());
        assert_eq!(*events.borrow(), ["completed", "buffer dropped"]);
    }
}

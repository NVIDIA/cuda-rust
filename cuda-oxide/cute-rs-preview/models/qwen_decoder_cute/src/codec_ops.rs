/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Checked time-major waveform decoder operations.

use crate::{codec_kernel, model::Batch, operator::Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, simt::LaunchConfig};
use cute_rs::Bf16;
use std::sync::Arc;

/// Checkpoint convolution orientation. These are different physical layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointLayout {
    /// Causal convolution weights stored as `[OUT, IN/groups, K]`.
    Conv,
    /// Transpose convolution weights stored as `[IN, OUT/groups, K]`.
    Transpose,
}

/// Validated channel, tap and group counts with a fixed checkpoint orientation.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    /// Total input channels per time step.
    input: usize,
    /// Total output channels per time step.
    output: usize,
    /// Number of temporal weight taps.
    kernel: usize,
    /// Nonzero group count dividing both channel counts.
    groups: usize,
    /// Physical weight orientation and convolution direction.
    layout: CheckpointLayout,
}
impl Geometry {
    /// Validate nonzero dimensions, divisibility by groups and addressable BF16 weights.
    fn new(
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
        layout: CheckpointLayout,
    ) -> std::result::Result<Self, &'static str> {
        if input == 0
            || output == 0
            || kernel == 0
            || groups == 0
            || !input.is_multiple_of(groups)
            || !output.is_multiple_of(groups)
        {
            return Err("invalid convolution dimensions or groups");
        }
        input
            .checked_mul(output / groups)
            .and_then(|n| n.checked_mul(kernel))
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or("convolution weight extent overflow")?;
        Ok(Self {
            input,
            output,
            kernel,
            groups,
            layout,
        })
    }
    /// Return the checked physical weight extent in BF16 elements.
    fn weight_len(self) -> usize {
        self.input * (self.output / self.groups) * self.kernel
    }
    /// Validate stride/dilation and addressable input/output activation extents.
    ///
    /// Ordinary convolution preserves rows with causal left padding; transpose
    /// returns rows times stride, cropping right padding. Rejects zero step,
    /// transpose stride above kernel length and arithmetic/allocation overflow.
    fn output_rows(self, rows: usize, step: usize) -> std::result::Result<usize, &'static str> {
        if step == 0 {
            return Err("zero stride or dilation");
        }
        let output_rows = match self.layout {
            CheckpointLayout::Conv => {
                (self.kernel - 1)
                    .checked_mul(step)
                    .and_then(|n| n.checked_add(rows))
                    .ok_or("convolution receptive field overflow")?;
                rows
            }
            CheckpointLayout::Transpose => {
                if step > self.kernel {
                    return Err("transpose stride exceeds kernel");
                }
                rows.checked_mul(step)
                    .ok_or("transpose output length overflow")?
            }
        };
        for (count, width) in [(rows, self.input), (output_rows, self.output)] {
            count
                .checked_mul(width)
                .and_then(|n| n.checked_mul(2))
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or("convolution activation extent overflow")?;
        }
        Ok(output_rows)
    }
}

/// A checked borrow of a physical convolution tensor. Group count and orientation
/// are immutable after construction; input/output/kernel widths are Rust types.
pub struct ConvWeights<'a, const IN: usize, const OUT: usize, const K: usize> {
    /// Borrowed physical weights whose length matches the validated geometry.
    weight: &'a DeviceBuffer<Bf16>,
    /// Optional OUT-element bias, added after rounding the convolution to BF16.
    bias: Option<&'a DeviceBuffer<Bf16>>,
    /// Immutable checked grouping, dimensions and orientation.
    geometry: Geometry,
}
impl<'a, const IN: usize, const OUT: usize, const K: usize> ConvWeights<'a, IN, OUT, K> {
    /// Validate group geometry and exact weight/bias lengths without copying data.
    ///
    /// The runner checks CUDA context compatibility when the view is used.
    pub fn new(
        weight: &'a DeviceBuffer<Bf16>,
        bias: Option<&'a DeviceBuffer<Bf16>>,
        layout: CheckpointLayout,
        groups: usize,
    ) -> Result<Self> {
        let geometry = Geometry::new(IN, OUT, K, groups, layout)?;
        if weight.len() != geometry.weight_len() || bias.is_some_and(|b| b.len() != OUT) {
            return Err("convolution checkpoint tensor extent mismatch".into());
        }
        Ok(Self {
            weight,
            bias,
            geometry,
        })
    }
}

/// Synchronous baseline. Every safe launch retains its inputs and owned output
/// until stream completion, including enqueue errors.
pub struct Runner {
    /// Stream on which all operations execute and are completed before return.
    stream: Arc<CudaStream>,
    /// Matching embedded waveform operation entry points.
    kernels: codec_kernel::kernels::LoadedModule,
}
impl Runner {
    /// Load waveform kernels into the stream context, propagating CUDA errors.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            // SAFETY: this binary loads its own matching embedded kernel artifact.
            kernels: unsafe { codec_kernel::kernels::load(stream.context())? },
            stream,
        })
    }
    /// Reject buffers from other contexts, then synchronize prior context work.
    fn ready(&self, buffers: &[&DeviceBuffer<Bf16>]) -> Result<()> {
        if buffers
            .iter()
            .any(|b| !Arc::ptr_eq(b.context(), self.stream.context()))
        {
            return Err("codec buffers must share runner context".into());
        }
        self.stream.context().synchronize()?;
        Ok(())
    }
    /// Synchronize even after enqueue failure; report the launch error before completion errors.
    fn finish(&self, launch: Result<()>) -> Result<()> {
        let completion = self.stream.synchronize();
        launch?;
        completion?;
        Ok(())
    }
    /// Run causal grouped convolution over `[time, IN]`, preserving time length.
    ///
    /// Dilation must be positive and weights must use `Conv` orientation. Returns
    /// geometry/context/launch errors and keeps inputs live through completion.
    pub fn conv<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        weights: &ConvWeights<'_, IN, OUT, K>,
        dilation: usize,
    ) -> Result<Batch<OUT>> {
        if weights.geometry.layout != CheckpointLayout::Conv {
            return Err("expected convolution checkpoint layout".into());
        }
        self.run_conv(input, weights, dilation)
    }
    /// Upsample `[time, IN]` to `[time * stride, OUT]` with right cropping.
    ///
    /// Requires `Transpose` weights and `1 <= stride <= K`; returns geometry,
    /// context or CUDA errors and waits for stream completion before return.
    pub fn transpose<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        weights: &ConvWeights<'_, IN, OUT, K>,
        stride: usize,
    ) -> Result<Batch<OUT>> {
        if weights.geometry.layout != CheckpointLayout::Transpose {
            return Err("expected transposed convolution checkpoint layout".into());
        }
        self.run_conv(input, weights, stride)
    }
    /// Normalize each row in FP32 and apply C-element affine weights before BF16 storage.
    ///
    /// Requires a finite positive epsilon, matching affine extents and one CUDA
    /// context. Returns validation or CUDA errors; empty inputs launch no kernel.
    pub fn layer_norm<const C: usize>(
        &self,
        input: &Batch<C>,
        weight: &DeviceBuffer<Bf16>,
        bias: &DeviceBuffer<Bf16>,
        epsilon: f32,
    ) -> Result<Batch<C>> {
        if weight.len() != C || bias.len() != C || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("invalid affine LayerNorm weights or epsilon".into());
        }
        self.ready(&[input.buffer(), weight, bias])?;
        let mut output = Batch::allocate(&self.stream, input.rows())?;
        if let Some(config) = grid(input.rows(), 4)? {
            // SAFETY: one complete warp owns each checked dense row.
            let prepared = self.kernels.prepare_layer_norm_bf16(LaunchConfig1D::new(
                config.grid_dim.0,
                128,
                0,
            ))?;
            let launch = {
                self.kernels
                    .layer_norm_bf16(
                        &self.stream,
                        &prepared,
                        input.buffer(),
                        weight,
                        bias,
                        output.buffer_mut(),
                        input.rows(),
                        C,
                        epsilon,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Normalize codebook accumulator rows once before embedding lookup.
    /// Divide each `[time, C]` row by its BF16 count clamped below by the
    /// BF16-rounded epsilon. Rejects denominator length/context mismatches and
    /// nonpositive/nonfinite epsilon; returns CUDA errors after completion.
    pub fn normalize_rows<const C: usize>(
        &self,
        input: &Batch<C>,
        denominator: &DeviceBuffer<Bf16>,
        epsilon: f32,
    ) -> Result<Batch<C>> {
        self.normalize_rows_buffer(input.buffer(), input.rows(), denominator, epsilon)
    }
    /// Borrow checkpoint storage directly; no staging readback is required.
    /// Input must contain exactly `rows * C` BF16 values and denominator must
    /// contain `rows` values. Epsilon is finite and positive and is rounded to
    /// BF16 before comparison with each count. Returns extent/context/CUDA errors.
    pub fn normalize_rows_buffer<const C: usize>(
        &self,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        denominator: &DeviceBuffer<Bf16>,
        epsilon: f32,
    ) -> Result<Batch<C>> {
        let shape = cute_rs::ffn::RowMajorShape::dense(rows, C)?;
        if input.len() != shape.required_len()
            || denominator.len() != rows
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err("invalid codebook row extents, denominators or epsilon".into());
        }
        self.unary_buffer(input, rows, denominator, input, 3, epsilon, 0.0)
    }
    /// Multiply matching BF16 batches in FP32 and round each result once.
    ///
    /// Mismatched row counts, contexts and CUDA failures return errors.
    pub fn multiply<const C: usize>(&self, left: &Batch<C>, right: &Batch<C>) -> Result<Batch<C>> {
        if left.rows() != right.rows() {
            return Err("elementwise product row mismatch".into());
        }
        self.unary(left, right.buffer(), left.buffer(), 4, 0.0, 0.0)
    }
    /// Apply per-channel SnakeBeta using BF16 log-alpha and log-beta vectors.
    ///
    /// Preserves eager BF16 intermediate rounding. Rejects wrong parameter
    /// extents or contexts and propagates CUDA failures.
    pub fn snake<const C: usize>(
        &self,
        input: &Batch<C>,
        log_alpha: &DeviceBuffer<Bf16>,
        log_beta: &DeviceBuffer<Bf16>,
    ) -> Result<Batch<C>> {
        if log_alpha.len() != C || log_beta.len() != C {
            return Err("SnakeBeta channel parameter extent mismatch".into());
        }
        self.unary(input, log_alpha, log_beta, 0, 0.0, 0.0)
    }
    /// Apply the erf-form GELU approximation and round its FP32 result to BF16.
    pub fn gelu<const C: usize>(&self, input: &Batch<C>) -> Result<Batch<C>> {
        self.unary(input, input.buffer(), input.buffer(), 1, 0.0, 0.0)
    }
    /// Clamp each value to finite ordered bounds, then store BF16.
    ///
    /// Rejects nonfinite or reversed bounds and propagates context/CUDA errors.
    pub fn clamp<const C: usize>(
        &self,
        input: &Batch<C>,
        lower: f32,
        upper: f32,
    ) -> Result<Batch<C>> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err("invalid waveform clamp bounds".into());
        }
        self.unary(input, input.buffer(), input.buffer(), 2, lower, upper)
    }
    /// Forward a typed batch to the private unary dispatcher with its checked row count.
    fn unary<const C: usize>(
        &self,
        input: &Batch<C>,
        alpha: &DeviceBuffer<Bf16>,
        beta: &DeviceBuffer<Bf16>,
        operation: u32,
        lower: f32,
        upper: f32,
    ) -> Result<Batch<C>> {
        self.unary_buffer(
            input.buffer(),
            input.rows(),
            alpha,
            beta,
            operation,
            lower,
            upper,
        )
    }
    /// Dispatch a validated private operation tag into a fresh `[rows, C]` batch.
    ///
    /// Callers establish operation-specific parameter lengths and input extent;
    /// this helper checks contexts and grid range and waits through launch errors.
    #[allow(clippy::too_many_arguments)]
    fn unary_buffer<const C: usize>(
        &self,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        alpha: &DeviceBuffer<Bf16>,
        beta: &DeviceBuffer<Bf16>,
        operation: u32,
        lower: f32,
        upper: f32,
    ) -> Result<Batch<C>> {
        self.ready(&[input, alpha, beta])?;
        let mut output = Batch::allocate(&self.stream, rows)?;
        if let Some(config) = grid(input.len(), 128)? {
            // SAFETY: private operation tags select only validated parameter
            // extents; one thread owns one output in the dense logical range.
            let launch = unsafe {
                self.kernels
                    .unary_bf16(
                        &self.stream,
                        config,
                        input,
                        alpha,
                        beta,
                        output.buffer_mut(),
                        input.len(),
                        C,
                        operation,
                        lower,
                        upper,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Compute residual plus a per-channel scaled input with two BF16 rounding steps.
    ///
    /// The product is rounded before addition. Rejects unequal row counts, scale
    /// extents or contexts, and propagates CUDA failures.
    pub fn scaled_add<const C: usize>(
        &self,
        input: &Batch<C>,
        residual: &Batch<C>,
        scale: &DeviceBuffer<Bf16>,
    ) -> Result<Batch<C>> {
        if input.rows() != residual.rows() || scale.len() != C {
            return Err("scaled residual extent mismatch".into());
        }
        self.ready(&[input.buffer(), residual.buffer(), scale])?;
        let mut output = Batch::allocate(&self.stream, input.rows())?;
        if let Some(config) = grid(input.buffer().len(), 128)? {
            // SAFETY: exact matching dense tensors and channel scale; disjoint
            // output ownership with one writer per logical element.
            let launch = unsafe {
                self.kernels
                    .scaled_add_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        residual.buffer(),
                        scale,
                        output.buffer_mut(),
                        input.buffer().len(),
                        C,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Validate output extent and schedule four MMA tiles per 128-thread block.
    ///
    /// Allocates a disjoint output and retains every operand through completion;
    /// orientation chooses whether step means dilation or transpose stride.
    fn run_conv<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        weights: &ConvWeights<'_, IN, OUT, K>,
        step: usize,
    ) -> Result<Batch<OUT>> {
        let g = weights.geometry;
        let rows = g.output_rows(input.rows(), step)?;
        let transpose = g.layout == CheckpointLayout::Transpose;
        let phases = if transpose { step } else { 1 };
        let tiles = input
            .rows()
            .div_ceil(16)
            .checked_mul((OUT / g.groups).div_ceil(8))
            .and_then(|n| n.checked_mul(g.groups))
            .and_then(|n| n.checked_mul(phases))
            .ok_or("convolution launch extent overflow")?;
        let config = grid(tiles, 4)?;
        let bias = weights.bias.unwrap_or(input.buffer());
        self.ready(&[input.buffer(), weights.weight, bias])?;
        let mut output = Batch::allocate(&self.stream, rows)?;
        if let Some(config) = config {
            // SAFETY: dimensions, groups, physical weight layout and all extents
            // are checked; full warps write disjoint MMA tiles.
            let prepared =
                self.kernels
                    .prepare_conv_bf16(LaunchConfig1D::new(config.grid_dim.0, 128, 0))?;
            let launch = {
                self.kernels
                    .conv_bf16(
                        &self.stream,
                        &prepared,
                        input.buffer(),
                        weights.weight,
                        bias,
                        output.buffer_mut(),
                        input.rows(),
                        IN,
                        OUT,
                        K,
                        step,
                        g.groups,
                        weights.bias.is_some(),
                        transpose,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
}
/// Build a 128-thread launch covering `work` logical items at positive `per_block`.
///
/// Zero work returns no launch; block counts above the CUDA grid-x limit fail.
fn grid(work: usize, per_block: usize) -> Result<Option<LaunchConfig>> {
    if work == 0 {
        return Ok(None);
    }
    let blocks = work.div_ceil(per_block);
    if blocks > i32::MAX as usize {
        return Err("codec grid exceeds CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Tiny asymmetric golden cases catch kernel reversal, dilation, channel layout,
/// transposed-convolution phase/cropping, grouping, bias and tile tails.
pub fn validate(stream: Arc<CudaStream>) -> Result<()> {
    /// Compare one asymmetric grouped convolution case against an FP64 scalar oracle.
    ///
    /// The oracle rounds the dot product before bias and checks every time/channel
    /// coordinate, including causal padding and transpose phases.
    fn case<const IN: usize, const OUT: usize, const K: usize>(
        runner: &Runner,
        rows: usize,
        groups: usize,
        step: usize,
        layout: CheckpointLayout,
    ) -> Result<()> {
        let input: Vec<_> = (0..rows * IN)
            .map(|i| Bf16::from_f32(((i * 7 % 17) as f32 - 8.0) / 8.0))
            .collect();
        let weight: Vec<_> = (0..IN * (OUT / groups) * K)
            .map(|i| Bf16::from_f32(((i * 3 % 13) as f32 - 6.0) / 16.0))
            .collect();
        let bias: Vec<_> = (0..OUT)
            .map(|i| Bf16::from_f32((i as f32 - 3.0) / 32.0))
            .collect();
        let x = Batch::<IN>::from_host(&runner.stream, rows, &input)?;
        let w = DeviceBuffer::from_host(&runner.stream, &weight)?;
        let b = DeviceBuffer::from_host(&runner.stream, &bias)?;
        let has_bias = K != 2;
        let weights = ConvWeights::<IN, OUT, K>::new(&w, has_bias.then_some(&b), layout, groups)?;
        let result = match layout {
            CheckpointLayout::Conv => runner.conv(&x, &weights, step)?,
            CheckpointLayout::Transpose => runner.transpose(&x, &weights, step)?,
        };
        let got = result.read(&runner.stream)?;
        for t in 0..result.rows() {
            for co in 0..OUT {
                let group = co / (OUT / groups);
                let mut expected = 0.0;
                for ci in 0..IN / groups {
                    for tap in 0..K {
                        let (source, wi) = match layout {
                            CheckpointLayout::Conv => (
                                t.checked_sub(step * (K - 1 - tap)),
                                (co * (IN / groups) + ci) * K + tap,
                            ),
                            CheckpointLayout::Transpose => (
                                (t >= tap && (t - tap).is_multiple_of(step))
                                    .then(|| (t - tap) / step),
                                ((group * (IN / groups) + ci) * (OUT / groups)
                                    + co % (OUT / groups))
                                    * K
                                    + tap,
                            ),
                        };
                        if let Some(source) = source.filter(|&s| s < rows) {
                            expected += input[source * IN + group * (IN / groups) + ci].to_f32()
                                as f64
                                * weight[wi].to_f32() as f64;
                        }
                    }
                }
                let dot = crate::reference::round(expected).to_f32();
                let expected = if has_bias {
                    crate::reference::round(dot as f64 + bias[co].to_f32() as f64).to_f32()
                } else {
                    dot
                };
                let actual = got[t * OUT + co].to_f32();
                if !actual.is_finite() || (actual - expected).abs() > 2e-5 + 0.008 * expected.abs()
                {
                    return Err(format!("codec convolution mismatch {layout:?} {IN}->{OUT} K{K} t{t} co{co}: {actual} != {expected}").into());
                }
            }
        }
        Ok(())
    }
    let runner = Runner::new(stream)?;
    case::<3, 5, 3>(&runner, 19, 1, 2, CheckpointLayout::Conv)?;
    case::<6, 10, 7>(&runner, 17, 2, 3, CheckpointLayout::Conv)?;
    case::<5, 5, 7>(&runner, 23, 5, 9, CheckpointLayout::Conv)?;
    case::<3, 5, 6>(&runner, 19, 1, 3, CheckpointLayout::Transpose)?;
    case::<6, 10, 5>(&runner, 17, 2, 3, CheckpointLayout::Transpose)?;
    case::<3, 5, 2>(&runner, 1, 1, 2, CheckpointLayout::Transpose)?;
    case::<3, 5, 3>(&runner, 0, 1, 1, CheckpointLayout::Conv)?;
    validate_conv_bias(&runner)?;
    validate_elements(&runner)?;
    validate_epsilon(&runner)?;
    println!("codec convolution, activation, normalization and epsilon fixtures passed");
    Ok(())
}

/// Distinguish dot-before-bias BF16 rounding from a fused FP32 bias addition.
fn validate_conv_bias(runner: &Runner) -> Result<()> {
    let input = Batch::<2>::from_host(&runner.stream, 1, &[1.0, 1.0 / 256.0].map(Bf16::from_f32))?;
    let weight = DeviceBuffer::from_host(&runner.stream, &[Bf16::from_f32(1.0); 2])?;
    let bias = DeviceBuffer::from_host(&runner.stream, &[Bf16::from_f32(1.0 / 1024.0)])?;
    for layout in [CheckpointLayout::Conv, CheckpointLayout::Transpose] {
        let w = ConvWeights::<2, 1, 1>::new(&weight, Some(&bias), layout, 1)?;
        let output = match layout {
            CheckpointLayout::Conv => runner.conv(&input, &w, 1)?,
            CheckpointLayout::Transpose => runner.transpose(&input, &w, 1)?,
        };
        let value = output.read(&runner.stream)?[0].to_f32();
        // cuDNN BF16 convolution rounds the dot before its separate bias add.
        // Fusing bias into FP32 would incorrectly produce 1.0078125 here.
        if value != 1.0 {
            return Err(format!("codec {layout:?} bias must round dot first: {value} != 1").into());
        }
    }
    Ok(())
}

/// Compare ragged-width activations, normalization and residual operations with FP64 oracles.
fn validate_elements(runner: &Runner) -> Result<()> {
    /// Channel count deliberately leaving a partial warp partition.
    const C: usize = 35;
    let values: Vec<_> = (0..C * 19)
        .map(|i| Bf16::from_f32(((i * 17 % 127) as f32 - 63.0) / 8.0))
        .collect();
    let alpha: Vec<_> = (0..C)
        .map(|i| Bf16::from_f32((i as f32 - 17.0) / 16.0))
        .collect();
    let beta: Vec<_> = (0..C)
        .map(|i| Bf16::from_f32((i as f32 - 17.0) / 32.0))
        .collect();
    let input = Batch::<C>::from_host(&runner.stream, 19, &values)?;
    let denominators: Vec<_> = (0..19)
        .map(|i| Bf16::from_f32(if i == 1 { 0.0 } else { (i + 1) as f32 / 4.0 }))
        .collect();
    let denom = DeviceBuffer::from_host(&runner.stream, &denominators)?;
    let a = DeviceBuffer::from_host(&runner.stream, &alpha)?;
    let b = DeviceBuffer::from_host(&runner.stream, &beta)?;
    let round = |x: f64| crate::reference::round(x).to_f32() as f64;
    // libc's double precision erf is independent of the GPU polynomial.
    #[link(name = "m")]
    unsafe extern "C" {
        /// Evaluate the system FP64 error function independently of the device approximation.
        fn erf(value: f64) -> f64;
    }
    for (name, output) in [
        ("gelu", runner.gelu(&input)?),
        (
            "normalize_rows",
            runner.normalize_rows(&input, &denom, 0.25)?,
        ),
        ("multiply", runner.multiply(&input, &input)?),
        ("snake", runner.snake(&input, &a, &b)?),
        ("clamp", runner.clamp(&input, -1.0, 1.0)?),
        ("layernorm", runner.layer_norm(&input, &a, &b, 1e-6)?),
        ("scaled_add", runner.scaled_add(&input, &input, &a)?),
    ] {
        let got = output.read(&runner.stream)?;
        for (i, &actual) in got.iter().enumerate() {
            let x = values[i].to_f32() as f64;
            let channel = i % C;
            let expected = match name {
                "gelu" => {
                    // SAFETY: libc erf accepts every scalar floating value.
                    round(0.5 * x * (1.0 + unsafe { erf(x * core::f64::consts::FRAC_1_SQRT_2) }))
                }
                "snake" => {
                    let a = round((alpha[channel].to_f32() as f64).exp());
                    let b = round((beta[channel].to_f32() as f64).exp());
                    let reciprocal = round(1.0 / round(b + 1e-9));
                    let sine = round(round(x * a).sin());
                    round(x + round(reciprocal * round(sine * sine)))
                }
                "clamp" => x.clamp(-1.0, 1.0),
                "normalize_rows" => round(x / (denominators[i / C].to_f32() as f64).max(0.25)),
                "multiply" => round(x * x),
                "scaled_add" => round(x + round(x * alpha[channel].to_f32() as f64)),
                _ => {
                    let row = &values[i / C * C..(i / C + 1) * C];
                    let mean = row.iter().map(|v| v.to_f32() as f64).sum::<f64>() / C as f64;
                    let variance = row
                        .iter()
                        .map(|v| (v.to_f32() as f64 - mean).powi(2))
                        .sum::<f64>()
                        / C as f64;
                    round(
                        (x - mean) / (variance + 1e-6).sqrt() * alpha[channel].to_f32() as f64
                            + beta[channel].to_f32() as f64,
                    )
                }
            } as f32;
            let actual = actual.to_f32();
            if !actual.is_finite() || (actual - expected).abs() > 2e-5 + 0.008 * expected.abs() {
                return Err(
                    format!("codec {name} mismatch element{i}: {actual} != {expected}").into(),
                );
            }
        }
    }
    Ok(())
}

/// Check codec/language RMSNorm epsilon values, the language default and invalid inputs.
fn validate_epsilon(runner: &Runner) -> Result<()> {
    use cute_rs::decoder::{FinalNorm, NormWeights};
    let values = [1e-4, -2e-4, 3e-4].map(crate::reference::round);
    let input = Batch::<3>::from_host(&runner.stream, 1, &values)?;
    let weights = DeviceBuffer::from_host(&runner.stream, &[Bf16::from_f32(1.0); 3])?;
    let weights = NormWeights::<FinalNorm, 3>::new(&weights)?;
    let model = crate::model::Runner::new(runner.stream.clone())?;
    let square_mean = values
        .iter()
        .map(|v| (v.to_f32() as f64).powi(2))
        .sum::<f64>()
        / 3.0;
    let mut outputs = Vec::new();
    for epsilon in [1e-6_f32, 1e-5_f32] {
        let got = model
            .norm_with_epsilon(&input, &weights, epsilon)?
            .read(&runner.stream)?;
        for (input, got) in values.iter().zip(&got) {
            let expected = crate::reference::round(
                input.to_f32() as f64 / (square_mean + epsilon as f64).sqrt(),
            )
            .to_f32();
            if (got.to_f32() - expected).abs() > 2e-5 + 0.008 * expected.abs() {
                return Err(format!(
                    "RMSNorm epsilon {epsilon} mismatch: {} != {expected}",
                    got.to_f32()
                )
                .into());
            }
        }
        outputs.push(got);
    }
    if outputs[0][0].to_bits() == outputs[1][0].to_bits() {
        return Err("RMSNorm ignored distinct language/codec epsilon".into());
    }
    let default = model.norm(&input, &weights)?.read(&runner.stream)?;
    if default
        .iter()
        .zip(&outputs[0])
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("default language RMSNorm epsilon changed".into());
    }
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        if model.norm_with_epsilon(&input, &weights, invalid).is_ok() {
            return Err("invalid RMSNorm epsilon accepted".into());
        }
    }
    Ok(())
}

/// Host-only convolution geometry and overflow checks.
#[cfg(test)]
mod tests {
    use super::*;
    /// Check causal convolution preserves time and accounts for grouped weight storage.
    #[test]
    fn convolution_shape_preserves_time_and_grouped_weight_extent() {
        let g = Geometry::new(6, 10, 7, 2, CheckpointLayout::Conv).unwrap();
        assert_eq!(g.weight_len(), 210);
        assert_eq!(g.output_rows(19, 3).unwrap(), 19);
        assert_eq!(g.output_rows(0, 9).unwrap(), 0);
    }
    /// Check transpose output length and reject unsupported strides.
    #[test]
    fn transpose_crops_right_padding_to_exact_stride_times_input() {
        let g = Geometry::new(3, 5, 6, 1, CheckpointLayout::Transpose).unwrap();
        assert_eq!(g.weight_len(), 90);
        assert_eq!(g.output_rows(17, 3).unwrap(), 51);
        assert_eq!(g.output_rows(0, 3).unwrap(), 0);
        assert!(g.output_rows(1, 0).is_err());
        assert!(g.output_rows(1, 7).is_err());
    }
    /// Reject zero geometry, indivisible groups and activation/receptive-field overflow.
    #[test]
    fn rejects_zero_dimensions_bad_groups_and_overflow() {
        for (i, o, k, g) in [
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (3, 4, 1, 2),
            (4, 3, 1, 2),
            (usize::MAX, 2, 7, 1),
        ] {
            assert!(Geometry::new(i, o, k, g, CheckpointLayout::Conv).is_err());
        }
        let g = Geometry::new(3, 5, 6, 1, CheckpointLayout::Transpose).unwrap();
        assert!(g.output_rows(usize::MAX, 3).is_err());
        let g = Geometry::new(3, 5, 7, 1, CheckpointLayout::Conv).unwrap();
        assert!(g.output_rows(1, usize::MAX).is_err());
    }
}

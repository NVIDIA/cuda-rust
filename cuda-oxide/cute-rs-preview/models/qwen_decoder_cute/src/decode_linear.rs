/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! One/two-row BF16 projections in the reference's per-shape reduction order.
use crate::{
    ffn_kernel::kernels,
    model::Batch,
    operator::Result,
    reference_linear::{ReductionPlan, emulate_fma_chains, fma_chain_blocks, reference_plan},
    reference_linear_kernel::kernels as reference,
};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cute_rs::{
    BFloat16,
    decoder::BiasWeights,
    ffn::{ProjectionRole, ProjectionWeights, RowMajorShape},
};
use std::sync::Arc;

/// Bounded per-kernel reuse of prepared launch configurations.
#[path = "prepared_cache.rs"]
mod prepared_cache;
use prepared_cache::PreparedCache;

/// Synchronous Q/K/V and gate/up submission groups sharing this runner.
#[path = "decode_linear_grouped.rs"]
mod grouped;

/// Validate one or two input rows and return one 32-thread CTA per eight output channels.
///
/// Rejects zero output width and a grid exceeding CUDA's signed grid-x limit.
fn launch_config(rows: usize, outputs: usize) -> Result<LaunchConfig1D> {
    if !(1..=2).contains(&rows) || outputs == 0 {
        return Err("decode projection requires one/two rows and positive output width".into());
    }
    let blocks = outputs.div_ceil(8);
    if blocks > i32::MAX as usize {
        return Err("decode projection grid exceeds CUDA grid-x range".into());
    }
    Ok(LaunchConfig1D::new(blocks as u32, 32, 0))
}

/// Synchronous one/two-row BF16 projection runner with pooled outputs and prepared launches.
///
/// Computes `input[rows, IN] * weight[OUT, IN]^T` in the reduction order the
/// reference uses for that shape (see `reference_linear`), adding any bias in
/// FP32 before the single BF16 output rounding.
pub struct Runner {
    /// Execution stream; context-wide readiness fences also cover external producers.
    stream: Arc<CudaStream>,
    /// Owned kernel module kept alive for its prepared projections.
    module: kernels::LoadedModule,
    /// Module with the FMA-chain and split-K entries.
    reference: reference::LoadedModule,
    /// Initialized output allocations shared only with runners in the same context.
    pool: crate::model::BufferPool,
    /// Unbiased launch configurations indexed by the number of eight-column CTAs.
    launches: PreparedCache<PreparedLaunch<kernels::__decode_linear_bf16_CudaKernel>>,
    /// Separate cache for the biased kernel, whose prepared launch type is distinct.
    biased_launches: PreparedCache<PreparedLaunch<kernels::__decode_linear_bias_bf16_CudaKernel>>,
    /// Sixteen-chain launches indexed by CTA count.
    fma16_launches: PreparedCache<PreparedLaunch<reference::__fma16_linear_bf16_CudaKernel>>,
    /// Biased sixteen-chain launches indexed by CTA count.
    fma16_bias_launches:
        PreparedCache<PreparedLaunch<reference::__fma16_linear_bias_bf16_CudaKernel>>,
    /// Thirty-two-chain launches indexed by CTA count.
    fma32_launches: PreparedCache<PreparedLaunch<reference::__fma32_linear_bf16_CudaKernel>>,
    /// Biased thirty-two-chain launches indexed by CTA count.
    fma32_bias_launches:
        PreparedCache<PreparedLaunch<reference::__fma32_linear_bias_bf16_CudaKernel>>,
    /// Split-K launches indexed by eight-column CTA count.
    split_k_launches: PreparedCache<PreparedLaunch<reference::__split_k_linear_bf16_CudaKernel>>,
    /// Biased split-K launches indexed by eight-column CTA count.
    split_k_bias_launches:
        PreparedCache<PreparedLaunch<reference::__split_k_linear_bias_bf16_CudaKernel>>,
}
impl Runner {
    /// Load the projection kernels and create a bounded private output pool.
    ///
    /// Returns CUDA module-loading errors; no model weights are loaded.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Self::with_pool(stream, crate::model::buffer_pool())
    }
    /// Construct a runner using an existing same-context pool and fresh launch caches.
    ///
    /// Pool context compatibility is rechecked when an output allocation is acquired.
    pub(crate) fn with_pool(
        stream: Arc<CudaStream>,
        pool: crate::model::BufferPool,
    ) -> Result<Self> {
        Ok(Self {
            // SAFETY: this example embeds the artifacts generated from the
            // matching kernel sources; every launch validates its full-warp contract.
            module: unsafe { kernels::load(stream.context())? },
            reference: unsafe { reference::load(stream.context())? },
            stream,
            pool,
            launches: PreparedCache::default(),
            biased_launches: PreparedCache::default(),
            fma16_launches: PreparedCache::default(),
            fma16_bias_launches: PreparedCache::default(),
            fma32_launches: PreparedCache::default(),
            fma32_bias_launches: PreparedCache::default(),
            split_k_launches: PreparedCache::default(),
            split_k_bias_launches: PreparedCache::default(),
        })
    }

    /// Widths and semantic roles remain in the borrowed weight/bias types.
    /// Strides are checked independently; input/output storage never aliases.
    ///
    /// `rows` must be one or two, and `input_stride` counts BF16 elements between
    /// input rows. Weights use `[OUT, IN]` order with their own checked stride.
    /// All buffers must share the runner context; a context fence precedes work
    /// and stream completion is attempted even when enqueue fails. Shape, grid,
    /// allocation, and CUDA errors propagate. Successful output is dense BF16.
    pub fn project_buffer<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &DeviceBuffer<BFloat16>,
        rows: usize,
        input_stride: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        let config = launch_config(rows, OUT)?;
        // Block shape and resources are fixed for each branded kernel. Only
        // grid-x varies; tensor dimensions, extents and contexts are still
        // checked on every invocation, independently of this cache key.
        let blocks = OUT.div_ceil(8) as u32;
        RowMajorShape::new(rows, IN, input_stride)?.validate_len(input.len())?;
        let bias_buffer = bias.map_or(input, |bias| bias.buffer());
        for buffer in [input, weight.buffer(), bias_buffer] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("decode projection buffers must share the runner context".into());
            }
        }
        self.stream.context().synchronize()?;
        let mut output = Batch::allocate_from(&self.pool, &self.stream, rows)?;
        // Keep all borrowed buffers live through completion on errors.
        let launched = match reference_plan(rows, IN, OUT) {
            ReductionPlan::FmaChains { chains } => {
                self.launch_fma_chains(chains, input, weight, bias, &mut output)
            }
            ReductionPlan::MmaSplitK { slice_width } => self.launch_split_k(
                slice_width,
                input,
                rows,
                input_stride,
                weight,
                bias,
                &mut output,
            ),
            // Checked full warps execute exactly the original 16x8 tile:
            // identical A/B packing, ascending K loop and FP32 epilogue. Only
            // CTA ownership changes.
            ReductionPlan::MmaChain => match bias {
                Some(bias) => {
                    let launch = self.biased_launches.get_or_prepare(blocks, || {
                        self.module.prepare_decode_linear_bias_bf16(config)
                    })?;
                    self.module
                        .decode_linear_bias_bf16(
                            &self.stream,
                            &launch,
                            input,
                            weight.buffer(),
                            bias.buffer(),
                            output.buffer_mut(),
                            rows,
                            IN,
                            OUT,
                            input_stride,
                            weight.shape().row_stride(),
                        )
                        .map_err(Into::into)
                }
                None => {
                    let launch = self.launches.get_or_prepare(blocks, || {
                        self.module.prepare_decode_linear_bf16(config)
                    })?;
                    self.module
                        .decode_linear_bf16(
                            &self.stream,
                            &launch,
                            input,
                            weight.buffer(),
                            output.buffer_mut(),
                            rows,
                            IN,
                            OUT,
                            input_stride,
                            weight.shape().row_stride(),
                        )
                        .map_err(Into::into)
                }
            },
        };
        let completed = self.stream.synchronize();
        launched?;
        completed?;
        Ok(output)
    }

    /// Enqueue the one-row FMA-chain entry for `chains` interleaved accumulators.
    ///
    /// The host repeats the entry's extent, pitch and alignment predicate so a
    /// rejected launch surfaces as an error instead of an untouched output.
    fn launch_fma_chains<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        chains: usize,
        input: &DeviceBuffer<BFloat16>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let stride = weight.shape().row_stride();
        if !(IN.is_multiple_of(chains)
            && stride >= IN
            && input.len() >= IN
            && OUT <= weight.buffer().len() / stride
            && bias.is_none_or(|bias| bias.buffer().len() >= OUT))
        {
            return Err("decode FMA-chain projection extents or pitch are unsupported".into());
        }
        let blocks = fma_chain_blocks(chains, OUT) as u32;
        let config = LaunchConfig1D::new(blocks, 32, 0);
        match (chains, bias) {
            (16, None) => {
                let launch = self
                    .fma16_launches
                    .get_or_prepare(blocks, || self.reference.prepare_fma16_linear_bf16(config))?;
                self.reference.fma16_linear_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            (16, Some(bias)) => {
                let launch = self.fma16_bias_launches.get_or_prepare(blocks, || {
                    self.reference.prepare_fma16_linear_bias_bf16(config)
                })?;
                self.reference.fma16_linear_bias_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    bias.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            (32, None) => {
                let launch = self
                    .fma32_launches
                    .get_or_prepare(blocks, || self.reference.prepare_fma32_linear_bf16(config))?;
                self.reference.fma32_linear_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            (32, Some(bias)) => {
                let launch = self.fma32_bias_launches.get_or_prepare(blocks, || {
                    self.reference.prepare_fma32_linear_bias_bf16(config)
                })?;
                self.reference.fma32_linear_bias_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    bias.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            _ => {
                return Err("decode FMA-chain projection has no entry for this chain count".into());
            }
        }
        Ok(())
    }

    /// Enqueue the split-K tensor-core entry with `slice_width` elements per slice.
    #[allow(clippy::too_many_arguments)]
    fn launch_split_k<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        slice_width: usize,
        input: &DeviceBuffer<BFloat16>,
        rows: usize,
        input_stride: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let stride = weight.shape().row_stride();
        if !(matches!(rows, 1 | 2)
            && IN.is_multiple_of(16)
            && slice_width.is_multiple_of(16)
            && slice_width != 0
            && OUT.is_multiple_of(8)
            && input_stride >= IN
            && stride >= IN
            && input_stride.is_multiple_of(2)
            && stride.is_multiple_of(2)
            && rows <= input.len() / input_stride
            && OUT <= weight.buffer().len() / stride
            && bias.is_none_or(|bias| bias.buffer().len() >= OUT))
        {
            return Err("decode split-K projection extents, pitch or slice are unsupported".into());
        }
        let blocks = OUT.div_ceil(8) as u32;
        let config = LaunchConfig1D::new(blocks, 32, 0);
        match bias {
            None => {
                let launch = self.split_k_launches.get_or_prepare(blocks, || {
                    self.reference.prepare_split_k_linear_bf16(config)
                })?;
                self.reference.split_k_linear_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    input_stride,
                    stride,
                    slice_width,
                )?;
            }
            Some(bias) => {
                let launch = self.split_k_bias_launches.get_or_prepare(blocks, || {
                    self.reference.prepare_split_k_linear_bias_bf16(config)
                })?;
                self.reference.split_k_linear_bias_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    bias.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    input_stride,
                    stride,
                    slice_width,
                )?;
            }
        }
        Ok(())
    }
}

/// Compare nonempty equal-length outputs to an FP64 oracle narrowed through FP32 to BF16.
///
/// Rejects nonfinite values or absolute error above `2e-5 + 0.008 * abs(expected)`;
/// returns the maximum absolute error. This tolerance does not prove bit identity.
fn compare(actual: &[BFloat16], expected: &[f64], label: &str) -> Result<f64> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!("{label}: comparison requires equal nonempty lengths").into());
    }
    let mut maximum = 0.0f64;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual.to_f32());
        let expected = f64::from(BFloat16::from_f32(*expected as f32).to_f32());
        let difference = (actual - expected).abs();
        // The existing per-operation synthetic gate; no new tolerance for
        // changing the reduction order. FP64 oracle does not use warp grouping.
        if !actual.is_finite()
            || !expected.is_finite()
            || difference > 2e-5 + 0.008 * expected.abs()
        {
            return Err(format!(
                "{label}[{index}]: {actual} != FP64/BF16 reference {expected} (error {difference})"
            )
            .into());
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}

/// Require nonempty equal extents and exact BF16 bits against `what`.
fn exact_bits(actual: &[BFloat16], expected: &[BFloat16], label: &str, what: &str) -> Result<()> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!("{label}: exact comparison requires equal nonempty lengths").into());
    }
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "{label}[{index}]: bits {:04x} differ from {what} {:04x}",
                a.to_bits(),
                b.to_bits()
            )
            .into());
        }
    }
    Ok(())
}

/// Deterministic data families for numerical and arithmetic-preservation fixtures.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    /// Small moderate-magnitude values checked against the FP64 accuracy bound.
    Regular,
    /// Alternating terms that exercise cancellation and the FP32 bias epilogue.
    Cancellation,
    /// Signed varied-exponent values checked for original-MMA bit identity and finiteness.
    Varied,
}

/// Advance a deterministic wrapping generator and produce a finite signed BF16 test value.
///
/// Exponent fields span 116 through 127; this is fixture data, not model sampling.
fn varied_bf16(seed: &mut u32) -> BFloat16 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let sign = (*seed >> 16) as u16 & 0x8000;
    let exponent = 116 + ((*seed >> 7) % 12) as u16;
    BFloat16::from_bits(sign | (exponent << 7) | (*seed as u16 & 127))
}

/// Run a strided projection with NaN-poisoned padding against exact and host references.
///
/// The exact gate follows the shape's reduction plan: FMA-chain shapes must
/// match the host emulation of the probed order, tensor-core-chain shapes must
/// match the original four-warp entry, and split-K shapes are checked for exact
/// slice sums by the projection-worker replay against reference captures.
/// Checks immutable input/bias/weight storage and finiteness. Regular and
/// cancellation patterns additionally require the FP64 accuracy bound;
/// varied-exponent cases report FP64 error without claiming that accuracy gate.
fn fixture<const IN: usize, const OUT: usize>(
    runner: &Runner,
    rows: usize,
    input_stride: usize,
    weight_stride: usize,
    has_bias: bool,
    pattern: Pattern,
) -> Result<usize> {
    use cute_rs::ffn::Query;
    let poison = BFloat16::from_bits(0x7fc1);
    let mut input = vec![poison; rows * input_stride + 7];
    let mut weight = vec![poison; OUT * weight_stride + 7];
    let mut bias = vec![poison; OUT + 7];
    let mut seed = 0x42b3_f991;
    for row in 0..rows {
        for k in 0..IN {
            input[row * input_stride + k] = match pattern {
                Pattern::Cancellation => {
                    BFloat16::from_f32([16.0, 0.03125, -16.0, -0.015625][k % 4] * (row + 1) as f32)
                }
                Pattern::Regular => {
                    BFloat16::from_f32(((k * 7 + row * 11) % 41) as f32 / 64.0 - 0.3125)
                }
                Pattern::Varied => varied_bf16(&mut seed),
            };
        }
    }
    for column in 0..OUT {
        bias[column] = BFloat16::from_f32((column % 7) as f32 / 128.0 - 0.0234375);
        for k in 0..IN {
            weight[column * weight_stride + k] = match pattern {
                Pattern::Cancellation => BFloat16::from_f32((column % 5 + 1) as f32 / 8.0),
                Pattern::Regular => {
                    BFloat16::from_f32(((k * 13 + column * 17) % 37) as f32 / 128.0 - 0.140625)
                }
                Pattern::Varied => varied_bf16(&mut seed),
            };
        }
    }
    let mut expected = vec![0.0f64; rows * OUT];
    for row in 0..rows {
        for column in 0..OUT {
            let mut sum = if has_bias {
                f64::from(bias[column].to_f32())
            } else {
                0.0
            };
            for k in 0..IN {
                sum += f64::from(input[row * input_stride + k].to_f32())
                    * f64::from(weight[column * weight_stride + k].to_f32());
            }
            expected[row * OUT + column] = sum;
        }
    }
    let input_dev = DeviceBuffer::from_host(&runner.stream, &input)?;
    let weight_dev = DeviceBuffer::from_host(&runner.stream, &weight)?;
    let bias_dev = DeviceBuffer::from_host(&runner.stream, &bias)?;
    let weights = ProjectionWeights::<Query, IN, OUT>::new(&weight_dev, weight_stride)?;
    let biases = BiasWeights::<Query, OUT>::new(&bias_dev)?;
    let output = runner.project_buffer(
        &input_dev,
        rows,
        input_stride,
        &weights,
        has_bias.then_some(&biases),
    )?;
    let plan = reference_plan(rows, IN, OUT);
    let label = format!(
        "decode linear {rows}x{IN}→{OUT}, bias={has_bias}, pattern={pattern:?}, plan={plan:?}"
    );
    let actual = output.read(&runner.stream)?;

    let exact_gate = match plan {
        ReductionPlan::FmaChains { chains } => {
            // One row: the kernel must reproduce the host emulation of the
            // probed chain-and-butterfly order bit for bit.
            let emulated = emulate_fma_chains(
                &input,
                &weight,
                weight_stride,
                IN,
                OUT,
                chains,
                has_bias.then_some(&bias[..]),
            );
            exact_bits(&actual, &emulated, &label, "host FMA-chain emulation")?;
            "FMA-chain emulation bit identity"
        }
        ReductionPlan::MmaChain => {
            // Run the original four-warp entry on the same buffers. The new
            // launch must be bitwise equal, not merely within the oracle's bound.
            let mut original = Batch::<OUT>::allocate(&runner.stream, rows)?;
            let config = cuda_core::LaunchConfig {
                grid_dim: (OUT.div_ceil(8).div_ceil(4) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the successful checked launch above established valid
            // one/two-row extents, pitches and contexts. The original block has
            // four complete warps, and a fresh owned output makes the original
            // accumulator map disjoint.
            let launched = unsafe {
                if has_bias {
                    runner.module.linear_bias_bf16(
                        &runner.stream,
                        config,
                        &input_dev,
                        &weight_dev,
                        &bias_dev,
                        original.buffer_mut(),
                        rows,
                        IN,
                        OUT,
                        input_stride,
                        weight_stride,
                    )
                } else {
                    runner.module.linear_bf16(
                        &runner.stream,
                        config,
                        &input_dev,
                        &weight_dev,
                        original.buffer_mut(),
                        rows,
                        IN,
                        OUT,
                        input_stride,
                        weight_stride,
                    )
                }
            };
            let completed = runner.stream.synchronize();
            launched?;
            completed?;
            exact_bits(
                &actual,
                &original.read(&runner.stream)?,
                &label,
                "original four-warp MMA",
            )?;
            "original-MMA bit identity"
        }
        ReductionPlan::MmaSplitK { .. } => "split-K (exact slice sums via worker replay)",
    };
    let error = if matches!(pattern, Pattern::Varied) {
        // These wide-dynamic-range cancellation cases are an arithmetic-
        // preservation gate. The original MMA itself exceeds the synthetic
        // absolute bound near zero, as confirmed before this classification.
        // Require original bits and finite values, and report FP64 error
        // without claiming an independent numerical-accuracy pass here.
        let mut maximum = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            let actual = f64::from(actual.to_f32());
            let expected = f64::from(BFloat16::from_f32(*expected as f32).to_f32());
            if !actual.is_finite() || !expected.is_finite() {
                return Err("decode MMA bit-preservation fixture produced nonfinite values".into());
            }
            maximum = maximum.max((actual - expected).abs());
        }
        maximum
    } else {
        compare(&actual, &expected, &label)?
    };
    for (before, buffer) in [
        (&input, &input_dev),
        (&weight, &weight_dev),
        (&bias, &bias_dev),
    ] {
        if !before
            .iter()
            .zip(buffer.to_host_vec(&runner.stream)?)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(
                "decode projection changed input/weight/bias storage or its padding".into(),
            );
        }
    }
    let gate = if matches!(pattern, Pattern::Varied) {
        format!("finite outputs and {exact_gate} (FP64 error reported only)")
    } else {
        format!("FP64 accuracy and {exact_gate}")
    };
    println!(
        "{label}: {} values, {gate}, max FP64/BF16 abs {error:.8}",
        expected.len()
    );
    Ok(expected.len())
}

/// Require the split-K entry with one slice to equal the ordinary decode chain.
///
/// A single slice is the whole width, so the entry must reproduce the ordinary
/// one-warp MMA kernel bit for bit; the production slice widths must then run
/// to finite results. Exact slice sums are checked against reference captures
/// by the projection-worker replay, which this synthetic check cannot replace.
fn split_k_single_slice_identity(runner: &Runner) -> Result<usize> {
    let mut checked = 0;
    let mut seed = 0x5a17_c0de;
    for (rows, width, outputs, slice_width) in [(1, 6144, 2048, 1280), (2, 3072, 1024, 192)] {
        let input: Vec<_> = (0..rows * width).map(|_| varied_bf16(&mut seed)).collect();
        let weight: Vec<_> = (0..outputs * width)
            .map(|_| varied_bf16(&mut seed))
            .collect();
        let input_dev = DeviceBuffer::from_host(&runner.stream, &input)?;
        let weight_dev = DeviceBuffer::from_host(&runner.stream, &weight)?;
        let mut plain = DeviceBuffer::from_host(
            &runner.stream,
            &vec![BFloat16::from_bits(0x7fc1); rows * outputs],
        )?;
        let mut single = DeviceBuffer::from_host(
            &runner.stream,
            &vec![BFloat16::from_bits(0x7fc1); rows * outputs],
        )?;
        let mut sliced = DeviceBuffer::from_host(
            &runner.stream,
            &vec![BFloat16::from_bits(0x7fc1); rows * outputs],
        )?;
        let blocks = outputs.div_ceil(8) as u32;
        let config = LaunchConfig1D::new(blocks, 32, 0);
        let plain_launch = runner.module.prepare_decode_linear_bf16(config)?;
        let split_launch = runner.reference.prepare_split_k_linear_bf16(config)?;
        runner.module.decode_linear_bf16(
            &runner.stream,
            &plain_launch,
            &input_dev,
            &weight_dev,
            &mut plain,
            rows,
            width,
            outputs,
            width,
            width,
        )?;
        for (output, slice) in [(&mut single, width), (&mut sliced, slice_width)] {
            runner.reference.split_k_linear_bf16(
                &runner.stream,
                &split_launch,
                &input_dev,
                &weight_dev,
                output,
                rows,
                width,
                outputs,
                width,
                width,
                slice,
            )?;
        }
        runner.stream.synchronize()?;
        let label = format!("split-K {rows}x{width}→{outputs}");
        exact_bits(
            &single.to_host_vec(&runner.stream)?,
            &plain.to_host_vec(&runner.stream)?,
            &label,
            "one-slice ordinary decode chain",
        )?;
        let sliced = sliced.to_host_vec(&runner.stream)?;
        if sliced.iter().any(|value| !value.to_f32().is_finite()) {
            return Err(
                format!("{label}: slice width {slice_width} produced nonfinite values").into(),
            );
        }
        checked += sliced.len();
        println!(
            "{label}: one-slice entry equals the ordinary decode chain; slice width {slice_width} finite"
        );
    }
    Ok(checked)
}

/// Independent FP64 oracle, poisoned padding, cancellation and bias-boundary checks.
pub fn validate(stream: Arc<CudaStream>) -> Result<()> {
    use Pattern::{Cancellation, Regular, Varied};
    use cute_rs::ffn::Query;
    stream.context().bind_to_thread()?;
    let runner = Runner::new(stream)?;
    let mut checked = 0;
    checked += fixture::<17, 7>(&runner, 1, 23, 22, false, Regular)?;
    checked += fixture::<33, 19>(&runner, 2, 41, 38, true, Regular)?;
    checked += fixture::<129, 5>(&runner, 2, 137, 133, false, Cancellation)?;
    checked += fixture::<1024, 3072>(&runner, 1, 1024, 1024, false, Regular)?;
    checked += fixture::<2048, 6144>(&runner, 2, 2048, 2048, false, Regular)?;
    checked += fixture::<6144, 2048>(&runner, 1, 6144, 6144, true, Cancellation)?;
    checked += fixture::<3072, 1024>(&runner, 2, 3072, 3072, false, Regular)?;
    let mut stress_checked = 0;
    stress_checked += fixture::<1024, 2048>(&runner, 1, 1031, 1029, true, Varied)?;
    stress_checked += fixture::<2048, 1024>(&runner, 2, 2055, 2053, false, Varied)?;
    stress_checked += fixture::<6144, 2048>(&runner, 2, 6144, 6144, true, Varied)?;
    stress_checked += fixture::<80, 16>(&runner, 2, 82, 84, true, Varied)?;
    // Exercise two/three complete packed K fragments without a prefetch batch.
    // Exact original-MMA comparisons catch skipped/repeated fragments and
    // preserve both bias paths across one/two-row ownership and even strides.
    stress_checked += fixture::<32, 8>(&runner, 1, 32, 32, false, Varied)?;
    stress_checked += fixture::<32, 16>(&runner, 2, 34, 36, true, Varied)?;
    stress_checked += fixture::<48, 16>(&runner, 1, 48, 50, true, Varied)?;
    stress_checked += fixture::<48, 8>(&runner, 2, 50, 48, false, Varied)?;
    // Cover the K512 prefetch boundary, complete batches and nonempty tails.
    // Varied fixtures compare exact bits with the original ordered MMA kernel;
    // even padded strides exercise packed addressing for both row/bias modes.
    stress_checked += fixture::<496, 16>(&runner, 1, 498, 500, false, Varied)?;
    stress_checked += fixture::<512, 8>(&runner, 1, 512, 514, false, Varied)?;
    stress_checked += fixture::<512, 16>(&runner, 2, 514, 516, true, Varied)?;
    stress_checked += fixture::<528, 16>(&runner, 1, 530, 532, true, Varied)?;
    stress_checked += fixture::<528, 8>(&runner, 2, 530, 528, false, Varied)?;
    stress_checked += fixture::<1040, 16>(&runner, 2, 1042, 1044, true, Varied)?;
    checked += fixture::<16, 8>(&runner, 1, 20, 22, false, Regular)?;
    // Every one-row production shape follows the reference's FMA-chain order.
    // Varied exponents stress the chain and butterfly rounding; a padded
    // pitch covers the 8-byte weight path and an odd pitch the scalar path.
    stress_checked += fixture::<1024, 2048>(&runner, 1, 1024, 1024, false, Varied)?;
    stress_checked += fixture::<1024, 1024>(&runner, 1, 1024, 1024, false, Varied)?;
    stress_checked += fixture::<1024, 1024>(&runner, 1, 1024, 1028, true, Varied)?;
    stress_checked += fixture::<1024, 3072>(&runner, 1, 1024, 1024, false, Varied)?;
    stress_checked += fixture::<2048, 1024>(&runner, 1, 2048, 2048, true, Varied)?;
    stress_checked += fixture::<2048, 1024>(&runner, 1, 2048, 2051, false, Varied)?;
    stress_checked += fixture::<2048, 2048>(&runner, 1, 2048, 2048, false, Varied)?;
    stress_checked += fixture::<2048, 3072>(&runner, 1, 2048, 2048, false, Varied)?;
    stress_checked += fixture::<2048, 6144>(&runner, 1, 2048, 2048, false, Varied)?;
    stress_checked += fixture::<3072, 1024>(&runner, 1, 3072, 3072, false, Varied)?;
    checked += fixture::<2048, 1024>(&runner, 1, 2048, 2048, true, Cancellation)?;
    checked += fixture::<1024, 1024>(&runner, 1, 1024, 1024, false, Regular)?;
    stress_checked += split_k_single_slice_identity(&runner)?;

    // The dot is halfway between BF16 values. Bias must be added in FP32
    // before rounding; rounding the dot first would incorrectly return 1.0.
    let input = DeviceBuffer::from_host(
        &runner.stream,
        &[
            BFloat16::from_f32(1.0),
            BFloat16::from_f32(1.0 / 256.0),
            BFloat16::from_f32(-1.0),
            BFloat16::from_f32(-1.0 / 256.0),
        ],
    )?;
    let weight = DeviceBuffer::from_host(&runner.stream, &[BFloat16::from_f32(1.0); 2])?;
    let bias = DeviceBuffer::from_host(&runner.stream, &[BFloat16::from_f32(1.0 / 1024.0)])?;
    let weights = ProjectionWeights::<Query, 2, 1>::new(&weight, 2)?;
    let biases = BiasWeights::<Query, 1>::new(&bias)?;
    let output = runner
        .project_buffer(&input, 2, 2, &weights, Some(&biases))?
        .read(&runner.stream)?;
    if output[0].to_bits() != BFloat16::from_f32(1.0078125).to_bits()
        || output[1].to_bits() != BFloat16::from_f32(-1.0).to_bits()
    {
        return Err("decode linear lost FP32 bias-before-BF16 rounding".into());
    }
    let mut bias_checked = output.len();
    for (rows, stride) in [(0, 2), (3, 2), (2, 1), (2, 4)] {
        if runner
            .project_buffer(&input, rows, stride, &weights, Some(&biases))
            .is_ok()
        {
            return Err("decode linear accepted invalid rows/stride/extent".into());
        }
    }

    // Put the same halfway dots in a full 2x16 -> 8 packed tile. Padding the
    // remaining K terms with zero keeps the expected results exact; rounding
    // the positive dot before bias would incorrectly produce 1.0 in row zero.
    let mut packed_input = [BFloat16::from_bits(0); 2 * 16];
    packed_input[0] = BFloat16::from_f32(1.0);
    packed_input[1] = BFloat16::from_f32(1.0 / 256.0);
    packed_input[16] = BFloat16::from_f32(-1.0);
    packed_input[17] = BFloat16::from_f32(-1.0 / 256.0);
    let mut packed_weight = [BFloat16::from_bits(0); 8 * 16];
    for column in 0..8 {
        packed_weight[column * 16] = BFloat16::from_f32(1.0);
        packed_weight[column * 16 + 1] = BFloat16::from_f32(1.0);
    }
    let packed_input = DeviceBuffer::from_host(&runner.stream, &packed_input)?;
    let packed_weight = DeviceBuffer::from_host(&runner.stream, &packed_weight)?;
    let packed_bias =
        DeviceBuffer::from_host(&runner.stream, &[BFloat16::from_f32(1.0 / 1024.0); 8])?;
    let packed_weights = ProjectionWeights::<Query, 16, 8>::new(&packed_weight, 16)?;
    let packed_biases = BiasWeights::<Query, 8>::new(&packed_bias)?;
    let packed_output = runner
        .project_buffer(&packed_input, 2, 16, &packed_weights, Some(&packed_biases))?
        .read(&runner.stream)?;
    if packed_output.len() != 16 {
        return Err("decode packed bias fixture returned an unexpected output extent".into());
    }
    for (index, actual) in packed_output.iter().enumerate() {
        let expected = BFloat16::from_f32(if index < 8 { 1.0078125 } else { -1.0 });
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "decode packed linear output {index} lost FP32 bias-before-BF16 rounding"
            )
            .into());
        }
    }
    bias_checked += packed_output.len();
    println!(
        "decode linear: {} FP64/bias accuracy checks, {} exact-order bit comparisons (including {} varied-exponent stress values), and invalid-extent checks passed",
        checked + bias_checked,
        checked + stress_checked,
        stress_checked,
    );
    grouped::validate(&runner)?;
    Ok(())
}

/// CPU-only launch-geometry and numerical-comparison contract checks.
#[cfg(test)]
mod tests {
    use super::*;
    /// Reject unsupported row counts, empty outputs, and oversized grids.
    #[test]
    fn decode_geometry_rejects_unsupported_rows_and_grid_overflow() {
        assert!(launch_config(1, 1).is_ok());
        assert!(launch_config(2, 7).is_ok());
        assert!(launch_config(0, 1).is_err());
        assert!(launch_config(3, 1).is_err());
        assert!(launch_config(1, 0).is_err());
        assert!(launch_config(1, usize::MAX).is_err());
    }
    /// Ensure nonfinite and inaccurate results fail, and bit identity is stricter than tolerance.
    #[test]
    fn fp64_oracle_rejects_nonfinite_and_wrong_results() {
        assert!(compare(&[BFloat16::from_f32(1.0)], &[1.0], "test").is_ok());
        assert!(compare(&[BFloat16::from_f32(1.0)], &[0.0], "test").is_err());
        assert!(compare(&[BFloat16::from_bits(0x7fc1)], &[0.0], "test").is_err());
        for expected in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(compare(&[BFloat16::from_f32(0.0)], &[expected], "test").is_err());
        }
        assert!(compare(&[], &[], "test").is_err());
        // A one-ULP BF16 change fits the numerical oracle's bound but must
        // still fail the arithmetic-preservation gate.
        let one = [BFloat16::from_f32(1.0)];
        let next = [BFloat16::from_f32(1.0078125)];
        assert!(compare(&next, &[1.0], "test").is_ok());
        assert!(exact_bits(&next, &one, "test", "fixture").is_err());
        assert!(exact_bits(&one, &one, "test", "fixture").is_ok());
    }
}

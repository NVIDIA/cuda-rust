/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Shared, ordered decoder-layer execution over exclusively owned GPU scratch.

use crate::model::{Batch, EPSILON};
use crate::operator::{GUARD_LEN, Result};
use crate::reference_linear::{ReductionPlan, reference_plan};
use crate::reference_linear_kernel::kernels as reference;
use crate::{
    attention_kernel, element_kernel, ffn_kernel, kernel, norm_kernel, predictor_graph_kernel,
};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, PreparedLaunch};
use cute_rs::{
    Bf16,
    decoder::{BiasWeights, DecoderWeights},
    ffn::{ProjectionRole, ProjectionWeights},
};
use std::sync::Arc;

/// Keep device guard offsets consistent with the shared attention state layout.
const _: () = assert!(GUARD_LEN == 8 && attention_kernel::STATE_SIZE == 130);

/// Prepared unbiased decode projection, branded to the matching kernel entry.
type ProjectionLaunch = PreparedLaunch<ffn_kernel::kernels::__decode_linear_bf16_CudaKernel>;
/// Prepared sixteen-chain FP32 projection launch.
type Fma16Launch = PreparedLaunch<reference::__fma16_linear_bf16_CudaKernel>;
/// Prepared biased sixteen-chain FP32 projection launch.
type Fma16BiasLaunch = PreparedLaunch<reference::__fma16_linear_bias_bf16_CudaKernel>;
/// Prepared thirty-two-chain FP32 projection launch.
type Fma32Launch = PreparedLaunch<reference::__fma32_linear_bf16_CudaKernel>;
/// Prepared split-K tensor-core projection launch.
type SplitKLaunch = PreparedLaunch<reference::__split_k_linear_bf16_CudaKernel>;
/// Prepared biased input adapter, with bias added before BF16 output rounding.
type BiasedLaunch = PreparedLaunch<ffn_kernel::kernels::__decode_linear_bias_bf16_CudaKernel>;
/// Prepared partial-attention reduction over one bounded cache partition.
type PartialLaunch = PreparedLaunch<attention_kernel::kernels::__partial_CudaKernel>;
/// Prepared attention merge producing BF16 head outputs.
type MergeLaunch = PreparedLaunch<attention_kernel::kernels::__merge_CudaKernel>;

/// Make a 128-thread elementwise launch for positive bounded decoder extents.
///
/// Callers supply compile-time/model-bounded work; this helper does not validate
/// zero work or truncation to the u32 grid dimension.
pub(crate) fn grid(work: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (work.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch exactly one 32-thread CTA for status, guard, and greedy-selection helpers.
pub(crate) fn one_warp() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Completion is installed before the first enqueue and precedes owner drops.
pub(crate) struct Completion(
    /// Retained stream until explicit completion or cleanup takes it exactly once.
    Option<Arc<CudaStream>>,
);
impl Completion {
    /// Install a completion guard before any enqueue and retain its stream owner.
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Self {
        Self(Some(stream.clone()))
    }
    /// Synchronize once, returning any completion failure and disarming drop cleanup.
    pub(crate) fn finish(mut self) -> Result<()> {
        Ok(self.0.take().expect("one completion").synchronize()?)
    }
}
impl Drop for Completion {
    /// Attempt to drain pending work if explicit completion was skipped, including unwinding.
    fn drop(&mut self) {
        if let Some(stream) = self.0.take() {
            let _ = stream.synchronize();
        }
    }
}

/// Guarded attention rows retain their logical width instead of carrying raw
/// indices into a universal buffer collection.
pub(crate) struct Guarded<const W: usize> {
    /// BF16 logical rows followed by `GUARD_LEN` sentinel elements checked on device.
    pub(crate) buffer: DeviceBuffer<Bf16>,
}
impl<const W: usize> Guarded<W> {
    /// Allocate `rows * W` initialized BF16 elements plus the operator guard tail.
    ///
    /// Rejects multiplication overflow and propagates CUDA allocation/upload errors.
    pub(crate) fn new(stream: &CudaStream, rows: usize) -> Result<Self> {
        Ok(Self {
            buffer: crate::operator::allocate(
                stream,
                rows.checked_mul(W).ok_or("guarded row overflow")?,
            )?,
        })
    }
}
/// Each live attention geometry owns its own guard boundary. A shared max-size
/// scratch would let the next layer overwrite evidence beyond the live extent.
pub(crate) struct LayerScratch<const H: usize, const I: usize> {
    /// Fixed live token-row count: two for first-step prefill or one for later decode.
    pub(crate) rows: usize,
    /// H-wide hidden rows, overwritten by input adaptation and each decoder layer.
    pub(crate) hidden: Batch<H>,
    /// Input RMSNorm output before Q/K/V projection.
    pub(crate) normalized: Batch<H>,
    /// 2048-wide query projections, corresponding to sixteen 128-value heads.
    pub(crate) query: Batch<2048>,
    /// 1024-wide key projections, corresponding to eight 128-value heads.
    pub(crate) key: Batch<1024>,
    /// 1024-wide unrotated value projections for cache append.
    pub(crate) value: Batch<1024>,
    /// Per-head RMS-normalized query values before RoPE.
    pub(crate) query_norm: Batch<2048>,
    /// Per-head RMS-normalized key values before RoPE.
    pub(crate) key_norm: Batch<1024>,
    /// RoPE-transformed queries with a guard immediately after the live extent.
    pub(crate) rotated_query: Guarded<2048>,
    /// RoPE-transformed keys with their own live-extent guard tail.
    pub(crate) rotated_key: Guarded<1024>,
    /// Merged 2048-wide attention result plus guard tail.
    pub(crate) attended: Guarded<2048>,
    /// FP32 `[rows, 16, splits, STATE_SIZE]` partial-attention states plus a guard tail.
    pub(crate) partials: DeviceBuffer<f32>,
    /// Attention output projected from 2048 back to H hidden channels.
    pub(crate) projected: Batch<H>,
    /// BF16 hidden-plus-attention residual, retained through the FFN.
    pub(crate) residual: Batch<H>,
    /// Post-attention RMSNorm output used by gate/up projections.
    pub(crate) post: Batch<H>,
    /// I-wide FFN gate projection before activation.
    pub(crate) gate: Batch<I>,
    /// I-wide FFN up projection multiplied by the rounded SiLU output.
    pub(crate) up: Batch<I>,
    /// BF16-rounded SiLU intermediate, preserving the baseline activation boundary.
    pub(crate) silu: Batch<I>,
    /// BF16-rounded product of the SiLU and up rows.
    pub(crate) product: Batch<I>,
    /// H-wide FFN down projection before the second residual add.
    pub(crate) down: Batch<H>,
    /// Final RMSNorm output for every live row after the caller finishes its layer stack.
    pub(crate) final_norm: Batch<H>,
    /// Exact number of 64-token cache partitions represented by the live guard.
    pub(crate) splits: usize,
    /// Prepared full-CTA reduction for this exact row/head/split geometry.
    pub(crate) partial_launch: PartialLaunch,
    /// Prepared one-warp attention merge for each live row and query head.
    pub(crate) merge_launch: MergeLaunch,
}
/// Checked attention geometry, including the guard immediately after live partials.
struct AttentionGeometry {
    /// Number of 64-token key partitions.
    splits: usize,
    /// FP32 allocation length, including the guard suffix.
    partial_values: usize,
    /// One 128-thread CTA per live query row, head, and key partition.
    partial_blocks: u32,
    /// One 32-thread CTA per live query row and head.
    merge_blocks: u32,
}

/// Validate decode rows, kernel ABI widths, grid bounds, and partial-state arithmetic.
///
/// Returns geometry for positive key prefixes without allocating or submitting work.
fn attention_geometry(rows: usize, key_tokens: usize) -> Result<AttentionGeometry> {
    if !matches!(rows, 1 | 2) || key_tokens == 0 || key_tokens > u32::MAX as usize {
        return Err("decoder scratch requires one/two rows and a positive u32 key prefix".into());
    }
    let splits = key_tokens.div_ceil(64);
    let heads = rows.checked_mul(16).ok_or("attention row/head overflow")?;
    let blocks = heads
        .checked_mul(splits)
        .ok_or("attention split-grid overflow")?;
    if blocks > i32::MAX as usize {
        return Err("attention partial grid exceeds CUDA grid-x range".into());
    }
    let partial_values = blocks
        .checked_mul(attention_kernel::STATE_SIZE)
        .and_then(|values| values.checked_add(GUARD_LEN))
        .ok_or("attention guarded partial extent overflow")?;
    Ok(AttentionGeometry {
        splits,
        partial_values,
        partial_blocks: blocks as u32,
        merge_blocks: heads as u32,
    })
}

impl<const H: usize, const I: usize> LayerScratch<H, I> {
    /// Allocate exactly one/two rows, preserving fixed Q2048/KV1024 head geometry.
    ///
    /// The FP32 guard follows the exact live partial extent for `key_tokens`.
    /// Rejects zero/overflowing attention geometry before allocation; allocation
    /// and launch-preparation errors propagate before any schedule submission.
    pub(crate) fn new(
        stream: &CudaStream,
        module: &attention_kernel::kernels::LoadedModule,
        rows: usize,
        key_tokens: usize,
    ) -> Result<Self> {
        let geometry = attention_geometry(rows, key_tokens)?;
        Ok(Self {
            rows,
            splits: geometry.splits,
            hidden: Batch::allocate(stream, rows)?,
            normalized: Batch::allocate(stream, rows)?,
            query: Batch::allocate(stream, rows)?,
            key: Batch::allocate(stream, rows)?,
            value: Batch::allocate(stream, rows)?,
            query_norm: Batch::allocate(stream, rows)?,
            key_norm: Batch::allocate(stream, rows)?,
            rotated_query: Guarded::new(stream, rows)?,
            rotated_key: Guarded::new(stream, rows)?,
            attended: Guarded::new(stream, rows)?,
            partials: DeviceBuffer::from_host(
                stream,
                &vec![f32::from_bits(0x7fc1_0001); geometry.partial_values],
            )?,
            projected: Batch::allocate(stream, rows)?,
            residual: Batch::allocate(stream, rows)?,
            post: Batch::allocate(stream, rows)?,
            gate: Batch::allocate(stream, rows)?,
            up: Batch::allocate(stream, rows)?,
            silu: Batch::allocate(stream, rows)?,
            product: Batch::allocate(stream, rows)?,
            down: Batch::allocate(stream, rows)?,
            final_norm: Batch::allocate(stream, rows)?,
            partial_launch: module.prepare_partial(LaunchConfig1D::new(
                geometry.partial_blocks,
                128,
                attention_kernel::PARTIAL_SHARED_BYTES,
            ))?,
            merge_launch: module.prepare_merge(LaunchConfig1D::new(
                geometry.merge_blocks,
                32,
                0,
            ))?,
        })
    }

    /// Prepare the exact partial guard boundary before the next layer schedule.
    ///
    /// Reuses storage and launches while the split count is unchanged. Otherwise
    /// all replacement allocation/preparation succeeds before any existing owner
    /// is replaced. Earlier unsafe schedule submissions must already be complete,
    /// and no captured graph may retain these replaceable partials. The stream
    /// and module must share the scratch context; errors leave this scratch intact.
    pub(crate) fn prepare_attention(
        &mut self,
        stream: &CudaStream,
        module: &attention_kernel::kernels::LoadedModule,
        key_tokens: usize,
    ) -> Result<()> {
        let geometry = attention_geometry(self.rows, key_tokens)?;
        if !Arc::ptr_eq(self.partials.context(), stream.context()) {
            return Err("decoder scratch context mismatch".into());
        }
        if geometry.splits == self.splits {
            return Ok(());
        }
        let partials = DeviceBuffer::from_host(
            stream,
            &vec![f32::from_bits(0x7fc1_0001); geometry.partial_values],
        )?;
        let partial_launch = module.prepare_partial(LaunchConfig1D::new(
            geometry.partial_blocks,
            128,
            attention_kernel::PARTIAL_SHARED_BYTES,
        ))?;
        self.partials = partials;
        self.partial_launch = partial_launch;
        self.splits = geometry.splits;
        Ok(())
    }
}

/// Owned kernel modules and fixed projection launches retained for eager execution and replay.
pub(crate) struct Modules {
    /// BF16 projection and activation module retained through graph destruction.
    pub(crate) linear: ffn_kernel::kernels::LoadedModule,
    /// Specialized/general RMSNorm module used by hidden and head normalization.
    pub(crate) norms: norm_kernel::kernels::LoadedModule,
    /// Residual, concatenation, and row-sum kernels.
    pub(crate) elements: element_kernel::kernels::LoadedModule,
    /// RoPE and K/V append kernels operating on guarded storage.
    pub(crate) rotary: kernel::kernels::LoadedModule,
    /// Partial and merge attention entries used with fixed typed cache geometry.
    pub(crate) attention: attention_kernel::kernels::LoadedModule,
    /// Status reset, guard checks, greedy selection, feedback, and range-copy kernels.
    pub(crate) schedule: predictor_graph_kernel::kernels::LoadedModule,
    /// Unbiased projection launch with 128 eight-column CTAs.
    projection_1024: ProjectionLaunch,
    /// Unbiased projection launch with 256 eight-column CTAs.
    projection_2048: ProjectionLaunch,
    /// Unbiased projection launch with 384 eight-column CTAs.
    projection_3072: ProjectionLaunch,
    /// Unbiased projection launch with 768 eight-column CTAs.
    projection_6144: ProjectionLaunch,
    /// Biased 2048-to-1024 adapter launch with 128 CTAs; two-row tensor-core chain.
    biased_1024: BiasedLaunch,
    /// FMA-chain and split-K entries in the reference reduction order.
    reference: reference::LoadedModule,
    /// Sixteen-chain launches for 1024 outputs (128 eight-output CTAs).
    fma16_1024: Fma16Launch,
    /// Sixteen-chain launches for 2048 outputs.
    fma16_2048: Fma16Launch,
    /// Sixteen-chain launches for 3072 outputs.
    fma16_3072: Fma16Launch,
    /// Sixteen-chain launches for 6144 outputs.
    fma16_6144: Fma16Launch,
    /// Thirty-two-chain launch for the 1024-to-1024 one-row projections (256 CTAs).
    fma32_1024: Fma32Launch,
    /// Biased sixteen-chain adapter launch for one row.
    fma16_bias_1024: Fma16BiasLaunch,
    /// Split-K launch for the two-row 3072-to-1024 projection (128 CTAs).
    split_k_1024: SplitKLaunch,
    /// Split-K launch for the one-row 6144-to-2048 projection (256 CTAs).
    split_k_2048: SplitKLaunch,
}
impl Modules {
    /// Load the embedded modules and prepare each fixed-width decode projection.
    ///
    /// All loading/preparation happens outside graph capture and propagates driver errors.
    pub(crate) fn new(stream: &CudaStream) -> Result<Self> {
        // SAFETY: modules embed this binary's matching kernel definitions.
        let linear = unsafe { ffn_kernel::kernels::load(stream.context())? };
        let attention = unsafe { attention_kernel::kernels::load(stream.context())? };
        let reference = unsafe { reference::load(stream.context())? };
        let warp = |blocks: u32| LaunchConfig1D::new(blocks, 32, 0);
        Ok(Self {
            projection_1024: linear.prepare_decode_linear_bf16(warp(128))?,
            projection_2048: linear.prepare_decode_linear_bf16(warp(256))?,
            projection_3072: linear.prepare_decode_linear_bf16(warp(384))?,
            projection_6144: linear.prepare_decode_linear_bf16(warp(768))?,
            biased_1024: linear.prepare_decode_linear_bias_bf16(warp(128))?,
            fma16_1024: reference.prepare_fma16_linear_bf16(warp(128))?,
            fma16_2048: reference.prepare_fma16_linear_bf16(warp(256))?,
            fma16_3072: reference.prepare_fma16_linear_bf16(warp(384))?,
            fma16_6144: reference.prepare_fma16_linear_bf16(warp(768))?,
            fma32_1024: reference.prepare_fma32_linear_bf16(warp(256))?,
            fma16_bias_1024: reference.prepare_fma16_linear_bias_bf16(warp(128))?,
            split_k_1024: reference.prepare_split_k_linear_bf16(warp(128))?,
            split_k_2048: reference.prepare_split_k_linear_bf16(warp(256))?,
            reference,
            linear,
            attention,
            norms: norm_kernel::kernels::load(stream.context())?,
            elements: element_kernel::kernels::load(stream.context())?,
            rotary: kernel::kernels::load(stream.context())?,
            schedule: predictor_graph_kernel::kernels::load(stream.context())?,
        })
    }

    /// Reject extents a prepared projection launch could not cover.
    #[allow(clippy::too_many_arguments)]
    fn check_projection_extents(
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        weight_len: usize,
        weight_stride: usize,
        bias_len: Option<usize>,
        output_len: usize,
        width: usize,
        outputs: usize,
    ) -> Result<()> {
        if !(matches!(rows, 1 | 2)
            && weight_stride >= width
            && input.len() >= rows * width
            && outputs <= weight_len / weight_stride
            && output_len >= rows * outputs
            && bias_len.is_none_or(|len| len >= outputs))
        {
            return Err("decoder projection extents do not cover the prepared launch".into());
        }
        Ok(())
    }

    /// Enqueue a bias-free projection in the shape's reference reduction order.
    ///
    /// One-row shapes use the FMA-chain entries, the one-row 6144-to-2048 and
    /// two-row 3072-to-1024 shapes use split K, and the remaining two-row
    /// shapes keep the tensor-core chain; see `reference_linear`.
    ///
    /// # Safety
    /// Caller has validated typed weight geometry/context, one/two-row dense input,
    /// separate output storage, and readiness. The stream must remain exclusively
    /// ordered, with all owners retained through completion or capture cleanup.
    pub(crate) unsafe fn project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let stride = weight.shape().row_stride();
        Self::check_projection_extents(
            input,
            rows,
            weight.buffer().len(),
            stride,
            None,
            output.buffer().len(),
            IN,
            OUT,
        )?;
        match reference_plan(rows, IN, OUT) {
            ReductionPlan::MmaChain => {
                let launch = match OUT {
                    1024 => &self.projection_1024,
                    2048 => &self.projection_2048,
                    3072 => &self.projection_3072,
                    6144 => &self.projection_6144,
                    _ => return Err("unsupported decoder projection width".into()),
                };
                self.linear.decode_linear_bf16(
                    stream,
                    launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    IN,
                    stride,
                )?;
            }
            ReductionPlan::FmaChains { chains: 16 } => {
                let launch = match OUT {
                    1024 => &self.fma16_1024,
                    2048 => &self.fma16_2048,
                    3072 => &self.fma16_3072,
                    6144 => &self.fma16_6144,
                    _ => return Err("unsupported decoder projection width".into()),
                };
                self.reference.fma16_linear_bf16(
                    stream,
                    launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            ReductionPlan::FmaChains { chains: 32 } if OUT == 1024 => {
                self.reference.fma32_linear_bf16(
                    stream,
                    &self.fma32_1024,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?;
            }
            ReductionPlan::FmaChains { .. } => {
                return Err("unsupported decoder projection chain count".into());
            }
            ReductionPlan::MmaSplitK { slice_width } => {
                let launch = match OUT {
                    1024 => &self.split_k_1024,
                    2048 => &self.split_k_2048,
                    _ => return Err("unsupported decoder split-K projection width".into()),
                };
                self.reference.split_k_linear_bf16(
                    stream,
                    launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    IN,
                    stride,
                    slice_width,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue the biased 2048-to-1024 adapter projection for one or two rows.
    ///
    /// Two rows keep the tensor-core chain; one row uses sixteen FP32 chains.
    /// Both add the bias in FP32 before the single BF16 rounding.
    ///
    /// # Safety
    /// As for `project`, with `bias` holding `OUT` ready values.
    pub(crate) unsafe fn project_biased<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        stream: &CudaStream,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: &BiasWeights<'_, Role, OUT>,
        output: &mut Batch<OUT>,
    ) -> Result<()> {
        let stride = weight.shape().row_stride();
        Self::check_projection_extents(
            input,
            rows,
            weight.buffer().len(),
            stride,
            Some(bias.buffer().len()),
            output.buffer().len(),
            IN,
            OUT,
        )?;
        match reference_plan(rows, IN, OUT) {
            ReductionPlan::MmaChain if OUT == 1024 => self.linear.decode_linear_bias_bf16(
                stream,
                &self.biased_1024,
                input,
                weight.buffer(),
                bias.buffer(),
                output.buffer_mut(),
                rows,
                IN,
                OUT,
                IN,
                stride,
            )?,
            ReductionPlan::FmaChains { chains: 16 } if OUT == 1024 => {
                self.reference.fma16_linear_bias_bf16(
                    stream,
                    &self.fma16_bias_1024,
                    input,
                    weight.buffer(),
                    bias.buffer(),
                    output.buffer_mut(),
                    IN,
                    OUT,
                    stride,
                )?
            }
            _ => return Err("unsupported biased decoder projection shape".into()),
        }
        Ok(())
    }
    /// Submit exactly the original norm/QKV/RoPE/attention/residual/FFN schedule.
    ///
    /// Hidden and FFN widths are typed independently of the fixed sixteen-query,
    /// eight-KV, 128-value head geometry. No allocation, upload, readback, or
    /// synchronization occurs here; arithmetic and BF16 boundaries are unchanged.
    ///
    /// # Safety
    /// All weights and buffers must share the stream context and be initialized
    /// and ready. Scratch/output/cache owners must be distinct and exclusively
    /// ordered on this stream, retained until completion including errors and
    /// unwinding. The cache prefix `0..start` must be valid; `start + rows` equals
    /// `key_tokens` and must fit `cache_capacity`. Cache guard tails immediately
    /// follow that capacity. Rotary tables hold `rows * 64` coefficients, and
    /// status contains one initialized word. The caller validates typed weight
    /// extents/strides and supported projection widths before submitting work.
    /// Partial storage/launches must have been prepared for the live key prefix;
    /// no captured graph may observe replacement of any referenced owner.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_layer<const H: usize, const I: usize>(
        &self,
        stream: &CudaStream,
        s: &mut LayerScratch<H, I>,
        weight: &DecoderWeights<'_, H, I, 2048, 1024, 128>,
        cache_key: &mut DeviceBuffer<Bf16>,
        cache_value: &mut DeviceBuffer<Bf16>,
        cache_capacity: usize,
        cos: &DeviceBuffer<Bf16>,
        sin: &DeviceBuffer<Bf16>,
        start: usize,
        key_tokens: usize,
        status: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let geometry = attention_geometry(s.rows, key_tokens)?;
        let live_end = start
            .checked_add(s.rows)
            .ok_or("decoder cache prefix overflow")?;
        let cache_elements = cache_capacity
            .checked_mul(1024)
            .and_then(|values| values.checked_add(GUARD_LEN))
            .ok_or("decoder guarded cache extent overflow")?;
        if live_end != key_tokens
            || key_tokens > cache_capacity
            || geometry.splits != s.splits
            || cache_key.len() != cache_elements
            || cache_value.len() != cache_elements
        {
            return Err(
                "decoder layer cache prefix, guard extent, or partial geometry mismatch".into(),
            );
        }
        let m = self;
        // SAFETY: the caller owns the checked schedule resources through completion;
        // the checks above prevent guard-address overflow and stale split geometry.
        unsafe {
            let a = weight.attention();
            let f = weight.ffn();
            crate::norm_specialized::enqueue(
                &m.norms,
                stream,
                s.hidden.buffer(),
                weight.input_norm().buffer(),
                s.normalized.buffer_mut(),
                s.rows,
                H,
                EPSILON,
            )?;
            m.project(
                stream,
                s.normalized.buffer(),
                s.rows,
                a.query(),
                &mut s.query,
            )?;
            m.project(stream, s.normalized.buffer(), s.rows, a.key(), &mut s.key)?;
            m.project(
                stream,
                s.normalized.buffer(),
                s.rows,
                a.value(),
                &mut s.value,
            )?;
            crate::norm_specialized::enqueue(
                &m.norms,
                stream,
                s.query.buffer(),
                a.query_norm().buffer(),
                s.query_norm.buffer_mut(),
                s.rows * 16,
                128,
                EPSILON,
            )?;
            crate::norm_specialized::enqueue(
                &m.norms,
                stream,
                s.key.buffer(),
                a.key_norm().buffer(),
                s.key_norm.buffer_mut(),
                s.rows * 8,
                128,
                EPSILON,
            )?;
            m.rotary.rotate_bf16(
                stream,
                grid(s.rows * 1024),
                s.query_norm.buffer(),
                cos,
                sin,
                &mut s.rotated_query.buffer,
                s.rows,
                16,
                128,
            )?;
            m.rotary.rotate_bf16(
                stream,
                grid(s.rows * 512),
                s.key_norm.buffer(),
                cos,
                sin,
                &mut s.rotated_key.buffer,
                s.rows,
                8,
                128,
            )?;
            m.rotary.append_bf16(
                stream,
                grid(s.rows * 1024),
                &s.rotated_key.buffer,
                s.value.buffer(),
                cache_key,
                cache_value,
                start * 1024,
                s.rows * 1024,
            )?;
            m.attention.partial(
                stream,
                &s.partial_launch,
                &s.rotated_query.buffer,
                cache_key,
                cache_value,
                &mut s.partials,
                s.rows as u32,
                key_tokens as u32,
                16,
                8,
                start as u32,
                true,
            )?;
            m.attention.merge(
                stream,
                &s.merge_launch,
                &s.partials,
                &mut s.attended.buffer,
                s.rows as u32,
                16,
                s.splits as u32,
            )?;
            m.schedule.attention_guards(
                stream,
                one_warp(),
                &s.partials,
                &s.rotated_query.buffer,
                &s.rotated_key.buffer,
                &s.attended.buffer,
                cache_key,
                cache_value,
                status,
                s.rows,
                s.splits,
                cache_capacity,
            )?;
            m.project(
                stream,
                &s.attended.buffer,
                s.rows,
                a.output(),
                &mut s.projected,
            )?;
            m.elements.add_bf16(
                stream,
                grid(s.rows * H),
                s.hidden.buffer(),
                s.projected.buffer(),
                s.residual.buffer_mut(),
                s.rows * H,
            )?;
            crate::norm_specialized::enqueue(
                &m.norms,
                stream,
                s.residual.buffer(),
                weight.post_attention_norm().buffer(),
                s.post.buffer_mut(),
                s.rows,
                H,
                EPSILON,
            )?;
            m.project(stream, s.post.buffer(), s.rows, f.gate(), &mut s.gate)?;
            m.project(stream, s.post.buffer(), s.rows, f.up(), &mut s.up)?;
            m.linear.activate_bf16(
                stream,
                grid((s.rows * I).div_ceil(8)),
                s.gate.buffer(),
                s.up.buffer(),
                s.silu.buffer_mut(),
                s.product.buffer_mut(),
                s.rows * I,
            )?;
            m.project(stream, s.product.buffer(), s.rows, f.down(), &mut s.down)?;
            m.elements.add_bf16(
                stream,
                grid(s.rows * H),
                s.residual.buffer(),
                s.down.buffer(),
                s.hidden.buffer_mut(),
                s.rows * H,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Split boundaries must move both the grid and the immediate live guard.
    #[test]
    fn attention_geometry_tracks_exact_live_split_boundaries() {
        for rows in [1, 2] {
            for (tokens, splits) in [(1, 1), (16, 1), (64, 1), (65, 2), (128, 2), (129, 3)] {
                let geometry = attention_geometry(rows, tokens).unwrap();
                assert_eq!(geometry.splits, splits);
                assert_eq!(geometry.partial_blocks, (rows * 16 * splits) as u32);
                assert_eq!(geometry.merge_blocks, (rows * 16) as u32);
                assert_eq!(geometry.partial_values, rows * 16 * splits * 130 + 8);
            }
        }
    }

    /// Zero work, ABI truncation and signed-grid overflow must fail before allocation.
    #[test]
    fn attention_geometry_rejects_invalid_or_overflowing_work() {
        for (rows, tokens) in [
            (0, 1),
            (3, 1),
            (1, 0),
            (1, usize::MAX),
            (2, u32::MAX as usize),
        ] {
            assert!(attention_geometry(rows, tokens).is_err());
        }
    }
}

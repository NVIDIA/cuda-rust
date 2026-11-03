/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed synchronous decoder composition.
//!
//! Dense `Batch<W>` values carry hidden widths through embedding, projection,
//! normalization, attention, and residual/FFN stages. Projection weights carry
//! their input/output widths and semantic roles independently of storage strides.
//! Context fences establish readiness; Rust borrows alone do not complete GPU work.
//!
//! Each layer appends its own K/V cache. `forward` publishes a new common token
//! prefix only after all layers and the final norm succeed, poisoning the cache
//! on a later failure. One/two-row projections use smaller decode CTAs while
//! preserving the general MMA reduction and BF16 rounding boundaries.
use crate::operator::Result;
use crate::{element_kernel, ffn_kernel, norm_kernel, operator, rotary::HostTables};
use cuda_core::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    BFloat16,
    decoder::{
        BiasWeights, DecoderShape, DecoderWeights, EmbeddingWeights, FinalNorm, NormRole,
        NormWeights, TokenIds,
    },
    ffn::{ProjectionRole, ProjectionWeights, RowMajorShape},
};
use std::{marker::PhantomData, sync::Arc};

/// Exact-size reuse of initialized allocations after asynchronous users finish.
#[path = "initialized_pool.rs"]
mod initialized_pool;
use initialized_pool::Lease;
/// Pool of exclusively leased BF16 device buffers, private to a runner and its helpers.
pub(crate) type BufferPool = initialized_pool::Pool<DeviceBuffer<BFloat16>>;

/// Create a pool retaining at most 64 MiB total, 4 MiB per allocation, and 16 per exact size.
///
/// Only idle buffers count toward limits. A context-wide fence must succeed
/// before return-to-pool reuse, covering reads enqueued through shared borrows.
pub(crate) fn buffer_pool() -> BufferPool {
    // Keep large prefill/codec shapes transient. Idle storage is bounded even
    // when a runner serves requests with many different prompt lengths.
    BufferPool::new(64 * 1024 * 1024, 4 * 1024 * 1024, 16, |buffer| {
        // A shared buffer borrow can enqueue a read on another stream. Rust's
        // borrow ending alone does not establish CUDA completion. Synchronize
        // the entire context before making this allocation writable again;
        // synchronize also binds the context on a fresh host thread.
        buffer.context().synchronize().is_ok()
    })
}

/// Language decoder RMSNorm epsilon, added to the FP32 mean square before reciprocal square root.
pub const EPSILON: f32 = 1e-6;

/// Owned dense BF16 rows. Width cannot be changed by a caller, nor can the
/// backing buffer be mutably borrowed while an operator reads it.
pub struct Batch<const W: usize> {
    /// Exclusive dense storage; a weak pool lease allows this batch to outlive its runner.
    buffer: Lease<DeviceBuffer<BFloat16>>,
    /// Number of contiguous rows, each containing exactly `W` BF16 elements.
    rows: usize,
}

impl<const W: usize> Batch<W> {
    /// Upload exactly `rows * W` initialized BF16 values as dense row-major storage.
    ///
    /// Rejects invalid/overflowing shapes or mismatched lengths and returns CUDA
    /// errors. The upload completes before the borrowed host slice can be released.
    pub fn from_host(stream: &CudaStream, rows: usize, values: &[BFloat16]) -> Result<Self> {
        let shape = RowMajorShape::dense(rows, W)?;
        if values.len() != shape.required_len() {
            return Err("input must contain exactly rows * typed width values".into());
        }
        stream.context().bind_to_thread()?;
        Ok(Self {
            buffer: Lease::owned(DeviceBuffer::from_host(stream, values)?),
            rows,
        })
    }
    /// Allocate dense rows initialized with the diagnostic BF16 guard pattern.
    ///
    /// This nonpooled owner is used by probes and persistent graph storage. Values
    /// are initialized but are not meaningful activations until overwritten.
    pub(crate) fn allocate(stream: &CudaStream, rows: usize) -> Result<Self> {
        let shape = RowMajorShape::dense(rows, W)?;
        stream.context().bind_to_thread()?;
        Ok(Self {
            buffer: Lease::owned(DeviceBuffer::from_host(
                stream,
                &vec![BFloat16::from_bits(operator::GUARD); shape.required_len()],
            )?),
            rows,
        })
    }
    /// Acquire exact-size initialized storage or allocate fresh guard-filled rows.
    ///
    /// Reused contents are retained, so output kernels must overwrite every logical
    /// element. Checks shape/byte overflow and the acquired buffer's CUDA context.
    pub(crate) fn allocate_from(
        pool: &BufferPool,
        stream: &CudaStream,
        rows: usize,
    ) -> Result<Self> {
        let shape = RowMajorShape::dense(rows, W)?;
        let len = shape.required_len();
        let bytes = len
            .checked_mul(std::mem::size_of::<BFloat16>())
            .ok_or("batch byte extent overflow")?;
        stream.context().bind_to_thread()?;
        // Pool ownership is private to a Runner and its decode projection
        // helper, which always use the same context. Kernels fully overwrite
        // logical output elements; fresh storage retains sentinel initialization.
        let buffer = pool.acquire(len, bytes, || {
            DeviceBuffer::from_host(stream, &vec![BFloat16::from_bits(operator::GUARD); len])
        })?;
        if !Arc::ptr_eq(buffer.context(), stream.context()) {
            return Err("pooled batch context mismatch".into());
        }
        Ok(Self { buffer, rows })
    }
    /// Return the logical row count; the width remains the type parameter `W`.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Borrow dense storage without transferring ownership.
    ///
    /// This borrow alone does not establish completion of external asynchronous users.
    pub fn buffer(&self) -> &DeviceBuffer<BFloat16> {
        &self.buffer
    }
    /// Borrow storage exclusively for checked writes; callers still establish device readiness.
    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<BFloat16> {
        &mut self.buffer
    }
    /// Fence all work in the matching context and copy dense BF16 values to the host.
    ///
    /// Rejects a foreign-context stream and propagates synchronization/copy errors.
    pub fn read(&self, stream: &CudaStream) -> Result<Vec<BFloat16>> {
        if !Arc::ptr_eq(self.buffer.context(), stream.context()) {
            return Err("read stream context mismatch".into());
        }
        stream.context().synchronize()?;
        Ok(self.buffer.to_host_vec(stream)?)
    }
}

/// A failed multi-layer append may leave different layer cursors. Such a cache
/// cannot be reused until reset; a successful call publishes a common prefix.
#[derive(Debug)]
struct CacheState {
    /// Common prefix published only after all decoder layers and final norm succeed.
    len: usize,
    /// Maximum token rows per layer cache.
    capacity: usize,
    /// Number of private layer caches required by every append.
    layers: usize,
    /// Whether an interrupted append or prediction failure requires a reset before reuse.
    poisoned: bool,
}

impl CacheState {
    /// Validate poison state, layer count, and checked append extent without advancing the prefix.
    fn check(&self, layers: usize, tokens: usize) -> Result<usize> {
        if self.poisoned {
            return Err("failed decoder cache must be reset before reuse".into());
        }
        if layers == 0 || layers != self.layers {
            return Err("decoder layer count does not match owned caches".into());
        }
        let end = self
            .len
            .checked_add(tokens)
            .ok_or("cache cursor overflow")?;
        if end > self.capacity {
            return Err("decoder cache capacity exceeded".into());
        }
        Ok(end)
    }

    /// Validate every physical layer's completed prefix and fixed token capacity.
    fn check_layer_prefixes(&self, layers: impl IntoIterator<Item = (usize, usize)>) -> Result<()> {
        let mut count = 0;
        for (len, capacity) in layers {
            if count >= self.layers || len != self.len || capacity != self.capacity {
                return Err("inconsistent decoder layer cache geometry".into());
            }
            count += 1;
        }
        if count != self.layers {
            return Err("decoder layer count does not match owned caches".into());
        }
        Ok(())
    }

    /// Require a healthy complete stack before changing its private ownership model.
    fn check_execution(
        &self,
        expected_layers: usize,
        layers: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<()> {
        self.check(expected_layers, 0)?;
        self.check_layer_prefixes(layers)
    }

    /// Check a completed transaction's proposed end without publishing any metadata.
    fn check_publication(&self, end: usize) -> Result<()> {
        if !self.poisoned {
            return Err("execution cache publication requires an active transaction".into());
        }
        if end <= self.len || end > self.capacity {
            return Err("execution cache publication must advance within capacity".into());
        }
        Ok(())
    }
}

/// One private K/V allocation per layer. Hidden and intermediate widths prevent
/// accidental Talker/predictor cache interchange even though both have eight KV heads.
pub struct DecoderCache<const H: usize, const I: usize, const K: usize> {
    /// Published common prefix and poison-on-failure metadata.
    state: CacheState,
    /// Separate key/value buffers and append cursors for each layer, with 128-value heads.
    layers: Vec<operator::KvCache<K, 128>>,
    /// Bind cache identity to hidden/FFN widths even though storage is sized by KV heads.
    widths: PhantomData<([(); H], [(); I])>,
}

impl<const H: usize, const I: usize, const K: usize> DecoderCache<H, I, K> {
    /// Allocate one K/V cache per nonzero layer count, each holding `capacity` token rows.
    ///
    /// `K` is the number of KV heads and each head has 128 values. Validates decoder
    /// geometry and propagates allocation/CUDA errors before returning an empty cache.
    pub fn new(stream: &CudaStream, layers: usize, capacity: usize) -> Result<Self> {
        if layers == 0 {
            return Err("decoder needs at least one layer".into());
        }
        DecoderShape::<H, I, K, K, 128>::new(0, H)?;
        Ok(Self {
            state: CacheState {
                len: 0,
                capacity,
                layers,
                poisoned: false,
            },
            layers: (0..layers)
                .map(|_| operator::KvCache::new(stream, capacity))
                .collect::<Result<_>>()?,
            widths: PhantomData,
        })
    }
    /// Return the last successfully published common prefix, in token rows.
    pub fn len(&self) -> usize {
        self.state.len
    }
    /// Report whether no common prefix has been published yet.
    #[allow(
        dead_code,
        reason = "Pairs with `len` for library consumers; this binary never asks"
    )]
    pub fn is_empty(&self) -> bool {
        self.state.len == 0
    }
    /// Report whether a failed operation prevents reuse until `reset`.
    pub fn is_poisoned(&self) -> bool {
        self.state.poisoned
    }
    /// A caller that consumes a completed decoder step but fails before
    /// returning its prediction must not silently retry against the new prefix.
    pub(crate) fn invalidate(&mut self) {
        self.state.poisoned = true;
    }
    /// Reset every layer cursor and clear poison state without reallocating or clearing storage.
    ///
    /// Old values outside the new empty prefix are not logically accessible.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.state.len = 0;
        self.state.poisoned = false;
    }
    /// Read each layer's complete K/V backing buffers, including unused capacity and guards.
    ///
    /// `len` reports the last common published prefix separately; the returned
    /// vectors are not truncated to it. A poisoned cache may contain partially
    /// advanced layers. Readback errors propagate, and snapshots do not repair
    /// or publish a consistent sequence.
    pub fn snapshots(&self, stream: &CudaStream) -> Result<Vec<(Vec<BFloat16>, Vec<BFloat16>)>> {
        self.layers
            .iter()
            .map(|layer| layer.snapshot(stream))
            .collect()
    }

    /// Consume a healthy completed stack into private execution-region ownership.
    ///
    /// Checks all layer counts, prefixes, capacities, contexts and guarded storage
    /// before a single context-wide readiness fence. No buffer is copied, reset or
    /// exposed through the synchronous cache API afterward. Failure consumes only
    /// buffers whose public cache operations have already completed.
    pub(crate) fn into_execution(
        self,
        stream: &CudaStream,
        expected_layers: usize,
    ) -> Result<ExecutionCache<H, I, K>> {
        self.state.check_execution(
            expected_layers,
            self.layers
                .iter()
                .map(|layer| (layer.len(), layer.capacity())),
        )?;
        for layer in &self.layers {
            layer.validate_execution(stream)?;
        }
        stream.context().synchronize()?;
        let layers = self
            .layers
            .into_iter()
            .map(|layer| layer.into_execution(stream))
            .collect::<Result<_>>()?;
        Ok(ExecutionCache {
            state: self.state,
            layers,
            widths: self.widths,
        })
    }
}

/// Private cache storage whose pending writes belong to one exclusive execution region.
///
/// Hidden/FFN widths retain the decoder role. There is no conversion back to a
/// synchronous `DecoderCache`; the plan must retain this owner until all submitted
/// work completes, including errors and unwinding.
pub(crate) struct ExecutionCache<const H: usize, const I: usize, const K: usize> {
    /// Last completed common prefix and sticky transaction-failure marker.
    state: CacheState,
    /// Owned guarded K/V allocations, reused without temporary pool leases.
    layers: Vec<operator::OwnedKvCache<K, 128>>,
    /// Decoder hidden/FFN width branding survives the consuming transfer.
    widths: PhantomData<([(); H], [(); I])>,
}

impl<const H: usize, const I: usize, const K: usize> ExecutionCache<H, I, K> {
    /// Return the last successfully completed common prefix in token rows.
    pub(crate) fn len(&self) -> usize {
        self.state.len
    }

    /// Return the fixed token capacity shared by every layer.
    pub(crate) fn capacity(&self) -> usize {
        self.state.capacity
    }

    /// Report an active or failed transaction; a failed plan must not be reused.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.state.poisoned
    }

    /// Mark the stack before its first enqueue; failure leaves this marker set.
    pub(crate) fn poison(&mut self) {
        self.state.poisoned = true;
    }

    /// Borrow private typed allocations for the region's ordered kernel launches.
    ///
    /// This does not establish device readiness or completion. Launch safety
    /// requires the plan to retain every owner and prohibit external overlapping
    /// uses until its unconditional completion boundary. Cursors remain unchanged.
    pub(crate) fn layers_mut(&mut self) -> &mut [operator::OwnedKvCache<K, 128>] {
        &mut self.layers
    }

    /// Publish all layer cursors and the common prefix after a successful step.
    ///
    /// Rejects absent transactions, nonadvancing/out-of-capacity ends, and changed
    /// layer geometry before modifying any cursor. Success clears poison atomically
    /// with publication at this exclusive host boundary; no CUDA work is submitted.
    ///
    /// # Safety
    /// Every layer's K/V range through `end` must be initialized, and all region
    /// work, final output work and device guard validation must have completed
    /// successfully. The caller must retain storage shape/context invariants and
    /// must never use this to recover a failed or incompletely submitted step.
    pub(crate) unsafe fn publish(&mut self, end: usize) -> Result<()> {
        self.state.check_publication(end)?;
        self.state.check_layer_prefixes(
            self.layers
                .iter()
                .map(|layer| (layer.len(), layer.capacity())),
        )?;
        for layer in &mut self.layers {
            // SAFETY: all cursors/capacities were checked before any mutation;
            // the caller proves successful initialized completion through end.
            unsafe { layer.publish_completed(end) };
        }
        self.state.len = end;
        self.state.poisoned = false;
        Ok(())
    }
}

/// Optional synchronous readbacks for this numerical example, disabled in normal composition.
#[derive(Default)]
pub struct Trace {
    /// Enable blocking stage readbacks; disabled by default.
    pub enabled: bool,
    /// Stage names and flattened logical BF16 values, in execution order.
    pub stages: Vec<(&'static str, Vec<BFloat16>)>,
}
impl Trace {
    /// When enabled, read a buffer and retain at most `length` elements under its stage name.
    ///
    /// Callers provide completed stage buffers and logical lengths excluding guard tails.
    fn record(
        &mut self,
        stream: &CudaStream,
        name: &'static str,
        buffer: &DeviceBuffer<BFloat16>,
        length: usize,
    ) -> Result<()> {
        if self.enabled {
            let mut values = buffer.to_host_vec(stream)?;
            values.truncate(length);
            self.stages.push((name, values));
        }
        Ok(())
    }
}

/// Round nonzero work up to 128-thread CTAs, returning no launch for empty work.
///
/// `per_block` is a nonzero amount of logical work per CTA supplied by each
/// caller; reject a grid beyond CUDA's signed grid-x limit.
fn grid(work: usize, per_block: usize) -> Result<Option<LaunchConfig>> {
    if work == 0 {
        return Ok(None);
    }
    let blocks = work.div_ceil(per_block);
    if blocks > i32::MAX as usize {
        return Err("grid exceeds CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Reuses the tensor-core FFN projections, RMSNorm, RoPE and GQA kernels.
/// Safe methods establish input readiness and retain borrows through completion,
/// including launch errors. This deliberately synchronous baseline is B=1.
pub struct Runner {
    /// Execution stream whose context must match all borrowed inputs and weights.
    stream: Arc<CudaStream>,
    /// Bounded idle output storage shared with the decode projection helper.
    pool: BufferPool,
    /// Embedding, row selection, concatenation, residual, and elementwise kernels.
    elements: element_kernel::kernels::LoadedModule,
    /// General multirow MMA projections and BF16 activation kernels.
    linear: ffn_kernel::kernels::LoadedModule,
    /// One/two-row MMA path retaining the general projection's arithmetic order.
    decode_linear: crate::decode_linear::Runner,
    /// General and specialized RMSNorm entry points.
    norms: norm_kernel::kernels::LoadedModule,
    /// Checked RoPE, cache append, and causal grouped-query attention composition.
    attention: operator::Runner,
}

impl Runner {
    /// Borrow the execution stream for compatible uploads, caches, and synchronous readbacks.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    /// Load all composition kernels for one stream and initialize bounded output reuse.
    ///
    /// Module-loading and CUDA errors propagate; no checkpoint tensors are loaded.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let pool = buffer_pool();
        Ok(Self {
            elements: element_kernel::kernels::load(stream.context())?,
            // SAFETY: the module embeds the artifact compiled from the shared
            // FFN source; the decode runner prepares its contracted launches.
            linear: unsafe { ffn_kernel::kernels::load(stream.context())? },
            decode_linear: crate::decode_linear::Runner::with_pool(stream.clone(), pool.clone())?,
            norms: norm_kernel::kernels::load(stream.context())?,
            attention: operator::Runner::new(stream.clone())?,
            stream,
            pool,
        })
    }
    /// Acquire dense output rows from this runner's same-context initialized pool.
    fn allocate<const W: usize>(&self, rows: usize) -> Result<Batch<W>> {
        Batch::allocate_from(&self.pool, &self.stream, rows)
    }
    /// Require identical context handles and fence all producers before using borrowed buffers.
    fn ready(&self, buffers: &[&DeviceBuffer<BFloat16>]) -> Result<()> {
        if buffers
            .iter()
            .any(|buffer| !Arc::ptr_eq(buffer.context(), self.stream.context()))
        {
            return Err("decoder buffers must share the runner context".into());
        }
        self.stream.context().synchronize()?;
        Ok(())
    }
    /// Attempt stream completion after a launch, returning launch errors before fence errors.
    ///
    /// Callers keep all participating buffers alive through this function.
    fn finish(&self, launch: Result<()>) -> Result<()> {
        let completed = self.stream.synchronize();
        launch?;
        completed?;
        Ok(())
    }
    /// Gather vocabulary-checked token rows from a dense `[V, W]` embedding table.
    ///
    /// Returns an independently owned dense batch in ID order, permitting repeated
    /// IDs. Rejects padded table strides or foreign contexts and propagates CUDA errors.
    pub fn embed<const V: usize, const W: usize>(
        &self,
        weights: &EmbeddingWeights<'_, V, W>,
        ids: TokenIds<'_, V>,
    ) -> Result<Batch<W>> {
        let shape = weights.output_shape(ids)?;
        if weights.shape().table().row_stride() != W {
            return Err("embedding kernel requires dense table rows".into());
        }
        self.ready(&[weights.buffer()])?;
        let mut output = self.allocate(ids.len())?;
        if let Some(config) = grid(shape.required_len(), 128)? {
            let ids_dev = DeviceBuffer::from_host(&self.stream, ids.as_slice())?;
            // SAFETY: TokenIds proves every index is in the checked table;
            // output ownership and one-thread-per-element writes are disjoint.
            let launch = unsafe {
                self.elements
                    .embedding_bf16(
                        &self.stream,
                        config,
                        weights.buffer(),
                        &ids_dev,
                        &mut output.buffer,
                        ids.len(),
                        W,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Copy checked row indices into a new dense batch, preserving their order and duplicates.
    ///
    /// Rejects out-of-range IDs; an empty selection produces an empty batch.
    pub fn select<const W: usize>(&self, input: &Batch<W>, ids: &[u32]) -> Result<Batch<W>> {
        if ids.iter().any(|&id| id as usize >= input.rows) {
            return Err("selected row exceeds batch length".into());
        }
        self.ready(&[input.buffer()])?;
        let mut output = self.allocate(ids.len())?;
        if let Some(config) = grid(output.buffer.len(), 128)? {
            let indices = DeviceBuffer::from_host(&self.stream, ids)?;
            // SAFETY: all row IDs were checked against the owned dense input.
            let launch = unsafe {
                self.elements
                    .embedding_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        &indices,
                        &mut output.buffer,
                        ids.len(),
                        W,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Append right rows after left rows without changing BF16 bits.
    ///
    /// Checks row-count overflow and context readiness, then returns distinct dense storage.
    pub fn concat<const W: usize>(&self, left: &Batch<W>, right: &Batch<W>) -> Result<Batch<W>> {
        let rows = left
            .rows
            .checked_add(right.rows)
            .ok_or("concatenated row count overflow")?;
        self.ready(&[left.buffer(), right.buffer()])?;
        let mut output = self.allocate(rows)?;
        let total = output.buffer.len();
        if let Some(config) = grid(total, 128)? {
            // SAFETY: checked total extent and disjoint destination allocation.
            let launch = unsafe {
                self.elements
                    .concat_bf16(
                        &self.stream,
                        config,
                        left.buffer(),
                        right.buffer(),
                        &mut output.buffer,
                        left.buffer.len(),
                        total,
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Sum codec embedding rows in FP32, rounding only the final sum to BF16.
    pub fn sum_rows<const W: usize>(&self, input: &Batch<W>) -> Result<Batch<W>> {
        self.ready(&[input.buffer()])?;
        let mut output = self.allocate(1)?;
        let config = grid(W, 128)?.ok_or("zero row width")?;
        // SAFETY: one thread owns each column and visits only valid input rows.
        let launch = unsafe {
            self.elements
                .sum_rows_bf16(
                    &self.stream,
                    config,
                    input.buffer(),
                    &mut output.buffer,
                    input.rows,
                    W,
                )
                .map_err(Into::into)
        };
        self.finish(launch)?;
        Ok(output)
    }
    /// Project dense `[rows, IN]` activations through `[OUT, IN]` role-typed weights.
    ///
    /// Optional same-role bias is added to FP32 accumulators before BF16 output
    /// rounding. One/two-row inputs use decode CTAs; larger inputs use the general
    /// MMA launch with the same reduction order. Returns dense `[rows, OUT]` output.
    pub fn project<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        self.project_buffer(input.buffer(), input.rows(), weight, bias)
    }
    /// Check dense input storage and dispatch to decode or general MMA projection.
    ///
    /// Weights may have a checked padded row stride. Readiness, launch limits, and
    /// completion are checked before returning independently owned output rows.
    fn project_buffer<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &DeviceBuffer<BFloat16>,
        rows: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        RowMajorShape::dense(rows, IN)?.validate_len(input.len())?;
        if matches!(rows, 1 | 2) {
            return self
                .decode_linear
                .project_buffer(input, rows, IN, weight, bias);
        }
        self.ready(&[input, weight.buffer()])?;
        if let Some(bias) = bias {
            self.ready(&[bias.buffer()])?;
        }
        let tiles = rows
            .div_ceil(16)
            .checked_mul(OUT.div_ceil(8))
            .ok_or("projection tile overflow")?;
        let config = grid(tiles, 4)?;
        let mut output = self.allocate(rows)?;
        if let Some(config) = config {
            // SAFETY: checked [out,in] weights and dense input/output extents;
            // the full-warp MMA map gives each output exactly one writer.
            let launch = unsafe {
                match bias {
                    Some(bias) => self.linear.linear_bias_bf16(
                        &self.stream,
                        config,
                        input,
                        weight.buffer(),
                        bias.buffer(),
                        &mut output.buffer,
                        rows,
                        IN,
                        OUT,
                        IN,
                        weight.shape().row_stride(),
                    ),
                    None => self.linear.linear_bf16(
                        &self.stream,
                        config,
                        input,
                        weight.buffer(),
                        &mut output.buffer,
                        rows,
                        IN,
                        OUT,
                        IN,
                        weight.shape().row_stride(),
                    ),
                }
                .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Apply row-wise RMSNorm with role-checked `W`-wide scales and language-model `EPSILON`.
    ///
    /// Accumulation and normalization use FP32 before each result narrows to BF16.
    pub fn norm<Role: NormRole, const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &NormWeights<'_, Role, W>,
    ) -> Result<Batch<W>> {
        self.norm_rows(input, weight.buffer(), W)
    }
    /// Apply row-wise RMSNorm with an explicit finite positive checkpoint epsilon.
    ///
    /// Codec and language decoders retain their distinct epsilon values. Rejects
    /// nonpositive/nonfinite epsilon before launching; returns BF16-rounded rows.
    pub fn norm_with_epsilon<Role: NormRole, const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &NormWeights<'_, Role, W>,
        epsilon: f32,
    ) -> Result<Batch<W>> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("normalization epsilon must be finite and positive".into());
        }
        self.norm_rows_epsilon(input, weight.buffer(), W, epsilon)
    }
    /// Normalize contiguous subrows using the language decoder epsilon.
    ///
    /// `norm_width` is in elements and must divide `W`; head norms use 128.
    fn norm_rows<const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &DeviceBuffer<BFloat16>,
        norm_width: usize,
    ) -> Result<Batch<W>> {
        self.norm_rows_epsilon(input, weight, norm_width, EPSILON)
    }
    /// Validate subrow geometry, fence inputs, and enqueue the selected RMSNorm implementation.
    ///
    /// Each of `input.rows() * (W / norm_width)` subrows shares the same scale
    /// vector. Internal callers supply a finite positive epsilon; the helper checks
    /// width divisibility, scale extent, and multiplication overflow.
    fn norm_rows_epsilon<const W: usize>(
        &self,
        input: &Batch<W>,
        weight: &DeviceBuffer<BFloat16>,
        norm_width: usize,
        epsilon: f32,
    ) -> Result<Batch<W>> {
        if norm_width == 0 || !W.is_multiple_of(norm_width) || weight.len() < norm_width {
            return Err("normalization width does not divide the typed row".into());
        }
        let rows = input
            .rows
            .checked_mul(W / norm_width)
            .ok_or("norm row overflow")?;
        self.ready(&[input.buffer(), weight])?;
        let mut output = self.allocate(input.rows)?;
        // SAFETY: ready establishes input readiness; typed rows divide exactly
        // into norm_width rows, output is exclusive, and finish retains all
        // buffers until completion on either enqueue success or failure.
        let launch = unsafe {
            crate::norm_specialized::enqueue(
                &self.norms,
                &self.stream,
                input.buffer(),
                weight,
                &mut output.buffer,
                rows,
                norm_width,
                epsilon,
            )
        };
        self.finish(launch)?;
        Ok(output)
    }
    /// Add matching dense batches in FP32 and narrow each residual result to BF16.
    ///
    /// Rejects unequal row counts or foreign contexts; inputs remain unchanged.
    pub fn add<const W: usize>(&self, left: &Batch<W>, right: &Batch<W>) -> Result<Batch<W>> {
        if left.rows != right.rows {
            return Err("residual row count mismatch".into());
        }
        self.ready(&[left.buffer(), right.buffer()])?;
        let mut output = self.allocate(left.rows)?;
        if let Some(config) = grid(left.buffer.len(), 128)? {
            // SAFETY: exact matching extents, distinct output, disjoint threads.
            let launch = unsafe {
                self.elements
                    .add_bf16(
                        &self.stream,
                        config,
                        left.buffer(),
                        right.buffer(),
                        &mut output.buffer,
                        left.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Evaluate SiLU with cute-rs stable FP32 sigmoid branches, then round once to BF16.
    ///
    /// The device helper retains its approximate base-two exponential and
    /// reciprocal operations; this is not an exact host `exp` reference.
    pub fn silu<const W: usize>(&self, input: &Batch<W>) -> Result<Batch<W>> {
        self.ready(&[input.buffer()])?;
        let mut output = self.allocate(input.rows)?;
        if let Some(config) = grid(input.buffer.len(), 128)? {
            // SAFETY: matching extents and one output element per thread.
            let launch = unsafe {
                self.elements
                    .silu_bf16(
                        &self.stream,
                        config,
                        input.buffer(),
                        &mut output.buffer,
                        input.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(output)
    }
    /// Apply the already validated BF16 SiLU/product rounding sequence.
    pub fn swiglu<const W: usize>(&self, gate: &Batch<W>, up: &Batch<W>) -> Result<Batch<W>> {
        if gate.rows() != up.rows() {
            return Err("SwiGLU row counts differ".into());
        }
        self.ready(&[gate.buffer(), up.buffer()])?;
        let mut silu = self.allocate::<W>(gate.rows())?;
        let mut product = self.allocate::<W>(gate.rows())?;
        if let Some(config) = grid(gate.buffer.len().div_ceil(8), 128)? {
            // SAFETY: typed equal-size buffers; each lane owns disjoint aligned
            // eight-element tiles, with the validated kernel's scalar tail.
            let launch = unsafe {
                self.linear
                    .activate_bf16(
                        &self.stream,
                        config,
                        gate.buffer(),
                        up.buffer(),
                        &mut silu.buffer,
                        &mut product.buffer,
                        gate.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        Ok(product)
    }
    /// Check all layer-weight contexts and complete their pending producers.
    ///
    /// Typed dimensions and semantic roles are validated separately by `DecoderShape`.
    fn validate_layer<const H: usize, const I: usize, const QW: usize, const KW: usize>(
        &self,
        weight: &DecoderWeights<'_, H, I, QW, KW, 128>,
    ) -> Result<()> {
        let a = weight.attention();
        let f = weight.ffn();
        self.ready(&[
            weight.input_norm().buffer(),
            weight.post_attention_norm().buffer(),
            a.query().buffer(),
            a.key().buffer(),
            a.value().buffer(),
            a.output().buffer(),
            a.query_norm().buffer(),
            a.key_norm().buffer(),
            f.gate().buffer(),
            f.up().buffer(),
            f.down().buffer(),
        ])
    }
    #[allow(clippy::too_many_arguments)]
    /// Single-layer numerical probe. An error after append may have advanced
    /// this raw cache; discard/reset it on error. Model calls use `forward`,
    /// which guards the complete stack with poison-on-failure state.
    pub(crate) fn layer<
        const H: usize,
        const I: usize,
        const Q: usize,
        const K: usize,
        const QW: usize,
        const KW: usize,
    >(
        &self,
        input: &Batch<H>,
        weight: &DecoderWeights<'_, H, I, QW, KW, 128>,
        cache: &mut operator::KvCache<K, 128>,
        tables: &HostTables<128>,
        trace: &mut Trace,
    ) -> Result<Batch<H>> {
        let tables = operator::DeviceTables::new(&self.stream, tables)?;
        self.layer_with_tables::<H, I, Q, K, QW, KW>(input, weight, cache, &tables, trace)
    }

    /// Execute pre-norm attention, residual addition, pre-norm SwiGLU, and final residual.
    ///
    /// `H/I` are hidden/FFN widths; `Q/K` count 128-value heads, while `QW/KW`
    /// are flattened projection widths. Reuses uploaded rotary tables and records
    /// optional stage traces. Cache append is not rolled back on later failures.
    #[allow(clippy::too_many_arguments)]
    fn layer_with_tables<
        const H: usize,
        const I: usize,
        const Q: usize,
        const K: usize,
        const QW: usize,
        const KW: usize,
    >(
        &self,
        input: &Batch<H>,
        weight: &DecoderWeights<'_, H, I, QW, KW, 128>,
        cache: &mut operator::KvCache<K, 128>,
        tables: &operator::DeviceTables<128>,
        trace: &mut Trace,
    ) -> Result<Batch<H>> {
        let shape = DecoderShape::<H, I, Q, K, 128>::new(input.rows, H)?;
        shape.validate_weights(weight)?;
        self.validate_layer(weight)?;
        if tables.tokens() != input.rows || input.rows > cache.capacity() - cache.len() {
            return Err("decoder positions or cache capacity mismatch".into());
        }
        let start = cache.len();
        let a = weight.attention();
        let normalized = self.norm(input, weight.input_norm())?;
        trace.record(
            &self.stream,
            "normalized",
            normalized.buffer(),
            input.rows * H,
        )?;
        let (q, k, v) = if matches!(input.rows, 1 | 2) {
            self.decode_linear.project_qkv(&normalized, a)?
        } else {
            (
                self.project(&normalized, a.query(), None)?,
                self.project(&normalized, a.key(), None)?,
                self.project(&normalized, a.value(), None)?,
            )
        };
        trace.record(&self.stream, "q", q.buffer(), input.rows * QW)?;
        trace.record(&self.stream, "k", k.buffer(), input.rows * KW)?;
        trace.record(&self.stream, "v", v.buffer(), input.rows * KW)?;
        let qn = self.norm_rows(&q, a.query_norm().buffer(), 128)?;
        let kn = self.norm_rows(&k, a.key_norm().buffer(), 128)?;
        trace.record(&self.stream, "q_normed", qn.buffer(), input.rows * QW)?;
        trace.record(&self.stream, "k_normed", kn.buffer(), input.rows * KW)?;
        let query = self.attention.append_prepared(
            cache,
            operator::AttentionInputs::<Q, K, 128>::new(
                qn.buffer(),
                kn.buffer(),
                v.buffer(),
                input.rows,
            )?,
            tables,
        )?;
        trace.record(
            &self.stream,
            "rotated_q",
            query.buffer(),
            query.tokens() * QW,
        )?;
        let attended = self.attention.attend(&query, cache, Some(start))?;
        trace.record(
            &self.stream,
            "attention",
            attended.buffer(),
            input.rows * QW,
        )?;
        let projected =
            self.project_buffer(attended.buffer(), attended.tokens(), a.output(), None)?;
        trace.record(
            &self.stream,
            "projected",
            projected.buffer(),
            input.rows * H,
        )?;
        let residual = self.add(input, &projected)?;
        trace.record(&self.stream, "residual", residual.buffer(), input.rows * H)?;
        let post = self.norm(&residual, weight.post_attention_norm())?;
        trace.record(&self.stream, "post_normed", post.buffer(), input.rows * H)?;
        let (gate, up) = if matches!(input.rows, 1 | 2) {
            self.decode_linear.project_gate_up(&post, weight.ffn())?
        } else {
            (
                self.project(&post, weight.ffn().gate(), None)?,
                self.project(&post, weight.ffn().up(), None)?,
            )
        };
        trace.record(&self.stream, "gate", gate.buffer(), input.rows * I)?;
        trace.record(&self.stream, "up", up.buffer(), input.rows * I)?;
        let mut silu = self.allocate::<I>(input.rows)?;
        let mut product = self.allocate::<I>(input.rows)?;
        if let Some(config) = grid(gate.buffer.len().div_ceil(8), 128)? {
            // SAFETY: dense typed FFN buffers, aligned vector tiles plus tails;
            // each thread owns a distinct eight-element output tile.
            let launch = unsafe {
                self.linear
                    .activate_bf16(
                        &self.stream,
                        config,
                        gate.buffer(),
                        up.buffer(),
                        &mut silu.buffer,
                        &mut product.buffer,
                        gate.buffer.len(),
                    )
                    .map_err(Into::into)
            };
            self.finish(launch)?;
        }
        trace.record(&self.stream, "silu", silu.buffer(), input.rows * I)?;
        trace.record(&self.stream, "product", product.buffer(), input.rows * I)?;
        let down = self.project(&product, weight.ffn().down(), None)?;
        trace.record(&self.stream, "down", down.buffer(), input.rows * H)?;
        let output = self.add(&residual, &down)?;
        trace.record(&self.stream, "output", output.buffer(), input.rows * H)?;
        Ok(output)
    }
    /// Append dense token rows through every decoder layer and apply the final RMSNorm.
    ///
    /// Validates weights, rotary row counts, and the common cache prefix before
    /// execution. Rotary tables are uploaded once for the stack. Once execution
    /// begins, failure leaves the cache poisoned and its published length unchanged;
    /// `reset` is required before reuse. Success publishes the new common prefix
    /// and returns all final-normalized rows with the original typed width.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<
        const H: usize,
        const I: usize,
        const Q: usize,
        const K: usize,
        const QW: usize,
        const KW: usize,
    >(
        &self,
        input: &Batch<H>,
        weights: &[DecoderWeights<'_, H, I, QW, KW, 128>],
        final_norm: &NormWeights<'_, FinalNorm, H>,
        cache: &mut DecoderCache<H, I, K>,
        tables: &HostTables<128>,
    ) -> Result<Batch<H>> {
        let end = cache.state.check(weights.len(), input.rows)?;
        let shape = DecoderShape::<H, I, Q, K, 128>::new(input.rows, H)?;
        if tables.tokens() != input.rows {
            return Err("rotary rows do not match decoder input".into());
        }
        self.ready(&[input.buffer(), final_norm.buffer()])?;
        for weight in weights {
            shape.validate_weights(weight)?;
            self.validate_layer(weight)?;
        }
        if cache
            .layers
            .iter()
            .any(|layer| layer.len() != cache.state.len)
        {
            return Err("inconsistent layer cache prefix".into());
        }
        // Every layer uses the same positions. Upload their immutable checked
        // coefficients once, before poisoning or changing any cache prefix.
        // The owner remains live through every synchronous layer append.
        let tables = operator::DeviceTables::new(&self.stream, tables)?;
        // Once a layer has appended, a later failure must invalidate the whole
        // sequence. Keep the old published length until every layer completes.
        cache.state.poisoned = true;
        let mut trace = Trace::default();
        let mut hidden = self.layer_with_tables::<H, I, Q, K, QW, KW>(
            input,
            &weights[0],
            &mut cache.layers[0],
            &tables,
            &mut trace,
        )?;
        for (weight, layer) in weights[1..].iter().zip(&mut cache.layers[1..]) {
            hidden = self.layer_with_tables::<H, I, Q, K, QW, KW>(
                &hidden, weight, layer, &tables, &mut trace,
            )?;
        }
        let output = self.norm(&hidden, final_norm)?;
        cache.state.len = end;
        cache.state.poisoned = false;
        Ok(output)
    }
}

/// Cache-state CPU checks and opt-in CUDA checks for pooled ownership/readiness.
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify pool return fences foreign-stream readers and leases survive pool destruction.
    #[test]
    #[ignore = "requires an available CUDA GPU"]
    fn pooled_batches_complete_foreign_stream_reads_and_survive_owner_drop() {
        /// Compile-time assertion that batches may be sent and shared across host threads.
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Batch<4>>();
        let context = cuda_core::CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let other = context.new_stream().unwrap();
        let pool = buffer_pool();
        let rows = 65536;
        let original_values = vec![BFloat16::from_f32(1.0); rows * 4];
        let replaced_values = vec![BFloat16::from_f32(3.0); rows * 4];
        let mut original = Batch::<4>::allocate_from(&pool, &stream, rows).unwrap();
        original
            .buffer_mut()
            .copy_from_host(&stream, &original_values)
            .unwrap();
        let address = original.buffer().cu_deviceptr();
        let mut copied = DeviceBuffer::<BFloat16>::zeroed(&other, rows * 4).unwrap();
        other.synchronize().unwrap();
        let (release, wait_release) = std::sync::mpsc::channel();
        let (started, wait_started) = std::sync::mpsc::channel();
        other
            .launch_host_function(move || {
                // No CUDA calls or panics in the callback. Its bounded wait also
                // guarantees cleanup if an assertion or worker unexpectedly fails.
                let _ = started.send(());
                let _ = wait_release.recv_timeout(std::time::Duration::from_secs(5));
            })
            .unwrap();
        copied
            .copy_from_device_async(original.buffer(), &other)
            .unwrap();
        wait_started
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(!other.query().unwrap());
        let (dropped, wait_dropped) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            // The shared source borrow has ended, but foreign-stream reads may
            // remain pending. Return on a thread with no current CUDA context.
            drop(original);
            dropped.send(()).unwrap();
            let mut reused = Batch::<4>::allocate_from(&pool, &stream, rows).unwrap();
            assert_eq!(reused.buffer().cu_deviceptr(), address);
            reused
                .buffer_mut()
                .copy_from_host(&stream, &replaced_values)
                .unwrap();
            let copied = copied.to_host_vec(&other).unwrap();
            assert!(
                copied
                    .iter()
                    .zip(&original_values)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            drop(reused);
            let retained = Batch::<4>::allocate_from(&pool, &stream, rows).unwrap();
            assert_eq!(retained.buffer().cu_deviceptr(), address);
            drop(pool); // its checked-out output remains independently owned
            assert!(
                retained
                    .read(&stream)
                    .unwrap()
                    .iter()
                    .zip(&replaced_values)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        });
        // The foreign stream cannot finish until released. A missing or
        // stream-only return fence would make Drop finish prematurely.
        let premature = wait_dropped.recv_timeout(std::time::Duration::from_millis(100));
        let _ = release.send(());
        worker.join().unwrap();
        assert!(matches!(
            premature,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
    }

    /// Ensure an idle allocation cannot be reused through a different context handle.
    #[test]
    #[ignore = "requires an available CUDA GPU"]
    fn pooled_batches_reject_a_different_context_handle() {
        let first = cuda_core::CudaContext::new(0).unwrap();
        let second = cuda_core::CudaContext::new(0).unwrap();
        let pool = buffer_pool();
        drop(Batch::<4>::allocate_from(&pool, &first.default_stream(), 1).unwrap());
        assert!(Batch::<4>::allocate_from(&pool, &second.default_stream(), 1).is_err());
        // Rejection must not lose or replace the first context's allocation.
        let retained = Batch::<4>::allocate_from(&pool, &first.default_stream(), 1).unwrap();
        assert!(Arc::ptr_eq(retained.buffer().context(), &first));
    }

    /// Verify invalid layer counts, capacity overflow, and poison state never advance the prefix.
    #[test]
    fn cache_step_rejects_inconsistent_layers_capacity_and_poison() {
        let mut state = CacheState {
            len: 3,
            capacity: 5,
            layers: 2,
            poisoned: false,
        };
        assert_eq!(state.check(2, 2).unwrap(), 5);
        assert_eq!(state.check(2, 0).unwrap(), 3);
        assert!(state.check(1, 1).is_err());
        assert!(state.check(2, 3).is_err());
        assert!(state.check(2, usize::MAX).is_err());
        assert_eq!(state.len, 3);
        state.poisoned = true;
        assert!(state.check(2, 0).is_err());
        assert!(state.check(2, 1).is_err());
    }

    /// Reject transfer metadata that could expose an incomplete or differently sized prefix.
    #[test]
    fn execution_transfer_requires_every_completed_layer_to_match() {
        let mut state = CacheState {
            len: 3,
            capacity: 5,
            layers: 2,
            poisoned: false,
        };
        assert!(state.check_execution(2, [(3, 5), (3, 5)]).is_ok());
        for layers in [
            vec![],
            vec![(3, 5)],
            vec![(3, 5); 3],
            vec![(3, 5), (2, 5)],
            vec![(3, 5), (3, 4)],
        ] {
            assert!(state.check_execution(2, layers).is_err());
        }
        assert!(state.check_execution(1, [(3, 5), (3, 5)]).is_err());
        assert!(state.check_execution(0, []).is_err());
        state.poisoned = true;
        assert!(state.check_execution(2, [(3, 5), (3, 5)]).is_err());
        state.poisoned = false;
        state.len = 6;
        assert!(state.check_execution(2, [(6, 5), (6, 5)]).is_err());
    }

    /// Publication requires an active transaction and a strictly advancing bounded prefix.
    #[test]
    fn execution_publication_rejects_invalid_end_without_changing_state() {
        let mut state = CacheState {
            len: 3,
            capacity: 5,
            layers: 2,
            poisoned: false,
        };
        assert!(state.check_publication(4).is_err());
        state.poisoned = true;
        assert!(state.check_publication(4).is_ok());
        assert!(state.check_publication(5).is_ok());
        for end in [0, 2, 3, 6, usize::MAX] {
            assert!(state.check_publication(end).is_err());
            assert_eq!(state.len, 3);
            assert!(state.poisoned);
        }
        state.len = usize::MAX;
        state.capacity = usize::MAX;
        assert!(state.check_publication(usize::MAX).is_err());
        assert!(state.check(2, 1).is_err());
    }
}

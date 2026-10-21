/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Synchronous BF16 rotary composition with privately owned key/value caches.
use crate::{kernel::kernels, rotary::HostTables};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    Bf16,
    attention::{AttentionShape, CacheShape},
};
use std::sync::Arc;

/// Host result carrying CUDA errors and rejected shape, position, or context contracts.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// BF16 quiet-NaN bits used as unwritten-storage and trailing-guard sentinels.
pub const GUARD: u16 = 0x7fc1;
/// Number of BF16 sentinel elements after each logical allocation.
pub const GUARD_LEN: usize = 8;

/// Immutable, completed rotary coefficients that may be reused across layers.
///
/// Construction preserves the checked host table's head width and token count.
/// The allocations stay private: callers cannot mutate them or enqueue writes
/// between appends, and each synchronous append retains this borrow to completion.
pub struct DeviceTables<const D: usize> {
    /// Number of coefficient rows; each row has `D / 2` elements.
    tokens: usize,
    /// Completed, private `[tokens, D / 2]` BF16 cosine allocation.
    cos: DeviceBuffer<Bf16>,
    /// Completed, private `[tokens, D / 2]` BF16 sine allocation.
    sin: DeviceBuffer<Bf16>,
}

impl<const D: usize> DeviceTables<D> {
    /// Upload both checked `[tokens, D / 2]` tables and complete the host copies.
    ///
    /// The resulting immutable allocations can be reused on another stream in the
    /// same context; their head width and token count remain tied to the host tables.
    pub fn new(stream: &CudaStream, tables: &HostTables<D>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        Ok(Self {
            tokens: tables.tokens(),
            cos: DeviceBuffer::from_host(stream, tables.cos())?,
            sin: DeviceBuffer::from_host(stream, tables.sin())?,
        })
    }

    /// Return the token-row count carried by these prepared coefficient tables.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Check the BF16 allocation extent including its trailing sentinel elements.
fn guarded_len(logical_len: usize) -> Result<usize> {
    let elements = logical_len
        .checked_add(GUARD_LEN)
        .ok_or("guard extent overflow")?;
    let bytes = elements
        .checked_mul(size_of::<Bf16>())
        .ok_or("guard byte extent overflow")?;
    if bytes > isize::MAX as usize {
        return Err("guarded BF16 allocation exceeds the slice address range".into());
    }
    Ok(elements)
}

/// One thread handles one rotation pair or one appended BF16 element.
fn launch_grid(work_items: usize) -> Result<Option<LaunchConfig>> {
    if work_items == 0 {
        return Ok(None);
    }
    let blocks = work_items.div_ceil(128);
    if blocks > i32::MAX as usize {
        return Err("work exceeds the CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// Allocate sentinel-filled storage only after checking the full guarded extent.
pub(crate) fn allocate(stream: &CudaStream, logical_len: usize) -> Result<DeviceBuffer<Bf16>> {
    let elements = guarded_len(logical_len)?;
    stream.context().bind_to_thread()?;
    Ok(DeviceBuffer::from_host(
        stream,
        &vec![Bf16::from_bits(GUARD); elements],
    )?)
}

/// Borrow dense BF16 inputs with distinct query/KV head dimensions.
///
/// Exact lengths reject accidental reinterpretation of trailing storage as
/// tokens. Context and pending producer work are checked by the runner.
pub struct AttentionInputs<'a, const Q: usize, const K: usize, const D: usize> {
    /// Unrotated `[tokens, Q, D]` queries; pending writes are ordered by the runner.
    query: &'a DeviceBuffer<Bf16>,
    /// Unrotated `[tokens, K, D]` keys to rotate before appending.
    key: &'a DeviceBuffer<Bf16>,
    /// `[tokens, K, D]` values copied unchanged into the cache.
    value: &'a DeviceBuffer<Bf16>,
    /// Checked grouping and element-count arithmetic for the const head dimensions.
    shape: AttentionShape<Q, K, D>,
    /// Number of new token rows across all three inputs.
    tokens: usize,
}

impl<'a, const Q: usize, const K: usize, const D: usize> AttentionInputs<'a, Q, K, D> {
    /// Borrow dense `[tokens, Q, D]` queries and `[tokens, K, D]` keys/values.
    ///
    /// Reject invalid head grouping, extent overflow, and any nonexact buffer length.
    /// Context identity and input readiness are established later by the runner.
    pub fn new(
        query: &'a DeviceBuffer<Bf16>,
        key: &'a DeviceBuffer<Bf16>,
        value: &'a DeviceBuffer<Bf16>,
        tokens: usize,
    ) -> Result<Self> {
        let shape = AttentionShape::new()?;
        if query.len() != shape.query_len(tokens)?
            || key.len() != shape.kv_len(tokens)?
            || value.len() != shape.kv_len(tokens)?
        {
            return Err(
                "attention input lengths must exactly match the token/head dimensions".into(),
            );
        }
        Ok(Self {
            query,
            key,
            value,
            shape,
            tokens,
        })
    }
}

/// Own cache storage and publish only the prefix completed by synchronous appends.
///
/// Buffer ownership and head dimensions cannot be replaced by callers. A reset
/// discards the logical prefix; it does not expose or clear old allocation data.
pub struct KvCache<const K: usize, const D: usize> {
    /// Rotated keys in `[capacity, K, D]` order, followed by guard elements.
    key: DeviceBuffer<Bf16>,
    /// Unrotated values with the same dense layout and trailing guards.
    value: DeviceBuffer<Bf16>,
    /// Checked fixed token capacity and per-token width in BF16 elements.
    shape: CacheShape<K, D>,
    /// Completed token prefix; only successful synchronized appends advance it.
    valid_tokens: usize,
}

/// Guarded K/V storage exclusively owned by a private asynchronous execution region.
///
/// Typed `CacheShape` preserves head count/width and fixed token capacity. The
/// region retains this owner through completion; no conversion back to the public
/// synchronous cache is provided while pending device work may exist.
#[allow(
    dead_code,
    reason = "Used by model consumers; this standalone example stays synchronous"
)]
pub(crate) struct OwnedKvCache<const K: usize, const D: usize> {
    /// Initialized rotated keys in `[capacity, K, D]` order plus guard suffix.
    pub(crate) key: DeviceBuffer<Bf16>,
    /// Initialized unrotated values with the same shape and guard suffix.
    pub(crate) value: DeviceBuffer<Bf16>,
    /// Checked dimensions retained independently of each buffer's element count.
    shape: CacheShape<K, D>,
    /// Last completed prefix, never advanced merely by enqueueing an append.
    valid_tokens: usize,
}

#[allow(
    dead_code,
    reason = "Used by model consumers; this standalone example stays synchronous"
)]
impl<const K: usize, const D: usize> OwnedKvCache<K, D> {
    /// Return the last published completed prefix in token rows.
    pub(crate) fn len(&self) -> usize {
        self.valid_tokens
    }

    /// Return the fixed token capacity, excluding trailing guards.
    pub(crate) fn capacity(&self) -> usize {
        self.shape.capacity()
    }

    /// Advance metadata only after the enclosing region validates every layer.
    ///
    /// # Safety
    /// The caller must prove `self.len() < end <= self.capacity()`, initialized
    /// K/V storage through `end`, and successful completion of all device users.
    /// No device operation or allocation occurs here, so a stack can validate all
    /// layers first and then publish their cursors without a fallible midpoint.
    pub(crate) unsafe fn publish_completed(&mut self, end: usize) {
        self.valid_tokens = end;
    }
}

impl<const K: usize, const D: usize> KvCache<K, D> {
    /// Allocate guarded `[capacity, K, D]` key/value buffers with an empty valid prefix.
    pub fn new(stream: &CudaStream, capacity: usize) -> Result<Self> {
        let shape = CacheShape::new(capacity)?;
        Ok(Self {
            key: allocate(stream, shape.allocation_len())?,
            value: allocate(stream, shape.allocation_len())?,
            shape,
            valid_tokens: 0,
        })
    }

    /// Return the number of tokens published by completed appends.
    pub fn len(&self) -> usize {
        self.valid_tokens
    }

    /// Return the fixed token capacity, excluding allocation guards.
    pub fn capacity(&self) -> usize {
        self.shape.capacity()
    }

    /// Validate completed private storage before transfer into an execution region.
    ///
    /// This performs no synchronization or mutation; public cache operations have
    /// already completed every append. The enclosing stack fences context readiness
    /// once, after every layer has passed these shape/context checks.
    #[allow(
        dead_code,
        reason = "Used by model consumers; this standalone example stays synchronous"
    )]
    pub(crate) fn validate_execution(&self, stream: &CudaStream) -> Result<()> {
        let allocation_len = guarded_len(self.shape.allocation_len())?;
        if self.valid_tokens > self.shape.capacity()
            || self.key.len() != allocation_len
            || self.value.len() != allocation_len
        {
            return Err("execution cache requires complete guarded K/V storage".into());
        }
        if !Arc::ptr_eq(self.key.context(), stream.context())
            || !Arc::ptr_eq(self.value.context(), stream.context())
        {
            return Err("execution cache must share the supplied stream context".into());
        }
        Ok(())
    }

    /// Consume completed storage without copying, clearing, or changing its prefix.
    ///
    /// Public cache operations finish before returning and expose no mutable
    /// backing storage, so consumption cannot abandon a pending append. The caller
    /// establishes region readiness once for the complete stack before submission.
    #[allow(
        dead_code,
        reason = "Used by model consumers; this standalone example stays synchronous"
    )]
    pub(crate) fn into_execution(self, stream: &CudaStream) -> Result<OwnedKvCache<K, D>> {
        self.validate_execution(stream)?;
        Ok(OwnedKvCache {
            key: self.key,
            value: self.value,
            shape: self.shape,
            valid_tokens: self.valid_tokens,
        })
    }

    /// Return all capacity and sentinel elements for this example's validation.
    /// Only the first `len()` tokens represent the current cache prefix.
    pub fn snapshot(&self, stream: &CudaStream) -> Result<(Vec<Bf16>, Vec<Bf16>)> {
        if !Arc::ptr_eq(self.key.context(), stream.context())
            || !Arc::ptr_eq(self.value.context(), stream.context())
        {
            return Err("snapshot stream must share the cache context".into());
        }
        stream.context().synchronize()?;
        let mut key = vec![Bf16::from_bits(0); self.key.len()];
        let mut value = vec![Bf16::from_bits(0); self.value.len()];
        self.key.copy_to_host(stream, &mut key)?;
        self.value.copy_to_host(stream, &mut value)?;
        Ok((key, value))
    }

    /// Discard the completed prefix; the next append starts at token zero.
    pub fn reset(&mut self) {
        self.valid_tokens = 0;
    }
}

/// Owned rotated query storage, retaining its query-head and head-width types.
pub struct RotatedQuery<const Q: usize, const D: usize> {
    /// Completed dense `[tokens, Q, D]` rotated queries, plus trailing guards.
    buffer: DeviceBuffer<Bf16>,
    /// Number of valid query token rows, excluding guards.
    tokens: usize,
}

impl<const Q: usize, const D: usize> RotatedQuery<Q, D> {
    /// Read-only buffer including the trailing validation sentinel elements.
    pub fn buffer(&self) -> &DeviceBuffer<Bf16> {
        &self.buffer
    }

    /// Return the number of rotated query token rows.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Bind a loaded kernel module to one stream and context.
pub struct Runner {
    /// Rotation and append kernels loaded in the execution context.
    module: kernels::LoadedModule,
    /// Split-attention executor sharing this runner's stream and context.
    attention: crate::attention::AttentionRunner,
    /// Stream whose completion makes returned queries and cache prefixes readable.
    stream: Arc<CudaStream>,
}

impl Runner {
    /// Load rotary/cache and attention kernels in the supplied stream's context.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            module: kernels::load(stream.context())?,
            attention: crate::attention::AttentionRunner::new(stream.clone())?,
            stream,
        })
    }

    /// Attend over the completed cache prefix with fixed 128-wide BF16 heads.
    /// `query_start` is a physical causal cache offset, independent of RoPE
    /// positions. `None` selects all available keys for every query.
    /// Queries and cache must share this runner's context. Pending work is
    /// ordered before reading, and input borrows remain live through completion,
    /// including enqueue failures. Empty queries launch no attention kernels.
    pub fn attend<const Q: usize, const K: usize>(
        &self,
        query: &RotatedQuery<Q, 128>,
        cache: &KvCache<K, 128>,
        query_start: Option<usize>,
    ) -> Result<crate::attention::AttentionOutput<Q>> {
        self.attention.run::<Q, K>(
            &query.buffer,
            query.tokens,
            &cache.key,
            &cache.value,
            cache.valid_tokens,
            query_start,
        )
    }

    /// Rotate Q/K and append completed K/V, retaining every borrow through completion.
    ///
    /// Coefficients are explicit inputs. Their token count and head dimension
    /// are checked, while their semantic positions remain the caller's choice.
    /// The current synchronous runner orders prior work in its whole context.
    pub fn append<const Q: usize, const K: usize, const D: usize>(
        &self,
        cache: &mut KvCache<K, D>,
        inputs: AttentionInputs<'_, Q, K, D>,
        tables: &HostTables<D>,
    ) -> Result<RotatedQuery<Q, D>> {
        let tables = DeviceTables::new(&self.stream, tables)?;
        self.append_prepared(cache, inputs, &tables)
    }

    /// Reuse completed coefficients without changing the synchronous append
    /// contract. Input readiness, cache publication and error completion remain
    /// the same as [`Self::append`]; the tables may originate on another stream
    /// in this context because their constructor completed both uploads.
    pub fn append_prepared<const Q: usize, const K: usize, const D: usize>(
        &self,
        cache: &mut KvCache<K, D>,
        inputs: AttentionInputs<'_, Q, K, D>,
        tables: &DeviceTables<D>,
    ) -> Result<RotatedQuery<Q, D>> {
        if !D.is_multiple_of(2) {
            return Err("split-half rotary head width must be even".into());
        }
        if tables.tokens() != inputs.tokens {
            return Err("rotary table token count does not match attention inputs".into());
        }
        for buffer in [
            inputs.query,
            inputs.key,
            inputs.value,
            &cache.key,
            &cache.value,
            &tables.cos,
            &tables.sin,
        ] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err(
                    "attention inputs, cache and tables must share the runner context".into(),
                );
            }
        }
        let range = cache
            .shape
            .append_range(cache.valid_tokens, inputs.tokens)?;
        let query_len = inputs.shape.query_len(inputs.tokens)?;
        let kv_len = inputs.shape.kv_len(inputs.tokens)?;
        let query_grid = launch_grid(query_len / 2)?;
        let key_grid = launch_grid(kv_len / 2)?;
        let append_grid = launch_grid(kv_len)?;
        let start_element = range
            .start
            .checked_mul(cache.shape.width())
            .ok_or("cache offset overflow")?;
        // Safe asynchronous writes may be pending on another stream. Establish
        // readiness before any kernel reads the borrowed input allocations.
        self.stream.context().synchronize()?;
        let mut query = RotatedQuery {
            buffer: allocate(&self.stream, query_len)?,
            tokens: inputs.tokens,
        };
        if inputs.tokens == 0 {
            return Ok(query);
        }
        let query_grid = query_grid.ok_or("missing query rotation grid")?;
        let key_grid = key_grid.ok_or("missing key rotation grid")?;
        let append_grid = append_grid.ok_or("missing cache append grid")?;
        let mut rotated_key = allocate(&self.stream, kv_len)?;
        // SAFETY: lengths, contexts, grids and capacity were checked above.
        // Inputs remain borrowed; every output is a distinct owned allocation.
        // Rotation threads own disjoint pairs, and append threads own disjoint
        // elements beyond the completed cache prefix. One stream orders all
        // launches; unconditional synchronization retains their buffers on error.
        let enqueued = (|| -> Result<()> {
            unsafe {
                self.module.rotate_bf16(
                    &self.stream,
                    query_grid,
                    inputs.query,
                    &tables.cos,
                    &tables.sin,
                    &mut query.buffer,
                    inputs.tokens,
                    Q,
                    D,
                )?;
                self.module.rotate_bf16(
                    &self.stream,
                    key_grid,
                    inputs.key,
                    &tables.cos,
                    &tables.sin,
                    &mut rotated_key,
                    inputs.tokens,
                    K,
                    D,
                )?;
                self.module.append_bf16(
                    &self.stream,
                    append_grid,
                    &rotated_key,
                    inputs.value,
                    &mut cache.key,
                    &mut cache.value,
                    start_element,
                    kv_len,
                )?;
            }
            Ok(())
        })();
        let completed = self.stream.synchronize();
        enqueued?;
        completed?;
        cache.valid_tokens = range.end;
        Ok(query)
    }
}

/// GPU validation lives in the executable rather than the CPU test harness,
/// which does not link the compiled kernel artifact.
pub(crate) fn validate_prepared_tables() -> Result<()> {
    prepared_tables_reuse_completed_coefficients_across_streams_exactly()?;
    prepared_table_rejection_preserves_the_completed_cache_prefix()?;
    println!("PASS prepared rotary table reuse, exact rounding, guards and rejection");
    Ok(())
}

/// Check cross-stream table reuse, exact rotary rounding, and cache/query guards.
fn prepared_tables_reuse_completed_coefficients_across_streams_exactly() -> Result<()> {
    let context = cuda_core::CudaContext::new(0)?;
    let upload = context.new_stream()?;
    let stream = context.new_stream()?;
    let runner = Runner::new(stream.clone())?;
    let host = HostTables::<2>::from_coefficients(
        1,
        vec![Bf16::from_bits(0x3f35)],
        vec![Bf16::from_bits(0x3f35)],
    )?;
    let tables = DeviceTables::new(&upload, &host)?;
    drop(host);
    let input = DeviceBuffer::from_host(&stream, &[0x3f81, 0x3f02].map(Bf16::from_bits))?;
    let mut cache = KvCache::<1, 2>::new(&stream, 2)?;
    for prefix in 1..=2 {
        let query = runner.append_prepared(
            &mut cache,
            AttentionInputs::<1, 1, 2>::new(&input, &input, &input, 1)?,
            &tables,
        )?;
        let actual = query.buffer().to_host_vec(&stream)?;
        // Discriminates separate BF16 product rounding from cast-once rotation.
        assert_eq!([actual[0].to_bits(), actual[1].to_bits()], [0x3eb4, 0x3f89]);
        assert!(actual[2..].iter().all(|value| value.to_bits() == GUARD));
        assert_eq!(cache.len(), prefix);
    }
    let (key, value) = cache.snapshot(&stream)?;
    let bits = |values: &[Bf16]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&key[..4]), [0x3eb4, 0x3f89, 0x3eb4, 0x3f89]);
    assert_eq!(bits(&value[..4]), [0x3f81, 0x3f02, 0x3f81, 0x3f02]);
    assert!(
        key[4..]
            .iter()
            .chain(&value[4..])
            .all(|value| value.to_bits() == GUARD)
    );
    Ok(())
}

/// Check that wrong table token counts or contexts leave the published cache unchanged.
fn prepared_table_rejection_preserves_the_completed_cache_prefix() -> Result<()> {
    let context = cuda_core::CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::new(stream.clone())?;
    let input = DeviceBuffer::from_host(&stream, &[Bf16::from_f32(1.0); 2])?;
    let mut cache = KvCache::<1, 2>::new(&stream, 2)?;
    let rotary = crate::rotary::RotarySpec::<2>::standard(1e6, 8)?;
    let host = rotary.tables(&[[0; 3]])?;
    runner.append(
        &mut cache,
        AttentionInputs::<1, 1, 2>::new(&input, &input, &input, 1)?,
        &host,
    )?;
    let before = cache.snapshot(&stream)?;
    let wrong_tokens = DeviceTables::new(&stream, &rotary.tables(&[[0; 3]; 2])?)?;
    let other = cuda_core::CudaContext::new(0)?;
    let wrong_context = DeviceTables::new(&other.default_stream(), &host)?;
    for tables in [&wrong_tokens, &wrong_context] {
        assert!(
            runner
                .append_prepared(
                    &mut cache,
                    AttentionInputs::<1, 1, 2>::new(&input, &input, &input, 1)?,
                    tables,
                )
                .is_err()
        );
        assert_eq!(cache.len(), 1);
        let after = cache.snapshot(&stream)?;
        assert!(
            before
                .0
                .iter()
                .chain(&before.1)
                .zip(after.0.iter().chain(&after.1))
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
    }
    Ok(())
}

/// CPU checks for guarded allocation extents and scalar launch geometry.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check eight-element guards against both integer and slice-byte overflow.
    #[test]
    fn guards_cover_empty_and_nonempty_buffers_without_exceeding_slice_extent() {
        assert_eq!(guarded_len(0).unwrap(), 8);
        assert_eq!(guarded_len(17).unwrap(), 25);
        let largest = isize::MAX as usize / size_of::<cute_rs::Bf16>();
        assert_eq!(guarded_len(largest - GUARD_LEN).unwrap(), largest);
        assert!(guarded_len(largest - GUARD_LEN + 1).is_err());
        assert!(guarded_len(usize::MAX).is_err());
    }

    /// Check 128-thread block rounding and the empty-work no-launch policy.
    #[test]
    fn launch_grid_covers_partial_blocks_and_skips_empty_work() {
        assert!(launch_grid(0).unwrap().is_none());
        for (elements, blocks) in [(1, 1), (127, 1), (128, 1), (129, 2), (257, 3)] {
            let grid = launch_grid(elements).unwrap().unwrap();
            assert_eq!(grid.grid_dim, (blocks, 1, 1));
            assert_eq!(grid.block_dim, (128, 1, 1));
            assert_eq!(grid.shared_mem_bytes, 0);
        }
    }

    /// Require a representable CUDA grid before narrowing the block count to `u32`.
    #[test]
    fn launch_grid_rejects_cuda_grid_overflow_before_casting() {
        let largest = i32::MAX as usize * 128;
        assert_eq!(
            launch_grid(largest).unwrap().unwrap().grid_dim.0,
            i32::MAX as u32
        );
        assert!(launch_grid(largest + 1).is_err());
        assert!(launch_grid(usize::MAX).is_err());
    }
}

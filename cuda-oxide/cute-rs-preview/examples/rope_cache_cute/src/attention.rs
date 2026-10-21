/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Checked synchronous attention over borrowed, completed BF16 cache prefixes.
use crate::attention_kernel::{PARTIAL_SHARED_BYTES, STATE_SIZE, kernels};
use crate::operator::{GUARD_LEN, Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cute_rs::Bf16;
use cute_rs::attention::{AttentionShape, AttentionWindow};
use std::sync::Arc;

/// Host-checked launch sizes for the fixed 128-wide split-KV schedule.
#[derive(Debug)]
struct Geometry {
    /// One CTA for each `(query token, query head, 64-key split)`.
    partial_blocks: u32,
    /// One CTA for each `(query token, query head)` output row.
    merge_blocks: u32,
    /// Number of 64-key partitions covering the visible cache allocation prefix.
    splits: u32,
    /// FP32 scratch element count, excluding guards: `partial_blocks * STATE_SIZE`.
    partial_elements: usize,
    /// BF16 result element count, excluding guards: `query_tokens * Q * 128`.
    output_elements: usize,
}

/// Validate head grouping, ABI widths, grid limits, and guarded scratch bytes.
///
/// Returns `None` for empty queries after checking shape validity. Counts describe
/// 64-key splits over the supplied window; all buffer extents are element counts.
fn geometry<const Q: usize, const K: usize>(window: AttentionWindow) -> Result<Option<Geometry>> {
    let shape = AttentionShape::<Q, K, 128>::new()?;
    let output_elements = shape.query_len(window.query_tokens())?;
    shape.kv_len(window.key_tokens())?;
    for value in [Q, K, window.query_tokens(), window.key_tokens()] {
        u32::try_from(value).map_err(|_| "attention dimension exceeds kernel ABI")?;
    }
    if window.query_tokens() == 0 {
        return Ok(None);
    }
    let splits = window.key_tokens().div_ceil(64);
    let merge_blocks = window
        .query_tokens()
        .checked_mul(Q)
        .ok_or("query grid overflow")?;
    let partial_blocks = merge_blocks
        .checked_mul(splits)
        .ok_or("split grid overflow")?;
    if partial_blocks > i32::MAX as usize || merge_blocks > i32::MAX as usize {
        return Err("attention grid exceeds CUDA grid-x range".into());
    }
    let partial_elements = partial_blocks
        .checked_mul(STATE_SIZE)
        .ok_or("scratch overflow")?;
    let guarded = partial_elements
        .checked_add(GUARD_LEN)
        .ok_or("scratch guard overflow")?;
    let bytes = guarded
        .checked_mul(size_of::<f32>())
        .ok_or("scratch byte overflow")?;
    if bytes > isize::MAX as usize {
        return Err("attention scratch exceeds addressable byte extent".into());
    }
    Ok(Some(Geometry {
        partial_blocks: partial_blocks as u32,
        merge_blocks: merge_blocks as u32,
        splits: splits as u32,
        partial_elements,
        output_elements,
    }))
}

/// Owned BF16 output; query-head count is retained and head width is always 128.
pub struct AttentionOutput<const Q: usize> {
    /// Dense `[tokens, Q, 128]` completed BF16 results, followed by validation guards.
    buffer: DeviceBuffer<Bf16>,
    /// Number of query token rows, excluding allocation guards.
    tokens: usize,
}

impl<const Q: usize> AttentionOutput<Q> {
    /// Read-only allocation including trailing validation guards.
    pub fn buffer(&self) -> &DeviceBuffer<Bf16> {
        &self.buffer
    }

    /// Return the number of completed query token rows.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Bind the split and merge kernels to one stream and context.
pub(crate) struct AttentionRunner {
    /// Partial and merge kernels compiled with their fixed CTA launch contracts.
    module: kernels::LoadedModule,
    /// Stream ordering partial and merge work in the owning context.
    stream: Arc<CudaStream>,
}

impl AttentionRunner {
    /// Load the split-attention bundle in the supplied stream's CUDA context.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        // SAFETY: this binary's named embedded bundle is compiled from this
        // non-generic cuda_module, including its declared launch contracts.
        Ok(Self {
            module: unsafe { kernels::load(stream.context())? },
            stream,
        })
    }

    /// Run FP32 split attention and return dense `[query_tokens, Q, 128]` BF16 output.
    ///
    /// The cache owner supplies initialized K/V prefixes in `[key_tokens, K, 128]`
    /// order. `query_start` is a physical causal token offset; `None` exposes all
    /// keys. Checked buffers share this runner's context. A context wait orders
    /// prior writes, and an unconditional stream wait retains all borrows through
    /// completion even after enqueue errors. Only the final result rounds to BF16;
    /// empty queries allocate guarded output without launching attention kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn run<const Q: usize, const K: usize>(
        &self,
        query: &DeviceBuffer<Bf16>,
        query_tokens: usize,
        key: &DeviceBuffer<Bf16>,
        value: &DeviceBuffer<Bf16>,
        key_tokens: usize,
        query_start: Option<usize>,
    ) -> Result<AttentionOutput<Q>> {
        let window = match query_start {
            Some(start) => AttentionWindow::causal(query_tokens, key_tokens, start)?,
            None => AttentionWindow::noncausal(query_tokens, key_tokens)?,
        };
        let shape = AttentionShape::<Q, K, 128>::new()?;
        let query_len = shape.query_len(query_tokens)?;
        let kv_len = shape.kv_len(key_tokens)?;
        if query.len() < query_len || key.len() < kv_len || value.len() < kv_len {
            return Err("attention allocations do not contain the requested prefix".into());
        }
        for buffer in [query, key, value] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("attention inputs must share the runner context".into());
            }
        }
        let geometry = geometry::<Q, K>(window)?;
        self.stream.context().synchronize()?;
        let mut output = AttentionOutput {
            buffer: crate::operator::allocate(&self.stream, query_len)?,
            tokens: query_tokens,
        };
        let Some(g) = geometry else {
            return Ok(output);
        };
        debug_assert_eq!(g.output_elements, query_len);
        // NaN sentinels reveal incomplete scratch writes; keep a trailing guard.
        let mut partials = DeviceBuffer::from_host(
            &self.stream,
            &vec![f32::from_bits(0x7fc1_0001); g.partial_elements + GUARD_LEN],
        )?;
        let partial_grid = self.module.prepare_partial(LaunchConfig1D::new(
            g.partial_blocks,
            128,
            PARTIAL_SHARED_BYTES,
        ))?;
        let merge_grid = self
            .module
            .prepare_merge(LaunchConfig1D::new(g.merge_blocks, 32, 0))?;
        // Checked lengths, dimensions, contexts, grids and scratch
        // extents. The cache owner guarantees the prefix is initialized. Each
        // CTA owns one split or query/head output. Every allocation/borrow stays
        // live until the unconditional stream wait, including enqueue errors.
        let enqueued = (|| -> Result<()> {
            self.module.partial(
                &self.stream,
                &partial_grid,
                query,
                key,
                value,
                &mut partials,
                query_tokens as u32,
                key_tokens as u32,
                Q as u32,
                K as u32,
                query_start.unwrap_or(0) as u32,
                query_start.is_some(),
            )?;
            self.module.merge(
                &self.stream,
                &merge_grid,
                &partials,
                &mut output.buffer,
                query_tokens as u32,
                Q as u32,
                g.splits,
            )?;
            Ok(())
        })();
        let completed = self.stream.synchronize();
        enqueued?;
        completed?;
        let mut scratch = vec![0.0; partials.len()];
        partials.copy_to_host(&self.stream, &mut scratch)?;
        if !scratch[g.partial_elements..]
            .iter()
            .all(|x| x.to_bits() == 0x7fc1_0001)
        {
            return Err("attention scratch guard overwritten".into());
        }
        Ok(output)
    }
}

/// CPU checks of split scratch sizing, empty queries, and launch integer limits.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check each query/head/split gets one independent 130-element FP32 state.
    #[test]
    fn splits_and_query_heads_determine_distinct_scratch_rows() {
        let window = AttentionWindow::causal(3, 129, 126).unwrap();
        let g = geometry::<16, 8>(window).unwrap().unwrap();
        assert_eq!((g.partial_blocks, g.merge_blocks, g.splits), (144, 48, 3));
        assert_eq!(g.partial_elements, 144 * 130);
        assert_eq!(g.output_elements, 3 * 16 * 128);
    }

    /// Require valid query/KV grouping even when no attention work is launched.
    #[test]
    fn empty_queries_launch_nothing_but_invalid_grouping_is_rejected() {
        let window = AttentionWindow::causal(0, 0, 0).unwrap();
        assert!(geometry::<16, 8>(window).unwrap().is_none());
        assert!(geometry::<3, 2>(window).is_err());
    }

    /// Reject dimensions and split grids that cannot be represented by the kernel ABI.
    #[test]
    fn reject_abi_and_grid_overflow_before_allocation() {
        let too_long = AttentionWindow::noncausal(1, u32::MAX as usize + 1).unwrap();
        assert!(geometry::<16, 8>(too_long).is_err());
        let too_many_queries = AttentionWindow::noncausal(i32::MAX as usize, 1).unwrap();
        assert!(geometry::<16, 8>(too_many_queries).is_err());
        let split_grid = AttentionWindow::noncausal(2, u32::MAX as usize).unwrap();
        assert!(geometry::<16, 8>(split_grid).is_err());
    }
}

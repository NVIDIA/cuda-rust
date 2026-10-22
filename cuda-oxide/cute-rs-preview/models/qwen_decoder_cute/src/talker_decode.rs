/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Private one-row Talker execution with one completion boundary across all layers.
//!
//! A completed prefill cache is consumed into this plan. Scratch and cache storage
//! never escape; returned hidden/logit batches are independent completed owners.
//! The public synchronous runners and their cross-stream readiness rules remain.

use crate::checkpoint::QwenWeights;
use crate::decoder_execution::{Completion, LayerScratch, Modules, grid, one_warp};
use crate::generation::{TALKER_LAYERS, TalkerStep, TalkerWeights};
use crate::model::{Batch, DecoderCache, EPSILON, ExecutionCache};
use crate::operator::Result;
use crate::rotary::RotarySpec;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{BFloat16, decoder::DecoderShape};
use std::sync::Arc;

/// Require exact Talker layer geometry, initialized weight extents and one context.
fn validate_weights(stream: &CudaStream, weights: &TalkerWeights<'_>) -> Result<()> {
    if weights.layers.len() != TALKER_LAYERS {
        return Err("Talker execution requires 28 layers".into());
    }
    let context = |buffer: &DeviceBuffer<BFloat16>| -> Result<()> {
        if Arc::ptr_eq(buffer.context(), stream.context()) {
            Ok(())
        } else {
            Err("Talker weight context mismatch".into())
        }
    };
    let shape = DecoderShape::<2048, 6144, 16, 8, 128>::new(1, 2048)?;
    for layer in &weights.layers {
        shape.validate_weights(layer)?;
        let a = layer.attention();
        let f = layer.ffn();
        for buffer in [
            layer.input_norm().buffer(),
            layer.post_attention_norm().buffer(),
            a.query().buffer(),
            a.key().buffer(),
            a.value().buffer(),
            a.output().buffer(),
            a.query_norm().buffer(),
            a.key_norm().buffer(),
            f.gate().buffer(),
            f.up().buffer(),
            f.down().buffer(),
        ] {
            context(buffer)?;
        }
    }
    weights
        .codec_head
        .shape()
        .validate_len(weights.codec_head.buffer().len())?;
    context(weights.codec_head.buffer())?;
    context(weights.norm.buffer())?;
    Ok(())
}

/// Owns the Talker cache and reusable layer storage on an exclusive nondefault stream.
///
/// `&mut self` serializes calls. The completed prefix advances only after every
/// layer, final norm, head, completion and guard check succeeds. A failed submitted
/// step poisons the plan; unlike the synchronous helper's head failure, it leaves
/// the previously published prefix unchanged. Rebuild after any such failure.
pub(crate) struct TalkerDecodePlan {
    /// Private stream on which cleanup attempts completion before owners drop.
    stream: Arc<CudaStream>,
    /// Retains every immutable checkpoint address used by submitted kernels.
    weights: Arc<QwenWeights>,
    /// Loaded modules and prepared projection entries for the exact kernel ABI.
    modules: Modules,
    /// One row of intermediates shared in stream order across all 28 layers.
    scratch: LayerScratch<2048, 6144>,
    /// Consumed completed prefill, branded with hidden/FFN/KV widths.
    cache: ExecutionCache<2048, 6144, 8>,
    /// Exact three-axis interleaved rotary policy of the synchronous Talker.
    rotary: RotarySpec<128>,
    /// Persistent 64-value cosine row, uploaded before layer submission.
    cos: DeviceBuffer<BFloat16>,
    /// Matching sine row; its host source stays live through completion.
    sin: DeviceBuffer<BFloat16>,
    /// Sticky per-step device guard status, checked after all layers finish.
    status: DeviceBuffer<u32>,
}

impl Drop for TalkerDecodePlan {
    /// Attempt completion before cache, scratch, modules and checkpoint owners drop.
    /// CUDA failures are unreportable from `Drop`; successful calls check completion.
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl TalkerDecodePlan {
    /// Transfer a completed compatible cache and allocate private reusable storage.
    ///
    /// Weight/cache validation and context readiness precede transfer. Setup is
    /// per request and belongs inside synthesis timing; no graph is captured.
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        weights: Arc<QwenWeights>,
        cache: DecoderCache<2048, 6144, 8>,
    ) -> Result<Self> {
        let stream = stream.context().new_stream()?;
        validate_weights(&stream, &weights.talker()?)?;
        let cache = cache.into_execution(&stream, TALKER_LAYERS)?;
        let modules = Modules::new(&stream)?;
        let scratch = LayerScratch::new(&stream, &modules.attention, 1, cache.len().max(1))?;
        Ok(Self {
            scratch,
            cache,
            modules,
            weights,
            rotary: RotarySpec::interleaved(1_000_000.0, 32768, [24, 20, 20])?,
            cos: DeviceBuffer::from_host(&stream, &[BFloat16::from_bits(0); 64])?,
            sin: DeviceBuffer::from_host(&stream, &[BFloat16::from_bits(0); 64])?,
            status: DeviceBuffer::from_host(&stream, &[0u32])?,
            stream,
        })
    }

    /// Return the token prefix published by the last fully successful step.
    pub(crate) fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check input shape/context, position count and cache capacity without submission.
    ///
    /// Returns the checked next prefix; rejection does not poison a healthy plan.
    /// `run_into` validates rotary position values when constructing the tables.
    fn validate_input(&self, input: &Batch<2048>, positions: &[[u32; 3]]) -> Result<usize> {
        if self.cache.is_poisoned() {
            return Err("failed Talker plan must be rebuilt before reuse".into());
        }
        if input.rows() != 1 || input.buffer().len() != 2048 || positions.len() != 1 {
            return Err("Talker decode requires one dense row and one rotary position".into());
        }
        if !Arc::ptr_eq(input.buffer().context(), self.stream.context()) {
            return Err("Talker input context mismatch".into());
        }
        let end = self
            .cache
            .len()
            .checked_add(1)
            .ok_or("Talker prefix overflow")?;
        if end > self.cache.capacity() || u32::try_from(end).is_err() {
            return Err("Talker cache capacity or attention index exceeded".into());
        }
        Ok(end)
    }

    /// Decode one input row into independent completed hidden/logit owners.
    ///
    /// Context readiness covers external producers. Exclusive storage and a
    /// completion guard keep all borrowed inputs alive through errors/unwinding.
    pub(crate) fn run(
        &mut self,
        input: &Batch<2048>,
        positions: &[[u32; 3]],
    ) -> Result<TalkerStep> {
        self.validate_input(input, positions)?;
        let output = TalkerStep {
            last_hidden: Batch::allocate(&self.stream, 1)?,
            logits: Batch::allocate(&self.stream, 1)?,
        };
        self.run_into(input, positions, output)
    }

    /// Submit into separately owned outputs allocated before the execution boundary.
    ///
    /// Only internal callers supply outputs. All geometry/context checks precede
    /// enqueue; a completion guard precedes asynchronous uploads and kernel work.
    fn run_into(
        &mut self,
        input: &Batch<2048>,
        positions: &[[u32; 3]],
        mut output: TalkerStep,
    ) -> Result<TalkerStep> {
        let end = self.validate_input(input, positions)?;
        if output.last_hidden.rows() != 1
            || output.last_hidden.buffer().len() != 2048
            || output.logits.rows() != 1
            || output.logits.buffer().len() != 3072
            || !Arc::ptr_eq(output.last_hidden.buffer().context(), self.stream.context())
            || !Arc::ptr_eq(output.logits.buffer().context(), self.stream.context())
        {
            return Err(
                "Talker output must contain one dense hidden/logit row in the plan context".into(),
            );
        }
        let tables = self.rotary.tables(positions)?;
        // Reallocate only at a live attention split boundary, before submission.
        self.scratch
            .prepare_attention(&self.stream, &self.modules.attention, end)?;
        let weights_owner = self.weights.clone();
        let views = weights_owner.talker()?;
        self.stream.context().synchronize()?;
        let start = self.cache.len();
        let capacity = self.cache.capacity();
        self.cache.poison();
        let completion = Completion::new(&self.stream);
        let enqueued = (|| -> Result<()> {
            // SAFETY: private, distinct typed allocations; checked one-row inputs,
            // layer weights and live cache prefix. Host rotary tables and all
            // device owners outlive Completion, including errors and unwinding.
            unsafe {
                self.cos
                    .copy_from_host_async_unchecked(&self.stream, tables.cos())?;
                self.sin
                    .copy_from_host_async_unchecked(&self.stream, tables.sin())?;
                self.modules
                    .schedule
                    .reset_status(&self.stream, one_warp(), &mut self.status)?;
                self.modules.schedule.copy_range(
                    &self.stream,
                    grid(2048),
                    input.buffer(),
                    self.scratch.hidden.buffer_mut(),
                    0,
                    0,
                    2048,
                )?;
                for (weight, cache) in views.layers.iter().zip(self.cache.layers_mut()) {
                    self.modules.enqueue_layer(
                        &self.stream,
                        &mut self.scratch,
                        weight,
                        &mut cache.key,
                        &mut cache.value,
                        capacity,
                        &self.cos,
                        &self.sin,
                        start,
                        end,
                        &mut self.status,
                    )?;
                }
                crate::norm_specialized::enqueue(
                    &self.modules.norms,
                    &self.stream,
                    self.scratch.hidden.buffer(),
                    views.norm.buffer(),
                    output.last_hidden.buffer_mut(),
                    1,
                    2048,
                    EPSILON,
                )?;
                self.modules.project(
                    &self.stream,
                    output.last_hidden.buffer(),
                    1,
                    &views.codec_head,
                    &mut output.logits,
                )?;
            }
            Ok(())
        })();
        let completed = completion.finish();
        enqueued?;
        completed?;
        let status = self.status.to_host_vec(&self.stream)?[0];
        if status != 0 {
            return Err(
                format!("Talker device validation failed (status {status}: guard=2)").into(),
            );
        }
        // SAFETY: every layer's append and dependent work completed, all active
        // guards passed, and end is the checked common prefix for this step.
        unsafe {
            self.cache.publish(end)?;
        }
        Ok(output)
    }
}

/// Exact cache/output parity, readiness, ownership and poison regression fixtures.
#[path = "talker_decode_validation.rs"]
mod validation;
pub(crate) use validation::validate;

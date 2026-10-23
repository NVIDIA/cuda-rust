/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Input, accumulator, and output ownership pipelines for paired GEMM.

use super::__compiler;
use cuda_device::barrier::Barrier;

/// Two-CTA TMA-to-MMA stage ring, with one completion arrival per stage.
///
/// `TX_BYTES` is the total number of bytes copied by both CTAs per stage.
/// One lane in the leader CTA sets this expected byte count. MMA completion
/// releases the stage in both CTAs so the next TMA copy can reuse it.
pub struct PipelineTmaUmma<const STAGES: u32, const TX_BYTES: u32> {
    /// Local array of full barriers tracking cluster-wide input-copy bytes.
    full: *mut Barrier,
    /// Local array of empty barriers released by group-two MMA completion.
    empty: *mut Barrier,
}
impl<const STAGES: u32, const TX_BYTES: u32> PipelineTmaUmma<STAGES, TX_BYTES> {
    /// Require at least one stage and a nonzero cluster transaction byte count.
    const VALID: () = assert!(STAGES > 0 && TX_BYTES > 0);
    /// Attach separate full and empty barrier arrays.
    ///
    /// # Safety
    ///
    /// Each pointer addresses `STAGES` aligned shared barriers exclusively
    /// used by this pipeline until all consumers and producers finish.
    #[inline(always)]
    pub const unsafe fn from_raw(full: *mut Barrier, empty: *mut Barrier) -> Self {
        let () = Self::VALID;
        Self { full, empty }
    }
    /// Initialize both arrays from one elected lane per CTA.
    ///
    /// # Safety
    ///
    /// Publish barrier initialization and synchronize the cluster before use.
    #[inline(always)]
    pub unsafe fn init(&self) {
        unsafe { __compiler::sm100_pipeline_init::<STAGES, TX_BYTES>(self.full, self.empty) }
    }
    /// Wait for a reusable stage, with producer phase initially one.
    ///
    /// # Safety
    ///
    /// `stage < STAGES`; `phase` must be the producer's current ring phase.
    #[inline(always)]
    pub unsafe fn producer_acquire(&self, stage: u32, phase: u32) {
        unsafe { __compiler::sm100_pipeline_acquire::<STAGES, TX_BYTES>(self.empty, stage, phase) }
    }
    /// Attach the complete cluster's expected copy bytes to this full barrier.
    ///
    /// # Safety
    ///
    /// Exactly one elected lane of rank zero calls this after acquire.
    #[inline(always)]
    pub unsafe fn producer_expect_tx(&self, stage: u32) {
        unsafe { __compiler::sm100_pipeline_expect::<STAGES, TX_BYTES>(self.full, stage) }
    }
    /// Return the stage's local completion barrier for typed TMA copies.
    ///
    /// # Safety
    ///
    /// `stage < STAGES`. Copies using this barrier must belong to the acquired
    /// stage and phase covered by the leader's one cluster-wide expectation.
    #[inline(always)]
    pub unsafe fn producer_get_barrier(&self, stage: u32) -> *mut Barrier {
        unsafe { self.full.add(stage as usize) }
    }
    /// Wait for input copies and order subsequent tensor-core operations.
    ///
    /// # Safety
    ///
    /// Rank zero's MMA warp calls with its current consumer stage and phase.
    #[inline(always)]
    pub unsafe fn consumer_wait(&self, stage: u32, phase: u32) {
        unsafe { __compiler::sm100_pipeline_wait::<STAGES, TX_BYTES>(self.full, stage, phase) }
    }
    /// Signal both CTAs' empty barriers after pending MMA reads finish.
    ///
    /// # Safety
    ///
    /// One elected lane in the leader CTA calls after issuing all stage MMAs.
    #[inline(always)]
    pub unsafe fn consumer_release(&self, stage: u32) {
        unsafe { __compiler::sm100_pipeline_release::<STAGES, TX_BYTES>(self.empty, stage) }
    }
}

/// One accumulator slot shared by the MMA producer and epilogue warps.
pub struct PipelineUmmaAsync<const CONSUMER_WARPS: u32> {
    /// Local barrier released when all epilogue warps finish reading the accumulator.
    empty: *mut Barrier,
    /// Local barrier signaled when the leader’s MMA work completes.
    full: *mut Barrier,
}
impl<const CONSUMER_WARPS: u32> PipelineUmmaAsync<CONSUMER_WARPS> {
    /// Require between one and 32 epilogue consumer warps per CTA.
    const VALID: () = assert!(CONSUMER_WARPS > 0 && CONSUMER_WARPS <= 32);
    /// Attach the shared empty and full barriers.
    ///
    /// # Safety
    ///
    /// Both pointers address distinct aligned barriers exclusively owned by
    /// this pipeline. Each CTA has `CONSUMER_WARPS` epilogue consumer warps.
    #[inline(always)]
    pub const unsafe fn from_raw(empty: *mut Barrier, full: *mut Barrier) -> Self {
        let () = Self::VALID;
        Self { empty, full }
    }
    /// Initialize from one elected lane in each CTA, before cluster publication.
    ///
    /// # Safety
    ///
    /// No pipeline operation may run until initialization is visible cluster-wide.
    #[inline(always)]
    pub unsafe fn init(&self) {
        unsafe { __compiler::sm100_accumulator_init::<CONSUMER_WARPS>(self.empty, self.full) }
    }
    /// Wait for all epilogue warps to release the previous accumulator tile.
    ///
    /// # Safety
    ///
    /// Only the leader CTA's MMA warp acquires; its initial phase is one.
    #[inline(always)]
    pub unsafe fn producer_acquire(&self, phase: u32) {
        unsafe { __compiler::sm100_accumulator_acquire::<CONSUMER_WARPS>(self.empty, phase) }
    }
    /// Publish the tile to both CTAs after all pending MMA operations finish.
    ///
    /// # Safety
    ///
    /// One elected lane in the leader CTA calls after the final K stage.
    #[inline(always)]
    pub unsafe fn producer_commit(&self) {
        unsafe { __compiler::sm100_accumulator_commit::<CONSUMER_WARPS>(self.full) }
    }
    /// Wait for a full accumulator tile and order tensor-memory loads.
    ///
    /// # Safety
    ///
    /// Consumer warps pass their current phase, initially zero.
    #[inline(always)]
    pub unsafe fn consumer_wait(&self, phase: u32) {
        unsafe { __compiler::sm100_accumulator_wait::<CONSUMER_WARPS>(self.full, phase) }
    }
    /// Release this warp's portion of the accumulator back to the leader.
    ///
    /// # Safety
    ///
    /// Every lane in every epilogue warp calls after completing its tensor-
    /// memory reads. The operation fences all lanes and internally elects
    /// one lane to arrive at the leader CTA barrier.
    #[inline(always)]
    pub unsafe fn consumer_release(&self) {
        unsafe { __compiler::sm100_accumulator_release::<CONSUMER_WARPS>(self.empty) }
    }
}

/// A ring of shared epilogue tiles tracked by asynchronous TMA store groups.
///
/// This is the SM100 form of CuTeDSL's `pipeline.PipelineTmaStore`, with a
/// `u32` stage count and the sm100 compiler hooks. The SM120 form with the
/// same methods over a `usize` stage count is [`crate::PipelineTmaStore`].
#[derive(Clone, Copy, Default)]
pub struct PipelineTmaStore<const STAGES: u32>;
impl<const STAGES: u32> PipelineTmaStore<STAGES> {
    /// Reject an empty ring of shared output buffers.
    const VALID: () = assert!(STAGES > 0);
    /// Define a ring containing at least one shared output buffer.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self
    }
    /// Commit the issuer's pending TMA stores to one copy group.
    #[inline(always)]
    pub fn producer_commit(&self) {
        unsafe { __compiler::sm100_store_commit::<STAGES>() }
    }
    /// Wait until at most `STAGES-1` store groups still read shared memory.
    #[inline(always)]
    pub fn producer_acquire(&self) {
        unsafe { __compiler::sm100_store_acquire::<STAGES>() }
    }
    /// Wait until every store group has stopped reading shared memory.
    #[inline(always)]
    pub fn producer_tail(&self) {
        unsafe { __compiler::sm100_store_tail::<STAGES>() }
    }
}

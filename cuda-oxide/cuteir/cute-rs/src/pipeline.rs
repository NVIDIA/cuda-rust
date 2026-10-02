/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Move Tensor Memory Accelerator (TMA) tiles through shared buffers safely.
//!
//! TMA is hardware that copies a complete tile between global memory and
//! shared memory.
//!
//! A thread is one GPU worker. A warp is 32 threads that run together. A CTA,
//! also called a block, is a group of warps that share memory.
//!
//! A *stage* is one reusable shared-memory buffer. Producer threads load the
//! next tile. Consumer warps use a full tile for computation. Two barriers
//! per stage act like ready signals:
//!
//! ```text
//! producer                              consumer warps
//! wait until stage is EMPTY             wait until stage is FULL
//! mark expected TMA byte count           read and compute
//! start TMA load                         lane 0 of each warp marks EMPTY
//! move to next stage                     move to next stage
//! ```
//!
//! Stages form a ring: `0 -> 1 -> ... -> 0`. A one-bit phase distinguishes
//! the new use of a slot from its previous use. New buffers are empty, so the
//! producer starts at phase 1 and the consumer at phase 0. The phase flips
//! only when the stage index wraps to slot 0. This matches CUTLASS
//! `PipelineTmaAsync`.

use core::marker::PhantomData;
use core::mem::{align_of, size_of};

use cuda_device::barrier::Barrier;

/// Marks a cursor used by the thread that starts TMA loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Producer {}

/// Marks a cursor used by warps that read loaded tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consumer {}

/// A cursor pointing to one slot in the circular stage buffer.
///
/// `Role` prevents using a consumer cursor in a producer operation, or the
/// reverse. The cursor stores only a slot number and a phase bit. The kernel
/// keeps its K-loop counter separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineState<Role, const STAGES: usize> {
    slot: u32,
    phase: u32,
    role: PhantomData<Role>,
}

impl<const STAGES: usize> PipelineState<Producer, STAGES> {
    /// Start writing at slot 0. Phase 1 sees a new buffer as empty.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        __compiler::pipeline_state_new_producer::<STAGES>()
    }
}

impl<const STAGES: usize> Default for PipelineState<Producer, STAGES> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const STAGES: usize> PipelineState<Consumer, STAGES> {
    /// Start reading the first tile produced in slot 0.
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        __compiler::pipeline_state_new_consumer::<STAGES>()
    }
}

impl<const STAGES: usize> Default for PipelineState<Consumer, STAGES> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<Role, const STAGES: usize> PipelineState<Role, STAGES> {
    /// Current shared-buffer slot.
    #[must_use]
    #[inline(always)]
    pub fn slot(&self) -> usize {
        __compiler::pipeline_state_slot(self)
    }

    /// Phase bit expected from the current slot's barrier.
    #[must_use]
    #[inline(always)]
    pub const fn phase(&self) -> u32 {
        self.phase
    }

    /// Move to the next slot and flip the phase when the ring wraps.
    #[inline(always)]
    pub fn advance(&mut self) {
        __compiler::pipeline_state_advance(self);
    }
}

/// A CTA-local TMA load pipeline with one full and one empty barrier per stage.
///
/// - `STAGES` is the number of shared tile buffers.
/// - `CONSUMER_WARPS` is the number of warps that read each tile. Lane 0 of
///   each warp releases the stage through [`Self::consumer_release`].
/// - `TX_BYTES` is the total number of bytes loaded into one stage.
///
/// This value stores only two pointers. The caller owns the barriers and tile
/// buffers in shared memory.
#[derive(Clone, Copy)]
pub struct TmaLoadPipeline<const STAGES: usize, const CONSUMER_WARPS: u32, const TX_BYTES: u32> {
    full: *mut Barrier,
    empty: *mut Barrier,
}

/// Track TMA stores from shared memory to global memory.
///
/// Hardware tracks these stores in copy groups, so this type stores no data
/// and uses no memory barriers:
///
/// ```text
/// issue one or more shared -> global copies
///                     │
///                     ▼
/// producer_commit()   put them in one group
///                     │
///                     ▼
/// producer_acquire()  wait until the oldest shared buffer is reusable
///                     │
///                     ▼
/// producer_tail()     wait until no store still reads shared memory
/// ```
///
/// `STAGES` is the number of shared result buffers. After
/// [`Self::producer_acquire`], at most `STAGES - 1` committed groups still
/// read shared memory. This matches CUTLASS `PipelineTmaStore`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TmaStorePipeline<const STAGES: usize>;

impl<const STAGES: usize> TmaStorePipeline<STAGES> {
    /// Most store groups that may still read shared memory after acquire.
    pub const MAX_PENDING: u32 = {
        assert!(STAGES > 0);
        assert!(STAGES <= u32::MAX as usize);
        (STAGES - 1) as u32
    };

    /// Create a store pipeline. The value occupies no bytes.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        let _ = Self::MAX_PENDING;
        Self
    }

    /// Put this thread's TMA stores since the last commit into one group.
    #[inline(always)]
    pub fn producer_commit(&self) {
        __compiler::tma_store_producer_commit::<STAGES>();
    }

    /// Wait until the oldest shared result buffer can be changed or reused.
    ///
    /// This waits for TMA to finish reading shared memory. The later global
    /// write may still be in progress; that does not prevent buffer reuse.
    #[inline(always)]
    pub fn producer_acquire(&self) {
        __compiler::tma_store_producer_acquire::<STAGES>();
    }

    /// Wait until every TMA store has finished reading its shared source.
    #[inline(always)]
    pub fn producer_tail(&self) {
        __compiler::tma_store_producer_tail::<STAGES>();
    }
}

impl<const STAGES: usize, const CONSUMER_WARPS: u32, const TX_BYTES: u32>
    TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>
{
    /// Required alignment of each shared-memory barrier array.
    pub const STORAGE_ALIGN: usize = align_of::<Barrier>();

    /// Bytes used by adjacent `FULL[STAGES]` and `EMPTY[STAGES]` arrays.
    ///
    /// Using this constant also checks the type parameters. Each empty
    /// barrier expects one arrival from lane 0 of every consumer warp.
    ///
    /// ```compile_fail
    /// use cute_rs::TmaLoadPipeline;
    /// const _: usize = TmaLoadPipeline::<0, 1, 16>::STORAGE_BYTES;
    /// ```
    pub const STORAGE_BYTES: usize = {
        assert!(STAGES > 0);
        assert!(STAGES <= u32::MAX as usize);
        assert!(CONSUMER_WARPS > 0);
        assert!(CONSUMER_WARPS <= 32);
        assert!(TX_BYTES > 0);
        2 * STAGES * size_of::<Barrier>()
    };

    /// Use one allocation containing the full barriers, then empty barriers.
    ///
    /// ```text
    /// base -> [FULL x STAGES][EMPTY x STAGES]
    /// ```
    ///
    /// # Safety
    ///
    /// `base` must point to at least [`Self::STORAGE_BYTES`] writable bytes in
    /// shared memory and be aligned to [`Self::STORAGE_ALIGN`]. The first
    /// `STAGES` entries must be the full barriers and the next `STAGES` the
    /// empty barriers. This pipeline must have exclusive use of that storage
    /// until all asynchronous work and [`Self::producer_tail`] have finished.
    #[must_use]
    #[inline(always)]
    pub unsafe fn from_raw_base(base: *mut Barrier) -> Self {
        unsafe { __compiler::tma_load_pipeline_from_raw_base(base) }
    }

    /// Use separately allocated full and empty barrier arrays.
    ///
    /// # Safety
    ///
    /// `full` and `empty` must point to different shared-memory arrays of
    /// `STAGES` barriers, each aligned to [`Self::STORAGE_ALIGN`]. This
    /// pipeline must have exclusive use of both arrays until all asynchronous
    /// work and [`Self::producer_tail`] have finished.
    #[must_use]
    #[inline(always)]
    pub unsafe fn from_raw_parts(full: *mut Barrier, empty: *mut Barrier) -> Self {
        let _ = Self::STORAGE_BYTES;
        Self { full, empty }
    }

    /// Initialize all full and empty barriers.
    ///
    /// A full barrier expects one producer arrival. An empty barrier expects
    /// one arrival from lane 0 of each consumer warp. Every thread in the CTA
    /// must call this method on the same control-flow path. Only `init_thread`
    /// writes the barriers; the method publishes those writes and synchronizes
    /// the full CTA before returning.
    ///
    /// # Safety
    ///
    /// The pointer rules of the constructor must still hold. `init_thread`
    /// must be a valid thread number in this CTA. Neither barrier array may
    /// already be active or accessed at the same time by another pipeline.
    #[inline(always)]
    pub unsafe fn init(self, init_thread: u32) {
        unsafe { __compiler::tma_load_pipeline_init(self, init_thread) };
    }

    /// Wait until the current write slot is empty.
    ///
    /// The first slot starts ready, so the first load uses the same operation
    /// as every later load.
    ///
    /// # Safety
    ///
    /// `state` must be this pipeline's current producer cursor, owned by one
    /// producer leader. An acquire for the same slot must not overlap another
    /// acquire or be repeated without advancing the cursor.
    #[inline(always)]
    pub unsafe fn producer_acquire(self, state: &PipelineState<Producer, STAGES>) {
        unsafe { __compiler::pipeline_producer_acquire(self, state) };
    }

    /// Tell the current full barrier to expect `TX_BYTES` from TMA.
    ///
    /// The producer leader calls this once after [`Self::producer_acquire`]
    /// and before starting the stage's TMA loads. Pass the returned barrier
    /// pointer to those loads.
    ///
    /// # Safety
    ///
    /// `state` must point to the slot just acquired from this pipeline. The
    /// TMA loads attached to the returned barrier must complete exactly
    /// `TX_BYTES` bytes in total.
    #[must_use]
    #[inline(always)]
    pub unsafe fn producer_expect_tx(
        self,
        state: &PipelineState<Producer, STAGES>,
    ) -> *mut Barrier {
        unsafe { __compiler::pipeline_producer_expect_tx(self, state) }
    }

    /// Wait until TMA has filled the consumer's current slot.
    ///
    /// # Safety
    ///
    /// `state` must be this pipeline's current consumer cursor. Every consumer
    /// warp must complete this wait before reading the matching shared buffer.
    #[inline(always)]
    pub unsafe fn consumer_wait(self, state: &PipelineState<Consumer, STAGES>) {
        unsafe { __compiler::pipeline_consumer_wait(self, state) };
    }

    /// Mark the current read slot empty after all consumers finish.
    ///
    /// Every lane in a consumer warp may call this method once, but only
    /// hardware lane 0 updates the barrier. Exactly `CONSUMER_WARPS` warps
    /// must each contribute one lane-0 update after their last read.
    ///
    /// # Safety
    ///
    /// `state` must point to the slot previously passed to
    /// [`Self::consumer_wait`]. The calling warp must have finished all reads
    /// from that shared buffer. Exactly `CONSUMER_WARPS` different warps must
    /// participate, with no warp contributing more than one lane-0 update.
    #[inline(always)]
    pub unsafe fn consumer_release(self, state: &PipelineState<Consumer, STAGES>) {
        unsafe { __compiler::pipeline_consumer_release(self, state) };
    }

    /// Wait until consumers have released every stage that may still be busy.
    ///
    /// Starting at the producer's current slot, this waits through one full
    /// turn of the ring. Slots never used by a short startup pass immediately;
    /// used slots wait for the matching phase. This is CUTLASS's
    /// producer-tail rule.
    ///
    /// # Safety
    ///
    /// `state` must be the producer cursor immediately after its final issued
    /// stage. Call this exactly once before the producer exits or any pipeline
    /// shared storage is reused.
    #[inline(always)]
    pub unsafe fn producer_tail(self, state: PipelineState<Producer, STAGES>) {
        unsafe { __compiler::pipeline_producer_tail(self, state) };
    }
}

/// Stable compiler boundaries for the TMA load-pipeline protocol.
///
/// Public methods keep the normal Rust API. Device compilation recognizes
/// these exact calls; protocol state math remains ordinary testable Rust.
#[doc(hidden)]
pub mod __compiler {
    use super::*;

    #[inline(never)]
    pub fn tma_store_producer_acquire<const STAGES: usize>() {
        let _ = TmaStorePipeline::<STAGES>::MAX_PENDING;
        unreachable!("cute-rs TMA-store acquire executed outside recognized device compilation")
    }

    #[inline(never)]
    pub fn tma_store_producer_commit<const STAGES: usize>() {
        let _ = TmaStorePipeline::<STAGES>::MAX_PENDING;
        unreachable!("cute-rs TMA-store commit executed outside recognized device compilation")
    }

    #[inline(never)]
    pub fn tma_store_producer_tail<const STAGES: usize>() {
        let _ = TmaStorePipeline::<STAGES>::MAX_PENDING;
        unreachable!("cute-rs TMA-store tail executed outside recognized device compilation")
    }

    #[inline(never)]
    pub fn pipeline_state_new_producer<const STAGES: usize>() -> PipelineState<Producer, STAGES> {
        const { assert!(STAGES > 0) };
        const { assert!(STAGES <= u32::MAX as usize) };
        PipelineState {
            slot: 0,
            phase: 1,
            role: PhantomData,
        }
    }

    #[inline(never)]
    pub fn pipeline_state_new_consumer<const STAGES: usize>() -> PipelineState<Consumer, STAGES> {
        const { assert!(STAGES > 0) };
        const { assert!(STAGES <= u32::MAX as usize) };
        PipelineState {
            slot: 0,
            phase: 0,
            role: PhantomData,
        }
    }

    #[inline(never)]
    pub fn pipeline_state_slot<Role, const STAGES: usize>(
        state: &PipelineState<Role, STAGES>,
    ) -> usize {
        state.slot as usize
    }

    #[inline(never)]
    pub fn pipeline_state_advance<Role, const STAGES: usize>(
        state: &mut PipelineState<Role, STAGES>,
    ) {
        const { assert!(STAGES > 0) };
        const { assert!(STAGES <= u32::MAX as usize) };
        state.slot = state.slot.wrapping_add(1);
        if state.slot as usize == STAGES {
            state.slot = 0;
            state.phase ^= 1;
        }
    }

    #[inline(never)]
    pub unsafe fn tma_load_pipeline_from_raw_base<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        base: *mut Barrier,
    ) -> TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES> {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        TmaLoadPipeline {
            full: base,
            empty: unsafe { base.add(STAGES) },
        }
    }

    #[inline(never)]
    pub unsafe fn tma_load_pipeline_init<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        init_thread: u32,
    ) {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline.full, pipeline.empty, init_thread);
        unreachable!("cute-rs TMA-load init executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn pipeline_producer_acquire<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        state: &PipelineState<Producer, STAGES>,
    ) {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline, state);
        unreachable!("cute-rs producer acquire executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn pipeline_producer_expect_tx<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        state: &PipelineState<Producer, STAGES>,
    ) -> *mut Barrier {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline, state);
        unreachable!("cute-rs producer expect-tx executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn pipeline_consumer_wait<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        state: &PipelineState<Consumer, STAGES>,
    ) {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline, state);
        unreachable!("cute-rs consumer wait executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn pipeline_consumer_release<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        state: &PipelineState<Consumer, STAGES>,
    ) {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline, state);
        unreachable!("cute-rs consumer release executed outside recognized device compilation")
    }

    #[inline(never)]
    pub unsafe fn pipeline_producer_tail<
        const STAGES: usize,
        const CONSUMER_WARPS: u32,
        const TX_BYTES: u32,
    >(
        pipeline: TmaLoadPipeline<STAGES, CONSUMER_WARPS, TX_BYTES>,
        state: PipelineState<Producer, STAGES>,
    ) {
        let _ = TmaLoadPipeline::<STAGES, CONSUMER_WARPS, TX_BYTES>::STORAGE_BYTES;
        let _ = (pipeline, state);
        unreachable!("cute-rs producer tail executed outside recognized device compilation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_and_consumer_use_opposite_initial_phases() {
        let producer = PipelineState::<Producer, 3>::new();
        let consumer = PipelineState::<Consumer, 3>::new();
        assert_eq!((producer.slot(), producer.phase()), (0, 1));
        assert_eq!((consumer.slot(), consumer.phase()), (0, 0));
    }

    #[test]
    fn slot_wrap_flips_phase_once_per_ring() {
        let mut state = PipelineState::<Consumer, 3>::new();
        let expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 0)];
        for (slot, phase) in expected {
            assert_eq!((state.slot(), state.phase()), (slot, phase));
            state.advance();
        }
    }

    #[test]
    fn state_is_exactly_two_device_register_values() {
        assert_eq!(size_of::<PipelineState<Producer, 3>>(), 8);
        assert_eq!(size_of::<PipelineState<Consumer, 3>>(), 8);
    }

    #[test]
    fn static_contract_sizes_two_aligned_barrier_rings() {
        type Pipeline = TmaLoadPipeline<3, 8, 32_768>;
        const BYTES: usize = Pipeline::STORAGE_BYTES;
        assert_eq!(Pipeline::STORAGE_ALIGN, 8);
        assert_eq!(BYTES, 6 * size_of::<Barrier>());
    }

    #[test]
    fn tma_store_pipeline_is_zero_storage_and_uses_stage_minus_one() {
        assert_eq!(size_of::<TmaStorePipeline<1>>(), 0);
        assert_eq!(TmaStorePipeline::<1>::MAX_PENDING, 0);
        assert_eq!(TmaStorePipeline::<4>::MAX_PENDING, 3);
    }
}

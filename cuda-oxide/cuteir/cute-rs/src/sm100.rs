/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 GEMM building blocks for a pair of thread blocks (CTAs).
//!
//! Rust types describe shared-memory layouts and the two output partitions.
//! The compiler turns these calls into CuTe operations, then uses CUTLASS to
//! produce GPU code. Call them only inside device code.
//!
//! ```text
//! global A/B -> TMA -> shared tiles -> MMA -> tensor memory -> output
//! ```

use crate::TmaDesc;
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;
use cuda_device::barrier::Barrier;

/// A shared tile used as the left input of a matrix product.
pub enum OperandA {}
/// A shared tile used as the right input of a matrix product.
pub enum OperandB {}

/// A non-owning FP16 shared tile with a layout and an operand role.
pub struct Sm100SharedTile<Layout, Role> {
    base: *mut f16,
    marker: PhantomData<(Layout, Role)>,
}

impl<Layout: ReifySmem2D, Role> Sm100SharedTile<Layout, Role> {
    /// Interpret existing shared storage using `Layout`.
    ///
    /// # Safety
    ///
    /// `base` must remain live and contain every element addressed by `Layout`.
    /// TMA input bases must be aligned to their full swizzle period; the
    /// supported B128 MMA layouts require 1024-byte alignment.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        Self {
            base,
            marker: PhantomData,
        }
    }
}

/// Allocate one tensor-memory region collectively in a two-CTA cluster.
///
/// Tensor memory has 128 rows per CTA. `COLUMNS` is a power of two from 32
/// through 512. Allocation and deallocation are collective warp operations.
pub struct Sm100Tmem<const COLUMNS: u32>;
impl<const COLUMNS: u32> Sm100Tmem<COLUMNS> {
    const VALID: () = assert!(COLUMNS >= 32 && COLUMNS <= 512 && COLUMNS.is_power_of_two());

    /// Allocate columns and relinquish the allocation permit.
    ///
    /// # Safety
    ///
    /// Every lane of the allocating warp in both CTAs must participate.
    /// `token` points to a live aligned shared `u32`. Synchronize readers
    /// before loading the address written there.
    #[inline(always)]
    pub unsafe fn allocate(token: *mut u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_alloc::<COLUMNS>(token) }
    }

    /// Return a previously allocated region after both CTAs finish using it.
    ///
    /// # Safety
    ///
    /// All asynchronous MMA and loads must be complete and the two CTAs must
    /// rendezvous before both allocating warps call this operation.
    #[inline(always)]
    pub unsafe fn deallocate(address: u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_dealloc::<COLUMNS>(address) }
    }
}

/// Two-CTA FP16 tiled MMA with two N partitions sharing the A collector.
///
/// The A collector holds an input so two MMA instructions can reuse it:
///
/// ```text
/// each K=16 step:  A x B0 -> first N0 columns   (save A)
///                 A x B1 -> next N1 columns    (reuse, then release A)
/// ```
///
/// Each step uses a fresh A input. The complete output is
/// `256 x (N0 + N1)` with FP32 accumulation.
pub struct Sm100TiledMma<AL, B0L, B1L, const N0: u32, const N1: u32, const K: u32 = 64>(
    PhantomData<(AL, B0L, B1L)>,
);
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    const VALID: () = {
        assert!(N0 >= 16 && N0 <= 256 && N0.is_multiple_of(16));
        assert!(N1 >= 16 && N1 <= 256 && N1.is_multiple_of(16));
        assert!(N0 + N1 <= 512 && K == 64);
        assert!(AL::ROWS == 128 && AL::COLS == K as i64);
        assert!(B0L::ROWS == (N0 / 2) as i64 && B0L::COLS == K as i64);
        assert!(B1L::ROWS == (N1 / 2) as i64 && B1L::COLS == K as i64);
    };

    /// Check the tile dimensions and create the MMA configuration.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }

    /// Accumulate one staged K tile into tensor memory.
    ///
    /// # Safety
    ///
    /// One elected lane in the leader CTA issues this operation after the
    /// input pipeline wait. Both CTAs' tiles must be full and remain unchanged
    /// until the asynchronous consumer release completes. `address` names a
    /// 512-column allocation; `accumulate=false` starts a new output tile.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        address: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b0: &Sm100SharedTile<B0L, OperandB>,
        b1: &Sm100SharedTile<B1L, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::sm100_tiled_mma::<AL, B0L, B1L, N0, N1, K>(
                address, a.base, b0.base, b1.base, accumulate,
            )
        }
    }
}
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Default for Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    fn default() -> Self {
        Self::new()
    }
}

/// A cluster TMA copy whose descriptor and destination have the same layout.
pub struct Sm100ClusterTmaCopy<Layout>(PhantomData<Layout>);
impl<Layout: ReifySmem2D> Sm100ClusterTmaCopy<Layout> {
    /// Copy an FP16 tile, routing completion to the leader CTA's barrier.
    ///
    /// Coordinates count elements. Each CTA supplies its rank and receives
    /// only its own tile; completion bytes from both CTAs accumulate at rank 0.
    ///
    /// # Safety
    ///
    /// All 32 lanes of one producer warp in each CTA call this operation
    /// together after acquiring the stage, with identical arguments within
    /// that warp. The leader must attach exactly one expectation covering both
    /// CTAs' copies to the same stage and phase. The native CuTe copy elects
    /// one lane to issue the copy and may synchronize the warp.
    /// Calling it from an already elected lane can deadlock when
    /// pipeline stages are reused.
    ///
    /// The descriptor must encode `Layout`, destination storage must be valid
    /// and aligned, and `barrier` names the local full barrier for this stage.
    #[inline(always)]
    pub unsafe fn copy<Role>(
        desc: *const TmaDesc<f16, Layout>,
        dst: &mut Sm100SharedTile<Layout, Role>,
        row: u32,
        column: u32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::sm100_cluster_tma_load::<Layout>(desc, dst.base, row, column, barrier, rank)
        }
    }
}

/// Convert one 128x32 FP32 accumulator slice to an FP16 tile in shared memory.
pub struct Sm100TmemEpilogue<Layout>(PhantomData<Layout>);
impl<Layout: ReifySmem2D> Sm100TmemEpilogue<Layout> {
    /// Load a lane's 32 accumulators, add its row bias, and convert to FP16.
    ///
    /// # Safety
    ///
    /// All lanes of the four epilogue warps participate after accumulator wait.
    /// `address` already selects the calling warp's 32 TMEM rows and current
    /// 32-column slice. `tid` is the CTA thread index in `0..128`. `dst` has
    /// the supported 128x32 layout and is reusable according to the store
    /// pipeline. Publish shared writes before issuing the TMA store.
    #[inline(always)]
    pub unsafe fn store(address: u32, dst: *mut f16, tid: u32, bias: f32) {
        unsafe { __compiler::sm100_tmem_epilogue::<Layout>(address, dst, tid, bias) }
    }

    /// Select one warp's 32 TMEM rows and one 32-column accumulator slice.
    ///
    /// `warp` is in `0..4` and the chosen columns must belong to the tile.
    #[inline(always)]
    pub const fn slice_address(base: u32, warp: u32, subtile: u32) -> u32 {
        (((base >> 16) + warp * 32) << 16) | ((base & 0xffff) + subtile * 32)
    }

    /// Start the FP16 tile's asynchronous shared-to-global TMA store.
    ///
    /// # Safety
    ///
    /// One elected store issuer calls after all writers publish their shared
    /// writes and synchronize. The descriptor encodes `Layout`; coordinates
    /// count elements. The shared source must remain live and unchanged until
    /// this copy's committed group finishes reading it. A store-pipeline
    /// acquire releases the oldest buffer in the ring and can leave this
    /// newest group pending; it does not immediately release this source.
    #[inline(always)]
    pub unsafe fn store_async(
        desc: *const TmaDesc<f16, Layout>,
        src: *const f16,
        row: u32,
        column: u32,
    ) {
        unsafe { __compiler::sm100_tma_store::<Layout>(desc, src, row, column) }
    }
}

/// Two-CTA TMA-to-MMA stage ring, with one completion arrival per stage.
///
/// `TX_BYTES` is the total number of bytes copied by both CTAs per stage.
/// One lane in the leader CTA sets this expected byte count. MMA completion
/// releases the stage in both CTAs so the next TMA copy can reuse it.
pub struct Sm100TmaMmaPipeline<const STAGES: u32, const TX_BYTES: u32> {
    full: *mut Barrier,
    empty: *mut Barrier,
}
impl<const STAGES: u32, const TX_BYTES: u32> Sm100TmaMmaPipeline<STAGES, TX_BYTES> {
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
    pub unsafe fn producer_expect(&self, stage: u32) {
        unsafe { __compiler::sm100_pipeline_expect::<STAGES, TX_BYTES>(self.full, stage) }
    }
    /// Return the stage's local completion barrier for typed TMA copies.
    ///
    /// # Safety
    ///
    /// `stage < STAGES`. Copies using this barrier must belong to the acquired
    /// stage and phase covered by the leader's one cluster-wide expectation.
    #[inline(always)]
    pub unsafe fn full_barrier(&self, stage: u32) -> *mut Barrier {
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
pub struct Sm100AccumulatorPipeline<const CONSUMER_WARPS: u32> {
    empty: *mut Barrier,
    full: *mut Barrier,
}
impl<const CONSUMER_WARPS: u32> Sm100AccumulatorPipeline<CONSUMER_WARPS> {
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
#[derive(Clone, Copy, Default)]
pub struct Sm100TmaStorePipeline<const STAGES: u32>;
impl<const STAGES: u32> Sm100TmaStorePipeline<STAGES> {
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

/// Calls recognized by the compiler; never execute these on the host.
#[doc(hidden)]
#[allow(missing_docs)]
pub mod __compiler {
    use super::*;
    #[inline(never)]
    pub unsafe fn sm100_tmem_alloc<const COLUMNS: u32>(token: *mut u32) {
        let _ = token;
        unreachable!("SM100 CuTe marker requires device compilation")
    }
    #[inline(never)]
    pub unsafe fn sm100_tmem_dealloc<const COLUMNS: u32>(tmem: u32) {
        let _ = tmem;
        unreachable!("SM100 CuTe marker requires device compilation")
    }
    #[inline(never)]
    pub unsafe fn sm100_tiled_mma<AL, BL0, BL1, const N0: u32, const N1: u32, const K: u32>(
        tmem: u32,
        a: *const f16,
        b0: *const f16,
        b1: *const f16,
        accumulate: bool,
    ) {
        let _ = (tmem, a, b0, b1, accumulate, PhantomData::<(AL, BL0, BL1)>);
        unreachable!("SM100 CuTe marker requires device compilation")
    }
    #[inline(never)]
    pub unsafe fn sm100_tmem_epilogue<L>(tmem: u32, dst: *mut f16, tid: u32, bias: f32) {
        let _ = (tmem, dst, tid, bias, PhantomData::<L>);
        unreachable!("SM100 CuTe marker requires device compilation")
    }
    // Every lane of the producer warp participates. CuTe selects the issuing
    // lane and performs any required warp synchronization.
    #[inline(never)]
    pub unsafe fn sm100_cluster_tma_load<L>(
        desc: *const TmaDesc<f16, L>,
        dst: *mut f16,
        row: u32,
        col: u32,
        bar: *mut Barrier,
        rank: u32,
    ) {
        let _ = (desc, dst, row, col, bar, rank);
        unreachable!("SM100 CuTe marker requires device compilation")
    }

    #[inline(never)]
    pub unsafe fn sm100_tma_store<L>(
        desc: *const TmaDesc<f16, L>,
        src: *const f16,
        row: u32,
        col: u32,
    ) {
        let _ = (desc, src, row, col);
        unreachable!("SM100 CuTe marker requires device compilation")
    }
    macro_rules! store_marker {
        ($name:ident) => {
            #[inline(never)]
            pub unsafe fn $name<const STAGES: u32>() {
                unreachable!("SM100 CuTe marker requires device compilation")
            }
        };
    }
    store_marker!(sm100_store_commit);
    store_marker!(sm100_store_acquire);
    store_marker!(sm100_store_tail);
    macro_rules! pipeline_marker {
        ($name:ident ($($arg:ident:$ty:ty),*)) => {
            #[inline(never)]
            pub unsafe fn $name<const STAGES:u32,const TX_BYTES:u32>($($arg:$ty),*) {
                let _ = ($($arg),*); unreachable!("SM100 CuTe marker requires device compilation")
            }
        };
    }
    pipeline_marker!(sm100_pipeline_init(full: *mut Barrier, empty: *mut Barrier));
    pipeline_marker!(sm100_pipeline_acquire(empty: *mut Barrier, stage: u32, phase: u32));
    pipeline_marker!(sm100_pipeline_expect(full: *mut Barrier, stage: u32));
    pipeline_marker!(sm100_pipeline_wait(full: *mut Barrier, stage: u32, phase: u32));
    pipeline_marker!(sm100_pipeline_release(empty: *mut Barrier, stage: u32));
    macro_rules! accumulator_marker {
        ($name:ident ($($arg:ident:$ty:ty),*)) => {
            #[inline(never)]
            pub unsafe fn $name<const CONSUMER_WARPS:u32>($($arg:$ty),*) {
                let _ = ($($arg),*); unreachable!("SM100 CuTe marker requires device compilation")
            }
        };
    }
    accumulator_marker!(sm100_accumulator_init(empty: *mut Barrier, full: *mut Barrier));
    accumulator_marker!(sm100_accumulator_acquire(empty: *mut Barrier, phase: u32));
    accumulator_marker!(sm100_accumulator_commit(full: *mut Barrier));
    accumulator_marker!(sm100_accumulator_wait(full: *mut Barrier, phase: u32));
    accumulator_marker!(sm100_accumulator_release(empty: *mut Barrier));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Composed, RowMajor, Swizzle};
    type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
    type B0 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
    type B1 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
    #[test]
    fn typed_split_plan_has_no_runtime_storage() {
        let mma = Sm100TiledMma::<A, B0, B1, 192, 160>::new();
        assert_eq!(core::mem::size_of_val(&mma), 0);
    }
}

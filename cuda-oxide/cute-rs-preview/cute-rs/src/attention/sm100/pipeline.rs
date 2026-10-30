/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-CTA SM100 barrier rings with independently typed full and empty edges.
//!
//! Policies restrict completion mechanisms and preserve memory ordering.
//! Kernels still supply correct participants, persistent positions, buffer
//! lifetimes, and completion waits for deferred operations.
//!
//! Choose each edge by what its receiver will access:
//!
//! | Handoff | Policy |
//! | --- | --- |
//! | TMA/MMA completion before MMA | [`AsyncCompletion`] |
//! | MMA completion before TMA overwrites shared memory | [`AsyncCopyReuse`] |
//! | Thread-written shared data before MMA | [`SharedToAsync`] |
//! | MMA completion before ordinary shared writes | [`SharedReuse`] |
//! | TMEM data between threads | [`TensorMemory`] |
//! | Local shared data, without / with a later TC access | [`LocalShared`] / [`LocalMemory`] |
//!
//! A [`RingCursor`] tracks one role's position; [`TwoCtaRing`] documents the
//! producer/consumer handoff. Advancing a cursor never signals a barrier.

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use ctm_rs::mbarrier::{self, Barrier, ManagedBarrier, Ready, Uninit};
pub use ctm_rs::mbarrier::{GeneralBarrier, TmaBarrier};
use ctm_rs::{async_copy, cluster, fence, groups::CtaGroup2, memory::SharedAddress, tcgen05, warp};

/// Restricts edge policies and cursor roles to the implementations in this module.
mod sealed {
    /// Marks supported receiver ordering and whether later TC access needs a fence.
    pub trait Wait {
        // Existing policies order a subsequent TC receiver. Only explicitly
        // non-TC receiver policies may omit this fence.
        /// Whether the ring inserts an after-thread-sync fence following this policy’s wait.
        const TC_AFTER: bool = true;
    }
    /// Admits per-thread remote arrivals for a policy that publishes the correct memory domain.
    pub trait Signal {}
    /// Admits group-two MMA completion as the signal for this edge.
    pub trait Mma {}
    /// Admits transaction-counted TMA completion as the signal for this edge.
    pub trait Tma {}
    /// Restricts cursors to producer and consumer phase conventions.
    pub trait Cursor {}
}

/// Sealed ordering policy for one barrier-ring edge.
pub trait WaitPolicy: sealed::Wait + Copy {
    /// Observe the phase and acquire this policy's protected memory.
    /// # Safety
    /// The local shared barrier must be live and initialized, with matching
    /// signalers and a current or immediately preceding phase. The ring supplies
    /// an after-thread-sync fence for policies with TC receivers. Non-TC policies
    /// require a separate fence before any later asynchronous TC operations.
    unsafe fn wait(bar: *const Barrier, phase: u32);
}
/// Sealed policy permitting each participating thread to signal rank zero.
pub trait SignalPolicy: WaitPolicy + sealed::Signal {
    /// Publish one thread's handoff to the leader's corresponding barrier.
    /// # Safety
    /// The local barrier must have the same offset in exactly two live CTAs.
    /// Complete deferred operations first and contribute exactly the initialized
    /// number of arrivals, with each participant arriving once per phase.
    unsafe fn arrive(bar: *mut Barrier);
}
/// Sealed policy accepting group-two MMA multicast completion.
pub trait MmaPolicy: WaitPolicy + sealed::Mma {}
/// Sealed policy accepting transaction-counted TMA completion.
pub trait TmaPolicy: WaitPolicy + sealed::Tma {}

/// CTA acquire for local shared data or local TMEM metadata exchanges.
/// Local arrivals and cp.async attachment use unsafe raw accessors. TMEM
/// signalers must complete their operations and fence before those arrivals.
#[derive(Clone, Copy)]
pub struct LocalMemory;
impl sealed::Wait for LocalMemory {}
impl WaitPolicy for LocalMemory {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity(bar, phase) } {}
    }
}

/// CTA acquire for ordinary shared-memory or cp.async page exchanges.
///
/// This retains local acquire semantics and cp.async completion visibility.
/// It supplies no TC after-thread-sync fence: the receiver must use ordinary
/// memory or asynchronous copies, with no subsequent asynchronous TC access
/// before a separate TC fence. Local signaling uses the unsafe raw accessors.
#[derive(Clone, Copy)]
pub struct LocalShared;
impl sealed::Wait for LocalShared {
    const TC_AFTER: bool = false;
}
impl WaitPolicy for LocalShared {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity(bar, phase) } {}
    }
}

/// Direct TMA/MMA completion without publishing an ordinary-memory payload.
/// The ring orders subsequent MMA using a TC fence. Subsequent TMA must be
/// issued after the wait. This policy is not a thread-written data handoff.
#[derive(Clone, Copy)]
pub struct AsyncCompletion;
impl sealed::Wait for AsyncCompletion {}
impl sealed::Mma for AsyncCompletion {}
impl sealed::Tma for AsyncCompletion {}
impl MmaPolicy for AsyncCompletion {}
impl TmaPolicy for AsyncCompletion {}
impl WaitPolicy for AsyncCompletion {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, phase) } {}
    }
}

/// Observe completed MMA reads before reusing shared storage with TMA copies.
///
/// The cluster-scoped completion poll is followed by the caller's asynchronous
/// copy issue. This policy supplies no TC fence and does not acquire an ordinary
/// memory payload. Use it for an empty edge whose receiver issues TMA copies;
/// ordinary shared overwrites require [`SharedReuse`], and subsequent TC access
/// requires a separate after-thread-sync fence.
#[derive(Clone, Copy)]
pub struct AsyncCopyReuse;
impl sealed::Wait for AsyncCopyReuse {
    const TC_AFTER: bool = false;
}
impl sealed::Mma for AsyncCopyReuse {}
impl MmaPolicy for AsyncCopyReuse {}
impl WaitPolicy for AsyncCopyReuse {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, phase) } {}
    }
}

/// TMEM handoff using cluster execution ordering and TC thread-sync fences.
/// Thread signalers must complete deferred TMEM loads/stores first. Ordinary
/// shared or global payload writes are not published by this policy.
#[derive(Clone, Copy)]
pub struct TensorMemory;
impl sealed::Wait for TensorMemory {}
impl sealed::Signal for TensorMemory {}
impl sealed::Mma for TensorMemory {}
impl MmaPolicy for TensorMemory {}
impl WaitPolicy for TensorMemory {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, phase) } {}
    }
}
impl SignalPolicy for TensorMemory {
    #[inline(always)]
    unsafe fn arrive(bar: *mut Barrier) {
        tcgen05::fence_before_thread_sync();
        unsafe { arrive_remote(bar) };
    }
}

/// Publish ordinary local shared stores to the two-CTA async MMA reader.
/// Every writer releases through the generic-to-async proxy fence; the wait
/// acquires cluster shared memory in the async proxy.
#[derive(Clone, Copy)]
pub struct SharedToAsync;
impl sealed::Wait for SharedToAsync {}
impl sealed::Signal for SharedToAsync {}
impl WaitPolicy for SharedToAsync {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, phase) } {}
        unsafe { fence::fence_proxy_async_acquire_sync_restrict() };
    }
}
impl SignalPolicy for SharedToAsync {
    #[inline(always)]
    unsafe fn arrive(bar: *mut Barrier) {
        unsafe {
            fence::fence_proxy_async_release_sync_restrict();
            arrive_remote(bar);
        }
    }
}

/// Acquire completed MMA shared reads before ordinary shared-buffer overwrites.
/// The generic acquire keeps cluster scope and restricts its memory domain to
/// shared objects; it does not acquire unrelated global memory.
#[derive(Clone, Copy)]
pub struct SharedReuse;
impl sealed::Wait for SharedReuse {}
impl sealed::Mma for SharedReuse {}
impl MmaPolicy for SharedReuse {}
impl WaitPolicy for SharedReuse {
    #[inline(always)]
    unsafe fn wait(bar: *const Barrier, phase: u32) {
        while !unsafe { mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, phase) } {}
        unsafe { fence::fence_sync_restrict_acquire() };
    }
}

/// Cursor roles shared with [`crate::pipeline::PipelineState`]: a producer
/// cursor starts at empty phase one, a consumer cursor at full phase zero.
pub use crate::pipeline::{Consumer, Producer};
impl sealed::Cursor for Producer {}
impl sealed::Cursor for Consumer {}

/// Sealed role fixing a cursor's initial phase and permitted ring operations.
pub trait CursorRole: sealed::Cursor + Copy {
    /// Starting barrier parity: one for producer empty waits, zero for consumer full waits.
    #[doc(hidden)]
    const INITIAL_PHASE: u32;
}
impl CursorRole for Producer {
    const INITIAL_PHASE: u32 = 1;
}
impl CursorRole for Consumer {
    const INITIAL_PHASE: u32 = 0;
}

/// Persistent position in a ring, with a bounded slot and explicit wait phase.
///
/// Advance exactly once after each stage handoff, and retain the cursor across
/// work items. Advancing wraps the slot and toggles the phase without division
/// or a monotonically increasing counter. Copies preserve position only; they
/// grant no extra ownership, participants, or permission to repeat a handoff.
///
/// ```
/// use cute_rs::attention::sm100::pipeline::{Producer, RingCursor};
/// let mut cursor = RingCursor::<2, Producer>::new();
/// cursor.advance();
/// cursor.advance();
/// assert_eq!((cursor.index(), cursor.phase()), (0, 0)); // Wrapped; phase flipped.
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingCursor<const STAGES: u32, Role: CursorRole> {
    /// Bounded barrier-array slot in 0..STAGES.
    index: u32,
    /// Parity this role waits for at the current slot.
    phase: u32,
    /// Records the role’s initial phase and permitted handoff operations.
    role: PhantomData<Role>,
}
impl<const STAGES: u32, Role: CursorRole> RingCursor<STAGES, Role> {
    /// Reject zero-stage cursors before any slot arithmetic.
    const VALID: () = assert!(STAGES > 0);

    /// Start at slot zero, with the producer or consumer's initial wait phase.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self {
            index: 0,
            phase: Role::INITIAL_PHASE,
            role: PhantomData,
        }
    }
    /// Current stage index, always less than STAGES.
    ///
    /// It is a `u32` like this ring's stage count; the TMA load pipeline's
    /// [`crate::PipelineState::index`] is a `usize`.
    #[inline(always)]
    pub const fn index(self) -> u32 {
        self.index
    }
    /// Phase awaited by this role at the current stage.
    #[inline(always)]
    pub const fn phase(self) -> u32 {
        self.phase
    }
    /// Move to the next stage, toggling phase on wrap.
    ///
    /// This updates software state only; it performs no barrier operation.
    #[inline(always)]
    pub const fn advance(&mut self) {
        self.index += 1;
        if self.index == STAGES {
            self.index = 0;
            self.phase ^= 1;
        }
    }
}
impl<const STAGES: u32, Role: CursorRole> Default for RingCursor<STAGES, Role> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Non-owning view of adjacent full and empty barrier arrays in two CTAs.
///
/// Both CTAs have identical shared-memory layouts. Thread arrivals target rank
/// zero; MMA completion multicasts to ranks zero and one. Local-only edges use
/// LocalMemory and the unsafe raw barrier accessors.
///
/// Tickets start at zero and continue across persistent work. The initial waited
/// full and empty parities are zero and one respectively; initialization puts
/// both hardware barriers in phase zero. Do not reset or overflow tickets
/// while phases remain live. Copies grant no additional ownership, participants,
/// or permission to signal a phase twice.
///
/// For a TMA-to-MMA B-operand ring, use `AsyncCompletion` for full and
/// `AsyncCopyReuse` for empty. One lane per CTA initializes with `init(1, 1)`;
/// publish initialization as required by [`Self::init`]. Keep both role cursors
/// across work items. The following device-only steps assume matching live
/// storage; the caller selects the B tile at the cursor's stage.
///
/// Every producer-warp lane in each CTA acquires and copies its 64×64 FP16
/// tile. One elected rank-zero lane expects 16384 bytes across the CTA pair:
///
/// ```no_run
/// # #![feature(f16)]
/// # use cute_rs::{OperandB, TensorMap, attention::sm100::{SharedLayout, SharedTile, TmaCopy, pipeline::*}};
/// type L = SharedLayout<64, 64>;
/// # type Ring = TwoCtaRing<2, AsyncCompletion, AsyncCopyReuse>;
/// # unsafe fn produce(ring: Ring, p: &mut RingCursor<2, Producer>, desc: &TensorMap<f16, L>, b: &mut SharedTile<L, OperandB>, rank: u32) {
/// unsafe {
///     ring.acquire_cursor(*p);
///     if rank == 0 && ctm_rs::warp::is_elected_sync(u32::MAX) {
///         ring.producer_expect_tx_cursor(*p, 16384);
///     }
///     let done = ring.full_at_cursor(*p);
///     TmaCopy::<L, 2>::load(desc, b, (rank * 64) as i32, 0, done, rank);
///     p.advance();
/// }
/// # }
/// ```
///
/// The leader CTA's MMA warp waits; its elected lane issues the product and
/// attaches the empty signal to MMA completion. A must already be ready; A, B
/// and destination TMEM stay live until completion. `accumulate = true`
/// requires initialized TMEM. The next producer acquire waits for this release:
///
/// ```no_run
/// # use cute_rs::{OperandA, OperandB, attention::sm100::{MmaSs, SharedLayout, SharedTile, pipeline::*}};
/// # type L = SharedLayout<64, 64>;
/// # type Ring = TwoCtaRing<2, AsyncCompletion, AsyncCopyReuse>;
/// # unsafe fn consume(ring: Ring, c: &mut RingCursor<2, Consumer>, mma: &MmaSs<L, L, 128, 128, 64, 2>, dst: u32, a: &SharedTile<L, OperandA>, b: &SharedTile<L, OperandB>, accumulate: bool) {
/// unsafe {
///     ring.wait_cursor(*c);
///     if ctm_rs::warp::is_elected_sync(u32::MAX) {
///         mma.gemm(dst, a, b, accumulate);
///         ring.mma_empty_cursor(*c);
///     }
///     c.advance();
/// }
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct TwoCtaRing<const STAGES: u32, Full: WaitPolicy, Empty: WaitPolicy = Full> {
    /// Local array of full barriers with identical offsets in both CTAs.
    full: *mut Barrier,
    /// Local array of empty barriers immediately after the full array.
    empty: *mut Barrier,
    /// Carries independent receiver-ordering policies for the two edges.
    ordering: PhantomData<(Full, Empty)>,
}

impl<const STAGES: u32, Full: WaitPolicy, Empty: WaitPolicy> TwoCtaRing<STAGES, Full, Empty> {
    /// Reject zero-stage rings before modulo or storage-size calculations.
    const VALID: () = assert!(STAGES > 0);
    /// Required alignment of the barrier storage.
    pub const STORAGE_ALIGN: usize = align_of::<Barrier>();
    /// Bytes occupied by adjacent full and empty arrays; rejects zero stages.
    pub const STORAGE_BYTES: usize = {
        let () = Self::VALID;
        2 * STAGES as usize * size_of::<Barrier>()
    };

    /// View full then empty arrays, starting at `base[first]`.
    /// # Safety
    /// The aligned allocation must contain at least first + 2 * STAGES barriers.
    /// Before device operations, it must be local shared storage with identical
    /// offsets in both CTAs, used exclusively by this pipeline until all async
    /// work and waits finish. Before repurposing this storage within the kernel,
    /// invalidate every initialized barrier after all participants have finished.
    /// This non-owning view neither initializes nor invalidates storage.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut Barrier, first: usize) -> Self {
        let () = Self::VALID;
        Self {
            full: unsafe { base.add(first) },
            empty: unsafe { base.add(first + STAGES as usize) },
            ordering: PhantomData,
        }
    }
    #[inline(always)]
    /// Reduce a persistent ticket to its barrier-array slot.
    const fn slot(ticket: u32) -> usize {
        (ticket % STAGES) as usize
    }
    #[inline(always)]
    /// Select the full barrier parity, flipping after each complete ring traversal.
    const fn full_phase(ticket: u32) -> u32 {
        (ticket / STAGES) & 1
    }
    #[inline(always)]
    /// Select the producer’s empty parity, opposite the full parity for the same ticket.
    const fn empty_phase(ticket: u32) -> u32 {
        Self::full_phase(ticket) ^ 1
    }

    /// Initialize every stage's counts from one thread in each CTA.
    /// # Safety
    /// Counts must be positive valid mbarrier counts matching the protocol.
    /// Each storage location must be uninitialized or explicitly invalidated;
    /// an idle but valid barrier must not be initialized again. No participant
    /// may still access a previous barrier in these arrays. Publish initialization
    /// and synchronize the cluster before use; constructor storage requirements
    /// remain in force.
    #[inline(always)]
    pub unsafe fn init(self, producers: u32, consumers: u32) {
        let mut stage = 0;
        while stage < STAGES {
            unsafe {
                mbarrier::mbarrier_init(self.full.add(stage as usize), producers);
                mbarrier::mbarrier_init(self.empty.add(stage as usize), consumers);
            }
            stage += 1;
        }
    }
    /// Borrow the local full barrier for raw TMA or cp.async attachment.
    /// # Safety
    /// Storage must satisfy the constructor's allocation contract. Raw operations
    /// must obey Full, counts, and ticket phase; this grants no extra arrival.
    #[inline(always)]
    pub unsafe fn full_at(self, ticket: u32) -> *mut Barrier {
        unsafe { self.full.add(Self::slot(ticket)) }
    }
    /// Borrow the local empty barrier for raw local release operations.
    /// # Safety
    /// Storage must satisfy the constructor's allocation contract. Raw releases
    /// must obey Empty, completion fences, counts, and ticket phase.
    #[inline(always)]
    pub unsafe fn empty_at(self, ticket: u32) -> *mut Barrier {
        unsafe { self.empty.add(Self::slot(ticket)) }
    }
    /// Wait for reuse with the empty policy's receiver ordering.
    /// # Safety
    /// Use the current producer ticket with matching consumer releases. This
    /// grants no additional ownership and does not order arbitrary generic
    /// accesses when Empty only tracks async/TMEM completion. Non-TC policies
    /// do not order subsequent asynchronous TC operations.
    #[inline(always)]
    pub unsafe fn acquire(self, ticket: u32) {
        unsafe { Empty::wait(self.empty_at(ticket), Self::empty_phase(ticket)) };
        after_wait::<Empty>();
    }
    /// Wait for production with the full policy's receiver ordering.
    /// # Safety
    /// Use the current consumer ticket with matching producer signals. Every
    /// consuming thread must wait before accessing the protected stage and obey
    /// Full's receiver restrictions; non-TC policies supply no TC fence.
    #[inline(always)]
    pub unsafe fn wait(self, ticket: u32) {
        unsafe { Full::wait(self.full_at(ticket), Self::full_phase(ticket)) };
        after_wait::<Full>();
    }
    /// Set expected TMA bytes and contribute one full-barrier arrival.
    /// # Safety
    /// One elected rank-zero thread calls after acquire. TMA must complete
    /// exactly bytes across the CTA pair on this full barrier.
    #[inline(always)]
    pub unsafe fn producer_expect_tx(self, ticket: u32, bytes: u32)
    where
        Full: TmaPolicy,
    {
        unsafe { mbarrier::mbarrier_arrive_expect_tx_cluster(self.full_at(ticket), 1, bytes) };
    }
    /// Signal full in both CTAs after prior group-two MMA completes.
    /// # Safety
    /// One elected rank-zero thread calls after all MMA producing the stage.
    /// Its one completion arrival must match the full barrier count.
    #[inline(always)]
    pub unsafe fn mma_full(self, ticket: u32)
    where
        Full: MmaPolicy,
    {
        unsafe { mma_arrive(self.full_at(ticket)) };
    }
    /// Signal empty in both CTAs after prior group-two MMA completes.
    /// # Safety
    /// One elected rank-zero thread calls after all MMA consuming the stage.
    /// Its one completion arrival must match the empty barrier count.
    #[inline(always)]
    pub unsafe fn mma_empty(self, ticket: u32)
    where
        Empty: MmaPolicy,
    {
        unsafe { mma_arrive(self.empty_at(ticket)) };
    }
    /// Publish one thread's production to the leader's full barrier.
    /// # Safety
    /// Complete deferred operations first. Each counted producer contributes
    /// one arrival after its last protected access, using the current ticket.
    #[inline(always)]
    pub unsafe fn thread_full(self, ticket: u32)
    where
        Full: SignalPolicy,
    {
        unsafe { Full::arrive(self.full_at(ticket)) };
    }
    /// Publish one thread's consumption to the leader's empty barrier.
    /// # Safety
    /// Complete deferred operations first. Each counted consumer contributes
    /// one arrival after its last protected access, using the current ticket.
    #[inline(always)]
    pub unsafe fn thread_empty(self, ticket: u32)
    where
        Empty: SignalPolicy,
    {
        unsafe { Empty::arrive(self.empty_at(ticket)) };
    }
}

impl<const STAGES: u32, Full: WaitPolicy, Empty: WaitPolicy> TwoCtaRing<STAGES, Full, Empty> {
    /// Borrow the full barrier at an already bounded cursor index.
    /// # Safety
    /// The storage and raw-operation requirements of [`Self::full_at`] apply.
    #[inline(always)]
    pub unsafe fn full_at_cursor<Role: CursorRole>(
        self,
        cursor: RingCursor<STAGES, Role>,
    ) -> *mut Barrier {
        unsafe { self.full.add(cursor.index as usize) }
    }
    /// Borrow the empty barrier at an already bounded cursor index.
    /// # Safety
    /// The storage and raw-operation requirements of [`Self::empty_at`] apply.
    #[inline(always)]
    pub unsafe fn empty_at_cursor<Role: CursorRole>(
        self,
        cursor: RingCursor<STAGES, Role>,
    ) -> *mut Barrier {
        unsafe { self.empty.add(cursor.index as usize) }
    }
    /// Wait for reuse using a producer cursor, without deriving slot or phase.
    /// # Safety
    /// The requirements of [`Self::acquire`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn acquire_cursor(self, cursor: RingCursor<STAGES, Producer>) {
        unsafe { Empty::wait(self.empty_at_cursor(cursor), cursor.phase) };
        after_wait::<Empty>();
    }
    /// Wait for production using a consumer cursor.
    /// # Safety
    /// The requirements of [`Self::wait`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn wait_cursor(self, cursor: RingCursor<STAGES, Consumer>) {
        unsafe { Full::wait(self.full_at_cursor(cursor), cursor.phase) };
        after_wait::<Full>();
    }
    /// Expect TMA bytes and arrive at the producer cursor's full barrier.
    /// # Safety
    /// The requirements of [`Self::producer_expect_tx`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn producer_expect_tx_cursor(self, cursor: RingCursor<STAGES, Producer>, bytes: u32)
    where
        Full: TmaPolicy,
    {
        unsafe {
            mbarrier::mbarrier_arrive_expect_tx_cluster(self.full_at_cursor(cursor), 1, bytes)
        };
    }
    /// Signal MMA production at the producer cursor's full barrier.
    /// # Safety
    /// The requirements of [`Self::mma_full`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn mma_full_cursor(self, cursor: RingCursor<STAGES, Producer>)
    where
        Full: MmaPolicy,
    {
        unsafe { mma_arrive(self.full_at_cursor(cursor)) };
    }
    /// Signal MMA consumption at the consumer cursor's empty barrier.
    /// # Safety
    /// The requirements of [`Self::mma_empty`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn mma_empty_cursor(self, cursor: RingCursor<STAGES, Consumer>)
    where
        Empty: MmaPolicy,
    {
        unsafe { mma_arrive(self.empty_at_cursor(cursor)) };
    }
    /// Publish a thread's production at the producer cursor's full barrier.
    /// # Safety
    /// The requirements of [`Self::thread_full`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn thread_full_cursor(self, cursor: RingCursor<STAGES, Producer>)
    where
        Full: SignalPolicy,
    {
        unsafe { Full::arrive(self.full_at_cursor(cursor)) };
    }
    /// Publish a thread's consumption at the consumer cursor's empty barrier.
    /// # Safety
    /// The requirements of [`Self::thread_empty`] apply to the current cursor position.
    #[inline(always)]
    pub unsafe fn thread_empty_cursor(self, cursor: RingCursor<STAGES, Consumer>)
    where
        Empty: SignalPolicy,
    {
        unsafe { Empty::arrive(self.empty_at_cursor(cursor)) };
    }
}

impl<const STAGES: u32> TwoCtaRing<STAGES, LocalMemory> {
    /// Order preceding TMEM stores and contribute one thread arrival to the local full barrier.
    ///
    /// # Safety
    /// The caller must have completed its TMEM metadata stores for `ticket` and
    /// be an expected producer participant for the local full barrier.
    #[inline(always)]
    pub unsafe fn commit_local_tmem(self, ticket: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                self.full_at(ticket),
            ));
        }
    }

    /// Order preceding TMEM loads and contribute one thread arrival to the local empty barrier.
    ///
    /// # Safety
    /// The caller must have completed its TMEM metadata reads for `ticket` and
    /// be an expected consumer participant for the local empty barrier.
    #[inline(always)]
    pub unsafe fn release_local_tmem(self, ticket: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                self.empty_at(ticket),
            ));
        }
    }
}

impl<const STAGES: u32> TwoCtaRing<STAGES, LocalShared> {
    /// Acquire a page-index slot, issue one 4-byte `cp.async.ca` on `lane == 0`, and attach
    /// completion to the local full barrier across all 32 warp lanes.
    ///
    /// Slots in `smem_base` are spaced 256 bytes apart (`(ticket % STAGES) * 256`).
    ///
    /// # Safety
    /// All 32 lanes of the page-loader warp must participate with matching `ticket`,
    /// `smem_base` covering `STAGES * 256` bytes, and `page_offset < page_table.len()`.
    #[inline(always)]
    pub unsafe fn stage_page_index(
        self,
        ticket: u32,
        smem_base: *mut u8,
        page_table: &[i32],
        page_offset: usize,
        lane: u32,
    ) {
        unsafe {
            self.acquire(ticket);
            let dst = smem_base
                .add((ticket % STAGES) as usize * 256)
                .cast::<u32>();
            if lane == 0 {
                async_copy::cp_async_ca_4(
                    SharedAddress::from_raw(dst),
                    &*page_table.as_ptr().add(page_offset).cast::<u32>(),
                );
            }
            async_copy::cp_async_mbarrier_arrive_noinc(
                &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(self.full_at(ticket)),
            );
        }
    }

    /// Wait for the page-index copy at `ticket` to complete and read the staged `i32` page index.
    ///
    /// # Safety
    /// `smem_base` must point to the 256-byte-strided page-index staging array
    /// paired with this ring, and a matching `stage_page_index` must have been issued.
    #[must_use]
    #[inline(always)]
    pub unsafe fn wait_and_read_page_index(self, ticket: u32, smem_base: *const u8) -> i32 {
        unsafe {
            self.wait(ticket);
            *smem_base
                .add((ticket % STAGES) as usize * 256)
                .cast::<i32>()
        }
    }

    /// Read the already-waited `i32` page index at `ticket` and arrive on the local empty barrier.
    ///
    /// # Safety
    /// The caller must have already waited on `ticket` (for example via
    /// [`Self::wait_and_read_page_index`]) and all 32 consumer-warp lanes must
    /// call this before the slot can be reused.
    #[must_use]
    #[inline(always)]
    pub unsafe fn read_page_index_and_release(self, ticket: u32, smem_base: *const u8) -> i32 {
        unsafe {
            let page = *smem_base
                .add((ticket % STAGES) as usize * 256)
                .cast::<i32>();
            mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                self.empty_at(ticket),
            ));
            page
        }
    }
}

/// Single-CTA SM100 full/empty stage ring for warp-specialized attention pipelines.
///
/// Mirrors CuTeDSL's single-CTA `PipelineTmaUmma`, `PipelineUmmaAsync`,
/// `PipelineAsyncUmma` and `PipelineAsync` in `fmha.py`. Producer counts start
/// at phase 1 (initially empty) and consumer counts at phase 0 (initially
/// waiting for production); counts persist across persistent work tiles.
pub struct SingleCtaRing<const STAGES: u32, FullKind = GeneralBarrier> {
    /// Contiguous stage barriers completed when producers publish their work.
    full: *mut Barrier,
    /// Contiguous stage barriers completed when consumers release their slots.
    empty: *mut Barrier,
    /// Distinguishes transaction-counted TMA producers (`TmaBarrier`) from ordinary arrivals.
    _full_kind: PhantomData<FullKind>,
}

impl<const STAGES: u32, K> Copy for SingleCtaRing<STAGES, K> {}
impl<const STAGES: u32, K> Clone for SingleCtaRing<STAGES, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const STAGES: u32, K> SingleCtaRing<STAGES, K> {
    /// Bind a ring at `offset` measured in eight-byte barrier slots from `storage`.
    ///
    /// # Safety
    /// The shared allocation must contain `2 * STAGES` aligned, nonoverlapping
    /// 8-byte barrier slots starting at `storage.add(offset * 8)`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn from_raw(storage: *mut u8, offset: usize) -> Self {
        const { assert!(STAGES > 0) };
        let full = unsafe { storage.add(offset * 8).cast::<Barrier>() };
        Self {
            full,
            empty: unsafe { full.add(STAGES as usize) },
            _full_kind: PhantomData,
        }
    }

    /// Initialize full and empty barriers with per-stage producer and consumer arrival counts.
    ///
    /// # Safety
    /// Call once from a single thread before publishing barrier initialization
    /// to the CTA (`cute_rs::arch::mbarrier_init_fence()`).
    #[inline(always)]
    pub unsafe fn init(self, producers: u32, consumers: u32) {
        let mut i = 0;
        while i < STAGES {
            unsafe {
                ManagedBarrier::<Uninit, K>::from_static(self.full.add(i as usize))
                    .init_instruction(producers);
                ManagedBarrier::<Uninit, GeneralBarrier>::from_static(self.empty.add(i as usize))
                    .init_instruction(consumers);
            }
            i += 1;
        }
    }

    /// Wait for consumers to release `count % STAGES`, using the producer's inverted parity.
    ///
    /// # Safety
    /// The barrier must be initialized and `count` must track the producer's
    /// monotonically increasing stage ticket.
    #[inline(always)]
    pub unsafe fn acquire(self, count: u32) {
        let ptr = unsafe { self.empty.add((count % STAGES) as usize) };
        unsafe { wait_local::<GeneralBarrier>(ptr, ((count / STAGES) & 1) ^ 1) };
    }

    /// Wait for producers to fill `count % STAGES`, then order subsequent TMEM accesses.
    ///
    /// # Safety
    /// The barrier must be initialized and `count` must track the consumer's
    /// monotonically increasing stage ticket.
    #[inline(always)]
    pub unsafe fn wait(self, count: u32) {
        unsafe { wait_local::<K>(self.full_ptr(count), (count / STAGES) & 1) };
        tcgen05::fence_after_thread_sync();
    }

    /// Select the full barrier pointer for the monotonically increasing stage count.
    ///
    /// # Safety
    /// The ring's barrier array must remain live in CTA shared memory.
    #[must_use]
    #[inline(always)]
    pub unsafe fn full_ptr(self, count: u32) -> *mut Barrier {
        unsafe { self.full.add((count % STAGES) as usize) }
    }

    /// Order prior TMEM accesses and contribute one thread arrival to the full barrier.
    ///
    /// # Safety
    /// Each expected producer thread must call this once after completing its
    /// stage writes for `count`.
    #[inline(always)]
    pub unsafe fn commit_threads(self, count: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, K>::adopt_ready(
                self.full_ptr(count),
            ));
        }
    }

    /// Order prior TMEM accesses and contribute one thread arrival to the empty barrier.
    ///
    /// # Safety
    /// Each expected consumer thread must call this once after completing its
    /// stage reads for `count`.
    #[inline(always)]
    pub unsafe fn release_threads(self, count: u32) {
        tcgen05::fence_before_thread_sync();
        unsafe {
            mbarrier::arrive(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                self.empty.add((count % STAGES) as usize),
            ));
        }
    }

    /// Have one elected lane signal the full barrier after outstanding MMA work completes.
    ///
    /// # Safety
    /// All 32 lanes of the MMA warp must participate convergently in the election.
    #[inline(always)]
    pub unsafe fn commit_mma(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                tcgen05::commit(&ManagedBarrier::<Ready, K>::adopt_ready(
                    self.full_ptr(count),
                ));
            }
        }
    }

    /// Have one elected lane release the slot after outstanding MMA work completes.
    ///
    /// # Safety
    /// All 32 lanes of the MMA warp must participate convergently in the election.
    #[inline(always)]
    pub unsafe fn release_mma(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                tcgen05::commit(&ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                    self.empty.add((count % STAGES) as usize),
                ));
            }
        }
    }
}

impl<const STAGES: u32> SingleCtaRing<STAGES, TmaBarrier> {
    /// Have one elected lane arrive and expect `TX_BYTES` TMA transaction bytes on `count`.
    ///
    /// # Safety
    /// All 32 lanes of the producer warp must participate convergently in the election.
    #[inline(always)]
    pub unsafe fn producer_expect_tx<const TX_BYTES: u32>(self, count: u32) {
        if warp::is_elected_sync(u32::MAX) {
            unsafe {
                mbarrier::arrive_expect_tx(
                    &ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(self.full_ptr(count)),
                    TX_BYTES,
                );
            }
        }
    }
}

/// Single-CTA TMA-to-UMMA pipeline (`pipeline.PipelineTmaUmma` in CuTeDSL `fmha.py`).
///
/// TMA producer warps arrive with expected transaction bytes on the full barrier,
/// and the UMMA warp releases the empty barrier upon MMA completion.
pub type PipelineTmaUmma<const STAGES: u32> = SingleCtaRing<STAGES, TmaBarrier>;

/// Single-CTA UMMA-to-thread pipeline (`pipeline.PipelineUmmaAsync` in CuTeDSL `fmha.py`).
///
/// The UMMA warp commits completion to the full barrier and consumer threads
/// release the empty barrier after TMEM reads.
pub type PipelineUmmaAsync<const STAGES: u32> = SingleCtaRing<STAGES, GeneralBarrier>;

/// Single-CTA thread-to-UMMA pipeline (`pipeline.PipelineAsyncUmma` in CuTeDSL `fmha.py`).
///
/// Producer threads commit TMEM/shared writes to the full barrier and the UMMA
/// warp releases the empty barrier upon MMA completion.
pub type PipelineAsyncUmma<const STAGES: u32> = SingleCtaRing<STAGES, GeneralBarrier>;

/// Single-CTA thread-to-thread pipeline (`pipeline.PipelineAsync` in CuTeDSL `fmha.py`).
///
/// Producer threads commit to the full barrier and consumer threads release the
/// empty barrier.
pub type PipelineAsync<const STAGES: u32> = SingleCtaRing<STAGES, GeneralBarrier>;

/// Spin until an initialized local shared barrier completes the requested parity phase.
#[inline(always)]
unsafe fn wait_local<K>(ptr: *const Barrier, phase: u32) {
    let barrier = unsafe { ManagedBarrier::<Ready, K>::adopt_ready(ptr.cast_mut()) };
    while !unsafe { mbarrier::try_wait_parity(&barrier, phase) } {}
}

#[inline(always)]
/// Insert the TC receiver fence only for policies that require it.
fn after_wait<Policy: WaitPolicy>() {
    if <Policy as sealed::Wait>::TC_AFTER {
        tcgen05::fence_after_thread_sync();
    }
}

#[inline(always)]
/// Signal the matching rank-zero barrier without an implicit memory release.
///
/// # Safety
/// The two CTAs must have matching live initialized barriers and the caller must supply the policy’s publication fences.
unsafe fn arrive_remote(bar: *mut Barrier) {
    unsafe {
        cluster::arrive_remote_relaxed(cluster::mapa(SharedAddress::from_raw(bar), 0));
    }
}
#[inline(always)]
/// Attach prior group-two MMA completion to both CTAs’ matching barriers.
///
/// # Safety
/// The elected leader issuer must have matching initialized barriers and expected completion counts in both CTAs.
unsafe fn mma_arrive(bar: *mut Barrier) {
    unsafe {
        tcgen05::commit_multicast_cg2(
            SharedAddress::from_raw(bar),
            cluster::CtaMask::<CtaGroup2>::try_from_bits(3).unwrap_unchecked(),
        );
    }
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;
    /// Ticket arithmetic must wrap stage indices and preserve opposite full/empty parities.
    #[test]
    fn tickets_wrap_slots_and_flip_opposite_phases() {
        /// Stage-count and ordering-policy fixture for checking ticket parity arithmetic.
        type Ring = TwoCtaRing<3, AsyncCompletion>;
        for (ticket, slot, full, empty) in [
            (0, 0, 0, 1),
            (1, 1, 0, 1),
            (2, 2, 0, 1),
            (3, 0, 1, 0),
            (4, 1, 1, 0),
            (5, 2, 1, 0),
            (6, 0, 0, 1),
        ] {
            assert_eq!(Ring::slot(ticket), slot);
            assert_eq!(Ring::full_phase(ticket), full);
            assert_eq!(Ring::empty_phase(ticket), empty);
        }
    }
    /// A new work item must retain the prior ring position instead of resetting its parity.
    #[test]
    fn persistent_work_keeps_phase_at_a_partial_ring_boundary() {
        /// Stage-count and ordering-policy fixture for checking ticket parity arithmetic.
        type Ring = TwoCtaRing<15, AsyncCompletion>;
        // Two work items consume 17 stages each without restarting the ring.
        for (ticket, slot, full, empty) in [(17, 2, 1, 0), (34, 4, 0, 1)] {
            assert_eq!(Ring::slot(ticket), slot);
            assert_eq!(Ring::full_phase(ticket), full);
            assert_eq!(Ring::empty_phase(ticket), empty);
        }
    }
    /// A one-slot ring must alternate full and empty parities on each ticket.
    #[test]
    fn one_stage_ring_flips_phase_on_every_ticket() {
        /// Stage-count and ordering-policy fixture for checking ticket parity arithmetic.
        type Ring = TwoCtaRing<1, TensorMemory>;
        for (ticket, full, empty) in [(0, 0, 1), (1, 1, 0), (2, 0, 1), (3, 1, 0)] {
            assert_eq!(Ring::slot(ticket), 0);
            assert_eq!(Ring::full_phase(ticket), full);
            assert_eq!(Ring::empty_phase(ticket), empty);
        }
    }
}

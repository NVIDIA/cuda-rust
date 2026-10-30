/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit warp-level compiler facts built from supported CUDA operations.
//!
//! This module carries the `cute.arch` names of CuTeDSL over the PTX-named
//! ctm-rs layer, mirroring `cute.arch` over the NVVM primitives in CuTeDSL.
//! One primitives-layer name lives here as well, [`fence_sync_restrict`],
//! because its order is a type and the DSL has no `cute.arch` form of it.
//! Every forwarder is an `#[inline(always)]` call to the existing ctm-rs
//! primitive: it adds no instruction, no storage and no implicit election or
//! synchronization, and cute-rs never reaches past ctm-rs to the raw device
//! operations. Argument order follows `cute.arch`,
//! the value first and the participation mask last; Rust has no default
//! arguments, so the mask is spelled out (`u32::MAX` for the full warp).
//! Safety follows the primitive: a forwarder is `unsafe` when the function it
//! calls is, under the same contract.

use ctm_rs::mbarrier::{BarrierToken, ManagedBarrier, Ready, TmaBarrier};
use ctm_rs::memory::{ClusterSharedAddress, SharedAddress};

/// Return a warp-uniform value in a form that can favor uniform-register use.
///
/// This is CuTeDSL's `cute.arch.make_warp_uniform`: a full-warp shuffle
/// broadcasts lane zero's value. Pass the returned value to subsequent
/// calculations. The compiler may then keep those calculations in uniform
/// registers; this is an optimization opportunity, not a register-allocation
/// guarantee.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently with equal
/// `value` arguments. Do not call from an elected-lane or partially active
/// branch. This is an identity operation only under that uniformity promise;
/// it does not check or repair a divergent value.
#[inline(always)]
pub unsafe fn make_warp_uniform(value: u32) -> u32 {
    // SAFETY: the caller supplies the full-warp convergence contract above.
    unsafe { ctm_rs::warp::shuffle_sync(u32::MAX, value, 0) }
}

/// Sum one FP32 contribution from each of the 32 lanes of a warp.
///
/// This is CuTeDSL's `cute.arch.warp_reduction_sum` for FP32. Every lane
/// receives the sum. Uses the existing CUDA butterfly reduction, with FP32
/// additions in XOR-offset order 16, 8, 4, 2, 1. The result need not equal a
/// sequential sum. No shared memory or block barrier is required. Inactive
/// logical elements must contribute zero while their lanes still participate.
/// This is a device operation, not a host reduction.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently. The block
/// must contain complete warps. Rust's type system cannot prove convergence;
/// do not invoke this in a lane-dependent branch or after a partial-warp exit.
#[must_use]
#[inline(always)]
pub unsafe fn warp_reduction_sum_f32(value: f32) -> f32 {
    // SAFETY: every lane participates under this function's caller contract.
    unsafe { ctm_rs::warp::reduce_sum_f32(value) }
}

/// Approximate base-two exponential, preserving the device instruction semantics.
#[inline(always)]
pub fn exp2_f32(value: f32) -> f32 {
    ctm_rs::float::ex2_approx_f32(value)
}

/// Approximate base-two exponential with subnormal flushing (`ex2.approx.ftz.f32`).
///
/// This is CuTeDSL's `cute.math.exp2(x, fastmath=True)` used in SM100 online
/// softmax and output-correction loops.
#[must_use]
#[inline(always)]
pub fn exp2_ftz_f32(value: f32) -> f32 {
    ctm_rs::float::ex2_approx_ftz_f32(value)
}

/// Approximate base-two logarithm (`lg2.approx.f32`).
///
/// This is CuTeDSL's `cute.math.log2(x, fastmath=True)` used to compute
/// base-two log-sum-exp statistics in SM100 attention.
#[must_use]
#[inline(always)]
pub fn log2_f32(value: f32) -> f32 {
    ctm_rs::float::lg2_approx_f32(value)
}

/// Approximate reciprocal with the device instruction's flush-to-zero behavior.
#[inline(always)]
pub fn reciprocal_ftz_f32(value: f32) -> f32 {
    ctm_rs::float::rcp_approx_ftz_f32(value)
}

/// Approximate reciprocal square root without changing surrounding arithmetic.
#[inline(always)]
pub fn rsqrt_f32(value: f32) -> f32 {
    ctm_rs::float::rsqrt_approx_f32(value)
}

/// Approximate sine; callers retain responsibility for argument-range reduction.
#[inline(always)]
pub fn sin_f32(value: f32) -> f32 {
    ctm_rs::float::sin_approx_f32(value)
}

/// Maximum across one full warp, using ordered comparisons and XOR offsets.
///
/// This is CuTeDSL's `cute.arch.warp_reduction_max` for FP32. The comparison
/// selects the right value when `left > right` is false; it deliberately
/// retains this behavior for NaNs and signed zero, so the tree stays here
/// instead of forwarding to a primitive with a different NaN rule.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[inline(always)]
pub unsafe fn warp_reduction_max_f32(mut value: f32) -> f32 {
    let mut offset = 16;
    while offset != 0 {
        let other = ctm_rs::warp::shuffle_xor_f32(value, offset);
        value = if value > other { value } else { other };
        offset /= 2;
    }
    value
}

/// Sum independent groups of four neighboring lanes, with XOR offsets 1 then 2.
///
/// This tree is intentionally different from the full-warp reduction tree.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[inline(always)]
pub unsafe fn quad_sum_f32(mut value: f32) -> f32 {
    value += ctm_rs::warp::shuffle_xor_f32(value, 1);
    value += ctm_rs::warp::shuffle_xor_f32(value, 2);
    value
}

/// A named CTA barrier with a statically fixed participant count.
///
/// This is CuTeDSL's `pipeline.NamedBarrier(barrier_id, num_threads)` with
/// both values as const generics; [`NamedBarrier::sync`] is its
/// `arrive_and_wait`. This marker describes the synchronization operation,
/// not an owned hardware resource. Callers must keep barrier phases and their
/// participating threads consistent; distinct logical barriers must use
/// distinct IDs.
pub struct NamedBarrier<const ID: u32, const THREADS: u32>;

impl<const ID: u32, const THREADS: u32> NamedBarrier<ID, THREADS> {
    /// Wait for all participating threads and make their shared writes visible.
    ///
    /// # Safety
    /// Exactly `THREADS` threads in the same CTA must execute matching barrier
    /// phases convergently. They must form complete warps, and no other logical
    /// barrier may concurrently reuse `ID`. No participant may exit early.
    #[inline(always)]
    pub unsafe fn sync() {
        const {
            assert!(ID < 16 && THREADS > 0 && THREADS <= 1024 && THREADS.is_multiple_of(32));
        }
        // SAFETY: the caller supplies the fixed complete-warp participant set.
        unsafe { ctm_rs::barrier::barrier_cta_sync(ID, THREADS) };
    }
}

// Thread, block and grid coordinates.

/// Return the thread index within the CTA as `(x, y, z)`.
///
/// This is `cute.arch.thread_idx()`; it reads `%tid.x`, `%tid.y` and `%tid.z`
/// through [`ctm_rs::thread`].
#[must_use]
#[inline(always)]
pub fn thread_idx() -> (u32, u32, u32) {
    (
        ctm_rs::thread::threadIdx_x(),
        ctm_rs::thread::threadIdx_y(),
        ctm_rs::thread::threadIdx_z(),
    )
}

/// Return the CTA index within the grid as `(x, y, z)`.
///
/// This is `cute.arch.block_idx()`; it reads `%ctaid.x`, `%ctaid.y` and
/// `%ctaid.z` through [`ctm_rs::thread`].
#[must_use]
#[inline(always)]
pub fn block_idx() -> (u32, u32, u32) {
    (
        ctm_rs::thread::blockIdx_x(),
        ctm_rs::thread::blockIdx_y(),
        ctm_rs::thread::blockIdx_z(),
    )
}

/// Return the number of CTAs in each grid dimension as `(x, y, z)`.
///
/// This is `cute.arch.grid_dim()`; it reads `%nctaid.x`, `%nctaid.y` and
/// `%nctaid.z` through [`ctm_rs::thread`].
#[must_use]
#[inline(always)]
pub fn grid_dim() -> (u32, u32, u32) {
    (
        ctm_rs::thread::gridDim_x(),
        ctm_rs::thread::gridDim_y(),
        ctm_rs::thread::gridDim_z(),
    )
}

/// Return the number of threads in each CTA dimension as `(x, y, z)`.
///
/// This is `cute.arch.block_dim()`; it reads `%ntid.x`, `%ntid.y` and
/// `%ntid.z` through [`ctm_rs::thread`].
#[must_use]
#[inline(always)]
pub fn block_dim() -> (u32, u32, u32) {
    (
        ctm_rs::thread::blockDim_x(),
        ctm_rs::thread::blockDim_y(),
        ctm_rs::thread::blockDim_z(),
    )
}

/// Return the logical warp index within the CTA, marked warp-uniform.
///
/// This is `cute.arch.warp_idx()`: the block-linear thread index
/// (`x + y * dim_x + z * dim_x * dim_y`) divided by 32, passed through
/// [`make_warp_uniform`]. For a one-dimensional block this is
/// `threadIdx.x / 32`. Hardware warps are formed from consecutive linear
/// thread indices, so the quotient is the same in every lane of a warp.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently, as
/// [`make_warp_uniform`] requires. Do not call from an elected-lane or
/// partially active branch.
#[must_use]
#[inline(always)]
pub unsafe fn warp_idx() -> u32 {
    let (x, y, z) = thread_idx();
    let dim_x = ctm_rs::thread::blockDim_x();
    let dim_y = ctm_rs::thread::blockDim_y();
    let linear = x + y * dim_x + z * dim_x * dim_y;
    // SAFETY: the caller supplies the full-warp convergence contract above.
    unsafe { make_warp_uniform(linear / 32) }
}

/// Return this thread's lane index within its warp (`%laneid`).
///
/// This is `cute.arch.lane_idx()` over [`ctm_rs::warp::lane_id`].
#[must_use]
#[inline(always)]
pub fn lane_idx() -> u32 {
    ctm_rs::warp::lane_id()
}

// Warp election and synchronization.

/// Elect one lane of the full warp and return whether this lane won.
///
/// This is the predicate form of `cute.arch.elect_one()`, over
/// [`ctm_rs::warp::is_elected_sync`] with the full 32-lane mask (`elect.sync`,
/// sm_90+). All 32 lanes must be converged at the call. Use
/// [`ctm_rs::warp::is_elected_sync`] directly for a partial mask.
#[must_use]
#[inline(always)]
pub fn elect_one_sync() -> bool {
    ctm_rs::warp::is_elected_sync(u32::MAX)
}

/// Synchronize the warp lanes named in `mask` (`bar.warp.sync`).
///
/// This is `cute.arch.sync_warp(mask)` over [`ctm_rs::warp::bar_warp_sync`]; pass
/// `u32::MAX` for the full warp.
///
/// # Safety
/// The caller's lane must be included in `mask`. Every lane named in `mask`
/// must reach this call with the same mask. All accesses ordered by the barrier
/// must satisfy their allocation contracts.
#[inline(always)]
pub unsafe fn sync_warp(mask: u32) {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::bar_warp_sync(mask) }
}

/// Synchronize `number_of_threads` CTA threads at the named barrier `barrier_id`.
///
/// This is `cute.arch.barrier(barrier_id, number_of_threads)` over
/// [`ctm_rs::barrier::barrier_cta_sync`] (`barrier.cta.sync`). [`NamedBarrier`]
/// fixes both values at compile time; the aligned form stays in ctm-rs as
/// `barrier_cta_sync_aligned`.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier(barrier_id: u32, number_of_threads: u32) {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::barrier::barrier_cta_sync(barrier_id, number_of_threads) }
}

/// Arrive at the named barrier `barrier_id` without waiting.
///
/// This is `cute.arch.barrier_arrive(barrier_id, number_of_threads)` over
/// [`ctm_rs::barrier::barrier_cta_arrive`] (`barrier.cta.arrive`, the form
/// that admits a subset of the CTA's warps); the aligned form stays in ctm-rs
/// as `barrier_cta_arrive_aligned`.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_arrive(barrier_id: u32, number_of_threads: u32) {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::barrier::barrier_cta_arrive(barrier_id, number_of_threads) }
}

// Mbarrier transactions.

/// Publish preceding mbarrier initialization at cluster scope.
///
/// This is `cute.arch.mbarrier_init_fence()` over
/// [`ctm_rs::fence::fence_mbarrier_init`]
/// (`fence.mbarrier_init.release.cluster`).
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`ctm_rs::fence::fence_mbarrier_init`].
#[inline(always)]
pub unsafe fn mbarrier_init_fence() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::fence::fence_mbarrier_init() }
}

/// Arrive once on a TMA barrier and add the expected transaction bytes.
///
/// This is `cute.arch.mbarrier_arrive_and_expect_tx(mbar, bytes)` over the typed
/// [`ctm_rs::mbarrier::arrive_expect_tx`]; the arrival decrement is one and the
/// hardware phase token is returned.
///
/// # Safety
/// The issuing participant must be expected, and `bytes` must match the
/// asynchronous transfers completing against this barrier phase.
#[inline(always)]
pub unsafe fn mbarrier_arrive_and_expect_tx<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::mbarrier::arrive_expect_tx(bar, bytes) }
}

// Sync-restricted fences.

/// Seals [`MemOrder`] to the orders the intrinsic catalog can emit.
mod sealed {
    /// Implemented by the order markers of this module only.
    pub trait MemOrder {}
}

/// Memory order of a sync-restricted fence, fixed at compile time.
///
/// This is the `order` argument of CuTeDSL's `fence_sync_restrict(order)`,
/// a `MemOrder` string enum (`"acquire"` or `"release"`) in the DSL and a
/// marker type here, so the instruction is chosen when the kernel compiles.
/// The trait is sealed and [`Acquire`] is its only implementor: the intrinsic
/// catalog has no `fence.release.sync_restrict::shared::cta.cluster` entry
/// yet, so the release order is not offered until that entry is added to the
/// catalog and bound in cuda-device.
pub trait MemOrder: sealed::MemOrder {
    /// Issue the sync-restricted fence in this order at cluster scope.
    ///
    /// # Safety
    /// Pair this with a compatible cluster-scoped synchronization protocol.
    /// This does not substitute for a proxy fence or asynchronous completion
    /// wait.
    unsafe fn fence_sync_restrict();
}

/// The acquire order, `MemOrder.ACQUIRE` in CuTeDSL.
///
/// Its fence is `fence.acquire.sync_restrict::shared::cluster.cluster`: it
/// acquires cluster-shared memory at cluster scope and leaves global memory
/// unordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acquire;

impl sealed::MemOrder for Acquire {}

impl MemOrder for Acquire {
    #[inline(always)]
    unsafe fn fence_sync_restrict() {
        // SAFETY: forwarded under the caller contract on the trait method.
        unsafe { ctm_rs::fence::fence_sync_restrict_acquire() }
    }
}

/// Fence cluster-shared memory at cluster scope in `Order`, excluding global
/// memory.
///
/// This is CuTeDSL's `fence_sync_restrict(order)` with the order as a type
/// parameter: `fence_sync_restrict::<Acquire>()` is
/// `fence.acquire.sync_restrict::shared::cluster.cluster` over
/// [`ctm_rs::fence::fence_sync_restrict_acquire`]. The DSL defines the
/// function in its primitives layer (`nvvm.fence_sync_restrict`) rather than
/// in `cute.arch`; cute-rs keeps it beside the other fences here.
///
/// # Safety
/// Pair this with a compatible cluster-scoped synchronization protocol. This
/// does not substitute for a proxy fence or asynchronous completion wait.
#[inline(always)]
pub unsafe fn fence_sync_restrict<Order: MemOrder>() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { Order::fence_sync_restrict() }
}

// Cluster membership and barriers.

/// Arrive at the cluster barrier with release ordering.
///
/// This is `cute.arch.cluster_arrive()` over
/// [`ctm_rs::cluster::barrier_cluster_arrive`]; the aligned form stays in
/// ctm-rs as `barrier_cluster_arrive_aligned`.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait.
#[inline(always)]
pub unsafe fn cluster_arrive() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::cluster::barrier_cluster_arrive() }
}

/// Arrive at the cluster barrier without a release guarantee.
///
/// This is `cute.arch.cluster_arrive_relaxed()` over
/// [`ctm_rs::cluster::barrier_cluster_arrive_relaxed`]; the aligned form stays
/// in ctm-rs as `barrier_cluster_arrive_relaxed_aligned`.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute the matching wait. Publish earlier memory accesses with the
/// required cluster-scope fence first.
#[inline(always)]
pub unsafe fn cluster_arrive_relaxed() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::cluster::barrier_cluster_arrive_relaxed() }
}

/// Wait at the cluster barrier with acquire ordering.
///
/// This is `cute.arch.cluster_wait()` over
/// [`ctm_rs::cluster::barrier_cluster_wait`]; the aligned form stays in ctm-rs
/// as `barrier_cluster_wait_aligned`.
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion,
/// then execute this matching wait.
#[inline(always)]
pub unsafe fn cluster_wait() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::cluster::barrier_cluster_wait() }
}

/// Return the linear rank of this CTA within its cluster.
///
/// This is `cute.arch.block_idx_in_cluster()` over
/// [`ctm_rs::cluster::cluster_ctarank`], the value `%cluster_ctarank` holds, in
/// `0..cluster_size`.
#[must_use]
#[inline(always)]
pub fn block_idx_in_cluster() -> u32 {
    ctm_rs::cluster::cluster_ctarank()
}

/// Return this CTA's index within its cluster as `(x, y, z)`.
///
/// This is `cute.arch.block_in_cluster_idx()`; it reads `%cluster_ctaid.x`,
/// `%cluster_ctaid.y` and `%cluster_ctaid.z` through [`ctm_rs::cluster`].
/// [`block_idx_in_cluster`] is the linear rank of the same CTA
/// (`%cluster_ctarank`); [`cluster_idx`] is the cluster's own `(x, y, z)`
/// within the grid.
#[must_use]
#[inline(always)]
pub fn block_in_cluster_idx() -> (u32, u32, u32) {
    (
        ctm_rs::cluster::cluster_ctaidX(),
        ctm_rs::cluster::cluster_ctaidY(),
        ctm_rs::cluster::cluster_ctaidZ(),
    )
}

/// Return the number of CTAs in each cluster dimension as `(x, y, z)`.
///
/// This is `cute.arch.block_in_cluster_dim()`; it reads `%cluster_nctaid.x`,
/// `%cluster_nctaid.y` and `%cluster_nctaid.z` through [`ctm_rs::cluster`].
/// The compiler's own `cluster_dim` vocabulary, the `#[cluster_launch]`
/// attribute and the intrinsic family behind it, means this value, CTAs per
/// cluster; CuTeDSL's `cute.arch.cluster_dim()` is [`cluster_dim`], the
/// number of clusters in the grid.
#[must_use]
#[inline(always)]
pub fn block_in_cluster_dim() -> (u32, u32, u32) {
    (
        ctm_rs::cluster::cluster_nctaidX(),
        ctm_rs::cluster::cluster_nctaidY(),
        ctm_rs::cluster::cluster_nctaidZ(),
    )
}

/// Return this CTA's cluster index within the grid as `(x, y, z)`.
///
/// This is `cute.arch.cluster_idx()`; it reads `%clusterid.x`, `%clusterid.y`
/// and `%clusterid.z` through [`ctm_rs::cluster`]. The linear form of the
/// same index stays `ctm_rs::cluster::cluster_idx()`, and
/// [`block_in_cluster_idx`] is the CTA's index within that cluster.
#[must_use]
#[inline(always)]
pub fn cluster_idx() -> (u32, u32, u32) {
    (
        ctm_rs::cluster::cluster_idx_x(),
        ctm_rs::cluster::cluster_idx_y(),
        ctm_rs::cluster::cluster_idx_z(),
    )
}

/// Return the number of clusters in each grid dimension as `(x, y, z)`.
///
/// This is `cute.arch.cluster_dim()`; it reads `%nclusterid.x`,
/// `%nclusterid.y` and `%nclusterid.z` through [`ctm_rs::cluster`]. The
/// compiler's own `cluster_dim` vocabulary, the `#[cluster_launch]` attribute
/// of cuda-macros and the cuda-intrinsics-gen family behind it, means CTAs
/// per cluster, which is [`block_in_cluster_dim`] here; the DSL's
/// `cluster_dim` means clusters in the grid.
#[must_use]
#[inline(always)]
pub fn cluster_dim() -> (u32, u32, u32) {
    (
        ctm_rs::cluster::cluster_grid_dim_x(),
        ctm_rs::cluster::cluster_grid_dim_y(),
        ctm_rs::cluster::cluster_grid_dim_z(),
    )
}

/// Map local shared storage to the same offset in cluster peer `rank`.
///
/// This is `cute.arch.map_dsmem_ptr(ptr, rank)` over
/// [`ctm_rs::cluster::mapa`].
///
/// # Safety
/// `rank` must name an active CTA in this cluster with matching live storage.
/// Remote initialization, readiness and lifetime must be coordinated explicitly.
#[inline(always)]
pub unsafe fn map_dsmem_ptr<T>(
    address: SharedAddress<'_, T>,
    rank: u32,
) -> ClusterSharedAddress<'_, T> {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::cluster::mapa(address, rank) }
}

// Register budget.

/// Decrease the warpgroup's register limit to the immediate `N`.
///
/// This is `cute.arch.setmaxregister_decrease(n)` over
/// [`ctm_rs::thread::setmaxnreg_dec`]; `N` must be a multiple of eight in
/// `24..=256`.
///
/// # Safety
/// Every thread in the warpgroup must execute the same operation and count
/// on supported hardware. The current register limit must be at least `N`.
/// Synchronize the warpgroup before another register-limit adjustment.
/// Subsequent code must respect the selected limit.
#[inline(always)]
pub unsafe fn setmaxregister_decrease<const N: u32>()
where
    (): ctm_rs::thread::ValidRegisterCount<N>,
{
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::thread::setmaxnreg_dec::<N>() }
}

/// Increase the warpgroup's register limit to the immediate `N`.
///
/// This is `cute.arch.setmaxregister_increase(n)` over
/// [`ctm_rs::thread::setmaxnreg_inc`]; `N` must be a multiple of eight in
/// `24..=256`.
///
/// # Safety
/// Every thread in the warpgroup must execute the same operation and count
/// on supported hardware. The current register limit must be at most `N`.
/// Synchronize the warpgroup before another register-limit adjustment.
/// Coordinate releases by other warpgroups to avoid waiting indefinitely
/// for unavailable registers.
#[inline(always)]
pub unsafe fn setmaxregister_increase<const N: u32>()
where
    (): ctm_rs::thread::ValidRegisterCount<N>,
{
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::thread::setmaxnreg_inc::<N>() }
}

// Bulk-copy groups.

/// Commit this thread's pending bulk copies as one group; does not wait.
///
/// This is `cute.arch.cp_async_bulk_commit_group()` over
/// [`ctm_rs::tma::cp_async_bulk_commit_group`].
#[inline(always)]
pub fn cp_async_bulk_commit_group() {
    ctm_rs::tma::cp_async_bulk_commit_group()
}

/// Wait until at most `N` committed bulk groups still read their shared sources.
///
/// This is `cute.arch.cp_async_bulk_wait_group(N, read=True)` over
/// [`ctm_rs::tma::cp_async_bulk_wait_group_read`]; `N` is an immediate in `0..=7`. It
/// is not proof that the groups' global writes are visible to other threads.
#[inline(always)]
pub fn cp_async_bulk_wait_group_read<const N: u32>()
where
    (): ctm_rs::tma::ValidWaitCount<N>,
{
    ctm_rs::tma::cp_async_bulk_wait_group_read::<N>()
}

// Tensor memory completion.

/// Complete this thread's previously issued SM100 TMEM loads.
///
/// This is `cute.arch.fence_view_async_tmem_load()` over
/// [`ctm_rs::tcgen05::tmem::load_wait`] (`tcgen05.wait::ld`). No CTA barrier,
/// producer wait or memory/proxy fence is inserted.
///
/// # Safety
/// All 32 warp lanes must execute convergently. The loads must target live,
/// valid TMEM allocations with correctly ordered producers. All tcgen05
/// operations must use one CTA-group mode.
#[inline(always)]
pub unsafe fn fence_view_async_tmem_load() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::tcgen05::tmem::load_wait() }
}

/// Complete this thread's previously issued SM100 TMEM stores.
///
/// This is `cute.arch.fence_view_async_tmem_store()` over
/// [`ctm_rs::tcgen05::tmem::store_wait`] (`tcgen05.wait::st`). No CTA barrier,
/// consumer synchronization or memory/proxy fence is inserted.
///
/// # Safety
/// All 32 warp lanes must execute convergently. The stores must target live,
/// valid TMEM allocations with no conflicting accesses. All tcgen05 operations
/// must use one CTA-group mode.
#[inline(always)]
pub unsafe fn fence_view_async_tmem_store() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::tcgen05::tmem::store_wait() }
}

// Warp votes.

/// Ballot across the lanes in `mask`: bit `k` is set when lane `k` participates
/// and its `predicate` is true.
///
/// This is `cute.arch.vote_ballot_sync(pred, mask)` over
/// [`ctm_rs::warp::vote_sync_ballot`] (`vote.sync.ballot`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn vote_ballot_sync(predicate: bool, mask: u32) -> u32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::vote_sync_ballot(mask, predicate) }
}

/// Return whether `predicate` holds for at least one lane in `mask`.
///
/// This is `cute.arch.vote_any_sync(pred, mask)` over
/// [`ctm_rs::warp::vote_sync_any`] (`vote.sync.any`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn vote_any_sync(predicate: bool, mask: u32) -> bool {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::vote_sync_any(mask, predicate) }
}

/// Return whether `predicate` holds for every lane in `mask`.
///
/// This is `cute.arch.vote_all_sync(pred, mask)` over
/// [`ctm_rs::warp::vote_sync_all`] (`vote.sync.all`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn vote_all_sync(predicate: bool, mask: u32) -> bool {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::vote_sync_all(mask, predicate) }
}

/// Return the mask of the warp's currently active lanes (`activemask`).
///
/// This is `cute.arch.activemask()` over [`ctm_rs::warp::activemask`]. Bit `k`
/// is set when lane `k` has neither exited nor diverged away at this point of
/// execution.
#[must_use]
#[inline(always)]
pub fn activemask() -> u32 {
    ctm_rs::warp::activemask()
}

// Warp shuffles. The DSL's one generic function becomes a `u32` form and an
// `_f32` form, the two widths the preview kernels exchange.

/// Exchange a `u32` with the lane whose index is this lane's XOR `offset`.
///
/// This is `cute.arch.shuffle_sync_bfly(value, offset, mask)` over
/// [`ctm_rs::warp::shfl_sync_bfly`] (`shfl.sync.bfly.b32`); the clamp is the
/// full warp. No memory synchronization is inserted.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller. `offset` must select a source lane within the warp.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_bfly(value: u32, offset: u32, mask: u32) -> u32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_bfly(mask, value, offset) }
}

/// FP32 form of [`shuffle_sync_bfly`] over [`ctm_rs::warp::shfl_sync_bfly_f32`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller. `offset` must select a source lane within the warp.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_bfly_f32(value: f32, offset: u32, mask: u32) -> f32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_bfly_f32(mask, value, offset) }
}

/// Read a `u32` from the lane `offset` below this lane; lanes below `offset`
/// keep their own value.
///
/// This is `cute.arch.shuffle_sync_up(value, offset, mask)` over
/// [`ctm_rs::warp::shfl_sync_up`] (`shfl.sync.up.b32`). No memory
/// synchronization is inserted.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_up(value: u32, offset: u32, mask: u32) -> u32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_up(mask, value, offset) }
}

/// FP32 form of [`shuffle_sync_up`] over [`ctm_rs::warp::shfl_sync_up_f32`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_up_f32(value: f32, offset: u32, mask: u32) -> f32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_up_f32(mask, value, offset) }
}

/// Read a `u32` from the lane `offset` above this lane; the top `offset` lanes
/// keep their own value.
///
/// This is `cute.arch.shuffle_sync_down(value, offset, mask)` over
/// [`ctm_rs::warp::shfl_sync_down`] (`shfl.sync.down.b32`). No memory
/// synchronization is inserted.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_down(value: u32, offset: u32, mask: u32) -> u32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_down(mask, value, offset) }
}

/// FP32 form of [`shuffle_sync_down`] over [`ctm_rs::warp::shfl_sync_down_f32`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`,
/// including the caller.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_down_f32(value: f32, offset: u32, mask: u32) -> f32 {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::warp::shfl_sync_down_f32(mask, value, offset) }
}

// Proxy fences and packed FP32x2 / FP16x2 arithmetic.

/// Publish preceding generic-proxy shared-memory writes to the CTA async proxy.
///
/// This is CuTeDSL's `cute.arch.fence_view_async_shared()` over
/// [`ctm_rs::fence::fence_proxy_async_shared_cta`] (`fence.proxy.async.shared::cta`).
///
/// # Safety
/// Call before signaling or issuing an asynchronous TMA or UMMA operation that
/// reads ordinary shared-memory stores written by this thread.
#[inline(always)]
pub unsafe fn fence_view_async_shared() {
    // SAFETY: forwarded under the caller contract above.
    unsafe { ctm_rs::fence::fence_proxy_async_shared_cta() }
}

/// Two packed `f32` values carried in a 64-bit register pair for `f32x2` instructions.
///
/// Mirrors the tuple register pairs passed to `cute.arch.add_packed_f32x2`,
/// `cute.arch.mul_packed_f32x2` and `cute.arch.fma_packed_f32x2` in CuTeDSL.
pub use ctm_rs::f32x2::PackedF32x2;

/// Pack two `f32` scalars (`lo`, `hi`) into one [`PackedF32x2`] register pair.
///
/// Preserves the exact bit pattern of both lanes without arithmetic conversion.
#[must_use]
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> PackedF32x2 {
    ctm_rs::f32x2::pack_f32x2(lo, hi)
}

/// Extract the low `f32` lane of a [`PackedF32x2`] register pair.
///
/// Preserves the exact bit pattern of the low 32 bits.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_lo(pair: PackedF32x2) -> f32 {
    ctm_rs::f32x2::unpack_f32x2_lo(pair)
}

/// Extract the high `f32` lane of a [`PackedF32x2`] register pair.
///
/// Preserves the exact bit pattern of the high 32 bits.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_hi(pair: PackedF32x2) -> f32 {
    ctm_rs::f32x2::unpack_f32x2_hi(pair)
}

/// Add two [`PackedF32x2`] register pairs lane-by-lane (`add.rn.f32x2`).
///
/// This is CuTeDSL's `cute.arch.add_packed_f32x2(a, b)` over
/// [`ctm_rs::f32x2::add_f32x2`].
#[must_use]
#[inline(always)]
pub fn add_packed_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    ctm_rs::f32x2::add_f32x2(a, b)
}

/// Multiply two [`PackedF32x2`] register pairs lane-by-lane (`mul.rn.f32x2`).
///
/// This is CuTeDSL's `cute.arch.mul_packed_f32x2(a, b)` over
/// [`ctm_rs::f32x2::mul_f32x2`].
#[must_use]
#[inline(always)]
pub fn mul_packed_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    ctm_rs::f32x2::mul_f32x2(a, b)
}

/// Fused multiply-add `a * b + c` on [`PackedF32x2`] register pairs (`fma.rn.f32x2`).
///
/// This is CuTeDSL's `cute.arch.fma_packed_f32x2(a, b, c)` over
/// [`ctm_rs::f32x2::fma_f32x2`].
#[must_use]
#[inline(always)]
pub fn fma_packed_f32x2(a: PackedF32x2, b: PackedF32x2, c: PackedF32x2) -> PackedF32x2 {
    ctm_rs::f32x2::fma_f32x2(a, b, c)
}

/// Convert two `f32` values to packed round-to-nearest-even `f16x2` bits (`cvt.rn.f16x2.f32`).
///
/// `lo` occupies the low 16 bits and `hi` occupies the high 16 bits of the
/// returned `u32`, matching the packed `f16x2` storage expected by SM100 TMEM
/// and shared-memory probability tiles.
#[must_use]
#[inline(always)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    ctm_rs::convert::cvt_f16x2_f32(lo, hi)
}

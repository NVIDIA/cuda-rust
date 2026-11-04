/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit warp-level and architecture primitives (`cutlass.cute.arch`).
//!
//! This module provides the `cute.arch` functions of CuTeDSL over the compiler's
//! `cuda_device` intrinsics, mirroring `cute.arch` over `nvvm` in CuTeDSL.

use core::marker::PhantomData;
pub use cuda_device::barrier::Barrier;
use cuda_device::barrier::{BarrierToken, ManagedBarrier, Ready, TmaBarrier};

/// Non-owning typed shared-memory pointer within the calling CTA (`addrspace(3)`).
#[derive(Clone, Copy)]
pub struct SharedAddress<'a, T> {
    /// Raw pointer into the calling CTA's shared memory.
    ptr: *mut T,
    /// Retains the borrow lifetime and element type `T`.
    _marker: PhantomData<&'a T>,
}

impl<'a, T> SharedAddress<'a, T> {
    /// Wrap a raw CTA-shared pointer without dereferencing it.
    ///
    /// # Safety
    /// `ptr` must address live CTA-shared storage for `'a`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Return the underlying raw CTA-shared pointer.
    #[must_use]
    #[inline(always)]
    pub const fn as_raw(self) -> *mut T {
        self.ptr
    }
}

/// Non-owning typed distributed-shared-memory (`DSMEM`) pointer in a peer cluster CTA.
#[derive(Clone, Copy)]
pub struct ClusterSharedAddress<'a, T> {
    /// Raw pointer mapped into a peer CTA's shared memory via `mapa`.
    ptr: *mut T,
    /// Retains the borrow lifetime and element type `T`.
    _marker: PhantomData<&'a T>,
}

impl<'a, T> ClusterSharedAddress<'a, T> {
    /// Return the underlying mapped peer-CTA shared pointer.
    #[must_use]
    #[inline(always)]
    pub const fn as_raw(self) -> *mut T {
        self.ptr
    }
}

/// Return a warp-uniform value in a form that can favor uniform-register use.
///
/// This is CuTeDSL's `cute.arch.make_warp_uniform`: a full-warp shuffle
/// broadcasts lane zero's value.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently with equal `value` arguments.
#[inline(always)]
pub unsafe fn make_warp_uniform(value: u32) -> u32 {
    cuda_device::warp::shuffle_sync(u32::MAX, value, 0)
}

/// First two butterfly steps (XOR offsets 16 and 8) of a full-warp FP32 sum.
#[inline(always)]
fn warp_upper_sum_f32(mut value: f32) -> f32 {
    value += cuda_device::warp::shuffle_xor_f32(value, 16);
    value + cuda_device::warp::shuffle_xor_f32(value, 8)
}

/// First two butterfly steps (XOR offsets 16 and 8) of a full-warp FP32 maximum.
#[inline(always)]
fn warp_upper_max_f32(mut value: f32) -> f32 {
    let other16 = cuda_device::warp::shuffle_xor_f32(value, 16);
    value = if value > other16 { value } else { other16 };
    let other8 = cuda_device::warp::shuffle_xor_f32(value, 8);
    if value > other8 { value } else { other8 }
}

/// Sum one FP32 contribution from each of the 32 lanes of a warp (`cute.arch.warp_reduction_sum`).
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently.
#[must_use]
#[inline(always)]
pub unsafe fn warp_reduction_sum_f32(value: f32) -> f32 {
    let mut value = warp_upper_sum_f32(value);
    value += cuda_device::warp::shuffle_xor_f32(value, 4);
    value += cuda_device::warp::shuffle_xor_f32(value, 2);
    value + cuda_device::warp::shuffle_xor_f32(value, 1)
}

/// Approximate base-two exponential, preserving the device instruction semantics (`ex2.approx.f32`).
#[inline(always)]
pub fn exp2_f32(value: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(value)
}

/// Approximate base-two exponential with subnormal flushing (`ex2.approx.ftz.f32`).
#[must_use]
#[inline(always)]
pub fn exp2_ftz_f32(value: f32) -> f32 {
    cuda_device::float::ex2_approx_ftz_f32(value)
}

/// Approximate base-two logarithm (`lg2.approx.f32`).
#[must_use]
#[inline(always)]
pub fn log2_f32(value: f32) -> f32 {
    cuda_device::float::lg2_approx_f32(value)
}

/// Select the larger of two FP32 values using the hardware `fmax` instruction (`cute.arch.fmax`).
#[must_use]
#[inline(always)]
pub fn fmax_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Approximate reciprocal with flush-to-zero behavior (`rcp.approx.ftz.f32`).
#[inline(always)]
pub fn reciprocal_ftz_f32(value: f32) -> f32 {
    cuda_device::float::rcp_approx_ftz_f32(value)
}

/// Approximate reciprocal square root (`rsqrt.approx.f32`).
#[inline(always)]
pub fn rsqrt_f32(value: f32) -> f32 {
    cuda_device::float::rsqrt_approx_f32(value)
}

/// Approximate sine (`sin.approx.f32`).
#[inline(always)]
pub fn sin_f32(value: f32) -> f32 {
    cuda_device::float::sin_approx_f32(value)
}

/// Maximum across one full warp (`cute.arch.warp_reduction_max`).
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[inline(always)]
pub unsafe fn warp_reduction_max_f32(value: f32) -> f32 {
    let mut value = warp_upper_max_f32(value);
    let other4 = cuda_device::warp::shuffle_xor_f32(value, 4);
    value = if value > other4 { value } else { other4 };
    let other2 = cuda_device::warp::shuffle_xor_f32(value, 2);
    value = if value > other2 { value } else { other2 };
    let other1 = cuda_device::warp::shuffle_xor_f32(value, 1);
    if value > other1 { value } else { other1 }
}

/// Sum independent groups of four neighboring lanes, with XOR offsets 1 then 2.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[inline(always)]
pub unsafe fn quad_sum_f32(mut value: f32) -> f32 {
    value += cuda_device::warp::shuffle_xor_f32(value, 1);
    value += cuda_device::warp::shuffle_xor_f32(value, 2);
    value
}

/// Reduce a maximum across each 4-lane thread quad using XOR offsets 2 then 1.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[must_use]
#[inline(always)]
pub unsafe fn threadquad_reduce_max_f32(mut value: f32) -> f32 {
    value = fmax_f32(value, cuda_device::warp::shuffle_xor_f32(value, 2));
    value = fmax_f32(value, cuda_device::warp::shuffle_xor_f32(value, 1));
    value
}

/// Sum values across each 4-lane thread quad using XOR offsets 2 then 1.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[must_use]
#[inline(always)]
pub unsafe fn threadquad_reduce_sum_f32(mut value: f32) -> f32 {
    value += cuda_device::warp::shuffle_xor_f32(value, 2);
    value += cuda_device::warp::shuffle_xor_f32(value, 1);
    value
}

/// Commit this thread's pending `cp.async` copies as one group (`cute.arch.cp_async_commit_group`).
///
/// # Safety
/// Copies must satisfy their source/destination lifetime and alignment contracts.
#[inline(always)]
pub unsafe fn cp_async_commit_group() {
    unsafe { cuda_device::async_copy::cp_async_commit_group() }
}

/// Wait until at most `N` committed `cp.async` groups remain pending (`cute.arch.cp_async_wait_group`).
///
/// # Safety
/// Access only destinations whose copy groups this wait completes.
#[inline(always)]
pub unsafe fn cp_async_wait_group<const N: u32>() {
    const { assert!(N <= 7, "cp.async wait count must be in 0..=7") };
    unsafe { cuda_device::async_copy::cp_async_wait_group(N) }
}

/// A named CTA barrier with a statically fixed participant count (`pipeline.NamedBarrier`).
pub struct NamedBarrier<const ID: u32, const THREADS: u32>;

impl<const ID: u32, const THREADS: u32> NamedBarrier<ID, THREADS> {
    /// Wait for all participating threads and make their shared writes visible.
    ///
    /// # Safety
    /// Exactly `THREADS` threads in the same CTA must execute matching barrier phases convergently.
    #[inline(always)]
    pub unsafe fn sync() {
        const {
            assert!(ID < 16 && THREADS > 0 && THREADS <= 1024 && THREADS.is_multiple_of(32));
        }
        unsafe { cuda_device::barrier::barrier_cta_sync(ID, THREADS) };
    }
}

/// Return the thread index within the CTA as `(x, y, z)` (`cute.arch.thread_idx`).
#[must_use]
#[inline(always)]
pub fn thread_idx() -> (u32, u32, u32) {
    (
        cuda_device::thread::threadIdx_x(),
        cuda_device::thread::threadIdx_y(),
        cuda_device::thread::threadIdx_z(),
    )
}

/// Return the CTA index within the grid as `(x, y, z)` (`cute.arch.block_idx`).
#[must_use]
#[inline(always)]
pub fn block_idx() -> (u32, u32, u32) {
    (
        cuda_device::thread::blockIdx_x(),
        cuda_device::thread::blockIdx_y(),
        cuda_device::thread::blockIdx_z(),
    )
}

/// Return the number of CTAs in each grid dimension as `(x, y, z)` (`cute.arch.grid_dim`).
#[must_use]
#[inline(always)]
pub fn grid_dim() -> (u32, u32, u32) {
    (
        cuda_device::thread::gridDim_x(),
        cuda_device::thread::gridDim_y(),
        cuda_device::thread::gridDim_z(),
    )
}

/// Return the number of threads in each CTA dimension as `(x, y, z)` (`cute.arch.block_dim`).
#[must_use]
#[inline(always)]
pub fn block_dim() -> (u32, u32, u32) {
    (
        cuda_device::thread::blockDim_x(),
        cuda_device::thread::blockDim_y(),
        cuda_device::thread::blockDim_z(),
    )
}

/// Return the logical warp index within the CTA, marked warp-uniform (`cute.arch.warp_idx`).
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently.
#[must_use]
#[inline(always)]
pub unsafe fn warp_idx() -> u32 {
    let (x, y, z) = thread_idx();
    let dim_x = cuda_device::thread::blockDim_x();
    let dim_y = cuda_device::thread::blockDim_y();
    let linear = x + y * dim_x + z * dim_x * dim_y;
    unsafe { make_warp_uniform(linear / 32) }
}

/// Return this thread's lane index within its warp (`cute.arch.lane_idx`).
#[must_use]
#[inline(always)]
pub fn lane_idx() -> u32 {
    cuda_device::warp::lane_id()
}

/// Elect one lane of the full warp and return whether this lane won (`cute.arch.elect_one`).
#[must_use]
#[inline(always)]
pub fn elect_one_sync() -> bool {
    cuda_device::warp::is_elected_sync(u32::MAX)
}

/// Synchronize the warp lanes named in `mask` (`cute.arch.sync_warp`).
///
/// # Safety
/// The caller's lane must be included in `mask` and every lane named in `mask` must reach this call.
#[inline(always)]
pub unsafe fn sync_warp(mask: u32) {
    cuda_device::warp::sync_mask(mask)
}

/// Synchronize `number_of_threads` CTA threads at the named barrier `barrier_id` (`cute.arch.barrier`).
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier(barrier_id: u32, number_of_threads: u32) {
    unsafe { cuda_device::barrier::barrier_cta_sync(barrier_id, number_of_threads) }
}

/// Arrive at the named barrier `barrier_id` without waiting (`cute.arch.barrier_arrive`).
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_arrive(barrier_id: u32, number_of_threads: u32) {
    unsafe { cuda_device::barrier::barrier_cta_arrive(barrier_id, number_of_threads) }
}

/// Publish preceding mbarrier initialization at cluster scope (`cute.arch.mbarrier_init_fence`).
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of `fence.mbarrier_init.release.cluster`.
#[inline(always)]
pub unsafe fn mbarrier_init_fence() {
    unsafe { cuda_device::barrier::fence_mbarrier_init_release_cluster() }
}

/// Arrive once on a TMA barrier and add the expected transaction bytes (`cute.arch.mbarrier_arrive_and_expect_tx`).
///
/// # Safety
/// The issuing participant must be expected, and `bytes` must match the asynchronous transfers.
#[inline(always)]
pub unsafe fn mbarrier_arrive_and_expect_tx<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    bar.arrive_expect_tx(bytes)
}

/// Seals [`MemOrder`], [`ValidRegisterCount`], and [`ValidWaitCount`].
mod sealed {
    /// Implemented by the order markers of this module only.
    pub trait MemOrder {}
    /// Implemented for valid `setmaxnreg` counts (`24..=256`, multiple of 8).
    pub trait RegisterCount<const N: u32> {}
    /// Implemented for valid bulk wait counts (`0..=7`).
    pub trait WaitCount<const N: u32> {}
}

/// Memory order of a sync-restricted fence, fixed at compile time.
pub trait MemOrder: sealed::MemOrder {
    /// Issue the sync-restricted fence in this order at cluster scope.
    ///
    /// # Safety
    /// Pair this with a compatible cluster-scoped synchronization protocol.
    unsafe fn fence_sync_restrict();
}

/// The acquire order (`MemOrder.ACQUIRE` in CuTeDSL).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acquire;

impl sealed::MemOrder for Acquire {}

impl MemOrder for Acquire {
    #[inline(always)]
    unsafe fn fence_sync_restrict() {
        unsafe { cuda_device::barrier::fence_acquire_shared_cluster_cluster() }
    }
}

/// Fence cluster-shared memory at cluster scope in `Order`, excluding global memory.
///
/// # Safety
/// Pair this with a compatible cluster-scoped synchronization protocol.
#[inline(always)]
pub unsafe fn fence_sync_restrict<Order: MemOrder>() {
    unsafe { Order::fence_sync_restrict() }
}

/// Arrive at the cluster barrier with release ordering (`cute.arch.cluster_arrive`).
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion, then execute the matching wait.
#[inline(always)]
pub unsafe fn cluster_arrive() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive() }
}

/// Arrive at the cluster barrier without a release guarantee (`cute.arch.cluster_arrive_relaxed`).
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion, then execute the matching wait.
#[inline(always)]
pub unsafe fn cluster_arrive_relaxed() {
    unsafe { cuda_device::cluster::barrier_cluster_arrive_relaxed() }
}

/// Wait at the cluster barrier with acquire ordering (`cute.arch.cluster_wait`).
///
/// # Safety
/// Each non-exited cluster thread must arrive exactly once before completion, then execute this matching wait.
#[inline(always)]
pub unsafe fn cluster_wait() {
    unsafe { cuda_device::cluster::barrier_cluster_wait() }
}

/// Return the linear rank of this CTA within its cluster (`cute.arch.block_idx_in_cluster`).
#[must_use]
#[inline(always)]
pub fn block_idx_in_cluster() -> u32 {
    cuda_device::cluster::block_rank()
}

/// Return this CTA's index within its cluster as `(x, y, z)` (`cute.arch.block_in_cluster_idx`).
#[must_use]
#[inline(always)]
pub fn block_in_cluster_idx() -> (u32, u32, u32) {
    (
        cuda_device::cluster::cluster_ctaidX(),
        cuda_device::cluster::cluster_ctaidY(),
        cuda_device::cluster::cluster_ctaidZ(),
    )
}

/// Return the number of CTAs in each cluster dimension as `(x, y, z)` (`cute.arch.block_in_cluster_dim`).
#[must_use]
#[inline(always)]
pub fn block_in_cluster_dim() -> (u32, u32, u32) {
    (
        cuda_device::cluster::cluster_nctaidX(),
        cuda_device::cluster::cluster_nctaidY(),
        cuda_device::cluster::cluster_nctaidZ(),
    )
}

/// Return this CTA's cluster index within the grid as `(x, y, z)` (`cute.arch.cluster_idx`).
#[must_use]
#[inline(always)]
pub fn cluster_idx() -> (u32, u32, u32) {
    (
        cuda_device::cluster::cluster_idx_x(),
        cuda_device::cluster::cluster_idx_y(),
        cuda_device::cluster::cluster_idx_z(),
    )
}

/// Return the number of clusters in each grid dimension as `(x, y, z)` (`cute.arch.cluster_dim`).
#[must_use]
#[inline(always)]
pub fn cluster_dim() -> (u32, u32, u32) {
    (
        cuda_device::cluster::cluster_grid_dim_x(),
        cuda_device::cluster::cluster_grid_dim_y(),
        cuda_device::cluster::cluster_grid_dim_z(),
    )
}

/// Map local shared storage to the same offset in cluster peer `rank` (`cute.arch.map_dsmem_ptr`).
///
/// # Safety
/// `rank` must name an active CTA in this cluster with matching live storage.
#[inline(always)]
pub unsafe fn map_dsmem_ptr<T>(
    address: SharedAddress<'_, T>,
    rank: u32,
) -> ClusterSharedAddress<'_, T> {
    let ptr = unsafe { cuda_device::cluster::map_shared_rank_mut(address.as_raw(), rank) };
    ClusterSharedAddress {
        ptr,
        _marker: PhantomData,
    }
}

/// Sealed proof that `N` is a valid `setmaxnreg` register count (`24..=256`, multiple of 8).
pub trait ValidRegisterCount<const N: u32>: sealed::RegisterCount<N> {}

/// Implement [`ValidRegisterCount`] for each hardware-supported register limit.
macro_rules! valid_reg_counts {
    ($($n:literal),+) => {
        $(
            impl sealed::RegisterCount<$n> for () {}
            impl ValidRegisterCount<$n> for () {}
        )+
    };
}
valid_reg_counts!(
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184,
    192, 200, 208, 216, 224, 232, 240, 248, 256
);

/// Decrease the warpgroup's register limit to the immediate `N` (`cute.arch.setmaxregister_decrease`).
///
/// # Safety
/// Every thread in the warpgroup must execute the same operation and count on supported hardware.
#[inline(always)]
pub unsafe fn setmaxregister_decrease<const N: u32>()
where
    (): ValidRegisterCount<N>,
{
    unsafe { cuda_device::thread::setmaxnreg_dec::<N>() }
}

/// Increase the warpgroup's register limit to the immediate `N` (`cute.arch.setmaxregister_increase`).
///
/// # Safety
/// Every thread in the warpgroup must execute the same operation and count on supported hardware.
#[inline(always)]
pub unsafe fn setmaxregister_increase<const N: u32>()
where
    (): ValidRegisterCount<N>,
{
    unsafe { cuda_device::thread::setmaxnreg_inc::<N>() }
}

/// Commit this thread's pending bulk copies as one group (`cute.arch.cp_async_bulk_commit_group`).
#[inline(always)]
pub fn cp_async_bulk_commit_group() {
    cuda_device::tma::cp_async_bulk_commit_group()
}

/// Sealed proof that `N` is a valid bulk-copy wait count (`0..=7`).
pub trait ValidWaitCount<const N: u32>: sealed::WaitCount<N> {}

/// Implement [`ValidWaitCount`] for `0..=7`.
macro_rules! valid_wait_counts {
    ($($n:literal),+) => {
        $(
            impl sealed::WaitCount<$n> for () {}
            impl ValidWaitCount<$n> for () {}
        )+
    };
}
valid_wait_counts!(0, 1, 2, 3, 4, 5, 6, 7);

/// Wait until at most `N` committed bulk groups still read their shared sources (`cute.arch.cp_async_bulk_wait_group`).
#[inline(always)]
pub fn cp_async_bulk_wait_group_read<const N: u32>()
where
    (): ValidWaitCount<N>,
{
    cuda_device::tma::cp_async_bulk_wait_group_read(N)
}

/// Complete this thread's previously issued SM100 TMEM loads (`cute.arch.fence_view_async_tmem_load`).
///
/// # Safety
/// All 32 warp lanes must execute convergently.
#[inline(always)]
pub unsafe fn fence_view_async_tmem_load() {
    cuda_device::tcgen05::tcgen05_load_wait()
}

/// Complete this thread's previously issued SM100 TMEM stores (`cute.arch.fence_view_async_tmem_store`).
///
/// # Safety
/// All 32 warp lanes must execute convergently.
#[inline(always)]
pub unsafe fn fence_view_async_tmem_store() {
    cuda_device::tcgen05::tcgen05_store_wait()
}

/// Ballot across the lanes in `mask` (`cute.arch.vote_ballot_sync`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask.
#[must_use]
#[inline(always)]
pub unsafe fn vote_ballot_sync(predicate: bool, mask: u32) -> u32 {
    cuda_device::warp::ballot_sync(mask, predicate)
}

/// Return whether `predicate` holds for at least one lane in `mask` (`cute.arch.vote_any_sync`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask.
#[must_use]
#[inline(always)]
pub unsafe fn vote_any_sync(predicate: bool, mask: u32) -> bool {
    cuda_device::warp::any_sync(mask, predicate)
}

/// Return whether `predicate` holds for every lane in `mask` (`cute.arch.vote_all_sync`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask.
#[must_use]
#[inline(always)]
pub unsafe fn vote_all_sync(predicate: bool, mask: u32) -> bool {
    cuda_device::warp::all_sync(mask, predicate)
}

/// Return the mask of the warp's currently active lanes (`cute.arch.activemask`).
#[must_use]
#[inline(always)]
pub fn activemask() -> u32 {
    cuda_device::warp::active_mask()
}

/// Exchange a `u32` with the lane whose index is this lane's XOR `offset` (`cute.arch.shuffle_sync_bfly`).
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_bfly(value: u32, offset: u32, mask: u32) -> u32 {
    cuda_device::warp::shuffle_xor_sync(mask, value, offset)
}

/// FP32 form of [`shuffle_sync_bfly`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_bfly_f32(value: f32, offset: u32, mask: u32) -> f32 {
    cuda_device::warp::shuffle_xor_f32_sync(mask, value, offset)
}

/// Read a `u32` from the lane `offset` below this lane (`cute.arch.shuffle_sync_up`).
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_up(value: u32, offset: u32, mask: u32) -> u32 {
    cuda_device::warp::shuffle_up_sync(mask, value, offset)
}

/// FP32 form of [`shuffle_sync_up`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_up_f32(value: f32, offset: u32, mask: u32) -> f32 {
    cuda_device::warp::shuffle_up_f32_sync(mask, value, offset)
}

/// Read a `u32` from the lane `offset` above this lane (`cute.arch.shuffle_sync_down`).
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_down(value: u32, offset: u32, mask: u32) -> u32 {
    cuda_device::warp::shuffle_down_sync(mask, value, offset)
}

/// FP32 form of [`shuffle_sync_down`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `offset`.
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync_down_f32(value: f32, offset: u32, mask: u32) -> f32 {
    cuda_device::warp::shuffle_down_f32_sync(mask, value, offset)
}

/// Publish preceding generic-proxy shared-memory writes to the CTA async proxy (`cute.arch.fence_view_async_shared`).
///
/// # Safety
/// Call before signaling or issuing an asynchronous TMA or UMMA operation.
#[inline(always)]
pub unsafe fn fence_view_async_shared() {
    unsafe { cuda_device::barrier::fence_proxy_async_shared_cta() }
}

/// Two packed `f32` values carried in a 64-bit register pair for `f32x2` instructions.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct PackedF32x2(
    /// Packed 64-bit register pair holding `(lo, hi)` FP32 lanes.
    u64,
);

/// Pack two `f32` scalars (`lo`, `hi`) into one [`PackedF32x2`] register pair.
#[must_use]
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::pack_f32x2(lo, hi))
}

/// Extract the low `f32` lane of a [`PackedF32x2`] register pair.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_lo(pair: PackedF32x2) -> f32 {
    cuda_device::f32x2::unpack_f32x2_lo(pair.0)
}

/// Extract the high `f32` lane of a [`PackedF32x2`] register pair.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_hi(pair: PackedF32x2) -> f32 {
    cuda_device::f32x2::unpack_f32x2_hi(pair.0)
}

/// Add two [`PackedF32x2`] register pairs lane-by-lane (`cute.arch.add_packed_f32x2`).
#[must_use]
#[inline(always)]
pub fn add_packed_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::add_f32x2(a.0, b.0))
}

/// Multiply two [`PackedF32x2`] register pairs lane-by-lane (`cute.arch.mul_packed_f32x2`).
#[must_use]
#[inline(always)]
pub fn mul_packed_f32x2(a: PackedF32x2, b: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::mul_f32x2(a.0, b.0))
}

/// Fused multiply-add `a * b + c` on [`PackedF32x2`] register pairs (`cute.arch.fma_packed_f32x2`).
#[must_use]
#[inline(always)]
pub fn fma_packed_f32x2(a: PackedF32x2, b: PackedF32x2, c: PackedF32x2) -> PackedF32x2 {
    PackedF32x2(cuda_device::f32x2::fma_f32x2(a.0, b.0, c.0))
}

/// Convert two `f32` values to packed round-to-nearest-even `f16x2` bits (`cvt.rn.f16x2.f32`).
#[must_use]
#[inline(always)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    cuda_device::convert::cvt_f16x2_f32(lo, hi)
}

/// Convert two `f32` values to packed round-to-nearest-even saturating `e4m3x2` bits (`cvt.rn.satfinite.e4m3x2.f32`).
#[must_use]
#[inline(always)]
pub fn cvt_e4m3x2_f32(lo: f32, hi: f32) -> u16 {
    cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(lo, hi)
}

/// Initialize a CTA-shared `mbarrier` with expected arrival count `count` (`cute.arch.mbarrier_init`).
///
/// # Safety
/// Call from one elected thread on an uninitialized or invalidated 8-byte-aligned shared barrier.
#[inline(always)]
pub unsafe fn mbarrier_init(bar: *mut Barrier, count: u32) {
    unsafe { cuda_device::barrier::mbarrier_init(bar, count) }
}

/// Spin-wait on a CTA-shared `mbarrier` until it completes `phase` (`cute.arch.mbarrier_wait`).
///
/// # Safety
/// `bar` must point to a live initialized CTA-shared barrier.
#[inline(always)]
pub unsafe fn mbarrier_wait(bar: *const Barrier, phase: u32) {
    while !unsafe { cuda_device::barrier::mbarrier_try_wait_parity(bar, phase) } {}
}

/// Arrive once on a CTA-shared `mbarrier` (`cute.arch.mbarrier_arrive`).
///
/// # Safety
/// `bar` must point to a live initialized CTA-shared barrier and the caller must be an expected participant.
#[inline(always)]
pub unsafe fn mbarrier_arrive(bar: *mut Barrier) {
    let _ = unsafe { cuda_device::barrier::mbarrier_arrive(bar) };
}

/// Issue one 4-byte `cp.async.ca.shared.global` copy from `src` to `dst` (`cute.arch.cp_async_shared_global`).
///
/// # Safety
/// `dst` and `src` must be 4-byte-aligned live shared and global addresses.
#[inline(always)]
pub unsafe fn cp_async_shared_global(dst: *mut u32, src: *const u32) {
    unsafe {
        cuda_device::async_copy::cp_async_ca_4(dst, &*src);
    }
}

/// Attach pending `cp.async` copies from this thread to `bar` without incrementing its count.
///
/// # Safety
/// `bar` must point to a live initialized CTA-shared barrier.
#[inline(always)]
pub unsafe fn cp_async_mbarrier_arrive_noinc(bar: *mut Barrier) {
    unsafe {
        cuda_device::async_copy::cp_async_mbarrier_arrive_noinc(bar);
    }
}

/// Commit prior `tcgen05` operations issued by this thread to `bar` (`cute.nvgpu.tcgen05.commit`).
///
/// # Safety
/// `bar` must point to a live initialized CTA-shared barrier.
#[inline(always)]
pub unsafe fn tcgen05_commit(bar: *mut Barrier) {
    unsafe { cuda_device::tcgen05::tcgen05_commit(bar.cast()) }
}

/// Order prior `tcgen05` TMEM accesses before a thread synchronization (`tcgen05.fence::before_thread_sync`).
#[inline(always)]
pub fn tcgen05_fence_before_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_before_thread_sync();
}

/// Order subsequent `tcgen05` TMEM accesses after a thread synchronization (`tcgen05.fence::after_thread_sync`).
#[inline(always)]
pub fn tcgen05_fence_after_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_after_thread_sync();
}

/// Atomically add two adjacent `BFloat16` values `(lo, hi)` to `*dst` and `*dst.add(1)` (`cute.arch.atomic_add`).
///
/// # Safety
/// `dst` must be 4-byte aligned and point to two writable `BFloat16` values in global memory.
#[inline(always)]
pub unsafe fn atomic_add_bf16x2(
    dst: *mut crate::BFloat16,
    lo: crate::BFloat16,
    hi: crate::BFloat16,
) {
    let packed = (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16);
    let _ = unsafe { cuda_device::atomic::atom_add_bf16x2(dst.cast::<u32>(), packed) };
}

/// Prefetch a 128-byte TMA descriptor into the tensor-map cache (`prefetch.tensormap`).
///
/// # Safety
/// `desc` must point to a valid 64-byte-aligned 128-byte CUDA tensor map.
#[inline(always)]
pub unsafe fn prefetch_tma_descriptor(desc: *const u8) {
    unsafe { cuda_device::tma::prefetch_tma_descriptor(desc.cast::<cuda_device::TmaDescriptor>()) }
}

/// Issue one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` warp instruction.
///
/// # Safety
/// All 32 lanes must participate with valid fragment registers.
#[must_use]
#[inline(always)]
pub unsafe fn mma_m16n8k16_f32_f16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    unsafe { cuda_device::wmma::mma_m16n8k16_f32_f16(c, a, b) }
}

/// Issue one `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` warp instruction.
///
/// # Safety
/// All 32 lanes must participate with valid fragment registers.
#[must_use]
#[inline(always)]
pub unsafe fn mma_m16n8k16_f32_bf16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c, a, b) }
}

/// Load `2` 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x2.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x2(address: u32) -> cuda_device::CuSimd<u32, 2> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x2_raw(address) }
}

/// Load 4 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x4.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x4(address: u32) -> cuda_device::CuSimd<u32, 4> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x4_raw(address) }
}

/// Load 8 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x8.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x8(address: u32) -> cuda_device::CuSimd<u32, 8> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x8_raw(address) }
}

/// Load 16 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x16.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x16(address: u32) -> cuda_device::CuSimd<u32, 16> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x16_raw(address) }
}

/// Load 32 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x32.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x32(address: u32) -> cuda_device::CuSimd<u32, 32> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x32_raw(address) }
}

/// Load 64 32-bit registers per lane from TMEM (`tcgen05.ld.sync.aligned.32x32b.x64.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[must_use]
#[inline(always)]
pub unsafe fn tmem_load_32x32b_x64(address: u32) -> cuda_device::CuSimd<u32, 64> {
    unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x64_raw(address) }
}

/// Store 2 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x2.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x2(address: u32, raw: [u32; 2]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x2_raw(address, cuda_device::CuSimd::new(raw))
    }
}

/// Store 4 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x4.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x4(address: u32, raw: [u32; 4]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x4_raw(address, cuda_device::CuSimd::new(raw))
    }
}

/// Store 8 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x8.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x8(address: u32, raw: [u32; 8]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x8_raw(address, cuda_device::CuSimd::new(raw))
    }
}

/// Store 16 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x16.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x16(address: u32, raw: [u32; 16]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x16_raw(address, cuda_device::CuSimd::new(raw))
    }
}

/// Store 32 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x32.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x32(address: u32, raw: [u32; 32]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x32_raw(address, cuda_device::CuSimd::new(raw))
    }
}

/// Store 64 32-bit registers per lane to TMEM (`tcgen05.st.sync.aligned.32x32b.x64.b32`).
///
/// # Safety
/// All 32 lanes must participate with a valid TMEM row-group address.
#[inline(always)]
pub unsafe fn tmem_store_32x32b_x64(address: u32, raw: [u32; 64]) {
    unsafe {
        cuda_device::tcgen05::tcgen05_st_32x32b_x64_raw(address, cuda_device::CuSimd::new(raw))
    }
}

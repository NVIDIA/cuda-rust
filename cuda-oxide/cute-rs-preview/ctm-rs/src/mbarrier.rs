/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Canonical barrier states with explicit instruction-only lifecycle methods.
use crate::memory::SharedAddress;
pub use cuda_device::barrier::{
    Barrier, BarrierToken, GeneralBarrier, Invalidated, ManagedBarrier, Ready, TmaBarrier, Uninit,
};

/// Signal arrival at a numbered CTA barrier without waiting.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_cta_arrive(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_arrive(barrier_id, thread_count) }
}

/// Signal arrival at a numbered CTA barrier in aligned mode without waiting.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
/// Every non-exited thread in the warp must execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cta_arrive_aligned(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_arrive_aligned(barrier_id, thread_count) }
}

/// Synchronize the requested number of threads at a numbered CTA barrier.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_cta_sync(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_sync(barrier_id, thread_count) }
}

/// Synchronize at a numbered CTA barrier in aligned mode.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
/// Every non-exited thread in the warp must execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cta_sync_aligned(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_sync_aligned(barrier_id, thread_count) }
}

/// Publish preceding mbarrier initialization at cluster scope.
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`cuda_device::barrier::fence_mbarrier_init_release_cluster`].
#[inline(always)]
pub unsafe fn fence_mbarrier_init_release_cluster() {
    unsafe { cuda_device::barrier::fence_mbarrier_init_release_cluster() }
}

/// Make preceding CTA-shared generic-proxy writes visible to the async proxy.
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`cuda_device::barrier::fence_proxy_async_shared_cta`].
#[inline(always)]
pub unsafe fn fence_proxy_async_shared_cta() {
    unsafe { cuda_device::barrier::fence_proxy_async_shared_cta() }
}

/// Borrow typed shared addresses from initialized barrier handles.
pub trait BarrierAddressExt {
    /// Borrow a current-CTA shared address without extending this handle's lifetime.
    fn shared_address(&self) -> SharedAddress<'_, Barrier>;
}

impl<Kind, const ID: usize> BarrierAddressExt for ManagedBarrier<Ready, Kind, ID> {
    #[inline(always)]
    fn shared_address(&self) -> SharedAddress<'_, Barrier> {
        unsafe { SharedAddress::from_raw(self.as_ptr().cast_mut()) }
    }
}

/// Issue one arrival, returning the hardware phase token.
///
/// # Safety
/// This participant must be expected in the current live barrier phase.
#[inline(always)]
pub unsafe fn arrive<K, const ID: usize>(bar: &ManagedBarrier<Ready, K, ID>) -> BarrierToken {
    bar.arrive()
}
/// Issue one arrival and add the expected TMA byte count. Arrival decrement is one.
///
/// # Safety
/// The issuing participant must be expected, and `bytes` must match the
/// asynchronous transfers completing against this barrier phase.
#[inline(always)]
pub unsafe fn arrive_expect_tx<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    bar.arrive_expect_tx(bytes)
}
/// Try the supplied phase parity once; no loop or phase update is inserted.
///
/// # Safety
/// Parity must be 0 or 1 and agree with the participant's intended barrier phase.
#[inline(always)]
pub unsafe fn try_wait_parity<K, const ID: usize>(
    bar: &ManagedBarrier<Ready, K, ID>,
    parity: u32,
) -> bool {
    bar.try_wait_parity(parity)
}

/// Arrive once and add expected TMA bytes with relaxed cluster scope.
///
/// # Safety
/// One expected participant issues this for the current phase. `bytes` must
/// cover the asynchronous transfers completing at this live local barrier;
/// initialization must already be published to every cluster participant.
#[inline(always)]
pub unsafe fn arrive_expect_tx_cluster<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    BarrierToken::from_raw(unsafe {
        cuda_device::barrier::mbarrier_arrive_expect_tx_cluster(bar.as_ptr(), 1, bytes)
    })
}

/// Test parity once with cluster-scope acquire ordering; no software wait loop.
///
/// # Safety
/// The live local barrier must track this participant's intended phase.
/// `parity` must be zero or one; coordinate remote participants explicitly.
#[inline(always)]
pub unsafe fn try_wait_parity_cluster<K, const ID: usize>(
    bar: &ManagedBarrier<Ready, K, ID>,
    parity: u32,
) -> bool {
    unsafe { cuda_device::barrier::mbarrier_try_wait_parity_cluster(bar.as_ptr(), parity) }
}

/// Initialize one raw CTA-shared barrier, without election or publication (sm_80+).
///
/// # Safety
/// `bar` must name aligned live CTA-shared storage. Exactly one thread must
/// initialize it with a nonzero expected count before any participant uses it.
/// Publish initialization and synchronize participants explicitly.
#[inline(always)]
pub unsafe fn mbarrier_init(bar: *mut Barrier, expected_count: u32) {
    unsafe { cuda_device::barrier::mbarrier_init(bar, expected_count) }
}

/// Issue one raw CTA-scoped release arrival and return its phase token (sm_80+).
///
/// # Safety
/// `bar` must name a live initialized CTA-shared barrier whose current phase
/// expects this arrival. Coordinate phase changes and keep it live through use.
#[inline(always)]
pub unsafe fn mbarrier_arrive(bar: *const Barrier) -> u64 {
    unsafe { cuda_device::barrier::mbarrier_arrive(bar) }
}

/// Issue one release arrival and add expected transaction bytes (sm_90+).
/// `tx_count` retains the canonical ABI; the instruction decrements arrivals by one.
///
/// # Safety
/// `bar` must name a live initialized CTA-shared barrier expecting this arrival.
/// Pass `tx_count = 1`; `bytes` must match the asynchronous transfers completing
/// at this phase. Keep all resources live until those transfers complete.
#[inline(always)]
pub unsafe fn mbarrier_arrive_expect_tx(bar: *const Barrier, tx_count: u32, bytes: u32) -> u64 {
    unsafe { cuda_device::barrier::mbarrier_arrive_expect_tx(bar, tx_count, bytes) }
}

/// Issue one relaxed cluster-scope arrival with expected transaction bytes (sm_90+).
/// `tx_count` retains the canonical ABI; the instruction decrements arrivals by one.
///
/// # Safety
/// `bar` must name a live initialized local CTA-shared barrier expecting this
/// arrival. Pass `tx_count = 1` and match `bytes` to this phase's transfers.
/// Publish initialization and provide required data/proxy ordering explicitly;
/// this relaxed operation establishes no data-access ordering.
#[inline(always)]
pub unsafe fn mbarrier_arrive_expect_tx_cluster(
    bar: *const Barrier,
    tx_count: u32,
    bytes: u32,
) -> u64 {
    unsafe { cuda_device::barrier::mbarrier_arrive_expect_tx_cluster(bar, tx_count, bytes) }
}

/// Test raw CTA-shared barrier parity once; no wait loop is inserted (sm_90+).
///
/// # Safety
/// `bar` must name a live initialized local barrier and `parity` must be zero
/// or one for the intended phase. Coordinate phase transitions explicitly.
#[inline(always)]
pub unsafe fn mbarrier_try_wait_parity(bar: *const Barrier, parity: u32) -> bool {
    unsafe { cuda_device::barrier::mbarrier_try_wait_parity(bar, parity) }
}

/// Test local barrier parity once with relaxed cluster ordering (sm_90+).
///
/// # Safety
/// `bar` must name a live initialized local CTA-shared barrier; `parity` must
/// be zero or one for the intended phase. Publish initialization explicitly.
/// A successful test orders execution/counts only; provide the required data,
/// proxy and TMEM ordering before accessing the completed phase's resources.
#[inline(always)]
pub unsafe fn mbarrier_try_wait_parity_cluster_relaxed(bar: *const Barrier, parity: u32) -> bool {
    unsafe { cuda_device::barrier::mbarrier_try_wait_parity_cluster_relaxed(bar, parity) }
}

/// Acquire cluster-shared async-proxy writes through the generic proxy (sm_90+).
///
/// # Safety
/// Pair this fence with the compatible cluster-scoped release and barrier
/// protocol. It neither waits for asynchronous operations nor orders global memory.
#[inline(always)]
pub unsafe fn fence_proxy_async_generic_acquire_shared_cluster_cluster() {
    unsafe { cuda_device::barrier::fence_proxy_async_generic_acquire_shared_cluster_cluster() }
}

/// Release CTA-shared generic writes to the async proxy at cluster scope (sm_90+).
///
/// # Safety
/// Pair this fence with the compatible cluster-scoped acquire and barrier
/// protocol, and keep the shared storage live through its asynchronous uses.
#[inline(always)]
pub unsafe fn fence_proxy_async_generic_release_shared_cta_cluster() {
    unsafe { cuda_device::barrier::fence_proxy_async_generic_release_shared_cta_cluster() }
}

/// Acquire cluster-shared memory at cluster scope (sm_90+), excluding global memory.
///
/// # Safety
/// Pair this with a compatible cluster-scoped synchronization protocol. This
/// does not substitute for a proxy fence or asynchronous completion wait.
#[inline(always)]
pub unsafe fn fence_acquire_shared_cluster_cluster() {
    unsafe { cuda_device::barrier::fence_acquire_shared_cluster_cluster() }
}

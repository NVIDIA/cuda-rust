/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! The `mbarrier.*` instruction family: canonical barrier states with explicit
//! instruction-only lifecycle methods, plus the raw init, arrive, expect-tx
//! and parity-test forms on CTA-shared barrier objects.
//!
//! Named CTA barriers (`bar.*`, `barrier.cta.*`) live in [`crate::barrier`];
//! memory fences (`fence.*`) in [`crate::fence`].
use crate::memory::SharedAddress;
pub use cuda_device::barrier::{
    Barrier, BarrierToken, GeneralBarrier, Invalidated, ManagedBarrier, Ready, TmaBarrier, Uninit,
};

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

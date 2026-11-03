/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! The `fence.*` instruction family: `membar`/`fence` memory-consistency
//! instructions, including `fence.mbarrier_init` and the `fence.proxy.async`
//! forms.
//!
//! A fence orders memory accesses between threads, scopes or proxies; none of
//! these waits for asynchronous completion or substitutes for a barrier.

/// Publish preceding mbarrier initialization at cluster scope.
///
/// PTX `fence.mbarrier_init.release.cluster` is the only form of this fence,
/// so the name carries no ordering or scope suffix.
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`cuda_device::barrier::fence_mbarrier_init_release_cluster`].
#[inline(always)]
pub unsafe fn fence_mbarrier_init() {
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

/// Acquire cluster-shared async-proxy writes through the generic proxy (sm_90+).
///
/// PTX `fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`.
///
/// # Safety
/// Pair this fence with the compatible cluster-scoped release and barrier
/// protocol. It neither waits for asynchronous operations nor orders global memory.
#[inline(always)]
pub unsafe fn fence_proxy_async_acquire_sync_restrict() {
    unsafe { cuda_device::barrier::fence_proxy_async_generic_acquire_shared_cluster_cluster() }
}

/// Release CTA-shared generic writes to the async proxy at cluster scope (sm_90+).
///
/// PTX `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`.
///
/// # Safety
/// Pair this fence with the compatible cluster-scoped acquire and barrier
/// protocol, and keep the shared storage live through its asynchronous uses.
#[inline(always)]
pub unsafe fn fence_proxy_async_release_sync_restrict() {
    unsafe { cuda_device::barrier::fence_proxy_async_generic_release_shared_cta_cluster() }
}

/// Acquire cluster-shared memory at cluster scope (sm_90+), excluding global memory.
///
/// PTX `fence.acquire.sync_restrict::shared::cluster.cluster`; CuTeDSL's
/// `fence_sync_restrict(order)` covers acquire and release, and only the
/// acquire form has an intrinsic here.
///
/// # Safety
/// Pair this with a compatible cluster-scoped synchronization protocol. This
/// does not substitute for a proxy fence or asynchronous completion wait.
#[inline(always)]
pub unsafe fn fence_sync_restrict_acquire() {
    unsafe { cuda_device::barrier::fence_acquire_shared_cluster_cluster() }
}

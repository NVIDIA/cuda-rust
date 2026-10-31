/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! The `bar.*` and `barrier.cta.*` instruction family: named CTA barriers.
//!
//! Each function forwards one instruction with the caller's barrier ID and
//! thread count; the aligned forms require every non-exited thread of a warp
//! to execute with identical control flow. `bar.warp.sync` is a warp
//! collective and stays in [`crate::warp::bar_warp_sync`]; mbarrier objects
//! live in [`crate::mbarrier`] and memory fences in [`crate::fence`].

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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit lane identity, election and register exchange.
//! No other primitive elects implicitly.

/// Exchange an FP32 register with the lane selected by XORing `lane_mask`.
///
/// This forwards the full-warp shuffle with participation mask `u32::MAX`.
/// All 32 lanes must participate with the same lane mask; no reduction or
/// memory synchronization is inserted.
#[inline(always)]
pub fn shuffle_xor_f32(value: f32, lane_mask: u32) -> f32 {
    cuda_device::warp::shuffle_xor_f32(value, lane_mask)
}

/// Elect one lane from `mask`, returning its lane ID and whether this lane won.
/// Every lane named in the mask must be converged at the call (sm_90+).
#[inline(always)]
pub fn elect_sync(mask: u32) -> (u32, bool) {
    cuda_device::warp::elect_sync(mask)
}

/// Return whether this lane is elected from `mask` (sm_90+).
/// Every lane named in the mask must be converged at the call.
#[inline(always)]
pub fn is_elected_sync(mask: u32) -> bool {
    cuda_device::warp::is_elected_sync(mask)
}

/// Return `threadIdx.x / 32`, the warp ID for a one-dimensional thread block.
#[inline(always)]
pub fn warp_id() -> u32 {
    cuda_device::warp::warp_id()
}

/// Read a participating lane's integer register under an explicit warp mask.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask, including the
/// caller. Each requested source lane must be in `0..32` and participate.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shuffle_sync(u32::MAX, 1, 0);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_sync(mask: u32, value: u32, src_lane: u32) -> u32 {
    cuda_device::warp::shuffle_sync(mask, value, src_lane)
}

/// Sum FP32 values across a full warp in XOR-offset order 16, 8, 4, 2, 1.
/// Every lane receives the sum. No memory synchronization is inserted.
///
/// # Safety
/// All 32 lanes must be active and execute convergently. Missing logical
/// elements must contribute zero while their lanes continue to participate.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::reduce_sum_f32(1.0);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn reduce_sum_f32(value: f32) -> f32 {
    cuda_device::warp::reduce_sum_f32(value)
}

/// Return whether every participating lane's predicate is true, using a full-warp mask.
///
/// # Safety
/// All 32 lanes must be active and execute this vote convergently.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::all(true);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn all(predicate: bool) -> bool {
    cuda_device::warp::all(predicate)
}

/// Synchronize warp lanes selected by an explicit participation mask.
///
/// # Safety
/// The caller's lane must be included in `mask`. Every lane named in `mask`
/// must reach this call with the same mask. All accesses ordered by the barrier
/// must satisfy their allocation contracts.
///
/// ```compile_fail,E0133
/// ctm_rs::warp::sync_mask(u32::MAX);
/// ```
#[inline(always)]
pub unsafe fn sync_mask(mask: u32) {
    cuda_device::warp::sync_mask(mask)
}

/// Exchange a register with the lane selected by XORing `lane_mask`.
/// This inserts no memory synchronization.
///
/// # Safety
/// All 32 lanes must be active and execute convergently with the same lane
/// mask. `lane_mask` must select a source lane within the warp.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shuffle_xor(0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shuffle_xor(value: u32, lane_mask: u32) -> u32 {
    cuda_device::warp::shuffle_xor(value, lane_mask)
}

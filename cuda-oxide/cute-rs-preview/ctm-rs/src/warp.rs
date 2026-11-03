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

/// Synchronize warp lanes selected by an explicit participation mask (PTX `bar.warp.sync`).
///
/// # Safety
/// The caller's lane must be included in `mask`. Every lane named in `mask`
/// must reach this call with the same mask. All accesses ordered by the barrier
/// must satisfy their allocation contracts.
///
/// ```compile_fail,E0133
/// ctm_rs::warp::bar_warp_sync(u32::MAX);
/// ```
#[inline(always)]
pub unsafe fn bar_warp_sync(mask: u32) {
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

/// Return this lane's index within its warp, from the `%laneid` register.
#[must_use]
#[inline(always)]
pub fn lane_id() -> u32 {
    cuda_device::warp::lane_id()
}

/// Return the mask of the warp's currently active lanes (`activemask.b32`).
/// Bit `k` is set when lane `k` has neither exited nor diverged away here.
#[must_use]
#[inline(always)]
pub fn activemask() -> u32 {
    cuda_device::warp::active_mask()
}

/// Ballot across the lanes in `mask` (`vote.sync.ballot.b32`): bit `k` is set
/// when lane `k` is in `mask` and its predicate is true.
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::vote_sync_ballot(u32::MAX, true);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn vote_sync_ballot(mask: u32, predicate: bool) -> u32 {
    cuda_device::warp::ballot_sync(mask, predicate)
}

/// Return whether `predicate` holds for at least one lane in `mask` (`vote.sync.any.pred`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::vote_sync_any(u32::MAX, true);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn vote_sync_any(mask: u32, predicate: bool) -> bool {
    cuda_device::warp::any_sync(mask, predicate)
}

/// Return whether `predicate` holds for every lane in `mask` (`vote.sync.all.pred`).
///
/// # Safety
/// Every lane named in `mask` must execute this vote with the same mask,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::vote_sync_all(u32::MAX, true);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn vote_sync_all(mask: u32, predicate: bool) -> bool {
    cuda_device::warp::all_sync(mask, predicate)
}

/// Exchange a register with the lane selected by XORing `lane_mask`, under an
/// explicit participation mask (`shfl.sync.bfly.b32`). This inserts no memory
/// synchronization.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and lane mask,
/// including the caller. `lane_mask` must select a source lane within the warp.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_bfly(u32::MAX, 0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_bfly(mask: u32, value: u32, lane_mask: u32) -> u32 {
    cuda_device::warp::shuffle_xor_sync(mask, value, lane_mask)
}

/// FP32 form of [`shfl_sync_bfly`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and lane mask,
/// including the caller. `lane_mask` must select a source lane within the warp.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_bfly_f32(u32::MAX, 0.0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_bfly_f32(mask: u32, value: f32, lane_mask: u32) -> f32 {
    cuda_device::warp::shuffle_xor_f32_sync(mask, value, lane_mask)
}

/// Read the register of the lane `delta` below this lane, under an explicit
/// participation mask (`shfl.sync.up.b32`); lanes below `delta` keep their own
/// value. This inserts no memory synchronization.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `delta`,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_up(u32::MAX, 0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_up(mask: u32, value: u32, delta: u32) -> u32 {
    cuda_device::warp::shuffle_up_sync(mask, value, delta)
}

/// FP32 form of [`shfl_sync_up`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `delta`,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_up_f32(u32::MAX, 0.0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_up_f32(mask: u32, value: f32, delta: u32) -> f32 {
    cuda_device::warp::shuffle_up_f32_sync(mask, value, delta)
}

/// Read the register of the lane `delta` above this lane, under an explicit
/// participation mask (`shfl.sync.down.b32`); the top `delta` lanes keep their
/// own value. This inserts no memory synchronization.
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `delta`,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_down(u32::MAX, 0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_down(mask: u32, value: u32, delta: u32) -> u32 {
    cuda_device::warp::shuffle_down_sync(mask, value, delta)
}

/// FP32 form of [`shfl_sync_down`].
///
/// # Safety
/// Every lane named in `mask` must execute with the same mask and `delta`,
/// including the caller.
///
/// ```compile_fail,E0133
/// let _ = ctm_rs::warp::shfl_sync_down_f32(u32::MAX, 0.0, 1);
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn shfl_sync_down_f32(mask: u32, value: f32, delta: u32) -> f32 {
    cuda_device::warp::shuffle_down_f32_sync(mask, value, delta)
}

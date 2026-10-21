/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit warp-level compiler facts built from supported CUDA operations.

/// Return a warp-uniform value in a form that can favor uniform-register use.
///
/// This matches CuTe's `arch.make_warp_uniform`: a full-warp shuffle broadcasts
/// lane zero's value. Pass the returned value to subsequent calculations. The
/// compiler may then keep those calculations in uniform registers; this is an
/// optimization opportunity, not a register-allocation guarantee.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently with equal
/// `value` arguments. Do not call from an elected-lane or partially active
/// branch. This is an identity operation only under that uniformity promise;
/// it does not check or repair a divergent value.
#[inline(always)]
pub unsafe fn assume_warp_uniform(value: u32) -> u32 {
    // SAFETY: the caller supplies the full-warp convergence contract above.
    unsafe { ctm_rs::warp::shuffle_sync(u32::MAX, value, 0) }
}

/// Sum one FP32 contribution from each of the 32 lanes of a warp.
///
/// Every lane receives the sum. Uses the existing CUDA butterfly reduction,
/// with FP32 additions in XOR-offset order 16, 8, 4, 2, 1. The result need not
/// equal a sequential sum. No shared memory or block barrier is required.
/// Inactive logical elements must contribute zero while their lanes still
/// participate. This is a device operation, not a host reduction.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently. The block
/// must contain complete warps. Rust's type system cannot prove convergence;
/// do not invoke this in a lane-dependent branch or after a partial-warp exit.
#[must_use]
#[inline(always)]
pub unsafe fn warp_sum_f32(value: f32) -> f32 {
    // SAFETY: every lane participates under this function's caller contract.
    unsafe { ctm_rs::warp::reduce_sum_f32(value) }
}

/// Approximate base-two exponential, preserving the device instruction semantics.
#[inline(always)]
pub fn exp2_f32(value: f32) -> f32 {
    ctm_rs::float::ex2_approx_f32(value)
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
/// The comparison selects the right value when `left > right` is false;
/// it deliberately retains this behavior for NaNs and signed zero.
///
/// # Safety
/// All 32 lanes must execute this operation convergently.
#[inline(always)]
pub unsafe fn warp_max_f32(mut value: f32) -> f32 {
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
/// This marker describes the synchronization operation, not an owned hardware
/// resource. Callers must keep barrier phases and their participating threads
/// consistent; distinct logical barriers must use distinct IDs.
pub struct BlockBarrier<const ID: u32, const THREADS: u32>;

impl<const ID: u32, const THREADS: u32> BlockBarrier<ID, THREADS> {
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
        unsafe { ctm_rs::mbarrier::barrier_cta_sync(ID, THREADS) };
    }
}

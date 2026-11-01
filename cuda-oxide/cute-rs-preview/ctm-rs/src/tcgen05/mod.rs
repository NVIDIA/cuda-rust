/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! SM100 tensor-memory allocation, FP16 MMA and register readback.
//!
//! Instructions require PTX 8.6 and a tcgen05-capable target profile; the current
//! CTM consumer validates `sm_100a` and `sm_103a`. TCGEN05 is unavailable on SM120a.
//!
//! Start with [`descriptors::DenseSharedF16`] for supported shapes and
//! [`tmem::TmemGuard`] for allocation ownership. Issue, commit, wait and release
//! remain explicit; [`tmem::load_wait`] and [`tmem::store_wait`] are the two
//! completion waits. The crate README gives an issue-site example and the
//! supported compiler profiles.

pub mod descriptors;
pub mod mma;
pub mod tmem;

/// Commit the issuing thread's group-one MMA work to a ready local barrier.
///
/// # Safety
/// The elected issuer must own the pending group-one MMA operations. This live
/// barrier phase must expect their completion arrival, and operands must remain
/// live until completion. This function does not elect, wait or change phases.
/// All tcgen05 operations in the kernel must use group-one mode consistently.
///
/// ```compile_fail
/// use ctm_rs::mbarrier::{GeneralBarrier, ManagedBarrier, Uninit};
/// # fn rejected(barrier: &ManagedBarrier<Uninit, GeneralBarrier>) {
/// unsafe { ctm_rs::tcgen05::commit(barrier) };
/// # }
/// ```
#[inline(always)]
pub unsafe fn commit<Kind, const ID: usize>(
    barrier: &crate::mbarrier::ManagedBarrier<crate::mbarrier::Ready, Kind, ID>,
) {
    unsafe { cuda_device::tcgen05::tcgen05_commit(barrier.as_ptr().cast_mut().cast()) }
}

/// Relinquish the group-one allocation permit after allocation completes.
///
/// # Safety
/// The allocating warp must participate uniformly, and the kernel must use
/// group-one TMEM operations consistently. No deallocation is performed.
#[inline(always)]
pub unsafe fn relinquish_alloc_permit() {
    cuda_device::tcgen05::tcgen05_relinquish_alloc_permit()
}

/// Order later tcgen05 accesses after thread synchronization.
/// All tcgen05 operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn fence_after_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_after_thread_sync()
}

/// Order prior tcgen05 accesses before thread synchronization.
/// All tcgen05 operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn fence_before_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_before_thread_sync()
}

/// Release the group-2 tensor-memory allocation permit.
/// One full warp in each peer CTA must execute uniformly. All tcgen05
/// operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn relinquish_alloc_permit_cg2() {
    cuda_device::tcgen05::tcgen05_relinquish_alloc_permit_cg2()
}

/// Commit group-2 MMA completion to the selected CTAs' local barrier offsets.
///
/// # Safety
/// The MMA issuer must commit its own operations. Every selected CTA must be
/// active with a matching initialized barrier expecting the completion arrival.
/// Keep all associated resources live until the barrier signals completion.
#[inline(always)]
pub unsafe fn commit_multicast_cg2(
    barrier: super::memory::SharedAddress<'_, cuda_device::barrier::Barrier>,
    mask: super::cluster::CtaMask<super::groups::CtaGroup2>,
) {
    unsafe {
        cuda_device::tcgen05::tcgen05_commit_multicast_cg2(barrier.as_raw().cast(), mask.bits())
    }
}

/// Commit group-1 MMA completion to a local barrier (SM100), without waiting.
///
/// # Safety
/// The MMA issuer must commit its own operations exactly once to a live
/// initialized CTA-shared barrier expecting this completion arrival. Keep all
/// associated resources live until completion. All tcgen05 operations in the
/// kernel must use group-1 mode.
#[inline(always)]
pub unsafe fn commit_address(barrier: super::memory::SharedAddress<'_, crate::mbarrier::Barrier>) {
    unsafe { cuda_device::tcgen05::tcgen05_commit(barrier.as_raw().cast()) }
}

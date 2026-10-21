/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Static CTA-group selectors. They do not elect issuing threads.
/// Limits allocation dispatch to the supported one- and two-CTA modes.
mod sealed {
    /// Implemented only by the supported CTA-group selectors.
    pub trait Group {}
}
/// Instruction selector for one participating CTA.
pub enum CtaGroup1 {}
/// Instruction selector for a cooperating pair of CTAs.
pub enum CtaGroup2 {}
/// Reviewed hardware CTA-group modes.
pub trait CtaGroup: sealed::Group {
    #[doc(hidden)]
    /// # Safety
    /// A complete issuing warp in each required CTA must participate uniformly.
    unsafe fn alloc(slot: *mut u32, columns: u32);
    #[doc(hidden)]
    /// # Safety
    /// All accesses must be complete and the required warps must participate.
    unsafe fn dealloc(address: u32, columns: u32);
}
/// Bind a group selector to its literal allocation and deallocation intrinsics.
macro_rules! group {
    ($g:ty,$alloc:ident,$dealloc:ident) => {
        impl sealed::Group for $g {}
        impl CtaGroup for $g {
            #[inline(always)]
            unsafe fn alloc(slot: *mut u32, columns: u32) {
                unsafe { cuda_device::tcgen05::$alloc(slot, columns) }
            }
            #[inline(always)]
            unsafe fn dealloc(address: u32, columns: u32) {
                unsafe { cuda_device::tcgen05::$dealloc(address, columns) }
            }
        }
    };
}
group!(CtaGroup1, tcgen05_alloc, tcgen05_dealloc);
group!(CtaGroup2, tcgen05_alloc_cg2, tcgen05_dealloc_cg2);

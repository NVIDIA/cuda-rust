/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rank-three copies matching independent attention operand layouts.

use super::{__compiler, SharedTile, SmemLayout};
use crate::copy::tma::TmaDesc;
use core::marker::PhantomData;
use ctm_rs::mbarrier::Barrier;

/// A rank-three TMA copy into the same physical layout used by MMA.
pub struct TmaCopy<L, const GROUP: u32>(
    #[doc = "Retains the shared layout that must match the tensor-map descriptor."] PhantomData<L>,
);
impl<L: SmemLayout, const G: u32> TmaCopy<L, G> {
    /// Copy a physical row-major global tile into a typed shared tile.
    /// # Safety
    /// All producer-warp lanes participate. Coordinates are in physical
    /// global rows/columns; columns must be a multiple of `L::SPAN`. The leader
    /// barrier must expect copies from the complete group. For group two,
    /// rank is the local CTA rank and completion routes to CTA zero.
    #[inline(always)]
    pub unsafe fn load<R>(
        desc: &TmaDesc<f16, L>,
        dst: &mut SharedTile<L, R>,
        row: i32,
        col: i32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::tma_load::<L, G>(
                desc as *const _,
                dst.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
                barrier,
                rank,
            )
        }
    }
    /// # Safety
    /// One elected lane issues after stores to shared memory are visible to
    /// the async proxy. Storage stays live through the bulk-copy wait.
    #[inline(always)]
    pub unsafe fn store<R>(desc: &TmaDesc<f16, L>, src: &SharedTile<L, R>, row: i32, col: i32) {
        unsafe {
            __compiler::tma_store::<L, G>(
                desc as *const _,
                src.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
            )
        }
    }
}

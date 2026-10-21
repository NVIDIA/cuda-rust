/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-CTA tensor-memory allocation and accumulator epilogues.

use super::__compiler;
use crate::copy::tma::TmaDesc;
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;

/// Allocate one tensor-memory region collectively in a two-CTA cluster.
///
/// Tensor memory has 128 rows per CTA. `COLUMNS` is a power of two from 32
/// through 512. Allocation and deallocation are collective warp operations.
pub struct Sm100Tmem<const COLUMNS: u32>;
impl<const COLUMNS: u32> Sm100Tmem<COLUMNS> {
    /// Restrict allocation columns to supported powers of two from 32 through 512.
    const VALID: () = assert!(COLUMNS >= 32 && COLUMNS <= 512 && COLUMNS.is_power_of_two());

    /// Allocate columns and relinquish the allocation permit.
    ///
    /// # Safety
    ///
    /// Every lane of the allocating warp in both CTAs must participate.
    /// `token` points to a live aligned shared `u32`. Synchronize readers
    /// before loading the address written there.
    #[inline(always)]
    pub unsafe fn allocate(token: *mut u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_alloc::<COLUMNS>(token) }
    }

    /// Return a previously allocated region after both CTAs finish using it.
    ///
    /// # Safety
    ///
    /// All asynchronous MMA and loads must be complete and the two CTAs must
    /// rendezvous before both allocating warps call this operation.
    #[inline(always)]
    pub unsafe fn deallocate(address: u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_dealloc::<COLUMNS>(address) }
    }
}

/// Convert one 128x32 FP32 accumulator slice to an FP16 tile in shared memory.
pub struct Sm100TmemEpilogue<Layout>(
    #[doc = "Carries the shared output layout required by readback and TMA."] PhantomData<Layout>,
);
impl<Layout: ReifySmem2D> Sm100TmemEpilogue<Layout> {
    /// Load a lane's 32 accumulators, add its row bias, and convert to FP16.
    ///
    /// # Safety
    ///
    /// All lanes of the four epilogue warps participate after accumulator wait.
    /// `address` already selects the calling warp's 32 TMEM rows and current
    /// 32-column slice. `tid` is the CTA thread index in `0..128`. `dst` has
    /// the supported 128x32 layout and is reusable according to the store
    /// pipeline. Publish shared writes before issuing the TMA store.
    #[inline(always)]
    pub unsafe fn store(address: u32, dst: *mut f16, tid: u32, bias: f32) {
        unsafe { __compiler::sm100_tmem_epilogue::<Layout>(address, dst, tid, bias) }
    }

    /// Select one warp's 32 TMEM rows and one 32-column accumulator slice.
    ///
    /// `warp` is in `0..4` and the chosen columns must belong to the tile.
    #[inline(always)]
    pub const fn slice_address(base: u32, warp: u32, subtile: u32) -> u32 {
        (((base >> 16) + warp * 32) << 16) | ((base & 0xffff) + subtile * 32)
    }

    /// Start the FP16 tile's asynchronous shared-to-global TMA store.
    ///
    /// # Safety
    ///
    /// One elected store issuer calls after all writers publish their shared
    /// writes and synchronize. The descriptor encodes `Layout`; coordinates
    /// count elements. The shared source must remain live and unchanged until
    /// this copy's committed group finishes reading it. A store-pipeline
    /// acquire releases the oldest buffer in the ring and can leave this
    /// newest group pending; it does not immediately release this source.
    #[inline(always)]
    pub unsafe fn store_async(
        desc: *const TmaDesc<f16, Layout>,
        src: *const f16,
        row: u32,
        column: u32,
    ) {
        unsafe { __compiler::sm100_tma_store::<Layout>(desc, src, row, column) }
    }
}

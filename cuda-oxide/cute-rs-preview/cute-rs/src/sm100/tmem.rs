/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-CTA tensor-memory allocation and accumulator epilogues.

use super::__compiler;
use crate::copy::tma::TensorMap;
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
    pub unsafe fn alloc(token: *mut u32) {
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
    pub unsafe fn dealloc(address: u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_dealloc::<COLUMNS>(address) }
    }
}

/// Two-CTA cluster rendezvous barrier ordering collective TMEM deallocation.
///
/// In a two-CTA SM100 GEMM, neither CTA may free its TMEM columns until both
/// CTAs' epilogue warps have completed all TMEM reads. Each CTA initializes one
/// shared barrier with arrival count 32; warp zero's 32 lanes synchronize
/// locally, signal the peer CTA's barrier via DSMEM, and wait for parity 0
/// before calling [`Sm100Tmem::dealloc`].
#[derive(Clone, Copy)]
pub struct Sm100DeallocBarrier {
    /// Local shared 64-bit mbarrier at identical byte offsets in both CTAs.
    bar: *mut ctm_rs::mbarrier::Barrier,
}

impl Sm100DeallocBarrier {
    /// Wrap a shared-memory barrier slot reserved for two-CTA TMEM deallocation.
    ///
    /// # Safety
    ///
    /// `bar` must point to a live, 8-byte-aligned shared barrier at the same
    /// dynamic shared-memory offset in both CTAs of the cluster.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(bar: *mut ctm_rs::mbarrier::Barrier) -> Self {
        Self { bar }
    }

    /// Initialize the deallocation barrier for 32 remote warp-zero arrivals.
    ///
    /// # Safety
    ///
    /// Call once from one elected lane in each CTA before publishing barrier
    /// initialization across the cluster.
    #[inline(always)]
    pub unsafe fn init(&self) {
        unsafe {
            ctm_rs::mbarrier::ManagedBarrier::<
                ctm_rs::mbarrier::Uninit,
                ctm_rs::mbarrier::GeneralBarrier,
            >::from_static(self.bar)
            .init_instruction(32);
        }
    }

    /// Synchronize warp zero, arrive on the peer CTA's barrier, and wait until both CTAs finish.
    ///
    /// # Safety
    ///
    /// All 32 lanes of warp zero in both CTAs must call this operation after
    /// completing all TMEM loads and TMA store tails, with `rank` set to
    /// `cute_rs::arch::block_idx_in_cluster()`.
    #[inline(always)]
    pub unsafe fn sync_cluster(&self, rank: u32) {
        unsafe {
            crate::arch::NamedBarrier::<3, 32>::sync();
            let peer = crate::arch::map_dsmem_ptr(
                ctm_rs::memory::SharedAddress::from_raw(self.bar),
                rank ^ 1,
            );
            ctm_rs::cluster::arrive_remote(peer);
            let ready = ctm_rs::mbarrier::ManagedBarrier::<
                ctm_rs::mbarrier::Ready,
                ctm_rs::mbarrier::GeneralBarrier,
            >::adopt_ready(self.bar);
            while !ctm_rs::mbarrier::try_wait_parity_cluster(&ready, 0) {}
        }
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
        desc: *const TensorMap<f16, Layout>,
        src: *const f16,
        row: u32,
        column: u32,
    ) {
        unsafe { __compiler::sm100_tma_store::<Layout>(desc, src, row, column) }
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor-memory allocators (`cutlass.utils.TmemAllocator`).

use crate::nvgpu::tcgen05::__compiler;
use cuda_device::barrier::{Barrier, GeneralBarrier, ManagedBarrier, Ready, Uninit};
use cuda_device::tcgen05;

/// Sealed trait restricting `TmemAllocator` column counts to `32`, `64`, `128`, `256`, or `512`.
mod sealed {
    /// Implemented only for supported power-of-two TMEM column counts.
    pub trait Columns<const N: u32> {}
}

/// Proof that `COLUMNS` is a valid hardware TMEM allocation width (`32`, `64`, `128`, `256`, or `512`).
pub trait ValidColumns<const N: u32>: sealed::Columns<N> {}

/// Implement [`ValidColumns`] for each hardware-supported power-of-two column width.
macro_rules! valid_columns {
    ($($n:literal),+) => {
        $(
            impl sealed::Columns<$n> for () {}
            impl ValidColumns<$n> for () {}
        )+
    };
}
valid_columns!(32, 64, 128, 256, 512);

/// Collective tensor-memory allocator for SM100 kernels (`utils.TmemAllocator`).
///
/// Mirrors CuTeDSL's `cutlass.utils.TmemAllocator`: `COLUMNS` is a power of two in
/// `32..=512` and `GROUP` is `1` (single-CTA) or `2` (two-CTA).
pub struct TmemAllocator<const COLUMNS: u32, const GROUP: u32 = 1>;

impl<const COLUMNS: u32, const GROUP: u32> TmemAllocator<COLUMNS, GROUP>
where
    (): ValidColumns<COLUMNS>,
{
    /// Validate supported column counts and CTA group sizes at compile time.
    const VALID: () = {
        assert!(COLUMNS >= 32 && COLUMNS <= 512 && COLUMNS.is_power_of_two());
        assert!(GROUP == 1 || GROUP == 2);
    };

    /// Collectively allocate `COLUMNS` TMEM columns into the shared `token` slot (`TmemAllocator.allocate`).
    ///
    /// # Safety
    /// All 32 lanes of the allocating warp (in one CTA for `GROUP = 1`, or both
    /// CTAs for `GROUP = 2`) must participate with a live aligned shared `u32` slot.
    #[inline(always)]
    pub unsafe fn allocate(token: *mut u32) {
        let () = Self::VALID;
        unsafe {
            if GROUP == 1 {
                tcgen05::tcgen05_alloc(token, COLUMNS);
            } else {
                tcgen05::tcgen05_alloc_cg2(token, COLUMNS);
            }
        }
    }

    /// Relinquish the warp's TMEM allocation permit (`TmemAllocator.relinquish_alloc_permit`).
    ///
    /// # Safety
    /// All 32 lanes of the allocating warp must call this after [`Self::allocate`]
    /// and before kernel exit.
    #[inline(always)]
    pub unsafe fn relinquish_alloc_permit() {
        let () = Self::VALID;
        if GROUP == 1 {
            tcgen05::tcgen05_relinquish_alloc_permit();
        } else {
            tcgen05::tcgen05_relinquish_alloc_permit_cg2();
        }
    }

    /// Read the encoded TMEM base address from an initialized shared `token` slot (`TmemAllocator.retrieve_ptr`).
    ///
    /// # Safety
    /// Allocation into `token` must have completed and been synchronized across
    /// all reader warps.
    #[must_use]
    #[inline(always)]
    pub unsafe fn retrieve_ptr(token: *const u32) -> u32 {
        let () = Self::VALID;
        unsafe { *token }
    }

    /// Collectively deallocate `COLUMNS` TMEM columns (`TmemAllocator.free`).
    ///
    /// # Safety
    /// All asynchronous MMA and TMEM loads/stores must be complete and synchronized
    /// across all participating warps/CTAs before the allocating warp calls this.
    #[inline(always)]
    pub unsafe fn free(token: *const u32) {
        let () = Self::VALID;
        unsafe {
            let addr = *token;
            if GROUP == 1 {
                tcgen05::tcgen05_dealloc(addr, COLUMNS);
            } else {
                tcgen05::tcgen05_dealloc_cg2(addr, COLUMNS);
            }
        }
    }

    /// Bind the shared `u32` TMEM address slot at `smem_base + byte_offset` (`TmemAllocator.create`).
    ///
    /// # Safety
    /// `smem_base.add(byte_offset)` must be 4-byte aligned and point to a valid shared `u32` slot.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_bytes(smem_base: *mut u8, byte_offset: usize) -> *mut u32 {
        unsafe { smem_base.add(byte_offset).cast::<u32>() }
    }
}

/// Allocate one tensor-memory region collectively in a two-CTA cluster.
pub struct Sm100Tmem<const COLUMNS: u32>;
impl<const COLUMNS: u32> Sm100Tmem<COLUMNS> {
    /// Restrict allocation columns to supported powers of two from 32 through 512.
    const VALID: () = assert!(COLUMNS >= 32 && COLUMNS <= 512 && COLUMNS.is_power_of_two());

    /// Allocate columns and relinquish the allocation permit.
    ///
    /// # Safety
    /// Every lane of the allocating warp in both CTAs must participate.
    #[inline(always)]
    pub unsafe fn alloc(token: *mut u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_alloc::<COLUMNS>(token) }
    }

    /// Return a previously allocated region after both CTAs finish using it.
    ///
    /// # Safety
    /// All asynchronous MMA and loads must be complete and the two CTAs must rendezvous.
    #[inline(always)]
    pub unsafe fn dealloc(address: u32) {
        let () = Self::VALID;
        unsafe { __compiler::sm100_tmem_dealloc::<COLUMNS>(address) }
    }
}

/// Two-CTA cluster rendezvous barrier ordering collective TMEM deallocation.
#[derive(Clone, Copy)]
pub struct Sm100DeallocBarrier {
    /// Local shared 64-bit mbarrier at identical byte offsets in both CTAs.
    bar: *mut Barrier,
}

impl Sm100DeallocBarrier {
    /// Wrap a shared-memory barrier slot reserved for two-CTA TMEM deallocation.
    ///
    /// # Safety
    /// `bar` must point to a live, 8-byte-aligned shared barrier in both CTAs.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(bar: *mut Barrier) -> Self {
        Self { bar }
    }

    /// Initialize the deallocation barrier for 32 remote warp-zero arrivals.
    ///
    /// # Safety
    /// Call once from one elected lane in each CTA before cluster publication.
    #[inline(always)]
    pub unsafe fn init(&self) {
        unsafe {
            ManagedBarrier::<Uninit, GeneralBarrier>::from_static(self.bar).init_instruction(32);
        }
    }

    /// Synchronize warp zero, arrive on the peer CTA's barrier, and wait until both CTAs finish.
    ///
    /// # Safety
    /// All 32 lanes of warp zero in both CTAs must call this operation after completing all TMEM loads.
    #[inline(always)]
    pub unsafe fn sync_cluster(&self, rank: u32) {
        unsafe {
            crate::arch::NamedBarrier::<3, 32>::sync();
            let peer = cuda_device::cluster::map_shared_rank_mut(self.bar, rank ^ 1);
            let offset =
                cuda_device::shared::cvta_generic_to_cluster_shared_offset(peer.cast::<u8>());
            cuda_device::barrier::mbarrier_arrive_cluster(offset);
            let ready = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(self.bar);
            while !cuda_device::barrier::mbarrier_try_wait_parity_cluster(ready.as_ptr(), 0) {}
        }
    }
}

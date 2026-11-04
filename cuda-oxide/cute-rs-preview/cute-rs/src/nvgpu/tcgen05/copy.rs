/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor-memory (`TMEM`) copy atoms and register fragments (`cutlass.cute.nvgpu.tcgen05.copy`).

use super::__compiler;
use crate::arch;
use crate::layout::ReifySmem2D;
use crate::nvgpu::cpasync::TensorMap;
use core::marker::PhantomData;

/// Pack a warp's 32-row group offset and a column displacement into a live TMEM address.
#[must_use]
#[inline(always)]
pub const fn tmem_row_address(base: u32, warp_in_group: u32, column_offset: u32) -> u32 {
    base + ((warp_in_group * 32) << 16) + column_offset
}

/// Warp-collective 32-data-path TMEM register fragment (`tcgen05.Ld32x32bOp` / `St32x32bOp`).
pub struct TmemFragment<const REGS: usize> {
    /// This lane's FP32 register values.
    pub values: [f32; REGS],
}

/// Implement one native TMEM register width with explicit load/store completion variants.
macro_rules! fragment {
    ($n:literal,$load:ident,$store:ident, [$($idx:literal),+]) => {
        impl TmemFragment<$n> {
            /// Load a fragment and wait for the warp's preceding TMEM loads.
            /// # Safety
            /// All 32 lanes must execute with the same live, row-group-aligned
            /// TMEM address after the producer's completion and thread fence.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                unsafe { arch::fence_view_async_tmem_load() };
                result
            }
            /// Load without a completion wait, retaining register dependencies.
            /// # Safety
            /// Same participation and producer obligations as `load`.
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                let raw = unsafe { arch::$load(address) };
                Self {
                    values: [$(f32::from_bits(raw[$idx])),+],
                }
            }
            /// Store a fragment and wait for the warp's preceding TMEM stores.
            /// # Safety
            /// All 32 lanes must participate with the same live TMEM address.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                unsafe { arch::fence_view_async_tmem_store() };
            }
            /// Store without waiting, allowing disjoint TMEM chunks to overlap.
            /// # Safety
            /// Same participation and lifetime requirements as `store`.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let raw = [$(self.values[$idx].to_bits()),+];
                unsafe { Self::store_packed_deferred(address, raw) }
            }
            /// Store packed words unchanged (low FP16 lane in the low 16 bits).
            /// # Safety
            /// Same collective and lifetime requirements as `store`.
            #[inline(always)]
            pub unsafe fn store_packed(address: u32, raw: [u32; $n]) {
                unsafe { Self::store_packed_deferred(address, raw) };
                unsafe { arch::fence_view_async_tmem_store() };
            }
            /// Store packed words without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                unsafe { arch::$store(address, raw) };
            }
        }
    };
}
fragment!(2, tmem_load_32x32b_x2, tmem_store_32x32b_x2, [0, 1]);
fragment!(4, tmem_load_32x32b_x4, tmem_store_32x32b_x4, [0, 1, 2, 3]);
fragment!(
    8,
    tmem_load_32x32b_x8,
    tmem_store_32x32b_x8,
    [0, 1, 2, 3, 4, 5, 6, 7]
);
fragment!(
    16,
    tmem_load_32x32b_x16,
    tmem_store_32x32b_x16,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
);
fragment!(
    32,
    tmem_load_32x32b_x32,
    tmem_store_32x32b_x32,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31
    ]
);

/// Build a wider logical fragment from fixed x32 TMEM instruction chunks.
macro_rules! chunked_fragment {
    ($n:literal; $($offset:literal => $chunk:ident),+) => {
        impl TmemFragment<$n> {
            /// Load the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live row-group-aligned TMEM address.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                unsafe { arch::fence_view_async_tmem_load() };
                result
            }
            /// Load x32 chunks without a completion wait.
            /// # Safety
            /// Same participation and producer requirements as `load`.
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                $(let $chunk = unsafe { arch::tmem_load_32x32b_x32(address + $offset) };)+
                let mut values = [0.0; $n];
                $(
                    let mut i = 0;
                    while i < 32 {
                        values[$offset + i] = f32::from_bits($chunk[i]);
                        i += 1;
                    }
                )+
                Self { values }
            }
            /// Store the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live TMEM address.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                unsafe { arch::fence_view_async_tmem_store() };
            }
            /// Store x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and lifetime requirements as `store`.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let mut raw = [0; $n];
                let mut i = 0;
                while i < $n {
                    raw[i] = self.values[i].to_bits();
                    i += 1;
                }
                unsafe { Self::store_packed_deferred(address, raw) }
            }
            /// Store packed FP16 words in x32 chunks, preserving every bit.
            /// # Safety
            /// Same participation, lifetime, and fence obligations as `store`.
            #[inline(always)]
            pub unsafe fn store_packed(address: u32, raw: [u32; $n]) {
                unsafe { Self::store_packed_deferred(address, raw) };
                unsafe { arch::fence_view_async_tmem_store() };
            }
            /// Store packed x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                $(
                    let mut chunk = [0; 32];
                    let mut i = 0;
                    while i < 32 {
                        chunk[i] = raw[$offset + i];
                        i += 1;
                    }
                    unsafe { arch::tmem_store_32x32b_x32(address + $offset, chunk) };
                )+
            }
        }
    }
}
chunked_fragment!(64; 0 => a, 32 => b);
chunked_fragment!(128; 0 => a, 32 => b, 64 => c, 96 => d);

/// Convert one 128x32 FP32 accumulator slice to an FP16 tile in shared memory.
pub struct Sm100TmemEpilogue<Layout>(
    #[doc = "Carries the shared output layout required by readback and TMA."] PhantomData<Layout>,
);
impl<Layout: ReifySmem2D> Sm100TmemEpilogue<Layout> {
    /// Load a lane's 32 accumulators, add its row bias, and convert to FP16.
    ///
    /// # Safety
    /// All lanes of the four epilogue warps participate after accumulator wait.
    #[inline(always)]
    pub unsafe fn store(address: u32, dst: *mut f16, tid: u32, bias: f32) {
        unsafe { __compiler::sm100_tmem_epilogue::<Layout>(address, dst, tid, bias) }
    }

    /// Select one warp's 32 TMEM rows and one 32-column accumulator slice.
    #[inline(always)]
    pub const fn slice_address(base: u32, warp: u32, subtile: u32) -> u32 {
        (((base >> 16) + warp * 32) << 16) | ((base & 0xffff) + subtile * 32)
    }

    /// Start the FP16 tile's asynchronous shared-to-global TMA store.
    ///
    /// # Safety
    /// One elected store issuer calls after all writers publish their shared writes.
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

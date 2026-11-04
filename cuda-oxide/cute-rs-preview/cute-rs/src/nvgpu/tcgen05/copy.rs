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

impl<const REGS: usize> TmemFragment<REGS> {
    /// View this lane's TMEM fragment as an immutable [`TensorSSA`](crate::tensor::TensorSSA) register vector.
    #[must_use]
    #[inline(always)]
    pub const fn to_tensor_ssa(&self) -> crate::tensor::TensorSSA<f32, REGS> {
        crate::tensor::TensorSSA::from_array(self.values)
    }

    /// Extract a contiguous `LEN`-element [`TensorSSA`](crate::tensor::TensorSSA) sub-vector starting at `OFFSET`.
    #[must_use]
    #[inline(always)]
    pub fn slice<const OFFSET: usize, const LEN: usize>(
        &self,
    ) -> crate::tensor::TensorSSA<f32, LEN> {
        self.to_tensor_ssa().slice::<OFFSET, LEN>()
    }
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
                    values: [$(f32::from_bits(raw.data[$idx])),+],
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
fragment!(
    64,
    tmem_load_32x32b_x64,
    tmem_store_32x32b_x64,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    ]
);

/// Expand a 32-element raw `u32` chunk into 32 `f32::from_bits($c.data[idx])` expressions.
macro_rules! unpack32 {
    ($c:ident; $($idx:literal),+) => {
        [$(f32::from_bits($c.data[$idx])),+]
    };
}

/// Expand 32 `self.values[$offset + idx].to_bits()` expressions into a `[u32; 32]` array.
macro_rules! pack32_f32 {
    ($self:ident, $offset:literal; $($idx:literal),+) => {
        [$($self.values[$offset + $idx].to_bits()),+]
    };
}

/// Expand 32 `raw[$offset + idx]` expressions into a `[u32; 32]` array.
macro_rules! pack32_u32 {
    ($raw:ident, $offset:literal; $($idx:literal),+) => {
        [$($raw[$offset + $idx]),+]
    };
}

impl TmemFragment<128> {
    /// Load the 128-register fragment in four x32 chunks with one completion wait.
    /// # Safety
    /// All lanes participate with the same live row-group-aligned TMEM address.
    #[inline(always)]
    pub unsafe fn load(address: u32) -> Self {
        let result = unsafe { Self::load_deferred(address) };
        unsafe { arch::fence_view_async_tmem_load() };
        result
    }
    /// Load four x32 chunks without a completion wait using constant indices (`0` stack spills).
    /// # Safety
    /// Same participation and producer requirements as `load`.
    #[inline(always)]
    pub unsafe fn load_deferred(address: u32) -> Self {
        let a = unsafe { arch::tmem_load_32x32b_x32(address) };
        let b = unsafe { arch::tmem_load_32x32b_x32(address + 32) };
        let c = unsafe { arch::tmem_load_32x32b_x32(address + 64) };
        let d = unsafe { arch::tmem_load_32x32b_x32(address + 96) };
        let ua = unpack32!(a; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let ub = unpack32!(b; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let uc = unpack32!(c; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let ud = unpack32!(d; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        Self {
            values: [
                ua[0], ua[1], ua[2], ua[3], ua[4], ua[5], ua[6], ua[7], ua[8], ua[9], ua[10],
                ua[11], ua[12], ua[13], ua[14], ua[15], ua[16], ua[17], ua[18], ua[19], ua[20],
                ua[21], ua[22], ua[23], ua[24], ua[25], ua[26], ua[27], ua[28], ua[29], ua[30],
                ua[31], ub[0], ub[1], ub[2], ub[3], ub[4], ub[5], ub[6], ub[7], ub[8], ub[9],
                ub[10], ub[11], ub[12], ub[13], ub[14], ub[15], ub[16], ub[17], ub[18], ub[19],
                ub[20], ub[21], ub[22], ub[23], ub[24], ub[25], ub[26], ub[27], ub[28], ub[29],
                ub[30], ub[31], uc[0], uc[1], uc[2], uc[3], uc[4], uc[5], uc[6], uc[7], uc[8],
                uc[9], uc[10], uc[11], uc[12], uc[13], uc[14], uc[15], uc[16], uc[17], uc[18],
                uc[19], uc[20], uc[21], uc[22], uc[23], uc[24], uc[25], uc[26], uc[27], uc[28],
                uc[29], uc[30], uc[31], ud[0], ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7],
                ud[8], ud[9], ud[10], ud[11], ud[12], ud[13], ud[14], ud[15], ud[16], ud[17],
                ud[18], ud[19], ud[20], ud[21], ud[22], ud[23], ud[24], ud[25], ud[26], ud[27],
                ud[28], ud[29], ud[30], ud[31],
            ],
        }
    }
    /// Store the 128-register fragment in four x32 chunks with one completion wait.
    /// # Safety
    /// All lanes participate with the same live TMEM address.
    #[inline(always)]
    pub unsafe fn store(&self, address: u32) {
        unsafe { self.store_deferred(address) };
        unsafe { arch::fence_view_async_tmem_store() };
    }
    /// Store four x32 chunks without waiting for completion.
    /// # Safety
    /// Same participation and lifetime requirements as `store`.
    #[inline(always)]
    pub unsafe fn store_deferred(&self, address: u32) {
        let c0 = pack32_f32!(self, 0; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c1 = pack32_f32!(self, 32; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c2 = pack32_f32!(self, 64; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c3 = pack32_f32!(self, 96; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        unsafe {
            arch::tmem_store_32x32b_x32(address, c0);
            arch::tmem_store_32x32b_x32(address + 32, c1);
            arch::tmem_store_32x32b_x32(address + 64, c2);
            arch::tmem_store_32x32b_x32(address + 96, c3);
        }
    }
    /// Store packed FP16 words in four x32 chunks, preserving every bit.
    /// # Safety
    /// Same participation, lifetime, and fence obligations as `store`.
    #[inline(always)]
    pub unsafe fn store_packed(address: u32, raw: [u32; 128]) {
        unsafe { Self::store_packed_deferred(address, raw) };
        unsafe { arch::fence_view_async_tmem_store() };
    }
    /// Store packed x32 chunks without waiting for completion.
    /// # Safety
    /// Same participation and explicit completion obligations as `store_deferred`.
    #[inline(always)]
    pub unsafe fn store_packed_deferred(address: u32, raw: [u32; 128]) {
        let c0 = pack32_u32!(raw, 0; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c1 = pack32_u32!(raw, 32; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c2 = pack32_u32!(raw, 64; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let c3 = pack32_u32!(raw, 96; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        unsafe {
            arch::tmem_store_32x32b_x32(address, c0);
            arch::tmem_store_32x32b_x32(address + 32, c1);
            arch::tmem_store_32x32b_x32(address + 64, c2);
            arch::tmem_store_32x32b_x32(address + 96, c3);
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

    /// Start the FP16/BF16 tile's asynchronous shared-to-global TMA store.
    ///
    /// # Safety
    /// One elected store issuer calls after all writers publish their shared writes.
    #[inline(always)]
    pub unsafe fn store_async<T>(
        desc: *const TensorMap<T, Layout>,
        src: *const T,
        row: u32,
        column: u32,
    ) {
        unsafe {
            __compiler::sm100_tma_store::<Layout>(
                desc.cast::<TensorMap<f16, Layout>>(),
                src.cast::<f16>(),
                row,
                column,
            )
        }
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-collective register fragments for CTM tensor-memory loads and stores.
//! Conversion and chunking preserve the explicit completion choices of each API.

use ctm_rs::{
    groups::CtaGroup2,
    memory::CuSimd,
    tcgen05::{
        self,
        tmem::{self, TmemGuard, TmemReady, TmemUninit},
    },
    thread,
};

/// Pack a warp's 32-row group offset and a column displacement into a live TMEM address.
///
/// Tensor memory encodes the row offset in the upper 16 bits (`(warp_in_group * 32) << 16`)
/// and the column offset in the lower 16 bits. `warp_in_group` is in `0..4` within a
/// 128-thread warpgroup.
#[must_use]
#[inline(always)]
pub const fn tmem_row_address(base: u32, warp_in_group: u32, column_offset: u32) -> u32 {
    base + ((warp_in_group * 32) << 16) + column_offset
}

/// Collective tensor-memory allocator for SM100 attention kernels (`utils.TmemAllocator`).
///
/// Mirrors CuTeDSL's `cutlass.utils.TmemAllocator`: `COLUMNS` is a power of two in
/// `32..=512` and `GROUP` is `1` (single-CTA FMHA) or `2` (two-CTA MLA decode).
pub struct TmemAllocator<const COLUMNS: u32, const GROUP: u32 = 1>;

impl<const COLUMNS: u32, const GROUP: u32> TmemAllocator<COLUMNS, GROUP>
where
    (): tmem::ValidColumns<COLUMNS>,
{
    /// Validate supported column counts and CTA group sizes at compile time.
    const VALID: () = {
        assert!(COLUMNS >= 32 && COLUMNS <= 512 && COLUMNS.is_power_of_two());
        assert!(GROUP == 1 || GROUP == 2);
    };

    /// Collectively allocate `COLUMNS` TMEM columns into the shared `token` slot (`TmemAllocator.allocate`).
    ///
    /// Does not relinquish the allocation permit; pair with [`Self::relinquish_alloc_permit`]
    /// and [`Self::free`].
    ///
    /// # Safety
    /// All 32 lanes of the allocating warp (in one CTA for `GROUP = 1`, or both
    /// CTAs for `GROUP = 2`) must participate with a live aligned shared `u32` slot.
    #[inline(always)]
    pub unsafe fn allocate(token: *mut u32) {
        let () = Self::VALID;
        unsafe {
            if GROUP == 1 {
                TmemGuard::<TmemUninit, COLUMNS>::from_raw_slot(token).alloc_collective();
            } else {
                TmemGuard::<TmemUninit, COLUMNS, CtaGroup2>::from_raw_slot(token)
                    .alloc_collective();
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
        unsafe {
            if GROUP == 1 {
                tcgen05::relinquish_alloc_permit();
            } else {
                tcgen05::relinquish_alloc_permit_cg2();
            }
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
        unsafe {
            if GROUP == 1 {
                TmemGuard::<TmemReady, COLUMNS>::adopt_ready(token.cast_mut()).raw_address()
            } else {
                TmemGuard::<TmemReady, COLUMNS, CtaGroup2>::adopt_ready(token.cast_mut())
                    .raw_address()
            }
        }
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
            if GROUP == 1 {
                TmemGuard::<TmemReady, COLUMNS>::adopt_ready(token.cast_mut()).dealloc_collective();
            } else {
                TmemGuard::<TmemReady, COLUMNS, CtaGroup2>::adopt_ready(token.cast_mut())
                    .dealloc_collective();
            }
        }
    }
}

/// Warp-collective 32-data-path TMEM register fragment, one row per lane.
pub struct TmemFragment<const REGS: usize> {
    /// This lane's FP32 register values.
    pub values: [f32; REGS],
}
/// Implement one native TMEM register width with explicit load/store completion variants.
macro_rules! fragment {
    ($n:literal,$load:ident,$store:ident) => {
        impl TmemFragment<$n> {
            /// Load a fragment and wait for the warp's preceding TMEM loads.
            /// # Safety
            /// All 32 lanes must execute with the same live, row-group-aligned
            /// TMEM address after the producer's completion and thread fence.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::load_wait() };
                result
            }
            /// Load without a completion wait, retaining register dependencies.
            /// # Safety
            /// Same participation and producer obligations as `load`. Only true
            /// register dependencies order subsequent operations. The caller
            /// must wait before memory anti-dependencies or a consumer-release
            /// handoff, and apply the required before-thread-sync fence.
            /// See the [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                let raw = unsafe { tmem::$load(address) }.to_array();
                let mut values = [0.0; $n];
                let mut i = 0;
                while i < $n {
                    thread::__unroll_config::<0>();
                    values[i] = f32::from_bits(raw[i]);
                    i += 1;
                }
                Self { values }
            }
            /// Store a fragment and wait for the warp's preceding TMEM stores.
            /// # Safety
            /// All 32 lanes must participate with the same live TMEM address.
            /// Complete consumer handoffs only after the returned store wait.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::store_wait() };
            }
            /// Store without waiting, allowing disjoint TMEM chunks to overlap.
            /// # Safety
            /// Same participation and lifetime requirements as `store`. The
            /// caller must wait for stores before conflicting memory operations
            /// or completion handoffs, with the required thread-sync fence.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let mut raw = [0u32; $n];
                let mut i = 0;
                while i < $n {
                    thread::__unroll_config::<0>();
                    raw[i] = self.values[i].to_bits();
                    i += 1;
                }
                unsafe { Self::store_packed_deferred(address, raw) }
            }
            /// Store packed words unchanged (low FP16 lane in the low 16 bits).
            /// # Safety
            /// Same collective and lifetime requirements as `store`.
            #[inline(always)]
            pub unsafe fn store_packed(address: u32, raw: [u32; $n]) {
                unsafe { Self::store_packed_deferred(address, raw) };
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::store_wait() };
            }
            /// Store packed words without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as
            /// `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                unsafe { tmem::$store(address, CuSimd::new(raw)) };
            }
        }
    };
}
fragment!(2, load_32x32b_x2, store_32x32b_x2);
fragment!(4, load_32x32b_x4, store_32x32b_x4);
fragment!(8, load_32x32b_x8, store_32x32b_x8);
fragment!(16, load_32x32b_x16, store_32x32b_x16);
fragment!(32, load_32x32b_x32, store_32x32b_x32);
// The attention references transfer larger logical fragments as repeated x32
// instructions. Wide native transfers pass standalone probes but fail in the
// full attention compilation context with the pinned compiler.
/// Build a wider logical fragment from fixed x32 TMEM instruction chunks.
macro_rules! chunked_fragment {
    ($n:literal; $($offset:literal => $chunk:ident),+) => {
        impl TmemFragment<$n> {
            /// Load the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live row-group-aligned TMEM
            /// address after producer completion and the thread-sync fence.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::load_wait() };
                result
            }
            /// Load x32 chunks without a completion wait.
            /// # Safety
            /// Same participation and producer requirements as `load`. Only true
            /// register dependencies provide ordering. Wait before memory
            /// anti-dependencies or a consumer-release handoff, and apply the
            /// required before-thread-sync fence.
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                $(let $chunk = unsafe { tmem::load_32x32b_x32(address + $offset) };)+
                let mut values = [0.0; $n];
                $(
                    let raw = $chunk.to_array();
                    let mut i = 0;
                    while i < 32 {
                        thread::__unroll_config::<0>();
                        values[$offset + i] = f32::from_bits(raw[i]);
                        i += 1;
                    }
                )+
                Self { values }
            }
            /// Store the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live TMEM address. Publish
            /// completion only after a before-thread-sync fence by every lane.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::store_wait() };
            }
            /// Store x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and lifetime requirements as `store`. Wait
            /// before conflicting memory operations or completion handoffs,
            /// with the required thread-sync fence.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let mut raw = [0; $n];
                let mut i = 0;
                while i < $n {
                    thread::__unroll_config::<0>();
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
                // SAFETY: the caller supplies the full-warp participation contract.
                unsafe { tmem::store_wait() };
            }
            /// Store packed x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as
            /// `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                $(
                    let mut chunk = [0; 32];
                    let mut i = 0;
                    while i < 32 {
                        thread::__unroll_config::<0>();
                        chunk[i] = raw[$offset + i];
                        i += 1;
                    }
                    unsafe { tmem::store_32x32b_x32(address + $offset, CuSimd::new(chunk)) };
                )+
            }
        }
    }
}
chunked_fragment!(64; 0 => a, 32 => b);
chunked_fragment!(128; 0 => a, 32 => b, 64 => c, 96 => d);

impl TmemFragment<128> {
    /// Compute the per-head reciprocal root-mean-square scale `1 / sqrt(mean(x^2) + eps)`
    /// across all 128 FP32 columns of this thread's row fragment using `cute.arch.rsqrt`.
    #[must_use]
    #[inline(always)]
    pub fn rms_inv_std(&self, eps: f32) -> f32 {
        let mut sum_sq = 0.0f32;
        let mut j = 0usize;
        while j < 32 {
            thread::__unroll_config::<0>();
            let v0 = self.values[j];
            let v1 = self.values[32 + j];
            let v2 = self.values[64 + j];
            let v3 = self.values[96 + j];
            sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            j += 1;
        }
        crate::arch::rsqrt_f32(sum_sq * (1.0f32 / 128.0f32) + eps)
    }

    /// Scale all 128 FP32 columns by `scale` (and optional per-column `gamma` weights),
    /// quantize pairs to `Float8E4M3FN` (`cvt.rn.satfinite.e4m3x2.f32`), and issue four
    /// 256-bit (`32`-byte) `L1::no_allocate` global stores (`cute.copy`).
    ///
    /// # Safety
    /// `dst` must be 32-byte aligned and point to 128 writable bytes in global memory.
    /// When `gamma` is `Some(ptr)`, `ptr` must point to 128 readable `f32` values.
    #[inline(always)]
    pub unsafe fn quantize_and_store_e4m3_no_allocate(
        &self,
        scale: f32,
        gamma: Option<*const f32>,
        dst: *mut u8,
    ) {
        let dst_u32 = dst.cast::<u32>();
        let mut chunk = 0usize;
        while chunk < 4 {
            thread::__unroll_config::<0>();
            let base = chunk * 32;
            let mut packed = [0u32; 8];
            let mut w = 0usize;
            while w < 8 {
                thread::__unroll_config::<0>();
                let idx = base + w * 4;
                let (v0, v1, v2, v3) = if let Some(g_ptr) = gamma {
                    unsafe {
                        (
                            self.values[idx] * scale * (*g_ptr.add(idx)),
                            self.values[idx + 1] * scale * (*g_ptr.add(idx + 1)),
                            self.values[idx + 2] * scale * (*g_ptr.add(idx + 2)),
                            self.values[idx + 3] * scale * (*g_ptr.add(idx + 3)),
                        )
                    }
                } else {
                    (
                        self.values[idx] * scale,
                        self.values[idx + 1] * scale,
                        self.values[idx + 2] * scale,
                        self.values[idx + 3] * scale,
                    )
                };
                let lo16 = ctm_rs::convert::cvt_rn_satfinite_e4m3x2_f32(v0, v1) as u32;
                let hi16 = ctm_rs::convert::cvt_rn_satfinite_e4m3x2_f32(v2, v3) as u32;
                packed[w] = lo16 | (hi16 << 16);
                w += 1;
            }
            unsafe {
                crate::store_global_no_allocate(
                    dst_u32.add(chunk * 8),
                    &crate::GlobalStoreTile(packed),
                );
            }
            chunk += 1;
        }
    }
}

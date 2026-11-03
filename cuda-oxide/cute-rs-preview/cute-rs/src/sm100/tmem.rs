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

/// Typed group-one SM100 TMEM epilogue staging `[N, M]` FP32 accumulators through a compile-time
/// shared-memory `Layout` (`ReifySmem2D`) for fused `ReLU²` + NVFP4 requantization or BF16 output.
pub struct Sm100TmemStagedEpilogue<Layout, const M: usize = 128> {
    /// Shared-memory staging base (`[Layout::SHAPE.0, M]` `f32` elements).
    /// Written by Warps `0..3` during TMEM unload and read by all 16 compute warps during quantization.
    base: *mut f32,
    /// Zero-byte marker carrying the compile-time shared-memory epilogue layout.
    /// Verifies that the shared tile width matches `M = 128`.
    marker: PhantomData<Layout>,
}

impl<Layout: ReifySmem2D, const M: usize> Sm100TmemStagedEpilogue<Layout, M> {
    /// Compile-time check that the shared-memory layout matches the `M` accumulator rows.
    /// Ensures each unloaded token row occupies `M = 128` contiguous `f32` elements.
    const VALID: () = {
        assert!(M == 128);
        assert!(Layout::SHAPE.1 == M as i64);
    };

    /// Bind an existing shared-memory buffer as the typed TMEM epilogue staging tile.
    ///
    /// # Safety
    /// `base` must point to live shared memory with at least `Layout::SHAPE.0 * M` `f32` elements.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f32) -> Self {
        let () = Self::VALID;
        Self {
            base,
            marker: PhantomData,
        }
    }

    /// Unload `[limit, 128]` FP32 accumulators from TMEM (`2×` pipelined `tcgen05.ld.32x32b.x16` with
    /// a single `load_wait()` after the loop), apply `ReLU²` in the shadow of TMEM load latency,
    /// and stage into the shared-memory tile using the `4 × u128` quad-interleaved column permutation.
    ///
    /// # Safety
    /// Must be called after the MMA completion barrier wait; synchronizes all 512 CTA threads before returning.
    #[inline(always)]
    pub unsafe fn unload_relu2_to_shared(
        &self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        g_scale: f32,
    ) {
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            if tid < M {
                let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                let col_perm = ((tid & 12) << 3) | ((tid >> 4) << 2) | (tid & 3);
                let mut n_base = 0usize;
                while n_base + 32 <= limit {
                    let r0 =
                        ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                            .to_array();
                    let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                        tmem_row_base + ((n_base + 16) as u32),
                    )
                    .to_array();
                    let mut c = 0usize;
                    while c < 16 {
                        let a0 = f32::from_bits(r0[c]) * g_scale;
                        let a1 = f32::from_bits(r1[c]) * g_scale;
                        let u0 = if a0 > 0.0 { a0 } else { 0.0 };
                        let u1 = if a1 > 0.0 { a1 } else { 0.0 };
                        *self.base.add((n_base + c) * 128 + col_perm) = u0 * u0;
                        *self.base.add((n_base + 16 + c) * 128 + col_perm) = u1 * u1;
                        c += 1;
                    }
                    n_base += 32;
                }
                while n_base < limit {
                    let regs =
                        ctm_rs::tcgen05::tmem::load_32x32b_x8(tmem_row_base + (n_base as u32))
                            .to_array();
                    let mut c = 0usize;
                    while c < 8 {
                        if n_base + c < limit {
                            let acc = f32::from_bits(regs[c]) * g_scale;
                            let relu = if acc > 0.0 { acc } else { 0.0 };
                            *self.base.add((n_base + c) * 128 + col_perm) = relu * relu;
                        }
                        c += 1;
                    }
                    n_base += 8;
                }
                ctm_rs::tcgen05::tmem::load_wait();
            }
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 512);
        }
    }

    /// Unload `[limit, 128]` FP32 accumulators from TMEM (`2×` pipelined `tcgen05.ld.32x32b.x16` with
    /// a single `load_wait()`), apply `ReLU²`, convert to `BFloat16`, and stage into `Slot 1` (`self.base + 64 KB`)
    /// so `Slot 0` (`b_free_bar_ptr`) can be released before TMEM unload even begins!
    ///
    /// # Safety
    /// Must be called after the MMA completion barrier wait; synchronizes all 512 CTA threads before returning.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_relu2_bf16_slot1(
        &self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        g_scale: f32,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_b_free: bool,
    ) {
        unsafe {
            if signal_b_free {
                ctm_rs::fence::fence_proxy_async_shared_cta();
                if tid == 0 {
                    let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                }
            }
            let smem_u16 = self.base.cast::<u8>().add(65536).cast::<u16>();
            ctm_rs::tcgen05::fence_after_thread_sync();
            if tid < M {
                let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                let col_perm = ((tid & 12) << 3) | ((tid >> 4) << 2) | (tid & 3);
                let mut n_base = 0usize;
                while n_base + 32 <= limit {
                    let r0 =
                        ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                            .to_array();
                    let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                        tmem_row_base + ((n_base + 16) as u32),
                    )
                    .to_array();
                    let mut c = 0usize;
                    while c < 16 {
                        let a0 = f32::from_bits(r0[c]) * g_scale;
                        let a1 = f32::from_bits(r1[c]) * g_scale;
                        let u0 = if a0 > 0.0 { a0 } else { 0.0 };
                        let u1 = if a1 > 0.0 { a1 } else { 0.0 };
                        let v0 = (u0 * u0).to_bits();
                        let v1 = (u1 * u1).to_bits();
                        *smem_u16.add((n_base + c) * 128 + col_perm) =
                            (v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16) as u16;
                        *smem_u16.add((n_base + 16 + c) * 128 + col_perm) =
                            (v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16) as u16;
                        c += 1;
                    }
                    n_base += 32;
                }
                ctm_rs::tcgen05::tmem::load_wait();
            }
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 512);
        }
    }

    /// Return a `u128`-aligned pointer to the first 4-float quad of `(n, sub_tile16)` in the staged shared tile
    /// (with the 4 quads of 4 floats spaced by `8` `u128`s so 4 `ld.shared.v4.b32` loads fetch all 16 floats!).
    ///
    /// # Safety
    /// `n < Layout::SHAPE.0 as usize` and `sub_tile16 < M / 16` must hold after [`Self::unload_relu2_to_shared`].
    #[must_use]
    #[inline(always)]
    pub const unsafe fn group16_u128_ptr(&self, n: usize, sub_tile16: usize) -> *const u128 {
        unsafe {
            self.base
                .cast_const()
                .cast::<u128>()
                .add(n * 32 + sub_tile16)
        }
    }

    /// Return a `u128`-aligned pointer to the first 8-`bf16` octuple of `(n, sub_tile16)` in `Slot 1` (`self.base + 64 KB`).
    ///
    /// # Safety
    /// `n < Layout::SHAPE.0 as usize` and `sub_tile16 < M / 16` must hold after [`Self::unload_relu2_bf16_slot1`].
    #[must_use]
    #[inline(always)]
    pub const unsafe fn group16_bf16_slot1_u128_ptr(
        &self,
        n: usize,
        sub_tile16: usize,
    ) -> *const u128 {
        unsafe {
            self.base
                .cast_const()
                .cast::<u8>()
                .add(65536)
                .cast::<u128>()
                .add(n * 16 + sub_tile16)
        }
    }

    /// Unload `[limit, 128]` raw `u32` accumulators from TMEM (`2×` pipelined `tcgen05.ld.32x32b.x16`
    /// with a single `load_wait()` after the loop) into `[256, 128]` `u32`,
    /// then fuse `g_scale * routing_weights[route]` + `BFloat16` conversion + `atom.global.add.noftz.bf16x2`!
    ///
    /// # Safety
    /// Must be called by all 512 compute threads (`warp_id < 16`) after the MMA completion barrier wait;
    /// `output_u32` must point to a zero-initialized `[num_tokens, hidden_size]` `BFloat16` buffer.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_and_atomic_add_bf16x2(
        &self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        valid_n: usize,
        tile_m128: usize,
        group: usize,
        expert: usize,
        g_scale: f32,
        group_routes: *const i32,
        routing_weights: *const crate::BFloat16,
        output_u32: *mut u32,
        hidden_size: usize,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_b_free: bool,
        use_grouped_out: bool,
    ) {
        unsafe {
            if use_grouped_out {
                // Multi-group (`B >= 2048`): convert `f32 * g_scale -> bf16` (`u16`) inside the TMEM unload loop
                // into `Slot 1` (`self.base + 64 KB`), release `Slot 0` (`b_free_bar_ptr`) immediately before unload,
                // read `Rows 128..255` into registers to release `Slot 1` (`b_free_bar_ptr + 1`) early, and write
                // contiguous `128`-bit (`u128`) vectors to `output_u32` (`grouped_out`) with zero atomic adds!
                if signal_b_free {
                    ctm_rs::fence::fence_proxy_async_shared_cta();
                    if tid == 0 {
                        let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                    }
                }
                let smem_epi_u16 = self.base.cast::<u8>().add(65536).cast::<u16>();
                let smem_epi_u128 = smem_epi_u16.cast_const().cast::<u128>();
                ctm_rs::tcgen05::fence_after_thread_sync();
                if tid < M {
                    let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                    let mut n_base = 0usize;
                    while n_base + 32 <= limit {
                        let r0 =
                            ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                                .to_array();
                        let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                            tmem_row_base + ((n_base + 16) as u32),
                        )
                        .to_array();
                        let mut c = 0usize;
                        while c < 16 {
                            let v0 = (f32::from_bits(r0[c]) * g_scale).to_bits();
                            let v1 = (f32::from_bits(r1[c]) * g_scale).to_bits();
                            let b0 = (v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16) as u16;
                            let b1 = (v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16) as u16;
                            *smem_epi_u16.add((n_base + c) * M + tid) = b0;
                            *smem_epi_u16.add((n_base + 16 + c) * M + tid) = b1;
                            c += 1;
                        }
                        n_base += 32;
                    }
                    ctm_rs::tcgen05::tmem::load_wait();
                }
                ctm_rs::tcgen05::fence_before_thread_sync();
                ctm_rs::barrier::barrier_cta_sync(1, 512);

                let col16 = tid & 15;
                let row_off32 = tid >> 4; // 0..31
                let dst_col_u128 = tile_m128 * 16 + col16;
                let grouped_u128 = output_u32.cast::<u128>();

                let mut k = 0usize;
                while k < 4 {
                    let n = k * 32 + row_off32;
                    let q = *smem_epi_u128.add(n * 16 + col16);
                    *grouped_u128.add((group * 256 + n) * 128 + dst_col_u128) = q;
                    k += 1;
                }

                let n4 = 128 + row_off32;
                let q4 = core::ptr::read_volatile(smem_epi_u128.add(n4 * 16 + col16));
                let q5 = core::ptr::read_volatile(smem_epi_u128.add((n4 + 32) * 16 + col16));
                let q6 = core::ptr::read_volatile(smem_epi_u128.add((n4 + 64) * 16 + col16));
                let q7 = core::ptr::read_volatile(smem_epi_u128.add((n4 + 96) * 16 + col16));
                if signal_b_free {
                    ctm_rs::fence::fence_proxy_async_shared_cta();
                    ctm_rs::barrier::barrier_cta_sync(1, 512);
                    if tid == 0 {
                        let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr.add(1));
                    }
                }
                *grouped_u128.add((group * 256 + n4) * 128 + dst_col_u128) = q4;
                *grouped_u128.add((group * 256 + n4 + 32) * 128 + dst_col_u128) = q5;
                *grouped_u128.add((group * 256 + n4 + 64) * 128 + dst_col_u128) = q6;
                *grouped_u128.add((group * 256 + n4 + 96) * 128 + dst_col_u128) = q7;
                return;
            }
            let smem_u32 = self.base.cast::<u32>();
            ctm_rs::tcgen05::fence_after_thread_sync();
            if tid < M {
                let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
                let mut n_base = 0usize;
                while n_base + 32 <= limit {
                    let r0 =
                        ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                            .to_array();
                    let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                        tmem_row_base + ((n_base + 16) as u32),
                    )
                    .to_array();
                    let mut c = 0usize;
                    while c < 16 {
                        *smem_u32.add((n_base + c) * M + tid) = r0[c];
                        *smem_u32.add((n_base + 16 + c) * M + tid) = r1[c];
                        c += 1;
                    }
                    n_base += 32;
                }
                while n_base < limit {
                    let regs =
                        ctm_rs::tcgen05::tmem::load_32x32b_x8(tmem_row_base + (n_base as u32))
                            .to_array();
                    let mut c = 0usize;
                    while c < 8 {
                        if n_base + c < limit {
                            *smem_u32.add((n_base + c) * M + tid) = regs[c];
                        }
                        c += 1;
                    }
                    n_base += 8;
                }
                ctm_rs::tcgen05::tmem::load_wait();
            }
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 512);

            let col4 = tid & 3; // 0..3 (`32` f32s = `8 × u128` per thread)
            let row_off = tid >> 2; // 0..127 (`128` tokens processed in parallel per pass — ONLY 2 passes for 256 tokens!)
            let out_col_u32_base = tile_m128 * 64 + col4 * 2;
            let h_u32 = hidden_size >> 1;
            let smem_u128 = self.base.cast_const().cast::<u128>();

            let half_n = valid_n.min(128);
            let half_mask = half_n - 1;
            let n_rot = (expert * (half_n >> 2)) & half_mask;
            let mut n_step = row_off;
            while n_step < valid_n {
                let n = (n_step & 128) | ((n_step + n_rot) & half_mask);
                if n < limit {
                    let route = *group_routes.add(group * 256 + n) as usize;
                    let token = route >> 2;
                    let sw = g_scale
                        * f32::from_bits(((*routing_weights.add(route)).to_bits() as u32) << 16);
                    let row_u128 = smem_u128.add(n * 32 + col4);
                    let q0 = *row_u128;
                    let q1 = *row_u128.add(4);
                    let q2 = *row_u128.add(8);
                    let q3 = *row_u128.add(12);
                    let q4 = *row_u128.add(16);
                    let q5 = *row_u128.add(20);
                    let q6 = *row_u128.add(24);
                    let q7 = *row_u128.add(28);
                    let dst_ptr = output_u32.add(token * h_u32 + out_col_u32_base);

                    let mut q_idx = 0usize;
                    let qs = [q0, q1, q2, q3, q4, q5, q6, q7];
                    while q_idx < 8 {
                        let q = qs[q_idx];
                        let v0 = (f32::from_bits((q & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v1 = (f32::from_bits(((q >> 32) & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v2 = (f32::from_bits(((q >> 64) & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v3 = (f32::from_bits((q >> 96) as u32) * sw).to_bits();
                        let b0 = v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16;
                        let b1 = v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16;
                        let b2 = v2.wrapping_add(0x7FFF + ((v2 >> 16) & 1)) >> 16;
                        let b3 = v3.wrapping_add(0x7FFF + ((v3 >> 16) & 1)) >> 16;
                        let d = dst_ptr.add(q_idx * 8);
                        let _ = ctm_rs::memory::atom_add_bf16x2(d, b0 | (b1 << 16));
                        let _ = ctm_rs::memory::atom_add_bf16x2(d.add(1), b2 | (b3 << 16));
                        q_idx += 1;
                    }
                }
                if signal_b_free && n_step < 128 {
                    ctm_rs::fence::fence_proxy_async_shared_cta();
                    ctm_rs::barrier::barrier_cta_sync(1, 512);
                    if tid == 0 {
                        let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                    }
                }
                n_step += 128;
            }
            if signal_b_free {
                ctm_rs::fence::fence_proxy_async_shared_cta();
                ctm_rs::barrier::barrier_cta_sync(1, 512);
                if tid == 0 {
                    let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr.add(1));
                }
            }
        }
    }

    /// Unload a `[limit, 128]` FP32 accumulator tile from `tmem_addr`, apply `ReLU²(x * g_scale)`,
    /// stage through `128 KB` shared memory (`Layout`), and requantize all 16-element groups into
    /// `256`-row TMA burst layout (`vals_u64_ptr` and `scales_out`) for `B <= 1,024`.
    ///
    /// # Safety
    /// All 512 threads of `Warps 0..15` must call this convergently after `mma_bar` completes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_relu2_and_quantize_256(
        &self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        tile_m128: usize,
        group: usize,
        g_scale: f32,
        vals_u64_ptr: *mut u64,
        scales_out: &mut crate::tensor::partition::StridedTensorMut<'_, u8>,
    ) {
        let sub_tile16 = tid & 7;
        let row_off64 = tid >> 3;
        let grp = tile_m128 * 8 + sub_tile16;
        let burst_inter = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 1024 + ((grp >> 1) & 1) * 512 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        unsafe {
            self.unload_relu2_to_shared(tmem_addr, warp_id, tid, limit, g_scale);
            let mut n = row_off64;
            while n < limit {
                let smem_u128 = self.group16_u128_ptr(n, sub_tile16);
                let q0 = *smem_u128;
                let q1 = *smem_u128.add(8);
                let q2 = *smem_u128.add(16);
                let q3 = *smem_u128.add(24);
                let vals = [
                    f32::from_bits((q0 & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q0 >> 32) & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q0 >> 64) & 0xFFFF_FFFF) as u32),
                    f32::from_bits((q0 >> 96) as u32),
                    f32::from_bits((q1 & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q1 >> 32) & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q1 >> 64) & 0xFFFF_FFFF) as u32),
                    f32::from_bits((q1 >> 96) as u32),
                    f32::from_bits((q2 & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q2 >> 32) & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q2 >> 64) & 0xFFFF_FFFF) as u32),
                    f32::from_bits((q2 >> 96) as u32),
                    f32::from_bits((q3 & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q3 >> 32) & 0xFFFF_FFFF) as u32),
                    f32::from_bits(((q3 >> 64) & 0xFFFF_FFFF) as u32),
                    f32::from_bits((q3 >> 96) as u32),
                ];
                let mut max_abs = 0.0f32;
                let mut i = 0usize;
                while i < 16 {
                    if vals[i] > max_abs {
                        max_abs = vals[i];
                    }
                    i += 1;
                }
                let clamped = (max_abs * (1.0 / 6.0)).clamp(0.0, 448.0);
                let scaled_e4m3 = clamped * f32::from_bits(7 << 23);
                let raw_code = ((scaled_e4m3.to_bits() + 0x80000) >> 20) as u8;
                let scale_code = if raw_code > 0x7e { 0x7e } else { raw_code };
                let scale_val =
                    f32::from_bits((scale_code as u32) << 20) * f32::from_bits(247 << 23);
                let sv_bits = scale_val.to_bits();
                let sv_exp = ((sv_bits >> 23) & 0xFF).max(64);
                let sv_mant = (sv_bits >> 20) & 7;
                let diff = ((0x776E_6255_4633_1C00u64 >> (sv_mant * 8)) & 0xFF) as u32;
                let inv_scale_e2m1 =
                    f32::from_bits((0x2000 - diff) << 16) * f32::from_bits((191 - sv_exp) << 23);
                let scale_idx = ((group * 16 + k_atom) * 2 + (n >> 7)) * 512
                    + crate::block_scaled::Sm120ScaleAtom::word_offset(n & 127) * 4
                    + k_in_atom;
                scales_out.store_at(scale_idx, scale_code);
                let mut packed_u64 = 0u64;
                let mut idx = 0usize;
                while idx < 16 {
                    let bits = (vals[idx] * inv_scale_e2m1).to_bits().min(0x01C0_0000);
                    let nib = ((bits + 0x0020_0000) >> 22) as u64;
                    packed_u64 |= nib << (idx * 4);
                    idx += 1;
                }
                let u64_idx = (group * 2 + burst_inter) * 8192 + sub_u64_off + n * 2;
                *vals_u64_ptr.add(u64_idx) = packed_u64;
                n += 64;
            }
        }
    }

    /// Unload a `[128, 128]` FP32 accumulator slice from `tmem_d` (`2 × tcgen05.ld.32x32b.x64`) in
    /// `Warps 0..3` (`128` threads, where thread `tid` owns Token `n = tid`), immediately release
    /// `tmem_free_bar_ptr` after the second load, apply `ReLU²` + NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`)
    /// quantization in registers (`0 KB` `SMEM_EPI`), and store coalesced `u128` values and `u32` scales!
    ///
    /// # Safety
    /// Must be called by all 128 threads of `Warps 0..3` after waiting on `mma_bar` for `tmem_d`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_relu2_and_quantize_128(
        &self,
        tmem_d: u32,
        warp_id: u32,
        tid: usize,
        tile_m128: usize,
        group: usize,
        g_scale: f32,
        vals_u128_ptr: *mut u128,
        scales_u32_ptr: *mut u32,
        tmem_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_tmem_free: bool,
    ) {
        /// Quantize 16 contiguous `f32` accumulators (`r64[base..base+16]`) into one packed 64-bit
        /// `Float4E2M1FN` word (`u64`) and one `Float8E4M3FN` scale (`u8`) with constant array indices.
        macro_rules! quant16 {
            ($r64:ident, $b:expr, $g_scale:ident, $scale_sq_div6:ident) => {{
                let mut max_raw_bits = 0u32;
                let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
                macro_rules! step_v {
                    ($vi:ident, $idx:expr) => {
                        let pos_bits = ($r64[$b + $idx] as i32).max(0) as u32;
                        max_raw_bits = max_raw_bits.max(pos_bits);
                        let u = f32::from_bits(pos_bits) * $g_scale;
                        $vi = u * u;
                    };
                }
                step_v!(v0, 0);
                step_v!(v1, 1);
                step_v!(v2, 2);
                step_v!(v3, 3);
                step_v!(v4, 4);
                step_v!(v5, 5);
                step_v!(v6, 6);
                step_v!(v7, 7);
                step_v!(v8, 8);
                step_v!(v9, 9);
                step_v!(v10, 10);
                step_v!(v11, 11);
                step_v!(v12, 12);
                step_v!(v13, 13);
                step_v!(v14, 14);
                step_v!(v15, 15);

                let max_raw = f32::from_bits(max_raw_bits);
                let amax_div6 = max_raw * max_raw * $scale_sq_div6;
                let clamped = f32::from_bits(amax_div6.to_bits().min(0x43E0_0000));
                let scaled_e4m3 = clamped * f32::from_bits(7 << 23);
                let scale_code = (((scaled_e4m3.to_bits() + 0x80000) >> 20).min(0x7e)) as u32;
                let scale_val = f32::from_bits(scale_code << 20) * f32::from_bits(247 << 23);
                let sv_bits = scale_val.to_bits();
                let sv_exp = ((sv_bits >> 23) & 0xFF).max(64);
                let sv_mant = (sv_bits >> 20) & 7;
                let diff = ((0x776E_6255_4633_1C00u64 >> (sv_mant * 8)) & 0xFF) as u32;
                let inv_e2m1 =
                    f32::from_bits((0x2000 - diff) << 16) * f32::from_bits((191 - sv_exp) << 23);
                macro_rules! e2m1_nib {
                    ($v:ident) => {{
                        let bits = ($v * inv_e2m1).to_bits().min(0x01C0_0000);
                        ((bits + 0x0020_0000) >> 22) as u64
                    }};
                }
                let packed_u64 = e2m1_nib!(v0)
                    | (e2m1_nib!(v1) << 4)
                    | (e2m1_nib!(v2) << 8)
                    | (e2m1_nib!(v3) << 12)
                    | (e2m1_nib!(v4) << 16)
                    | (e2m1_nib!(v5) << 20)
                    | (e2m1_nib!(v6) << 24)
                    | (e2m1_nib!(v7) << 28)
                    | (e2m1_nib!(v8) << 32)
                    | (e2m1_nib!(v9) << 36)
                    | (e2m1_nib!(v10) << 40)
                    | (e2m1_nib!(v11) << 44)
                    | (e2m1_nib!(v12) << 48)
                    | (e2m1_nib!(v13) << 52)
                    | (e2m1_nib!(v14) << 56)
                    | (e2m1_nib!(v15) << 60);
                (packed_u64, scale_code)
            }};
        }

        let n = tid;
        let sfb_word = ((n & 31) * 4) + ((n >> 5) & 3);
        let burst_inter = tile_m128 >> 2;
        let scale_sq_div6 = g_scale * g_scale * (1.0f32 / 6.0f32);
        let tmem_row_base = tmem_d + ((warp_id * 32) << 16);
        /// Unload and quantize one 64-column half (`$half_m = 0` or `1`) without loop overhead.
        macro_rules! half_quant {
            ($half_m:expr, $is_second:expr) => {{
                let r64 =
                    ctm_rs::tcgen05::tmem::load_32x32b_x64(tmem_row_base + (($half_m * 64) as u32))
                        .to_array();
                ctm_rs::tcgen05::tmem::load_wait();
                if $is_second {
                    ctm_rs::tcgen05::fence_before_thread_sync();
                    ctm_rs::barrier::barrier_cta_sync(1, 128);
                    if signal_tmem_free && tid == 0 {
                        let _ = ctm_rs::mbarrier::mbarrier_arrive(tmem_free_bar_ptr);
                    }
                }
                let (p0, s0) = quant16!(r64, 0, g_scale, scale_sq_div6);
                let (p1, s1) = quant16!(r64, 16, g_scale, scale_sq_div6);
                let (p2, s2) = quant16!(r64, 32, g_scale, scale_sq_div6);
                let (p3, s3) = quant16!(r64, 48, g_scale, scale_sq_div6);
                let sc4 = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
                let k_atom = tile_m128 * 2 + $half_m;
                let s_in_burst = k_atom & 7;
                let scale_u32_idx = (group * 16 + k_atom) * 128 + sfb_word;
                *scales_u32_ptr.add(scale_u32_idx) = sc4;
                let u128_base = (group * 2 + burst_inter) * 2048 + s_in_burst * 256 + n;
                *vals_u128_ptr.add(u128_base) = (p0 as u128) | ((p1 as u128) << 64);
                *vals_u128_ptr.add(u128_base + 128) = (p2 as u128) | ((p3 as u128) << 64);
            }};
        }
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            half_quant!(0, false);
            half_quant!(1, true);
        }
    }

    /// Unload a `[128, 128]` FP32 accumulator slice from `tmem_d` (`2 × tcgen05.ld.32x32b.x64`) in
    /// `Warps 0..3` (`128` threads), immediately release `tmem_free_bar_ptr`, convert to `bf16x2`
    /// (`cvt.rn.bf16x2.f32`), stage through 2-stage `8 KB` `B64`-swizzled shared tiles, and issue
    /// asynchronous 2D TMA Stores (`copy_tma_s2g_2d`) to `out_tma`!
    ///
    /// # Safety
    /// Must be called by all 128 threads of `Warps 0..3` after waiting on `mma_bar` for `tmem_d`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_and_tma_store_bf16_128<OutLayout: ReifySmem2D>(
        &self,
        tmem_d: u32,
        warp_id: u32,
        tid: usize,
        tile_m128: usize,
        group: usize,
        g_scale: f32,
        smem_epi0_u128: *mut u128,
        smem_epi1_u128: *mut u128,
        out_tma: &TensorMap<u16, OutLayout>,
        tmem_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_tmem_free: bool,
    ) {
        /// Pack 8 contiguous `f32` accumulators (`r64[base..base+8]`) scaled by `g_scale` into one `u128`
        /// of 8 `BFloat16` values using 3-input `IADD3` RNE rounding and constant indices.
        macro_rules! pack8_bf16 {
            ($r64:ident, $b:expr, $g_scale:ident) => {{
                macro_rules! rne_pair {
                    ($i0:expr, $i1:expr) => {{
                        let v0 = (f32::from_bits($r64[$b + $i0]) * $g_scale).to_bits();
                        let v1 = (f32::from_bits($r64[$b + $i1]) * $g_scale).to_bits();
                        let r0 = v0.wrapping_add(0x7FFF).wrapping_add((v0 >> 16) & 1);
                        let r1 = v1.wrapping_add(0x7FFF).wrapping_add((v1 >> 16) & 1);
                        (r0 >> 16) | (r1 & 0xFFFF_0000)
                    }};
                }
                let p0 = rne_pair!(0, 1);
                let p1 = rne_pair!(2, 3);
                let p2 = rne_pair!(4, 5);
                let p3 = rne_pair!(6, 7);
                (p0 as u128) | ((p1 as u128) << 32) | ((p2 as u128) << 64) | ((p3 as u128) << 96)
            }};
        }

        let u_swz = (tid >> 1) & 3;
        let tmem_row_base = tmem_d + ((warp_id * 32) << 16);
        let out_desc = core::ptr::from_ref(out_tma).cast::<cuda_device::TmaDescriptor>();
        /// Unload and TMA-store one 64-column half (`$half_m = 0` or `1`) without loop overhead.
        macro_rules! half_store {
            ($half_m:expr, $is_second:expr) => {{
                let r64 =
                    ctm_rs::tcgen05::tmem::load_32x32b_x64(tmem_row_base + (($half_m * 64) as u32))
                        .to_array();
                ctm_rs::tcgen05::tmem::load_wait();
                if $is_second {
                    ctm_rs::tcgen05::fence_before_thread_sync();
                    ctm_rs::barrier::barrier_cta_sync(1, 128);
                    if signal_tmem_free && tid == 0 {
                        let _ = ctm_rs::mbarrier::mbarrier_arrive(tmem_free_bar_ptr);
                    }
                }

                if tid == 0 {
                    ctm_rs::tma::cp_async_bulk_wait_group_read::<1>();
                }
                ctm_rs::barrier::barrier_cta_sync(1, 128);
                *smem_epi0_u128.add(tid * 4 + u_swz) = pack8_bf16!(r64, 0, g_scale);
                *smem_epi0_u128.add(tid * 4 + (1 ^ u_swz)) = pack8_bf16!(r64, 8, g_scale);
                *smem_epi0_u128.add(tid * 4 + (2 ^ u_swz)) = pack8_bf16!(r64, 16, g_scale);
                *smem_epi0_u128.add(tid * 4 + (3 ^ u_swz)) = pack8_bf16!(r64, 24, g_scale);
                ctm_rs::fence::fence_proxy_async_shared_cta();
                ctm_rs::barrier::barrier_cta_sync(1, 128);
                if tid == 0 {
                    cuda_device::tma::cp_async_bulk_tensor_2d_s2g(
                        smem_epi0_u128.cast_const().cast::<u8>(),
                        out_desc,
                        (tile_m128 * 128 + $half_m * 64) as i32,
                        (group * 128) as i32,
                    );
                    ctm_rs::tma::cp_async_bulk_commit_group();
                }

                if tid == 0 {
                    ctm_rs::tma::cp_async_bulk_wait_group_read::<1>();
                }
                ctm_rs::barrier::barrier_cta_sync(1, 128);
                *smem_epi1_u128.add(tid * 4 + u_swz) = pack8_bf16!(r64, 32, g_scale);
                *smem_epi1_u128.add(tid * 4 + (1 ^ u_swz)) = pack8_bf16!(r64, 40, g_scale);
                *smem_epi1_u128.add(tid * 4 + (2 ^ u_swz)) = pack8_bf16!(r64, 48, g_scale);
                *smem_epi1_u128.add(tid * 4 + (3 ^ u_swz)) = pack8_bf16!(r64, 56, g_scale);
                ctm_rs::fence::fence_proxy_async_shared_cta();
                ctm_rs::barrier::barrier_cta_sync(1, 128);
                if tid == 0 {
                    cuda_device::tma::cp_async_bulk_tensor_2d_s2g(
                        smem_epi1_u128.cast_const().cast::<u8>(),
                        out_desc,
                        (tile_m128 * 128 + $half_m * 64 + 32) as i32,
                        (group * 128) as i32,
                    );
                    ctm_rs::tma::cp_async_bulk_commit_group();
                }
            }};
        }
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            half_store!(0, false);
            half_store!(1, true);
        }
    }

    /// Unload `[limit, 128]` FP32 accumulators from `tmem_addr` across `Warps 0..3` (`128` threads),
    /// apply `ReLU²(x * g_scale)`, stage into `[128, 128]` `f32` shared memory (`self.base`), and
    /// requantize all 16-element groups across `0..limit` into `128`-row TMA burst layout with
    /// `inter_bursts` `K = 512` bursts (`e.g. 3` bursts for `I = 1,536`).
    ///
    /// # Safety
    /// All 128 threads of `Warps 0..3` must call this convergently after `mma_bar` completes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_relu2_and_quantize_active_128(
        &self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        tile_m128: usize,
        group: usize,
        g_scale: f32,
        inter_bursts: usize,
        vals_u64_ptr: *mut u64,
        scales_out: &mut crate::tensor::partition::StridedTensorMut<'_, u8>,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_b_free: bool,
    ) {
        let tmem_row_base = tmem_addr + (warp_id << 21);
        let col_perm = ((tid & 12) << 3) | ((tid >> 4) << 2) | (tid & 3);
        let sub_tile16 = tid & 7;
        let row_off16 = tid >> 3;
        let grp = tile_m128 * 8 + sub_tile16;
        let burst_inter = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        let inter_atoms = inter_bursts * 8;
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            let mut n_base = 0usize;
            while n_base < limit {
                let regs =
                    ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                        .to_array();
                macro_rules! st_relu2 {
                    ($c:expr) => {{
                        let n = n_base + $c;
                        if n < limit {
                            let a = f32::from_bits(regs[$c]) * g_scale;
                            let u = if a > 0.0 { a } else { 0.0 };
                            *self.base.add(n * 128 + col_perm) = u * u;
                        }
                    }};
                }
                st_relu2!(0);
                st_relu2!(1);
                st_relu2!(2);
                st_relu2!(3);
                st_relu2!(4);
                st_relu2!(5);
                st_relu2!(6);
                st_relu2!(7);
                st_relu2!(8);
                st_relu2!(9);
                st_relu2!(10);
                st_relu2!(11);
                st_relu2!(12);
                st_relu2!(13);
                st_relu2!(14);
                st_relu2!(15);
                n_base += 16;
            }
            ctm_rs::tcgen05::tmem::load_wait();
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 128);

            let mut n = row_off16;
            while n < limit {
                let row_u128 = self.group16_u128_ptr(n, sub_tile16);
                let q0 = *row_u128;
                let q1 = *row_u128.add(8);
                let q2 = *row_u128.add(16);
                let q3 = *row_u128.add(24);
                let v0 = f32::from_bits((q0 & 0xFFFF_FFFF) as u32);
                let v1 = f32::from_bits(((q0 >> 32) & 0xFFFF_FFFF) as u32);
                let v2 = f32::from_bits(((q0 >> 64) & 0xFFFF_FFFF) as u32);
                let v3 = f32::from_bits((q0 >> 96) as u32);
                let v4 = f32::from_bits((q1 & 0xFFFF_FFFF) as u32);
                let v5 = f32::from_bits(((q1 >> 32) & 0xFFFF_FFFF) as u32);
                let v6 = f32::from_bits(((q1 >> 64) & 0xFFFF_FFFF) as u32);
                let v7 = f32::from_bits((q1 >> 96) as u32);
                let v8 = f32::from_bits((q2 & 0xFFFF_FFFF) as u32);
                let v9 = f32::from_bits(((q2 >> 32) & 0xFFFF_FFFF) as u32);
                let v10 = f32::from_bits(((q2 >> 64) & 0xFFFF_FFFF) as u32);
                let v11 = f32::from_bits((q2 >> 96) as u32);
                let v12 = f32::from_bits((q3 & 0xFFFF_FFFF) as u32);
                let v13 = f32::from_bits(((q3 >> 32) & 0xFFFF_FFFF) as u32);
                let v14 = f32::from_bits(((q3 >> 64) & 0xFFFF_FFFF) as u32);
                let v15 = f32::from_bits((q3 >> 96) as u32);
                let mut amax = v0;
                macro_rules! max_v {
                    ($v:ident) => {
                        if $v > amax {
                            amax = $v;
                        }
                    };
                }
                max_v!(v1);
                max_v!(v2);
                max_v!(v3);
                max_v!(v4);
                max_v!(v5);
                max_v!(v6);
                max_v!(v7);
                max_v!(v8);
                max_v!(v9);
                max_v!(v10);
                max_v!(v11);
                max_v!(v12);
                max_v!(v13);
                max_v!(v14);
                max_v!(v15);
                let scale_code = crate::Float8E4M3FN::encode_rne(amax * (1.0f32 / 6.0f32));
                let inv_scale = crate::Float4E2M1FN::exact_inv_scale(scale_code);
                macro_rules! enc_pos {
                    ($v:ident, $sh:expr) => {
                        (crate::Float4E2M1FN::encode_non_neg_scaled($v * inv_scale) as u64) << $sh
                    };
                }
                let packed_u64 = enc_pos!(v0, 0)
                    | enc_pos!(v1, 4)
                    | enc_pos!(v2, 8)
                    | enc_pos!(v3, 12)
                    | enc_pos!(v4, 16)
                    | enc_pos!(v5, 20)
                    | enc_pos!(v6, 24)
                    | enc_pos!(v7, 28)
                    | enc_pos!(v8, 32)
                    | enc_pos!(v9, 36)
                    | enc_pos!(v10, 40)
                    | enc_pos!(v11, 44)
                    | enc_pos!(v12, 48)
                    | enc_pos!(v13, 52)
                    | enc_pos!(v14, 56)
                    | enc_pos!(v15, 60);
                let u64_idx = (group * inter_bursts + burst_inter) * 4096 + sub_u64_off + n * 2;
                let sfb_word = crate::block_scaled::Sm120ScaleAtom::word_offset(n);
                let s_idx = (group * inter_atoms + k_atom) * 512 + sfb_word * 4 + k_in_atom;
                *vals_u64_ptr.add(u64_idx) = packed_u64;
                scales_out.store_at(s_idx, scale_code);
                n += 16;
            }
            ctm_rs::fence::fence_proxy_async_shared_cta();
            ctm_rs::barrier::barrier_cta_sync(1, 128);
            if signal_b_free && tid == 0 {
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr.add(1));
            }
        }
    }

    /// Unload a Dual-M-Tile (`D0` at `tmem_addr` and `D1` at `tmem_addr + 128`) for `limit <= 16`
    /// (`B <= 16`) across `Warps 0..3` (`128` threads) and fuse `routing_weight * g_scale` +
    /// `atom_add_bf16x2` directly into `output_u32`.
    ///
    /// # Safety
    /// All 128 threads of `Warps 0..3` must call this convergently after `mma_bar` completes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_dual_m_and_atomic_add_bf16x2_16(
        &self,
        epi1: &Self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        tile_m0: usize,
        tile_m1: usize,
        group: usize,
        group_tile_rows: usize,
        g_scale: f32,
        group_routes: *const i32,
        routing_weights: *const crate::BFloat16,
        output_u32: *mut u32,
        hidden_size: usize,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_b_free: bool,
    ) {
        let tmem_row_base = tmem_addr + (warp_id << 21);
        let smem_epi0_u32 = self.base.cast::<u32>();
        let smem_epi0_u128 = self.base.cast_const().cast::<u128>();
        let smem_epi1_u32 = epi1.base.cast::<u32>();
        let smem_epi1_u128 = epi1.base.cast_const().cast::<u128>();
        let col16 = tid & 15;
        let row8 = tid >> 4;
        let col_u32_0 = tile_m0 * 64 + col16 * 4;
        let col_u32_1 = tile_m1 * 64 + col16 * 4;
        let h_u32 = hidden_size >> 1;
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            let r0 = ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base).to_array();
            let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + 128).to_array();
            macro_rules! st_raw16 {
                ($c:expr) => {
                    if $c < limit {
                        *smem_epi0_u32.add($c * 128 + tid) = r0[$c];
                        *smem_epi1_u32.add($c * 128 + tid) = r1[$c];
                    }
                };
            }
            st_raw16!(0);
            st_raw16!(1);
            st_raw16!(2);
            st_raw16!(3);
            st_raw16!(4);
            st_raw16!(5);
            st_raw16!(6);
            st_raw16!(7);
            st_raw16!(8);
            st_raw16!(9);
            st_raw16!(10);
            st_raw16!(11);
            st_raw16!(12);
            st_raw16!(13);
            st_raw16!(14);
            st_raw16!(15);
            ctm_rs::tcgen05::tmem::load_wait();
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 128);

            let mut n = row8;
            while n < limit {
                let route = *group_routes.add(group * group_tile_rows + n) as usize;
                let token = route >> 2;
                let sw = g_scale
                    * f32::from_bits(((*routing_weights.add(route)).to_bits() as u32) << 16);
                let q0_a = *smem_epi0_u128.add(n * 32 + col16 * 2);
                let q0_b = *smem_epi0_u128.add(n * 32 + col16 * 2 + 1);
                let q1_a = *smem_epi1_u128.add(n * 32 + col16 * 2);
                let q1_b = *smem_epi1_u128.add(n * 32 + col16 * 2 + 1);
                let d0 = output_u32.add(token * h_u32 + col_u32_0);
                let d1 = output_u32.add(token * h_u32 + col_u32_1);

                macro_rules! atomic_quad {
                    ($q:ident, $dst:expr) => {{
                        let v0 = (f32::from_bits(($q & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v1 = (f32::from_bits((($q >> 32) & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v2 = (f32::from_bits((($q >> 64) & 0xFFFF_FFFF) as u32) * sw).to_bits();
                        let v3 = (f32::from_bits(($q >> 96) as u32) * sw).to_bits();
                        let b0 = v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16;
                        let b1 = v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16;
                        let b2 = v2.wrapping_add(0x7FFF + ((v2 >> 16) & 1)) >> 16;
                        let b3 = v3.wrapping_add(0x7FFF + ((v3 >> 16) & 1)) >> 16;
                        let _ = ctm_rs::memory::atom_add_bf16x2($dst, b0 | (b1 << 16));
                        let _ = ctm_rs::memory::atom_add_bf16x2(($dst).add(1), b2 | (b3 << 16));
                    }};
                }
                atomic_quad!(q0_a, d0);
                atomic_quad!(q0_b, d0.add(2));
                atomic_quad!(q1_a, d1);
                atomic_quad!(q1_b, d1.add(2));
                n += 8;
            }
            ctm_rs::fence::fence_proxy_async_shared_cta();
            ctm_rs::barrier::barrier_cta_sync(1, 128);
            if signal_b_free && tid == 0 {
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr.add(1));
            }
        }
    }

    /// Unload a Dual-M-Tile (`D0` at `tmem_addr` and `D1` at `tmem_addr + 128`) across `0..limit`
    /// in `Warps 0..3` (`128` threads), convert to `BFloat16` RNE in shared memory, and write
    /// coalesced `u128` (`8 × bf16`) vectors to `grouped_out_u128`.
    ///
    /// # Safety
    /// All 128 threads of `Warps 0..3` must call this convergently after `mma_bar` completes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_dual_m_to_grouped_bf16_128(
        &self,
        epi1: &Self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        tile_m0: usize,
        tile_m1: usize,
        group: usize,
        group_tile_rows: usize,
        g_scale: f32,
        grouped_out_u128: *mut u128,
        hidden_size: usize,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_b_free: bool,
    ) {
        let tmem_row_base = tmem_addr + (warp_id << 21);
        let smem_epi0_u16 = self.base.cast::<u16>();
        let smem_epi0_u128 = self.base.cast_const().cast::<u128>();
        let smem_epi1_u16 = epi1.base.cast::<u16>();
        let smem_epi1_u128 = epi1.base.cast_const().cast::<u128>();
        let col16 = tid & 15;
        let row8 = tid >> 4;
        let dst_col0_u128 = tile_m0 * 16 + col16;
        let dst_col1_u128 = tile_m1 * 16 + col16;
        let h_u128 = hidden_size >> 3;
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            let mut n_base = 0usize;
            while n_base < limit {
                let r0 =
                    ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                        .to_array();
                let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                    tmem_row_base + 128 + (n_base as u32),
                )
                .to_array();
                macro_rules! st_dual16 {
                    ($c:expr) => {{
                        let n = n_base + $c;
                        if n < limit {
                            let v0 = (f32::from_bits(r0[$c]) * g_scale).to_bits();
                            let v1 = (f32::from_bits(r1[$c]) * g_scale).to_bits();
                            *smem_epi0_u16.add(n * 128 + tid) =
                                (v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16) as u16;
                            *smem_epi1_u16.add(n * 128 + tid) =
                                (v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16) as u16;
                        }
                    }};
                }
                st_dual16!(0);
                st_dual16!(1);
                st_dual16!(2);
                st_dual16!(3);
                st_dual16!(4);
                st_dual16!(5);
                st_dual16!(6);
                st_dual16!(7);
                st_dual16!(8);
                st_dual16!(9);
                st_dual16!(10);
                st_dual16!(11);
                st_dual16!(12);
                st_dual16!(13);
                st_dual16!(14);
                st_dual16!(15);
                n_base += 16;
            }
            ctm_rs::tcgen05::tmem::load_wait();
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 128);

            let group_row_base = group * group_tile_rows;
            let mut n = row8;
            while n < limit {
                let q0 = *smem_epi0_u128.add(n * 16 + col16);
                let q1 = *smem_epi1_u128.add(n * 16 + col16);
                let row_off = (group_row_base + n) * h_u128;
                *grouped_out_u128.add(row_off + dst_col0_u128) = q0;
                *grouped_out_u128.add(row_off + dst_col1_u128) = q1;
                n += 8;
            }
            ctm_rs::fence::fence_proxy_async_shared_cta();
            ctm_rs::barrier::barrier_cta_sync(1, 128);
            if signal_b_free && tid == 0 {
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr.add(1));
            }
        }
    }

    /// Unload a Triple-M-Tile (`D0` at `tmem_addr`, `D1` at `tmem_addr + 128`, `D2` at `tmem_addr + 256`)
    /// across `0..limit` in `Warps 0..3` (`128` threads), convert to `BFloat16` RNE in shared memory,
    /// and write coalesced `u128` (`8 × bf16`) vectors to `grouped_out_u128`.
    ///
    /// # Safety
    /// All 128 threads of `Warps 0..3` must call this convergently after `mma_bar` completes.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn unload_triple_m_to_grouped_bf16_128(
        &self,
        epi1: &Self,
        epi2: &Self,
        tmem_addr: u32,
        warp_id: u32,
        tid: usize,
        limit: usize,
        tile_m0: usize,
        tile_m1: usize,
        tile_m2: usize,
        group: usize,
        group_tile_rows: usize,
        g_scale: f32,
        grouped_out_u128: *mut u128,
        hidden_size: usize,
        a_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        b_free_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
        signal_free: bool,
    ) {
        let tmem_row_base = tmem_addr + (warp_id << 21);
        let smem_epi0_u16 = self.base.cast::<u16>();
        let smem_epi0_u128 = self.base.cast_const().cast::<u128>();
        let smem_epi1_u16 = epi1.base.cast::<u16>();
        let smem_epi1_u128 = epi1.base.cast_const().cast::<u128>();
        let smem_epi2_u16 = epi2.base.cast::<u16>();
        let smem_epi2_u128 = epi2.base.cast_const().cast::<u128>();
        let col16 = tid & 15;
        let row8 = tid >> 4;
        let dst_col0_u128 = tile_m0 * 16 + col16;
        let dst_col1_u128 = tile_m1 * 16 + col16;
        let dst_col2_u128 = tile_m2 * 16 + col16;
        let h_u128 = hidden_size >> 3;
        unsafe {
            ctm_rs::tcgen05::fence_after_thread_sync();
            let mut n_base = 0usize;
            while n_base < limit {
                let r0 =
                    ctm_rs::tcgen05::tmem::load_32x32b_x16(tmem_row_base + (n_base as u32))
                        .to_array();
                let r1 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                    tmem_row_base + 128 + (n_base as u32),
                )
                .to_array();
                let r2 = ctm_rs::tcgen05::tmem::load_32x32b_x16(
                    tmem_row_base + 256 + (n_base as u32),
                )
                .to_array();
                macro_rules! st_triple16 {
                    ($c:expr) => {{
                        let n = n_base + $c;
                        if n < limit {
                            let v0 = (f32::from_bits(r0[$c]) * g_scale).to_bits();
                            let v1 = (f32::from_bits(r1[$c]) * g_scale).to_bits();
                            let v2 = (f32::from_bits(r2[$c]) * g_scale).to_bits();
                            *smem_epi0_u16.add(n * 128 + tid) =
                                (v0.wrapping_add(0x7FFF + ((v0 >> 16) & 1)) >> 16) as u16;
                            *smem_epi1_u16.add(n * 128 + tid) =
                                (v1.wrapping_add(0x7FFF + ((v1 >> 16) & 1)) >> 16) as u16;
                            *smem_epi2_u16.add(n * 128 + tid) =
                                (v2.wrapping_add(0x7FFF + ((v2 >> 16) & 1)) >> 16) as u16;
                        }
                    }};
                }
                st_triple16!(0);
                st_triple16!(1);
                st_triple16!(2);
                st_triple16!(3);
                st_triple16!(4);
                st_triple16!(5);
                st_triple16!(6);
                st_triple16!(7);
                st_triple16!(8);
                st_triple16!(9);
                st_triple16!(10);
                st_triple16!(11);
                st_triple16!(12);
                st_triple16!(13);
                st_triple16!(14);
                st_triple16!(15);
                n_base += 16;
            }
            ctm_rs::tcgen05::tmem::load_wait();
            ctm_rs::tcgen05::fence_before_thread_sync();
            ctm_rs::barrier::barrier_cta_sync(1, 128);

            let group_row_base = group * group_tile_rows;
            let mut n = row8;
            while n < limit {
                let q0 = *smem_epi0_u128.add(n * 16 + col16);
                let q1 = *smem_epi1_u128.add(n * 16 + col16);
                let q2 = *smem_epi2_u128.add(n * 16 + col16);
                let row_off = (group_row_base + n) * h_u128;
                *grouped_out_u128.add(row_off + dst_col0_u128) = q0;
                *grouped_out_u128.add(row_off + dst_col1_u128) = q1;
                *grouped_out_u128.add(row_off + dst_col2_u128) = q2;
                n += 8;
            }
            ctm_rs::fence::fence_proxy_async_shared_cta();
            ctm_rs::barrier::barrier_cta_sync(1, 128);
            if signal_free && tid == 0 {
                let _ = ctm_rs::mbarrier::mbarrier_arrive(a_free_bar_ptr);
                let _ = ctm_rs::mbarrier::mbarrier_arrive(a_free_bar_ptr.add(1));
                let _ = ctm_rs::mbarrier::mbarrier_arrive(b_free_bar_ptr);
            }
        }
    }
}

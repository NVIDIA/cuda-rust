/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired N-partition FP16 MMA with shared A-collector reuse.

use super::{__compiler, OperandA, OperandB, Sm100ScaleTile, Sm100SharedTile};
use crate::block_scaled::{Mkl, Nkl};
use crate::markers::ReifySmem2D;
use core::marker::PhantomData;

/// Two-CTA FP16 tiled MMA with two N partitions sharing the A collector.
///
/// The A collector holds an input so two MMA instructions can reuse it:
///
/// ```text
/// each K=16 step:  A x B0 -> first N0 columns   (save A)
///                 A x B1 -> next N1 columns    (reuse, then release A)
/// ```
///
/// Each step uses a fresh A input. The complete output is
/// `256 x (N0 + N1)` with FP32 accumulation.
/// `K` must be 64; each N partition is a multiple of 16 in `16..=256`.
/// Per-CTA layouts cover A `(128, 64)`, B0 `(N0 / 2, 64)` and B1
/// `(N1 / 2, 64)`, in FP16 elements, with the supported B128 swizzle:
///
/// ```
/// use cute_rs::{ComposedLayout, RowMajor, Sm100TiledMma, Swizzle};
/// type Tile<const R: i64> = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<R, 64>>;
/// let mma = Sm100TiledMma::<Tile<128>, Tile<96>, Tile<80>, 192, 160>::new();
/// ```
pub struct Sm100TiledMma<AL, B0L, B1L, const N0: u32, const N1: u32, const K: u32 = 64>(
    #[doc = "Carries the three shared operand maps; no device storage is allocated."]
    PhantomData<(AL, B0L, B1L)>,
);
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    /// Reject unsupported N partitions, K extents and per-CTA operand shapes at instantiation.
    const VALID: () = {
        assert!(N0 >= 16 && N0 <= 256 && N0.is_multiple_of(16));
        assert!(N1 >= 16 && N1 <= 256 && N1.is_multiple_of(16));
        assert!(N0 + N1 <= 512 && K == 64);
        assert!(AL::SHAPE.0 == 128 && AL::SHAPE.1 == K as i64);
        assert!(B0L::SHAPE.0 == (N0 / 2) as i64 && B0L::SHAPE.1 == K as i64);
        assert!(B1L::SHAPE.0 == (N1 / 2) as i64 && B1L::SHAPE.1 == K as i64);
    };

    /// Check the tile dimensions and create the MMA configuration.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }

    /// Accumulate one staged K tile into tensor memory.
    ///
    /// # Safety
    ///
    /// One elected lane in the leader CTA issues this operation after the
    /// input pipeline wait. Both CTAs' tiles must be full and remain unchanged
    /// until the asynchronous consumer release completes. `address` names a
    /// 512-column allocation; `accumulate=false` starts a new output tile.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        address: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b0: &Sm100SharedTile<B0L, OperandB>,
        b1: &Sm100SharedTile<B1L, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::sm100_tiled_mma::<AL, B0L, B1L, N0, N1, K>(
                address, a.base, b0.base, b1.base, accumulate,
            )
        }
    }
}
impl<
    AL: ReifySmem2D,
    B0L: ReifySmem2D,
    B1L: ReifySmem2D,
    const N0: u32,
    const N1: u32,
    const K: u32,
> Default for Sm100TiledMma<AL, B0L, B1L, N0, N1, K>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Group-one Blackwell block-scaled NVFP4 (`mxf4nvf4.block_scale.block16`) tiled MMA with
/// integrated `tcgen05.cp.32x128b.warpx4` scale staging from shared memory into TMEM.
///
/// Mirrors CuTeDSL's `Mxf4NvF4` `TiledMma` + `SmemToTmemCopy` (`tcgen05.cp`) while encoding
/// the A and B shared-memory tile layouts (`AL`, `BL`) and operand roles (`OperandA`, `OperandB`,
/// `Mkl`, `Nkl`) directly in Rust's type system.
#[derive(Clone, Copy, Debug)]
pub struct Sm100BlockScaledTiledMma<AL, BL, const M: u32 = 128, const K: u32 = 64> {
    /// Hardware instruction descriptor (`Tcgen05InstructionDescriptor`) for `(M, valid_n)`.
    /// Encodes `E2M1 × E2M1` operands, `E4M3` block scales (`SF_VEC = 16`), and `f32` accumulation.
    idesc: u32,
    /// Leading-dimension byte stride of the B shared-memory sub-tile (`valid_n * 16`).
    /// Separates the low 16 bytes (`K = 0..31`) from the high 16 bytes (`K = 32..63`) of each row.
    b_lbo_bytes: u32,
    /// Byte stride between consecutive `K = 64` B shared-memory sub-tiles (`valid_n * 32`).
    /// Advances the B `Tcgen05SmemDescriptor` base address across `step = 0..8`.
    b_sub_bytes: u64,
    /// Zero-byte marker carrying the compile-time A and B shared-memory layout types (`AL`, `BL`).
    /// Checked against the [`Sm100SharedTile`] arguments passed to [`Self::copy_scales_and_gemm`].
    marker: PhantomData<(AL, BL)>,
}

impl<AL: ReifySmem2D, BL: ReifySmem2D, const M: u32, const K: u32>
    Sm100BlockScaledTiledMma<AL, BL, M, K>
{
    /// Initialize `COUNT` consecutive CTA-shared `mbarrier` slots at `bar_ptr` with expected arrival count `1`.
    ///
    /// # Safety
    /// One elected thread (`tid == 0`) must call this before `PipelineTmaAsync::init` publishes the barriers.
    #[inline(always)]
    pub unsafe fn init_barriers<const COUNT: usize>(bar_ptr: *mut ctm_rs::mbarrier::Barrier) {
        unsafe {
            let mut i = 0usize;
            while i < COUNT {
                ctm_rs::mbarrier::mbarrier_init(bar_ptr.add(i), 1);
                i += 1;
            }
        }
    }

    /// Spin-wait on CTA-shared `bar_ptr` until its phase reaches `parity` (`0` or `1`).
    ///
    /// # Safety
    /// `bar_ptr` must name a live initialized CTA-shared barrier.
    #[inline(always)]
    pub unsafe fn wait_barrier(bar_ptr: *const ctm_rs::mbarrier::Barrier, parity: u32) {
        unsafe { while !ctm_rs::mbarrier::mbarrier_try_wait_parity(bar_ptr, parity) {} }
    }

    /// Commit all prior `tcgen05.cp` and `tcgen05.mma` operations issued by the calling thread to `bar_ptr`.
    ///
    /// # Safety
    /// One elected MMA issuer lane calls this with a live initialized CTA-shared barrier.
    #[inline(always)]
    pub unsafe fn commit_barrier(bar_ptr: *mut ctm_rs::mbarrier::Barrier) {
        unsafe {
            ctm_rs::tcgen05::commit_address(ctm_rs::memory::SharedAddress::from_raw(bar_ptr));
        }
    }

    /// Collectively allocate `512` group-one TMEM columns into `slot_ptr` from `Warp 0`.
    ///
    /// # Safety
    /// All 32 lanes of `Warp 0` must call this convergently before barrier publication.
    #[inline(always)]
    pub unsafe fn alloc_tmem_cg1(slot_ptr: *mut u32) {
        unsafe {
            let uninit = ctm_rs::tcgen05::tmem::TmemGuard::<
                ctm_rs::tcgen05::tmem::TmemUninit,
                512,
                ctm_rs::groups::CtaGroup1,
            >::from_raw_slot(slot_ptr);
            let _ready = uninit.alloc_collective();
        }
    }

    /// Collectively deallocate the `512` group-one TMEM columns recorded at `slot_ptr` from `Warp 0`.
    ///
    /// # Safety
    /// All TMEM reads/writes must be complete and all 32 lanes of `Warp 0` must call convergently.
    #[inline(always)]
    pub unsafe fn dealloc_tmem_cg1(slot_ptr: *mut u32) {
        unsafe {
            let ready = ctm_rs::tcgen05::tmem::TmemGuard::<
                ctm_rs::tcgen05::tmem::TmemReady,
                512,
                ctm_rs::groups::CtaGroup1,
            >::adopt_ready(slot_ptr);
            let _ = ready.dealloc_collective();
        }
    }

    /// Collectively allocate `256` group-one TMEM columns into `slot_ptr` from `Warp 0`.
    ///
    /// # Safety
    /// All 32 lanes of `Warp 0` must call this convergently before barrier publication.
    #[inline(always)]
    pub unsafe fn alloc_tmem_cg1_256(slot_ptr: *mut u32) {
        unsafe {
            let uninit = ctm_rs::tcgen05::tmem::TmemGuard::<
                ctm_rs::tcgen05::tmem::TmemUninit,
                256,
                ctm_rs::groups::CtaGroup1,
            >::from_raw_slot(slot_ptr);
            let _ready = uninit.alloc_collective();
        }
    }

    /// Collectively deallocate the `256` group-one TMEM columns recorded at `slot_ptr` from `Warp 0`.
    ///
    /// # Safety
    /// All TMEM reads/writes must be complete and all 32 lanes of `Warp 0` must call convergently.
    #[inline(always)]
    pub unsafe fn dealloc_tmem_cg1_256(slot_ptr: *mut u32) {
        unsafe {
            let ready = ctm_rs::tcgen05::tmem::TmemGuard::<
                ctm_rs::tcgen05::tmem::TmemReady,
                256,
                ctm_rs::groups::CtaGroup1,
            >::adopt_ready(slot_ptr);
            let _ = ready.dealloc_collective();
        }
    }

    /// Scatter one 16-element quantized token group (`packed_u64` + `scale_code`) across its 4 routed
    /// expert destinations in `128`-row TMA burst layout with `k_bursts` `K = 512` bursts per group.
    ///
    /// # Safety
    /// `b_u64_ptr` and `sfb_out` must cover the `max_groups * k_bursts` TMA burst buffers.
    #[inline(always)]
    pub unsafe fn store_routed_token_group_k(
        b_u64_ptr: *mut u64,
        sfb_out: &mut crate::tensor::partition::StridedTensorMut<'_, u8>,
        gp128: u128,
        grp: usize,
        packed_u64: u64,
        scale_code: u8,
        k_bursts: usize,
    ) {
        let burst = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        let k_atoms = k_bursts * 8;
        unsafe {
            let mut slot = 0usize;
            while slot < 4 {
                let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
                let g = gp >> 8;
                let r = gp & 127;
                let u64_idx = (g * k_bursts + burst) * 4096 + sub_u64_off + r * 2;
                let s_idx = (g * k_atoms + k_atom) * 512
                    + crate::block_scaled::Sm120ScaleAtom::word_offset(r) * 4
                    + k_in_atom;
                *b_u64_ptr.add(u64_idx) = packed_u64;
                sfb_out.store_at(s_idx, scale_code);
                slot += 1;
            }
        }
    }

    /// Execute a single-CTA (`256`-thread = `8`-warp) shared-memory route histogram, parallel
    /// power-of-two bit-shift prefix scan, and group/permutation table scatter across `num_experts` experts.
    ///
    /// # Safety
    /// `dyn_smem` must point to at least `6,176` bytes of 16-byte-aligned dynamic shared memory,
    /// and all 256 threads in the CTA must call this convergently.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn route_histogram_and_scan_shared(
        dyn_smem: *mut u8,
        expert_ids: *const i32,
        group_experts: *mut i32,
        group_limits: *mut i32,
        group_routes: *mut i32,
        tok_to_perm: *mut i32,
        group_count: *mut i32,
        tid: usize,
        routes: usize,
        num_experts: usize,
        tile_n: u32,
        tile_n_log2: u32,
        group_tile_rows: usize,
    ) {
        let warp_id = tid >> 5;
        let lane = tid & 31;
        let tn_mask = tile_n - 1;
        unsafe {
            let groups_ptr = dyn_smem.cast::<u32>();
            let offsets_ptr = dyn_smem.add(1024).cast::<u32>();
            let ranks_ptr = dyn_smem.add(2048).cast::<u16>();
            let exp_ptr = dyn_smem.add(4096).cast::<u16>();
            let sub_counts_ptr = dyn_smem.add(6144).cast::<u16>(); // [8 warps, 192 experts] = 3072 B
            let active_ptr = dyn_smem.add(9216); // [256] u8

            *active_ptr.add(tid) = 0;
            ctm_rs::barrier::barrier_cta_sync(1, 256);

            let mut r = tid;
            while r < routes {
                let e = *expert_ids.add(r) as usize;
                *exp_ptr.add(r) = e as u16;
                *active_ptr.add(e) = 1;
                r += 256;
            }
            ctm_rs::barrier::barrier_cta_sync(1, 256);

            // 8 warps split `routes` into 8 chunks of `chunk_routes = routes / 8` (`<= 128` routes = `32` u64s!).
            let chunk_u64 = (routes >> 2) >> 3;
            let u64_start = warp_id * chunk_u64;
            let u64_end = if warp_id == 7 {
                routes >> 2
            } else {
                u64_start + chunk_u64
            };
            let exp_u64 = exp_ptr.cast_const().cast::<u64>();

            let mut e_idx = lane;
            while e_idx < num_experts {
                let mut local_c = 0u16;
                if *active_ptr.add(e_idx) != 0 {
                    let target_exp = e_idx as u16;
                    let mut i4 = u64_start;
                    while i4 < u64_end {
                        let pack = *exp_u64.add(i4);
                        let base = i4 << 2;
                        if (pack & 0xFFFF) as u16 == target_exp {
                            *ranks_ptr.add(base) = local_c;
                            local_c += 1;
                        }
                        if ((pack >> 16) & 0xFFFF) as u16 == target_exp {
                            *ranks_ptr.add(base + 1) = local_c;
                            local_c += 1;
                        }
                        if ((pack >> 32) & 0xFFFF) as u16 == target_exp {
                            *ranks_ptr.add(base + 2) = local_c;
                            local_c += 1;
                        }
                        if (pack >> 48) as u16 == target_exp {
                            *ranks_ptr.add(base + 3) = local_c;
                            local_c += 1;
                        }
                        i4 += 1;
                    }
                }
                *sub_counts_ptr.add(warp_id * 192 + e_idx) = local_c;
                e_idx += 32;
            }
            ctm_rs::barrier::barrier_cta_sync(1, 256);

            let mut count = 0u32;
            if tid < num_experts && *active_ptr.add(tid) != 0 {
                let mut w = 0usize;
                while w < 8 {
                    let wc = *sub_counts_ptr.add(w * 192 + tid) as u32;
                    *sub_counts_ptr.add(w * 192 + tid) = count as u16;
                    count += wc;
                    w += 1;
                }
            }
            let my_groups = if count > 0 {
                (count + tn_mask) >> tile_n_log2
            } else {
                0
            };
            *groups_ptr.add(tid) = my_groups;
            ctm_rs::barrier::barrier_cta_sync(1, 256);

            if tid == num_experts - 1 || (tid < num_experts && my_groups > 0) {
                let groups_u128 = groups_ptr.cast_const().cast::<u128>();
                let full_q = tid >> 2;
                let mut excl_offset = 0u32;
                let mut q = 0usize;
                while q < full_q {
                    let v = *groups_u128.add(q);
                    excl_offset += (v & 0xFFFF_FFFF) as u32
                        + ((v >> 32) & 0xFFFF_FFFF) as u32
                        + ((v >> 64) & 0xFFFF_FFFF) as u32
                        + (v >> 96) as u32;
                    q += 1;
                }
                let mut j = full_q << 2;
                while j < tid {
                    excl_offset += *groups_ptr.add(j);
                    j += 1;
                }
                *offsets_ptr.add(tid) = excl_offset;
                if tid == num_experts - 1 {
                    *group_count = (excl_offset + my_groups) as i32;
                }
                if my_groups > 0 {
                    let mut sg = 0u32;
                    while sg < my_groups {
                        let g_idx = (excl_offset + sg) as usize;
                        let rem = count - (sg << tile_n_log2);
                        let lim = if rem < tile_n { rem } else { tile_n };
                        *group_experts.add(g_idx) = tid as i32;
                        *group_limits.add(g_idx) = lim as i32;
                        sg += 1;
                    }
                }
            }
            ctm_rs::barrier::barrier_cta_sync(1, 256);

            let chunk_r_start = u64_start << 2;
            let chunk_r_end = u64_end << 2;
            let mut r2 = chunk_r_start + lane;
            while r2 < chunk_r_end {
                let exp = *exp_ptr.add(r2) as usize;
                let rank = (*ranks_ptr.add(r2) + *sub_counts_ptr.add(warp_id * 192 + exp)) as u32;
                let g = *offsets_ptr.add(exp) + (rank >> tile_n_log2);
                let row = rank & tn_mask;
                *tok_to_perm.add(r2) = ((g << 8) | row) as i32;
                *group_routes.add((g as usize) * group_tile_rows + (row as usize)) = r2 as i32;
                r2 += 32;
            }
        }
    }

    /// Fused shared-memory route counting and `u128` (`4 × (group << 8 | rank)`) lookup for `B <= 16`
    /// (`routes <= 64`), with optional `output` zeroing (`zero_out != 0`).
    ///
    /// # Safety
    /// When `fuse_route != 0`, `smem` must point to at least `768` bytes of 16-byte-aligned shared memory
    /// and all 128 threads in the CTA must call convergently.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    #[inline(always)]
    pub unsafe fn route_and_load_gp128_shared(
        smem: *mut u8,
        expert_ids: *const i32,
        group_experts: *mut i32,
        group_limits: *mut i32,
        group_routes: *mut i32,
        group_count: *mut i32,
        tok_to_perm: *mut i32,
        output_u128: *mut u128,
        tid: usize,
        cta_id: usize,
        idx: usize,
        row: usize,
        total_groups: usize,
        num_tokens: usize,
        group_tile_rows: usize,
        fuse_route: u32,
        zero_out: u32,
    ) -> u128 {
        unsafe {
            if zero_out != 0 && idx < total_groups {
                *output_u128.add(idx * 2) = 0;
                *output_u128.add(idx * 2 + 1) = 0;
            }
            if fuse_route != 0 {
                let routes = num_tokens * 4;
                let s_ids = smem.cast::<i32>();
                let s_first = smem.add(256).cast::<i32>();
                let s_perm = smem.add(512).cast::<i32>();

                if tid < routes {
                    *s_ids.add(tid) = *expert_ids.add(tid);
                }
                ctm_rs::barrier::barrier_cta_sync(1, 128);

                let mut my_exp = 0i32;
                let mut my_rank = 0i32;
                let mut my_count = 0i32;
                if tid < routes {
                    my_exp = *s_ids.add(tid);
                    let mut j = 0usize;
                    while j < routes {
                        if *s_ids.add(j) == my_exp {
                            my_count += 1;
                            if j < tid {
                                my_rank += 1;
                            }
                        }
                        j += 1;
                    }
                    *s_first.add(tid) = if my_rank == 0 { my_exp } else { i32::MAX };
                }
                ctm_rs::barrier::barrier_cta_sync(1, 128);

                if cta_id == 0 {
                    if tid < routes {
                        let mut my_group = 0i32;
                        let mut total_g = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = *s_first.add(j);
                            if fj != i32::MAX {
                                total_g += 1;
                                if fj < my_exp {
                                    my_group += 1;
                                }
                            }
                            j += 1;
                        }
                        let perm = (my_group << 8) | my_rank;
                        *s_perm.add(tid) = perm;
                        *tok_to_perm.add(tid) = perm;
                        *group_routes
                            .add((my_group as usize) * group_tile_rows + (my_rank as usize)) =
                            tid as i32;
                        if my_rank == 0 {
                            *group_experts.add(my_group as usize) = my_exp;
                            *group_limits.add(my_group as usize) = my_count;
                        }
                        if tid == 0 {
                            *group_count = total_g;
                        }
                    }
                } else {
                    let r_start = row * 4;
                    if tid >= r_start && tid < r_start + 4 {
                        let mut my_group = 0i32;
                        let mut j = 0usize;
                        while j < routes {
                            let fj = *s_first.add(j);
                            if fj < my_exp {
                                my_group += 1;
                            }
                            j += 1;
                        }
                        *s_perm.add(tid) = (my_group << 8) | my_rank;
                    }
                }
                ctm_rs::barrier::barrier_cta_sync(1, 128);
                if idx < total_groups {
                    *s_perm.cast_const().cast::<u128>().add(row)
                } else {
                    0u128
                }
            } else if idx < total_groups {
                *tok_to_perm.cast_const().cast::<u128>().add(row)
            } else {
                0u128
            }
        }
    }

    /// Scatter one 16-element quantized token group (`packed_u64` + `scale_code`) across its 4 routed
    /// expert destinations in `256`-row TMA burst layout (`B <= 1,024`).
    ///
    /// # Safety
    /// `b_u64_ptr` and `sfb_out` must cover the `num_groups8 * 2` TMA burst buffers.
    #[inline(always)]
    pub unsafe fn store_routed_token_group_256(
        b_u64_ptr: *mut u64,
        sfb_out: &mut crate::tensor::partition::StridedTensorMut<'_, u8>,
        gp128: u128,
        grp: usize,
        packed_u64: u64,
        scale_code: u8,
    ) {
        let burst = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 1024 + ((grp >> 1) & 1) * 512 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        unsafe {
            let mut slot = 0usize;
            while slot < 4 {
                let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
                let g = gp >> 8;
                let r = gp & 255;
                let u64_idx = (g * 2 + burst) * 8192 + sub_u64_off + r * 2;
                let s_idx = ((g * 16 + k_atom) * 2 + (r >> 7)) * 512
                    + crate::block_scaled::Sm120ScaleAtom::word_offset(r & 127) * 4
                    + k_in_atom;
                *b_u64_ptr.add(u64_idx) = packed_u64;
                sfb_out.store_at(s_idx, scale_code);
                slot += 1;
            }
        }
    }

    /// Scatter one 16-element quantized token group (`packed_u64` + `scale_code`) across its 4 routed
    /// expert destinations in `128`-row TMA burst layout (`B >= 2,048`).
    ///
    /// # Safety
    /// `b_u64_ptr` and `sfb_out` must cover the `num_groups8 * 2` TMA burst buffers.
    #[inline(always)]
    pub unsafe fn store_routed_token_group_128(
        b_u64_ptr: *mut u64,
        sfb_out: &mut crate::tensor::partition::StridedTensorMut<'_, u8>,
        gp128: u128,
        grp: usize,
        packed_u64: u64,
        scale_code: u8,
    ) {
        let burst = grp >> 5;
        let s_in_burst = (grp >> 2) & 7;
        let sub_u64_off = s_in_burst * 512 + ((grp >> 1) & 1) * 256 + (grp & 1);
        let k_atom = grp >> 2;
        let k_in_atom = grp & 3;
        unsafe {
            let mut slot = 0usize;
            while slot < 4 {
                let gp = ((gp128 >> (slot * 32)) & 0xFFFF_FFFF) as usize;
                let g = gp >> 7;
                let r = gp & 127;
                let u64_idx = (g * 2 + burst) * 4096 + sub_u64_off + r * 2;
                let s_idx = (g * 16 + k_atom) * 512
                    + crate::block_scaled::Sm120ScaleAtom::word_offset(r) * 4
                    + k_in_atom;
                *b_u64_ptr.add(u64_idx) = packed_u64;
                sfb_out.store_at(s_idx, scale_code);
                slot += 1;
            }
        }
    }
    /// Compile-time verification of the packed FP4 shared-tile shapes (`32,768` bytes per `K = 512` A stage).
    /// Rejects tile layouts whose dimensions do not match `M = 128` and 8 sub-tiles of `K = 64`.
    const VALID: () = {
        assert!(M == 128 && K == 64);
        assert!((AL::SHAPE.0 * AL::SHAPE.1) % ((M * (K / 2)) as i64) == 0);
        assert!(BL::SHAPE.1 % ((K / 2) as i64) == 0);
    };

    /// Construct a group-one block-scaled NVFP4 (`SF_VEC = 16`) tiled MMA for `valid_n` columns
    /// with an explicit B shared-memory leading-dimension byte offset (`b_lbo_bytes`), used when
    /// TMA stages fixed-capacity `[128, 256]` B burst tiles into a 2-stage ring buffer.
    #[must_use]
    #[inline(always)]
    pub const fn new_with_lbo(valid_n: u32, b_lbo_bytes: u32) -> Self {
        let () = Self::VALID;
        Self {
            idesc: ctm_rs::tcgen05::descriptors::encode_cg1_mxf4nvf4_block16_idesc(M, valid_n),
            b_lbo_bytes,
            b_sub_bytes: (b_lbo_bytes as u64) * 2,
            marker: PhantomData,
        }
    }

    /// Stage the A-side (`sfa`) and B-side (`sfb`) scale atoms for all 8 `K = 64` sub-tiles of a
    /// `K = 512` burst into TMEM (`tcgen05.cp.32x128b.warpx4`) and issue all 8 hardware block-scaled
    /// `mxf4nvf4` MMAs (`nvvm.tcgen05.mma.block_scale`), encoding shared-memory base descriptors once.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after the shared-memory tiles `a`, `b`,
    /// `sfa`, and `sfb` are published and `tmem_d` / `sfa_tmem_base` / `sfb_tmem_base` are allocated.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_scales_and_gemm_burst(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa: &Sm100ScaleTile<Mkl>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfa_base = unsafe { sfa.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let a_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_a_base, 2048, 128);
        let b_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_b_base,
            self.b_lbo_bytes,
            128,
        );
        let sfa_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfa_base, 2048, 128);
        let sfb_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfb_base, 2048, 128);
        let b_desc_step = self.b_sub_bytes >> 4;
        unsafe {
            let mut s = 0usize;
            while s < 8 {
                let sfa_tmem = sfa_tmem_base + (s as u32) * 4;
                let sfb_tmem = sfb_tmem_base + (s as u32) * 8;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfa_tmem, sfa_desc0 + (s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0 + (s as u64) * 64);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(
                    sfb_tmem + 4,
                    sfb_desc0 + (s as u64) * 64 + 32,
                );
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    a_desc0 + (s as u64) * 256,
                    b_desc0 + (s as u64) * b_desc_step,
                    self.idesc,
                    !(first_burst && s == 0),
                    sfa_tmem,
                    sfb_tmem,
                );
                s += 1;
            }
        }
    }

    /// Stage one `K = 64` step's 512-byte A and B scale atoms from typed shared tiles into TMEM
    /// (`tcgen05.cp.32x128b.warpx4`) and issue one hardware block-scaled `mxf4nvf4` MMA.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after the shared value and scale
    /// tiles at `step` are published and live, and `tmem_d`, `sfa_tmem_base`, and `sfb_tmem_base`
    /// must name valid TMEM column allocations.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_scales_and_gemm(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa: &Sm100ScaleTile<Mkl>,
        sfb: &Sm100ScaleTile<Nkl>,
        step: usize,
        accumulate: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfa_base = unsafe { sfa.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let sfa_tmem = sfa_tmem_base + (step as u32) * 4;
        let sfa_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_sfa_base + (step as u64) * 512,
            2048,
            128,
        );
        let a_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_a_base + (step as u64) * 4096,
            2048,
            128,
        );
        let b_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_b_base + (step as u64) * self.b_sub_bytes,
            self.b_lbo_bytes,
            128,
        );
        unsafe {
            ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfa_tmem, sfa_desc);
            if self.b_lbo_bytes == 4096 {
                let sfb_tmem = sfb_tmem_base + (step as u32) * 8;
                let sfb_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 1024,
                    2048,
                    128,
                );
                let sfb_desc1 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 1024 + 512,
                    2048,
                    128,
                );
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem + 4, sfb_desc1);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d, a_desc, b_desc, self.idesc, accumulate, sfa_tmem, sfb_tmem,
                );
            } else {
                let sfb_tmem = sfb_tmem_base + (step as u32) * 4;
                let sfb_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 512,
                    2048,
                    128,
                );
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d, a_desc, b_desc, self.idesc, accumulate, sfa_tmem, sfb_tmem,
                );
            }
        }
    }

    /// Stage the 8 A-side (`sfa`) scale atoms of a `K = 512` burst into TMEM (`tcgen05.cp.32x128b.warpx4`).
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after the `sfa` shared tile is published.
    #[inline(always)]
    pub unsafe fn copy_a_scales_burst(&self, sfa_tmem_base: u32, sfa: &Sm100ScaleTile<Mkl>) {
        let smem_sfa_base = unsafe { sfa.shared_offset() };
        let sfa_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfa_base, 2048, 128);
        /// Copy one 512-byte A scale atom at compile-time step `$s` into TMEM.
        macro_rules! cp_sfa {
            ($s:expr) => {
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(
                    sfa_tmem_base + ($s as u32) * 4,
                    sfa_desc0 + ($s as u64) * 32,
                );
            };
        }
        unsafe {
            cp_sfa!(0);
            cp_sfa!(1);
            cp_sfa!(2);
            cp_sfa!(3);
            cp_sfa!(4);
            cp_sfa!(5);
            cp_sfa!(6);
            cp_sfa!(7);
        }
    }

    /// Stage only the B-side (`sfb`) scale atoms for all 8 `K = 64` sub-tiles of a `K = 512` burst
    /// into TMEM (`tcgen05.cp.32x128b.warpx4`) and issue all 8 hardware block-scaled `mxf4nvf4` MMAs
    /// reusing the resident A shared tile (`a`) and A TMEM scales (`sfa_tmem_base`).
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `sfa_tmem_base` is populated
    /// and the new `b` / `sfb` burst tiles are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_b_scales_and_gemm_burst(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let a_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_a_base, 2048, 128);
        let b_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_b_base,
            self.b_lbo_bytes,
            128,
        );
        let sfb_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfb_base, 2048, 128);
        let b_desc_step = self.b_sub_bytes >> 4;
        unsafe {
            let mut s = 0usize;
            while s < 8 {
                let sfa_tmem = sfa_tmem_base + (s as u32) * 4;
                let sfb_tmem = sfb_tmem_base + (s as u32) * 8;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0 + (s as u64) * 64);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(
                    sfb_tmem + 4,
                    sfb_desc0 + (s as u64) * 64 + 32,
                );
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    a_desc0 + (s as u64) * 256,
                    b_desc0 + (s as u64) * b_desc_step,
                    self.idesc,
                    !(first_burst && s == 0),
                    sfa_tmem,
                    sfb_tmem,
                );
                s += 1;
            }
        }
    }

    /// Stage only the B-side activation scale atom (`sfb`) into TMEM (`tcgen05.cp.32x128b.warpx4`)
    /// and issue one hardware block-scaled `mxf4nvf4` MMA reusing the already-staged A shared tile
    /// (`a`) and A TMEM scales (`sfa_tmem_base`).
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after [`Self::copy_scales_and_gemm`]
    /// has already populated `sfa_tmem_base + step * 4` for the current `a` stage and the new `b` /
    /// `sfb` tiles at `step` are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_b_scales_and_gemm(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfb: &Sm100ScaleTile<Nkl>,
        step: usize,
        accumulate: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let sfa_tmem = sfa_tmem_base + (step as u32) * 4;
        let a_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_a_base + (step as u64) * 4096,
            2048,
            128,
        );
        let b_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_b_base + (step as u64) * self.b_sub_bytes,
            self.b_lbo_bytes,
            128,
        );
        unsafe {
            if self.b_lbo_bytes == 4096 {
                let sfb_tmem = sfb_tmem_base + (step as u32) * 8;
                let sfb_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 1024,
                    2048,
                    128,
                );
                let sfb_desc1 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 1024 + 512,
                    2048,
                    128,
                );
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem + 4, sfb_desc1);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d, a_desc, b_desc, self.idesc, accumulate, sfa_tmem, sfb_tmem,
                );
            } else {
                let sfb_tmem = sfb_tmem_base + (step as u32) * 4;
                let sfb_desc = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
                    smem_sfb_base + (step as u64) * 512,
                    2048,
                    128,
                );
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d, a_desc, b_desc, self.idesc, accumulate, sfa_tmem, sfb_tmem,
                );
            }
        }
    }

    /// Issue the 2-burst (`K = 1024`) stationary weight (`w_tma`) and weight-scale (`sfa_tma`) 2D TMA
    /// transfers into `smem_a_ptr` (`64 KB`) and `smem_sfa_ptr` (`8 KB`), tracked by `w_bar_ptr`.
    ///
    /// # Safety
    /// Exactly one elected producer thread in the CTA must call this operation with an initialized `w_bar_ptr`.
    #[inline(always)]
    pub unsafe fn load_stationary_weights_2d<ScaleL: ReifySmem2D>(
        w_tma: &crate::TensorMap<u8, AL>,
        sfa_tma: &crate::TensorMap<u16, ScaleL>,
        smem_a_ptr: *mut u8,
        smem_sfa_ptr: *mut u8,
        w_outer: usize,
        w_bar_ptr: *mut ctm_rs::mbarrier::Barrier,
    ) {
        let w_desc = core::ptr::from_ref(w_tma).cast::<cuda_device::TmaDescriptor>();
        let sfa_desc = core::ptr::from_ref(sfa_tma).cast::<cuda_device::TmaDescriptor>();
        unsafe {
            let _ = ctm_rs::mbarrier::mbarrier_arrive_expect_tx(w_bar_ptr, 1, 73_728);
            let mut burst = 0usize;
            while burst < 2 {
                cuda_device::tma::cp_async_bulk_tensor_2d_g2s_cta(
                    smem_a_ptr.add(burst * 32768),
                    w_desc,
                    (burst * 256) as i32,
                    (w_outer * 128) as i32,
                    w_bar_ptr,
                );
                cuda_device::tma::cp_async_bulk_tensor_2d_g2s_cta(
                    smem_sfa_ptr.add(burst * 4096),
                    sfa_desc,
                    0,
                    ((w_outer * 2 + burst) * 8) as i32,
                    w_bar_ptr,
                );
                burst += 1;
            }
        }
    }

    /// Stage the 8 B-side (`sfb`) scale atoms for a `K = 512` burst of a `128`-token group into TMEM
    /// (`tcgen05.cp.32x128b.warpx4`) and issue all 8 hardware block-scaled `mxf4nvf4` MMAs with `b`
    /// (`Tokens`, `M = 128`) on `Operand A` and `a` (`Weights`, `N = 128`) on `Operand B`.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `sfa_tmem_base` is populated
    /// and the new `b` / `sfb` burst tiles are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_b_scales_and_gemm_128_burst(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let a_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_a_base, 2048, 128);
        let b_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_b_base, 2048, 128);
        let sfb_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfb_base, 2048, 128);
        /// Copy one 512-byte B scale atom and issue one block-scaled `mxf4nvf4` MMA at step `$s`.
        macro_rules! step_128 {
            ($s:expr, $acc:expr) => {{
                let sfa_tmem = sfa_tmem_base + ($s as u32) * 4;
                let sfb_tmem = sfb_tmem_base + ($s as u32) * 4;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    b_desc0 + ($s as u64) * 256,
                    a_desc0 + ($s as u64) * 256,
                    self.idesc,
                    $acc,
                    sfb_tmem,
                    sfa_tmem,
                );
            }};
        }
        unsafe {
            step_128!(0, !first_burst);
            step_128!(1, true);
            step_128!(2, true);
            step_128!(3, true);
            step_128!(4, true);
            step_128!(5, true);
            step_128!(6, true);
            step_128!(7, true);
        }
    }

    /// Stage the 8 A-side (`sfa`) and 8 B-side (`sfb`) 512-byte scale atoms for a `K = 512` burst
    /// of a `128`-row activation tile (`b_lbo_bytes == 2048`) into TMEM (`tcgen05.cp.32x128b.warpx4`)
    /// and issue all 8 hardware block-scaled `mxf4nvf4` MMAs.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `a`, `b`, `sfa`, and `sfb`
    /// are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_scales_and_gemm_burst_128(
        &self,
        tmem_d: u32,
        sfa_tmem_base: u32,
        sfb_tmem_base: u32,
        a: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa: &Sm100ScaleTile<Mkl>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let smem_a_base = unsafe { a.shared_offset() };
        let smem_b_base = unsafe { b.shared_offset() };
        let smem_sfa_base = unsafe { sfa.shared_offset() };
        let smem_sfb_base = unsafe { sfb.shared_offset() };
        let a_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_a_base, 2048, 128);
        let b_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            smem_b_base,
            self.b_lbo_bytes,
            128,
        );
        let sfa_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfa_base, 2048, 128);
        let sfb_desc0 =
            ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(smem_sfb_base, 2048, 128);
        let b_desc_step = self.b_sub_bytes >> 4;
        macro_rules! step_b128 {
            ($s:expr, $acc:expr) => {{
                let sfa_tmem = sfa_tmem_base + ($s as u32) * 4;
                let sfb_tmem = sfb_tmem_base + ($s as u32) * 4;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfa_tmem, sfa_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sfb_tmem, sfb_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    a_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sfa_tmem,
                    sfb_tmem,
                );
            }};
        }
        unsafe {
            step_b128!(0, !first_burst);
            step_b128!(1, true);
            step_b128!(2, true);
            step_b128!(3, true);
            step_b128!(4, true);
            step_b128!(5, true);
            step_b128!(6, true);
            step_b128!(7, true);
        }
    }

    /// Stage the A-side (`sfa0`, `sfa1`) and shared B-side (`sfb`) scale atoms for a `K = 512` burst
    /// of a Dual-M-Tile (`M = 256` = `2 × 128`) tile into TMEM and issue all 16 hardware block-scaled
    /// `mxf4nvf4` MMAs into `tmem_d` (`cols 0..127`) and `tmem_d + 128` (`cols 128..255`).
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `a0`, `a1`, `b`, `sfa0`, `sfa1`,
    /// and `sfb` are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_scales_and_gemm_dual_m_burst(
        &self,
        tmem_d: u32,
        sfa0_tmem_base: u32,
        sfa1_tmem_base: u32,
        sfb_tmem_base: u32,
        a0: &Sm100SharedTile<AL, OperandA>,
        a1: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa0: &Sm100ScaleTile<Mkl>,
        sfa1: &Sm100ScaleTile<Mkl>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let a0_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { a0.shared_offset() },
            2048,
            128,
        );
        let a1_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { a1.shared_offset() },
            2048,
            128,
        );
        let b_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { b.shared_offset() },
            self.b_lbo_bytes,
            128,
        );
        let sa0_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfa0.shared_offset() },
            2048,
            128,
        );
        let sa1_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfa1.shared_offset() },
            2048,
            128,
        );
        let sb_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfb.shared_offset() },
            2048,
            128,
        );
        let b_desc_step = self.b_sub_bytes >> 4;
        macro_rules! step_dual {
            ($s:expr, $acc:expr) => {{
                let sa0 = sfa0_tmem_base + ($s as u32) * 4;
                let sa1 = sfa1_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sa0, sa0_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sa1, sa1_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sb, sb_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    a0_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sa0,
                    sb,
                );
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d + 128,
                    a1_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sa1,
                    sb,
                );
            }};
        }
        unsafe {
            step_dual!(0, !first_burst);
            step_dual!(1, true);
            step_dual!(2, true);
            step_dual!(3, true);
            step_dual!(4, true);
            step_dual!(5, true);
            step_dual!(6, true);
            step_dual!(7, true);
        }
    }

    /// Stage the A-side (`sfa0`, `sfa1`, `sfa2`) and shared B-side (`sfb`) scale atoms for a `K = 512`
    /// burst of a Triple-M-Tile (`M = 384` = `3 × 128`) tile into TMEM and issue all 24 hardware
    /// block-scaled `mxf4nvf4` MMAs into `tmem_d`, `tmem_d + 128`, and `tmem_d + 256`.
    ///
    /// # Safety
    /// One elected lane in the CTA must issue this operation after `a0`, `a1`, `a2`, `b`,
    /// `sfa0`, `sfa1`, `sfa2`, and `sfb` are published in shared memory.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn copy_scales_and_gemm_triple_m_burst(
        &self,
        tmem_d: u32,
        sfa0_tmem_base: u32,
        sfa1_tmem_base: u32,
        sfa2_tmem_base: u32,
        sfb_tmem_base: u32,
        a0: &Sm100SharedTile<AL, OperandA>,
        a1: &Sm100SharedTile<AL, OperandA>,
        a2: &Sm100SharedTile<AL, OperandA>,
        b: &Sm100SharedTile<BL, OperandB>,
        sfa0: &Sm100ScaleTile<Mkl>,
        sfa1: &Sm100ScaleTile<Mkl>,
        sfa2: &Sm100ScaleTile<Mkl>,
        sfb: &Sm100ScaleTile<Nkl>,
        first_burst: bool,
    ) {
        let a0_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { a0.shared_offset() },
            2048,
            128,
        );
        let a1_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { a1.shared_offset() },
            2048,
            128,
        );
        let a2_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { a2.shared_offset() },
            2048,
            128,
        );
        let b_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { b.shared_offset() },
            self.b_lbo_bytes,
            128,
        );
        let sa0_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfa0.shared_offset() },
            2048,
            128,
        );
        let sa1_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfa1.shared_offset() },
            2048,
            128,
        );
        let sa2_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfa2.shared_offset() },
            2048,
            128,
        );
        let sb_desc0 = ctm_rs::tcgen05::descriptors::encode_no_swizzle_smem_desc(
            unsafe { sfb.shared_offset() },
            2048,
            128,
        );
        let b_desc_step = self.b_sub_bytes >> 4;
        macro_rules! step_triple {
            ($s:expr, $acc:expr) => {{
                let sa0 = sfa0_tmem_base + ($s as u32) * 4;
                let sa1 = sfa1_tmem_base + ($s as u32) * 4;
                let sa2 = sfa2_tmem_base + ($s as u32) * 4;
                let sb = sfb_tmem_base + ($s as u32) * 4;
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sa0, sa0_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sa1, sa1_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sa2, sa2_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::cp_32x128b_warpx4(sb, sb_desc0 + ($s as u64) * 32);
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d,
                    a0_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sa0,
                    sb,
                );
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d + 128,
                    a1_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sa1,
                    sb,
                );
                ctm_rs::tcgen05::mma::issue_cg1_mxf4nvf4_block16(
                    tmem_d + 256,
                    a2_desc0 + ($s as u64) * 256,
                    b_desc0 + ($s as u64) * b_desc_step,
                    self.idesc,
                    $acc,
                    sa2,
                    sb,
                );
            }};
        }
        unsafe {
            step_triple!(0, !first_burst);
            step_triple!(1, true);
            step_triple!(2, true);
            step_triple!(3, true);
            step_triple!(4, true);
            step_triple!(5, true);
            step_triple!(6, true);
            step_triple!(7, true);
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Size-specialized device kernels for the canonical gemm_sol_final example.
// include!d into main.rs so the #[cuda_module] macro sees an inline module
// (it rejects file modules). Host validation and benchmarking remain in main.rs.
#[cuda_module]
mod kernels {
    use super::*;

    /// Target kernel: CLC + TMA multicast with 4 SMEM stages and no MCAST_BAR.
    ///
    /// Blackwell GEMM kernel: CLC + cta_group::2 + 4-stage SMEM pipeline.
    ///
    /// Architecture: 6 warps per CTA, cluster size 2 (CTA pairs).
    ///   - Warp 4 (TMA): loads A/B tiles from global via TMA into shared memory.
    ///   - Warp 5 (MMA): consumes SMEM tiles via pair-UMMA (tcgen05, cta_group::2).
    ///   - Warps 0-3 (Epilogue): read accumulators from TMEM, convert f32->bf16, store to global.
    ///
    /// Pipeline stages: 4 SMEM stages (TMA_BAR0..3 / MMA_BAR0..3), 2 accumulator stages
    /// (ACCUM_FULL0/1, ACCUM_EMPTY0/1), plus TILE_READY for TMA→MMA/Epilogue tile handoff.
    ///
    /// Key synchronization protocol (cta_group::2 barrier aliasing):
    ///   Both CTAs in a pair issue TMA loads, but the barrier pointer is masked with
    ///   PEER_BIT_MASK (0xFEFFFFF8) before being passed to the TMA instruction. This
    ///   clears bit 24, redirecting both CTAs' completion signals to the leader CTA's
    ///   (rank 0) barrier. Consequently:
    ///     - Only the leader sets expect_tx (doubled: both CTAs' bytes land on one barrier).
    ///     - Only the leader waits on TMA barriers in the MMA warp.
    ///     - The follower's MMA warp skips TMA waits since pair-UMMA is leader-issued.
    ///   MMA barriers still use normal multicast (tcgen05_commit_multicast_cg2 with
    ///   CTA_MASK_PAIR=0b11), so both CTAs receive MMA completion signals.
    ///
    /// CLC work-stealing: rank 0 issues clc_try_cancel_multicast; both CTAs receive the
    /// response via CLC_BAR. Tile indices are derived by dividing the CLC first_ctaid_x
    /// by the cluster size (2), NOT using raw CTA IDs.
    ///
    /// # Shape contract
    ///
    /// `n` must be a multiple of 256. The fixed host grid is expressed in 128-column
    /// work IDs, and this kernel drains its upper half to form 256-column tiles.
    ///
    /// `k` must be a multiple of 256. With a K tile size of 64, this makes `k_iters`
    /// a multiple of four, so every output tile starts on pipeline stage 0. The MMA
    /// consumer relies on that reset to keep its unrolled stage selection constant.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast_4_stage_pipeline(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        unsafe {
            static mut SMEM_A0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_A3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_OUT: SharedArray<u32, 16384, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // 4-stage TMA <-> MMA pipeline
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut TMA_BAR2: Barrier = Barrier::UNINIT;
            static mut TMA_BAR3: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR2: Barrier = Barrier::UNINIT;
            static mut MMA_BAR3: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: Barrier = Barrier::UNINIT;

            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_PANEL_BYTES: u32 = 64 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2; // 128 B rows per CTA (split by rank)
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 256;
            const CTA_MASK_PAIR: u16 = 0b11;
            // Clears bit 24 (CTA rank within pair) + alignment bits 2:0 of a shared
            // memory barrier address, redirecting TMA completions to the leader CTA's
            // barrier.
            const PEER_BIT_MASK: u32 = 0xFEFFFFF8;
            // L2 cache-blocking: the 256-column output tile doubles the physical
            // width of each N tile, so preserve the accepted cache-band widths by
            // halving the prior logical group sizes.
            let swizzle_g: u32 = if tiles_m <= 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let _ctaid = thread::blockIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << (my_rank as u16);
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut TMA_BAR2, 1);
                mbarrier_init(&raw mut TMA_BAR3, 1);
                mbarrier_init(&raw mut MMA_BAR0, 1);
                mbarrier_init(&raw mut MMA_BAR1, 1);
                mbarrier_init(&raw mut MMA_BAR2, 1);
                mbarrier_init(&raw mut MMA_BAR3, 1);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 256);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 256);
                mbarrier_init(&raw mut TILE_READY, 1);
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-arrive all MMA stage barriers so producer can start immediately.
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
                mbarrier_arrive(&raw const MMA_BAR2);
                mbarrier_arrive(&raw const MMA_BAR3);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
            let elect_one_cta = my_rank == 0;

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            cluster::cluster_sync();

            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_seq: u32 = 0;
                let mut clc_iter: u32 = 0;

                // The fixed host grid still contains one raw CLC work ID per
                // 256x128 tile. The wide kernel needs half as many 256x256 tiles,
                // so reinterpret the first half of the raw ID space and drain the
                // remainder through CLC without publishing work to the consumers.
                let cluster_base_id = thread::blockIdx_x() - my_rank;
                let mut raw_tile_idx = cluster_base_id / 2;
                let wide_tiles_n = _tiles_n / 2;
                let wide_total = tiles_m * wide_tiles_n;
                let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;

                loop {
                    if raw_tile_idx < wide_total {
                        // L2 cache-blocking swizzle over logical 256-column tiles.
                        let group_tiles = swizzle_g * tiles_m;
                        let group = raw_tile_idx / group_tiles;
                        let in_group = raw_tile_idx % group_tiles;
                        let n_start = group * swizzle_g;
                        let band_w = if swizzle_g < wide_tiles_n - n_start {
                            swizzle_g
                        } else {
                            wide_tiles_n - n_start
                        };
                        let tile_m = in_group / band_w;
                        let tile_n = n_start + in_group % band_w;

                        if is_lane0 {
                            *(&raw mut TILE_INFO as *mut u32).add(0) = tile_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                            mbarrier_arrive(&raw const TILE_READY);
                        }

                        let m_offset = (tile_m * 256 + my_rank * 128) as i32;
                        let b_n_offset = (tile_n * 256 + my_rank * 128) as i32;

                        let mut k_idx: u32 = 0;
                        while k_idx < k_iters {
                            let global_k = tile_seq * k_iters + k_idx;
                            let stage = global_k & 3;
                            let mma_parity = (global_k >> 2) & 1;

                            let (
                                smem_a_ptr,
                                smem_b_ptr,
                                tma_bar_const,
                                tma_bar_mut,
                                mma_bar_const,
                            ): (
                                *mut u8,
                                *mut u8,
                                *const Barrier,
                                *mut Barrier,
                                *const Barrier,
                            ) = match stage {
                                0 => (
                                    &raw mut SMEM_A0 as *mut u8,
                                    &raw mut SMEM_B0 as *mut u8,
                                    &raw const TMA_BAR0 as *const Barrier,
                                    &raw mut TMA_BAR0 as *mut Barrier,
                                    &raw const MMA_BAR0 as *const Barrier,
                                ),
                                1 => (
                                    &raw mut SMEM_A1 as *mut u8,
                                    &raw mut SMEM_B1 as *mut u8,
                                    &raw const TMA_BAR1 as *const Barrier,
                                    &raw mut TMA_BAR1 as *mut Barrier,
                                    &raw const MMA_BAR1 as *const Barrier,
                                ),
                                2 => (
                                    &raw mut SMEM_A2 as *mut u8,
                                    &raw mut SMEM_B2 as *mut u8,
                                    &raw const TMA_BAR2 as *const Barrier,
                                    &raw mut TMA_BAR2 as *mut Barrier,
                                    &raw const MMA_BAR2 as *const Barrier,
                                ),
                                _ => (
                                    &raw mut SMEM_A3 as *mut u8,
                                    &raw mut SMEM_B3 as *mut u8,
                                    &raw const TMA_BAR3 as *const Barrier,
                                    &raw mut TMA_BAR3 as *mut Barrier,
                                    &raw const MMA_BAR3 as *const Barrier,
                                ),
                            };

                            // Both CTAs receive the MMA completion, so both producer
                            // warps can safely wait before reusing this SMEM stage.
                            while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}

                            if is_lane0 {
                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(
                                        tma_bar_const,
                                        1,
                                        (A_TILE_BYTES + B_TILE_BYTES) * 2,
                                    );
                                }
                                let aliased_bar =
                                    ((tma_bar_mut as u32) & PEER_BIT_MASK) as *mut Barrier;
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                // The fixed host descriptor exposes a 64x64 B panel.
                                // Concatenate two panels to form this CTA's 128 columns.
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr.add(B_PANEL_BYTES as usize),
                                    b_tma,
                                    k_base,
                                    b_n_offset + 64,
                                    aliased_bar,
                                    self_mask,
                                );
                            }

                            k_idx += 1;
                        }
                        tile_seq += 1;
                    }

                    // One cancellation query consumes one CLC barrier phase even
                    // when its raw work ID is outside the reduced wide-tile range.
                    let clc_parity = clc_iter & 1;
                    if is_lane0 {
                        mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                        if elect_one_cta {
                            clc_try_cancel_multicast(resp_ptr as *mut u8, &raw mut CLC_BAR);
                        }
                    }
                    while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}
                    clc_iter += 1;

                    let resp_lo = *resp_ptr;
                    let resp_hi = *resp_ptr.add(1);
                    let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                    if is_canceled == 0 {
                        if is_lane0 {
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                            mbarrier_arrive(&raw const TILE_READY);
                        }
                        break;
                    }

                    let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                    raw_tile_idx = first_stolen / 2;
                }
            }

            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    if has_work == 0 {
                        break;
                    }

                    let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    if elect_one_cta && tile_iter >= NUM_ACCUM_STAGES {
                        let empty_parity = ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                        if accum_stage == 0 {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {
                            }
                        } else {
                            while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY1, empty_parity) {
                            }
                        }
                    }

                    let tile_k_base = tile_iter * k_iters;
                    let mut k_idx: u32 = 0;
                    // Unroll one full pipeline cycle. The launch contract guarantees
                    // k_iters % 4 == 0, so the producer's global stage and this local
                    // stage agree at every tile boundary. Keeping this expression
                    // loop-local lets the unroll pass fold each stage match.
                    #[unroll(4)]
                    while k_idx < k_iters {
                        let global_k = tile_k_base + k_idx;
                        let stage = k_idx & 3;
                        let tma_parity = (global_k >> 2) & 1;

                        let (smem_a_base, smem_b_base, tma_bar_const, mma_bar_mut): (
                            u64,
                            u64,
                            *const Barrier,
                            *mut Barrier,
                        ) = match stage {
                            0 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8),
                                &raw const TMA_BAR0 as *const Barrier,
                                &raw mut MMA_BAR0 as *mut Barrier,
                            ),
                            1 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8),
                                &raw const TMA_BAR1 as *const Barrier,
                                &raw mut MMA_BAR1 as *mut Barrier,
                            ),
                            2 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A2 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B2 as *const u8),
                                &raw const TMA_BAR2 as *const Barrier,
                                &raw mut MMA_BAR2 as *mut Barrier,
                            ),
                            _ => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A3 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B3 as *const u8),
                                &raw const TMA_BAR3 as *const Barrier,
                                &raw mut MMA_BAR3 as *mut Barrier,
                            ),
                        };

                        // LEADER-ONLY TMA WAIT + MMA:
                        // Because TMA completions are aliased to the leader's barrier, only
                        // the leader can (and should) wait on tma_bar_const. The follower's
                        // TMA barrier is never signaled. The follower's MMA warp simply loops
                        // through the K iterations without doing work — pair-UMMA is issued
                        // by the leader and operates on both CTAs' SMEM simultaneously.
                        if elect_one_cta {
                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}

                            if is_lane0 {
                                let mut j: u32 = 0;
                                while j < 4 {
                                    let byte_offset = (j * 32) as u64;
                                    let a_desc = build_smem_descriptor(
                                        smem_a_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let b_desc = build_smem_descriptor(
                                        smem_b_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );

                                    let accumulate = k_idx > 0 || j > 0;
                                    tcgen05_mma_f16_cg2(
                                        tmem_addr + tmem_stage_offset,
                                        a_desc,
                                        b_desc,
                                        idesc,
                                        accumulate,
                                    );
                                    j += 1;
                                }

                                tcgen05_commit_multicast_cg2(
                                    mma_bar_mut as *mut u64,
                                    CTA_MASK_PAIR,
                                );
                            }
                        }

                        k_idx += 1;
                    }

                    if elect_one_cta && is_lane0 {
                        if accum_stage == 0 {
                            tcgen05_commit_multicast_cg2(
                                &raw mut ACCUM_FULL0 as *mut u64,
                                CTA_MASK_PAIR,
                            );
                        } else {
                            tcgen05_commit_multicast_cg2(
                                &raw mut ACCUM_FULL1 as *mut u64,
                                CTA_MASK_PAIR,
                            );
                        }
                    }

                    tile_iter += 1;
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let mut epi_tile_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                let leader_accum_empty0_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as *const u8);
                let leader_accum_empty1_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as *const u8);

                const TILE_N: usize = 256;
                let warp_row_base = (warp_id * 32) as usize;
                let row_stride_bytes = TILE_N * 2;
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;

                        let mut col_block = 0u32;
                        while col_block < 16 {
                            let col_offset = (col_block * 16) as usize;

                            let regs_a = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32,
                            );
                            tcgen05_load_wait();

                            let regs_b = tcgen05_ld_16x256b_pure(
                                tmem_addr
                                    + tmem_stage_offset
                                    + (tmem_row << 16)
                                    + col_offset as u32
                                    + 8,
                            );
                            tcgen05_load_wait();

                            let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                            let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                            let out_row_lo =
                                warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                            let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_lo * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                            let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                            let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                            let out_row_hi =
                                warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                            let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                                out_row_hi * row_stride_bytes
                                    + col_offset * 2
                                    + col_offset_for_matrix2,
                            );
                            stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                            col_block += 1;
                        }
                        tmem_row_block += 1;
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128) as usize;
                    let tile_col_base = (tile_n * 128) as usize;
                    let base_row = warp_id as usize * 32;

                    let mut elem = lane_id as usize * 2;
                    while elem < 4096 {
                        let local_row = elem / 128;
                        let local_col = elem % 128;
                        let smem_idx = (base_row + local_row) * 128 + local_col;
                        let global_row = tile_row_base + base_row + local_row;
                        let global_col = tile_col_base + local_col;
                        let global_idx = global_row * n_u32 + global_col;
                        let packed = (SMEM_OUT[smem_idx] as u64)
                            | ((SMEM_OUT[smem_idx + 1] as u64) << 32);
                        let global_ptr =
                            out.get_unchecked_mut(global_idx) as *mut u32 as *mut u64;

                        *global_ptr = packed;
                        elem += 64;
                    }

                    if elect_one_cta {
                        if accum_stage == 0 {
                            mbarrier_arrive(&raw const ACCUM_EMPTY0);
                        } else {
                            mbarrier_arrive(&raw const ACCUM_EMPTY1);
                        }
                    } else {
                        if accum_stage == 0 {
                            mbarrier_arrive_cluster(leader_accum_empty0_addr);
                        } else {
                            mbarrier_arrive_cluster(leader_accum_empty1_addr);
                        }
                    }

                    epi_tile_iter += 1;
                }
            }

            // BUG FIX: cluster_sync before exit prevents "Cluster target block not present"
            // (CUDA_EXCEPTION_17). Without this, a fast CTA can exit while its partner is still
            // executing cross-CTA operations (e.g., mbarrier_arrive_cluster in the epilogue).
            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut TMA_BAR2);
                mbarrier_inval(&raw mut TMA_BAR3);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR2);
                mbarrier_inval(&raw mut MMA_BAR3);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut CLC_BAR);
            }
        }
    }

    /// Target kernel: CLC + TMA multicast with 4 SMEM stages and no MCAST_BAR.
    ///
    /// Blackwell GEMM kernel: one cta_group::2 worker computes an M512xN256 macro tile.
    ///
    /// Architecture: 6 warps per CTA, cluster size 2 (CTA pairs).
    ///   - Warp 4 (TMA): loads two A panels and one shared B tile per K stage.
    ///   - Warp 5 (MMA): computes two adjacent M256xN256 outputs that reuse B.
    ///   - Warps 0-3 (Epilogue): stream the two fixed TMEM output halves to global memory.
    ///
    /// Pipeline stages: 4 SMEM stages (TMA_BAR0..3 / MMA_BAR0..3), two fixed TMEM
    /// output halves (FULL0/1, EMPTY0/1), plus TILE_READY for macro-tile handoff.
    ///
    /// Key synchronization protocol (cta_group::2 barrier aliasing):
    ///   Both CTAs in a pair issue TMA loads, but the barrier pointer is masked with
    ///   PEER_BIT_MASK (0xFEFFFFF8) before being passed to the TMA instruction. This
    ///   clears bit 24, redirecting both CTAs' completion signals to the leader CTA's
    ///   (rank 0) barrier. Consequently:
    ///     - Only the leader sets expect_tx (doubled: both CTAs' bytes land on one barrier).
    ///     - Only the leader waits on TMA barriers in the MMA warp.
    ///     - The follower's MMA warp skips TMA waits since pair-UMMA is leader-issued.
    ///   MMA barriers still use normal multicast (tcgen05_commit_multicast_cg2 with
    ///   CTA_MASK_PAIR=0b11), so both CTAs receive MMA completion signals.
    ///
    /// CLC work-stealing: rank 0 issues clc_try_cancel_multicast; both CTAs receive the
    /// response via CLC_BAR. Tile indices are derived by dividing the CLC first_ctaid_x
    /// by the cluster size (2), NOT using raw CTA IDs.
    ///
    /// # Shape contract
    ///
    /// `n` must be a multiple of 256 and `tiles_m` must be even. The fixed host grid
    /// remains expressed in M256xN128 work IDs; only its first quarter is reinterpreted
    /// as M512xN256 macro tiles and the remaining IDs are drained through CLC.
    ///
    /// `k` must be a multiple of 256. With a K tile size of 64, this makes `k_iters`
    /// a multiple of four, so every macro tile starts on pipeline stage 0.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn gemm_sol_clc_multicast_4_stage_pipeline_large(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        n: i32,
        k: i32,
        tiles_m: u32,
        _tiles_n: u32,
    ) {
        unsafe {
            static mut SMEM_A0: SharedArray<u8, 32768, 128> = SharedArray::UNINIT;
            static mut SMEM_A1: SharedArray<u8, 32768, 128> = SharedArray::UNINIT;
            static mut SMEM_A2: SharedArray<u8, 32768, 128> = SharedArray::UNINIT;
            static mut SMEM_A3: SharedArray<u8, 32768, 128> = SharedArray::UNINIT;
            static mut SMEM_B0: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B1: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B2: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            static mut SMEM_B3: SharedArray<u8, 16384, 128> = SharedArray::UNINIT;
            // Rolling 32 KiB epilogue scratch: each warp owns 32x128 BF16 values.
            static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 4, 4> = SharedArray::UNINIT;

            // 4-stage TMA <-> MMA pipeline
            static mut TMA_BAR0: Barrier = Barrier::UNINIT;
            static mut TMA_BAR1: Barrier = Barrier::UNINIT;
            static mut TMA_BAR2: Barrier = Barrier::UNINIT;
            static mut TMA_BAR3: Barrier = Barrier::UNINIT;
            static mut MMA_BAR0: Barrier = Barrier::UNINIT;
            static mut MMA_BAR1: Barrier = Barrier::UNINIT;
            static mut MMA_BAR2: Barrier = Barrier::UNINIT;
            static mut MMA_BAR3: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: Barrier = Barrier::UNINIT;

            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_PANEL_BYTES: u32 = 64 * 64 * 2;
            const B_TILE_BYTES: u32 = 128 * 64 * 2; // 128 B rows per CTA (split by rank)
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const ACCUM_HALF_COLS: u32 = 256;
            const CTA_MASK_PAIR: u16 = 0b11;
            // Clears bit 24 (CTA rank within pair) + alignment bits 2:0 of a shared
            // memory barrier address, redirecting TMA completions to the leader CTA's
            // barrier.
            const PEER_BIT_MASK: u32 = 0xFEFFFFF8;
            // L2 cache-blocking: the 256-column output tile doubles the physical
            // width of each N tile, so preserve the accepted cache-band widths by
            // halving the prior logical group sizes.
            let swizzle_g: u32 = if tiles_m <= 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let _ctaid = thread::blockIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << (my_rank as u16);
            if tid == 0 {
                mbarrier_init(&raw mut TMA_BAR0, 1);
                mbarrier_init(&raw mut TMA_BAR1, 1);
                mbarrier_init(&raw mut TMA_BAR2, 1);
                mbarrier_init(&raw mut TMA_BAR3, 1);
                mbarrier_init(&raw mut MMA_BAR0, 2);
                mbarrier_init(&raw mut MMA_BAR1, 2);
                mbarrier_init(&raw mut MMA_BAR2, 2);
                mbarrier_init(&raw mut MMA_BAR3, 2);
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 256);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 256);
                mbarrier_init(&raw mut TILE_READY, 1);
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // Pre-arrive all MMA stage barriers so producer can start immediately.
            if tid == 0 {
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
                mbarrier_arrive(&raw const MMA_BAR2);
                mbarrier_arrive(&raw const MMA_BAR3);
                mbarrier_arrive(&raw const MMA_BAR0);
                mbarrier_arrive(&raw const MMA_BAR1);
                mbarrier_arrive(&raw const MMA_BAR2);
                mbarrier_arrive(&raw const MMA_BAR3);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
            let elect_one_cta = my_rank == 0;

            let idesc = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;
            cluster::cluster_sync();

            if warp_id == TMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut macro_seq: u32 = 0;
                let mut clc_iter: u32 = 0;

                // The fixed host grid contains one raw CLC work ID per M256xN128
                // tile. Reinterpret its first quarter as M512xN256 macro tiles and
                // drain the remaining raw IDs without publishing consumer work.
                let cluster_base_id = thread::blockIdx_x() - my_rank;
                let mut raw_tile_idx = cluster_base_id / 2;
                let macro_tiles_m = tiles_m / 2;
                let wide_tiles_n = _tiles_n / 2;
                let macro_total = macro_tiles_m * wide_tiles_n;
                let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;

                loop {
                    if raw_tile_idx < macro_total {
                        // L2 cache-blocking swizzle over M512xN256 macro tiles.
                        let group_tiles = swizzle_g * macro_tiles_m;
                        let group = raw_tile_idx / group_tiles;
                        let in_group = raw_tile_idx % group_tiles;
                        let n_start = group * swizzle_g;
                        let band_w = if swizzle_g < wide_tiles_n - n_start {
                            swizzle_g
                        } else {
                            wide_tiles_n - n_start
                        };
                        let macro_m = in_group / band_w;
                        let tile_n = n_start + in_group % band_w;

                        if is_lane0 {
                            *(&raw mut TILE_INFO as *mut u32).add(0) = macro_m;
                            *(&raw mut TILE_INFO as *mut u32).add(1) = tile_n;
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 1;
                            mbarrier_arrive(&raw const TILE_READY);
                        }

                        let m_offset0 = (macro_m * 512 + my_rank * 128) as i32;
                        let m_offset1 = m_offset0 + 256;
                        let b_n_offset = (tile_n * 256 + my_rank * 128) as i32;

                        let mut k_idx: u32 = 0;
                        while k_idx < k_iters {
                            let global_k = macro_seq * k_iters + k_idx;
                            let stage = global_k & 3;
                            let mma_parity = (global_k >> 2) & 1;

                            let (
                                smem_a_ptr,
                                smem_b_ptr,
                                tma_bar_const,
                                tma_bar_mut,
                                mma_bar_const,
                            ): (
                                *mut u8,
                                *mut u8,
                                *const Barrier,
                                *mut Barrier,
                                *const Barrier,
                            ) = match stage {
                                0 => (
                                    &raw mut SMEM_A0 as *mut u8,
                                    &raw mut SMEM_B0 as *mut u8,
                                    &raw const TMA_BAR0 as *const Barrier,
                                    &raw mut TMA_BAR0 as *mut Barrier,
                                    &raw const MMA_BAR0 as *const Barrier,
                                ),
                                1 => (
                                    &raw mut SMEM_A1 as *mut u8,
                                    &raw mut SMEM_B1 as *mut u8,
                                    &raw const TMA_BAR1 as *const Barrier,
                                    &raw mut TMA_BAR1 as *mut Barrier,
                                    &raw const MMA_BAR1 as *const Barrier,
                                ),
                                2 => (
                                    &raw mut SMEM_A2 as *mut u8,
                                    &raw mut SMEM_B2 as *mut u8,
                                    &raw const TMA_BAR2 as *const Barrier,
                                    &raw mut TMA_BAR2 as *mut Barrier,
                                    &raw const MMA_BAR2 as *const Barrier,
                                ),
                                _ => (
                                    &raw mut SMEM_A3 as *mut u8,
                                    &raw mut SMEM_B3 as *mut u8,
                                    &raw const TMA_BAR3 as *const Barrier,
                                    &raw mut TMA_BAR3 as *mut Barrier,
                                    &raw const MMA_BAR3 as *const Barrier,
                                ),
                            };

                            // Both CTAs receive the MMA completion, so both producer
                            // warps can safely wait before reusing this SMEM stage.
                            while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}

                            if is_lane0 {
                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(
                                        tma_bar_const,
                                        1,
                                        (2 * A_TILE_BYTES + B_TILE_BYTES) * 2,
                                    );
                                }
                                let aliased_bar =
                                    ((tma_bar_mut as u32) & PEER_BIT_MASK) as *mut Barrier;
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset0,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr.add(A_TILE_BYTES as usize),
                                    a_tma,
                                    k_base,
                                    m_offset1,
                                    aliased_bar,
                                    self_mask,
                                );
                                // The fixed host descriptor exposes a 64x64 B panel.
                                // Concatenate two panels to form this CTA's 128 columns.
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr.add(B_PANEL_BYTES as usize),
                                    b_tma,
                                    k_base,
                                    b_n_offset + 64,
                                    aliased_bar,
                                    self_mask,
                                );
                            }

                            k_idx += 1;
                        }
                        macro_seq += 1;
                    }

                    // One cancellation query consumes one CLC barrier phase even
                    // when its raw work ID is outside the reduced wide-tile range.
                    let clc_parity = clc_iter & 1;
                    if is_lane0 {
                        mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                        if elect_one_cta {
                            clc_try_cancel_multicast(resp_ptr as *mut u8, &raw mut CLC_BAR);
                        }
                    }
                    while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}
                    clc_iter += 1;

                    let resp_lo = *resp_ptr;
                    let resp_hi = *resp_ptr.add(1);
                    let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                    if is_canceled == 0 {
                        if is_lane0 {
                            *(&raw mut TILE_INFO as *mut u32).add(2) = 0;
                            mbarrier_arrive(&raw const TILE_READY);
                        }
                        break;
                    }

                    let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                    raw_tile_idx = first_stolen / 2;
                }
            }

            if warp_id == MMA_WARP {
                let is_lane0 = lane_id == 0;
                let mut macro_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    if has_work == 0 {
                        break;
                    }

                    if elect_one_cta && macro_iter > 0 {
                        let empty_parity = (macro_iter - 1) & 1;
                        while !mbarrier_try_wait_parity(&raw const ACCUM_EMPTY0, empty_parity) {}
                    }

                    let macro_k_base = macro_iter * k_iters;
                    let mut k_idx: u32 = 0;
                    // Four K64 stages form one complete SMEM ring cycle. Each stage
                    // commits its half-0 and half-1 MMA groups separately.
                    #[unroll(4)]
                    while k_idx < k_iters {
                        let global_k = macro_k_base + k_idx;
                        let stage = k_idx & 3;
                        let tma_parity = (global_k >> 2) & 1;

                        let (smem_a_base, smem_b_base, tma_bar_const, mma_bar_mut): (
                            u64,
                            u64,
                            *const Barrier,
                            *mut Barrier,
                        ) = match stage {
                            0 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A0 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B0 as *const u8),
                                &raw const TMA_BAR0 as *const Barrier,
                                &raw mut MMA_BAR0 as *mut Barrier,
                            ),
                            1 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A1 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B1 as *const u8),
                                &raw const TMA_BAR1 as *const Barrier,
                                &raw mut MMA_BAR1 as *mut Barrier,
                            ),
                            2 => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A2 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B2 as *const u8),
                                &raw const TMA_BAR2 as *const Barrier,
                                &raw mut MMA_BAR2 as *mut Barrier,
                            ),
                            _ => (
                                cvta_generic_to_shared_offset(&raw const SMEM_A3 as *const u8),
                                cvta_generic_to_shared_offset(&raw const SMEM_B3 as *const u8),
                                &raw const TMA_BAR3 as *const Barrier,
                                &raw mut MMA_BAR3 as *mut Barrier,
                            ),
                        };

                        // TMA completions are aliased to the leader CTA. The leader
                        // issues both pair-UMMA outputs while the follower advances
                        // the same K-stage loop without touching the barriers.
                        if elect_one_cta {
                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}

                            if is_lane0 {
                                let mut j: u32 = 0;
                                while j < 4 {
                                    let byte_offset = (j * 32) as u64;
                                    let a_desc = build_smem_descriptor(
                                        smem_a_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let b_desc = build_smem_descriptor(
                                        smem_b_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let accumulate = k_idx > 0 || j > 0;
                                    tcgen05_mma_f16_cg2(
                                        tmem_addr,
                                        a_desc,
                                        b_desc,
                                        idesc,
                                        accumulate,
                                    );
                                    j += 1;
                                }
                                tcgen05_commit_multicast_cg2(
                                    mma_bar_mut as *mut u64,
                                    CTA_MASK_PAIR,
                                );
                                if k_idx + 1 == k_iters {
                                    tcgen05_commit_multicast_cg2(
                                        &raw mut ACCUM_FULL0 as *mut u64,
                                        CTA_MASK_PAIR,
                                    );
                                }
                            }

                            // Once half 0 of the new macro has started, wait until the
                            // prior macro's half 1 has drained before overwriting it.
                            if macro_iter > 0 && k_idx == 0 {
                                let empty_parity = (macro_iter - 1) & 1;
                                while !mbarrier_try_wait_parity(
                                    &raw const ACCUM_EMPTY1,
                                    empty_parity,
                                ) {}
                            }

                            if is_lane0 {
                                let mut j: u32 = 0;
                                while j < 4 {
                                    let byte_offset = (j * 32) as u64;
                                    let a_desc = build_smem_descriptor(
                                        smem_a_base + A_TILE_BYTES as u64 + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let b_desc = build_smem_descriptor(
                                        smem_b_base + byte_offset,
                                        LBO_BYTES,
                                        SBO_BYTES,
                                        SWIZZLE_128B,
                                    );
                                    let accumulate = k_idx > 0 || j > 0;
                                    tcgen05_mma_f16_cg2(
                                        tmem_addr + ACCUM_HALF_COLS,
                                        a_desc,
                                        b_desc,
                                        idesc,
                                        accumulate,
                                    );
                                    j += 1;
                                }
                                tcgen05_commit_multicast_cg2(
                                    mma_bar_mut as *mut u64,
                                    CTA_MASK_PAIR,
                                );
                                if k_idx + 1 == k_iters {
                                    tcgen05_commit_multicast_cg2(
                                        &raw mut ACCUM_FULL1 as *mut u64,
                                        CTA_MASK_PAIR,
                                    );
                                }
                            }
                        }

                        k_idx += 1;
                    }

                    macro_iter += 1;
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let mut epi_macro_iter: u32 = 0;
                let mut tile_parity: u32 = 0;

                let leader_accum_empty0_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as *const u8);
                let leader_accum_empty1_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as *const u8);

                const SCRATCH_BF16_COLS: usize = 128;
                const SCRATCH_U32_COLS: usize = SCRATCH_BF16_COLS / 2;
                const SCRATCH_U32_PER_WARP: usize = 32 * SCRATCH_U32_COLS;
                const SCRATCH_ROW_BYTES: usize = SCRATCH_BF16_COLS * 2;
                let warp_scratch_u32 = warp_id as usize * SCRATCH_U32_PER_WARP;
                let warp_scratch_u8 =
                    (&raw mut SMEM_OUT as *mut u8).add(warp_scratch_u32 * 4);
                let row_within_8 = (lane_id % 8) as usize;
                let is_second_matrix = (8..16).contains(&lane_id);
                let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

                loop {
                    while !mbarrier_try_wait_parity(&raw const TILE_READY, tile_parity) {}
                    tile_parity ^= 1;

                    let has_work = *(&raw const TILE_INFO as *const u32).add(2);
                    if has_work == 0 {
                        break;
                    }

                    let macro_m = *(&raw const TILE_INFO as *const u32).add(0);
                    let tile_n = *(&raw const TILE_INFO as *const u32).add(1);
                    let full_parity = epi_macro_iter & 1;
                    let n_u32 = (n / 2) as usize;
                    let tile_col_base = (tile_n * 128) as usize;

                    let mut macro_half = 0u32;
                    while macro_half < 2 {
                        if macro_half == 0 {
                            while !mbarrier_try_wait_parity(
                                &raw const ACCUM_FULL0,
                                full_parity,
                            ) {}
                        } else {
                            while !mbarrier_try_wait_parity(
                                &raw const ACCUM_FULL1,
                                full_parity,
                            ) {}
                        }

                        let tmem_half_offset = macro_half * ACCUM_HALF_COLS;
                        let macro_row_base =
                            (macro_m * 512 + macro_half * 256 + my_rank * 128) as usize;

                        let mut n_chunk = 0u32;
                        while n_chunk < 2 {
                            let tmem_chunk_offset = tmem_half_offset + n_chunk * 128;

                            let mut tmem_row_block = 0u32;
                            while tmem_row_block < 2 {
                                let tmem_row = warp_id * 32 + tmem_row_block * 16;
                                let scratch_row_base = tmem_row_block as usize * 16;

                                let mut col_block = 0u32;
                                while col_block < 8 {
                                    let col_offset = (col_block * 16) as usize;

                                    let regs_a = tcgen05_ld_16x256b_pure(
                                        tmem_addr
                                            + (tmem_row << 16)
                                            + tmem_chunk_offset
                                            + col_offset as u32,
                                    );
                                    tcgen05_load_wait();

                                    let regs_b = tcgen05_ld_16x256b_pure(
                                        tmem_addr
                                            + (tmem_row << 16)
                                            + tmem_chunk_offset
                                            + col_offset as u32
                                            + 8,
                                    );
                                    tcgen05_load_wait();

                                    let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                                    let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);
                                    let scratch_row_lo = scratch_row_base + row_within_8;
                                    let smem_addr_lo = warp_scratch_u8.add(
                                        scratch_row_lo * SCRATCH_ROW_BYTES
                                            + col_offset * 2
                                            + col_offset_for_matrix2,
                                    );
                                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                                    let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                                    let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);
                                    let scratch_row_hi =
                                        scratch_row_base + 8 + row_within_8;
                                    let smem_addr_hi = warp_scratch_u8.add(
                                        scratch_row_hi * SCRATCH_ROW_BYTES
                                            + col_offset * 2
                                            + col_offset_for_matrix2,
                                    );
                                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                                    col_block += 1;
                                }

                                tmem_row_block += 1;
                            }

                            warp::sync_mask(u32::MAX);

                            let global_row_base =
                                macro_row_base + warp_id as usize * 32;
                            let global_col_base =
                                tile_col_base + n_chunk as usize * SCRATCH_U32_COLS;

                            let mut elem = lane_id as usize * 2;
                            while elem < SCRATCH_U32_PER_WARP {
                                let local_row = elem / SCRATCH_U32_COLS;
                                let local_col = elem % SCRATCH_U32_COLS;
                                let smem_idx = warp_scratch_u32
                                    + local_row * SCRATCH_U32_COLS
                                    + local_col;
                                let global_idx = (global_row_base + local_row) * n_u32
                                    + global_col_base
                                    + local_col;
                                let packed = (SMEM_OUT[smem_idx] as u64)
                                    | ((SMEM_OUT[smem_idx + 1] as u64) << 32);
                                let global_ptr =
                                    out.get_unchecked_mut(global_idx) as *mut u32 as *mut u64;

                                *global_ptr = packed;
                                elem += 64;
                            }

                            warp::sync_mask(u32::MAX);
                            n_chunk += 1;
                        }

                        if elect_one_cta {
                            if macro_half == 0 {
                                mbarrier_arrive(&raw const ACCUM_EMPTY0);
                            } else {
                                mbarrier_arrive(&raw const ACCUM_EMPTY1);
                            }
                        } else if macro_half == 0 {
                            mbarrier_arrive_cluster(leader_accum_empty0_addr);
                        } else {
                            mbarrier_arrive_cluster(leader_accum_empty1_addr);
                        }

                        macro_half += 1;
                    }

                    epi_macro_iter += 1;
                }
            }

            // BUG FIX: cluster_sync before exit prevents "Cluster target block not present"
            // (CUDA_EXCEPTION_17). Without this, a fast CTA can exit while its partner is still
            // executing cross-CTA operations (e.g., mbarrier_arrive_cluster in the epilogue).
            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            if tid == 0 {
                mbarrier_inval(&raw mut TMA_BAR0);
                mbarrier_inval(&raw mut TMA_BAR1);
                mbarrier_inval(&raw mut TMA_BAR2);
                mbarrier_inval(&raw mut TMA_BAR3);
                mbarrier_inval(&raw mut MMA_BAR0);
                mbarrier_inval(&raw mut MMA_BAR1);
                mbarrier_inval(&raw mut MMA_BAR2);
                mbarrier_inval(&raw mut MMA_BAR3);
                mbarrier_inval(&raw mut ACCUM_FULL0);
                mbarrier_inval(&raw mut ACCUM_FULL1);
                mbarrier_inval(&raw mut ACCUM_EMPTY0);
                mbarrier_inval(&raw mut ACCUM_EMPTY1);
                mbarrier_inval(&raw mut TILE_READY);
                mbarrier_inval(&raw mut CLC_BAR);
            }
        }
    }
}

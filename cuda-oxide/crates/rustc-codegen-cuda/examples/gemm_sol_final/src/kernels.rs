// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cuda_module]
mod kernels {
    use super::*;

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
                static mut SMEM_BUF: SharedArray<u8, 229376, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 16, 4> = SharedArray::UNINIT;

            // Up to 8-stage TMA <-> MMA pipeline
            static mut TMA_BAR: SharedArray<u64, 8, 8> = SharedArray::UNINIT;
            static mut MMA_BAR: SharedArray<u64, 8, 8> = SharedArray::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: SharedArray<u64, 4, 8> = SharedArray::UNINIT;

            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_PANEL32_BYTES: u32 = 32 * 64 * 2;
            const B_TILE_BYTES_256: u32 = 128 * 64 * 2;
            const B_TILE_BYTES_192: u32 = 96 * 64 * 2;
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 256;
            const CTA_MASK_PAIR: u16 = 0b11;
            const PEER_BIT_MASK: u32 = 0xFEFFFFF8;
            let num_stages: usize = 7;
            let b_smem_offset: usize = 7 * 16384;
            let swizzle_g: u32 = if tiles_m == 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let _ctaid = thread::blockIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << (my_rank as u16);
            if tid == 0 {
                let mut s = 0usize;
                while s < 7 {
                    mbarrier_init((&raw mut TMA_BAR as *mut Barrier).add(s), 1);
                    mbarrier_init((&raw mut MMA_BAR as *mut Barrier).add(s), 1);
                    s += 1;
                }
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 8);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 8);
                let mut ts = 0usize;
                while ts < 4 {
                    mbarrier_init((&raw mut TILE_READY as *mut Barrier).add(ts), 1);
                    ts += 1;
                }
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            cluster::cluster_sync();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
            let elect_one_cta = my_rank == 0;

            let idesc_256 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_192 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 192 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_224 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 224 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            if warp_id == TMA_WARP {
                if lane_id == 0 {
                    let mut stage: usize = 0;
                    let mut mma_parity: u32 = 1;
                    let mut tile_seq: usize = 0;
                    let mut clc_iter: u32 = 0;

                    let cluster_base_id = thread::blockIdx_x() - my_rank;
                    let mut raw_tile_idx = cluster_base_id / 2;
                    let wide_tiles_n = _tiles_n / 2;
                    let wide_total = tiles_m * wide_tiles_n;
                    let num_waves = (wide_total + 73) / 74;
                    let work_total = num_waves * 74;
                    let num_convert = work_total - wide_total;
                    let use_n224 = tiles_m >= 32;
                    let full_tiles = if use_n224 {
                        wide_total - 7 * num_convert
                    } else {
                        wide_total - 3 * num_convert
                    };
                    let head_cols = if use_n224 {
                        wide_tiles_n - 7
                    } else if tiles_m == 16 {
                        wide_tiles_n - 9
                    } else {
                        wide_tiles_n - 3
                    };
                    let head_tiles = tiles_m * head_cols;
                    let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;
                    let aliased_tma_bar0 = (((&raw mut TMA_BAR as *mut Barrier) as usize)
                        & (PEER_BIT_MASK as usize))
                        as *mut Barrier;

                    loop {
                        let clc_parity = clc_iter & 1;
                        if raw_tile_idx < work_total {
                            let (tile_m, col_bf16_base, tile_kind) = if raw_tile_idx < head_tiles {
                                let group_tiles = swizzle_g * tiles_m;
                                let group = raw_tile_idx / group_tiles;
                                let in_group = raw_tile_idx % group_tiles;
                                let n_start = group * swizzle_g;
                                let band_w = if swizzle_g < head_cols - n_start {
                                    swizzle_g
                                } else {
                                    head_cols - n_start
                                };
                                let raw_m = in_group / band_w;
                                let tm = if (group & 1) != 0 {
                                    tiles_m - 1 - raw_m
                                } else {
                                    raw_m
                                };
                                let tn = n_start + in_group % band_w;
                                (tm, tn * 256, 0u32)
                                } else if use_n224 {
                                if raw_tile_idx < full_tiles {
                                    let rem = raw_tile_idx - head_tiles;
                                    let t = rem / 7;
                                    let sub = rem % 7;
                                    (t, (head_cols + sub) * 256, 0u32)
                                } else {
                                    let tail = raw_tile_idx - full_tiles;
                                    let t = (full_tiles - head_tiles) / 7 + (tail / 8);
                                    let sub = tail & 7;
                                    (t, head_cols * 256 + sub * 224, 2u32)
                                }
                            } else if raw_tile_idx < full_tiles {
                                let rem = raw_tile_idx - head_tiles;
                                let t = rem / 3;
                                let sub = rem % 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn = head_cols + band * 3 + sub;
                                (tm, tn * 256, 0u32)
                            } else {
                                let tail = raw_tile_idx - full_tiles;
                                let t = (full_tiles - head_tiles) / 3 + (tail / 4);
                                let sub = tail & 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn_base = head_cols + band * 3;
                                (tm, tn_base * 256 + sub * 192, 1u32)
                            };

                            let packed_col = (col_bf16_base >> 5) | (tile_kind << 30);

                            let t_slot = tile_seq & 3;
                            let info_base = (&raw mut TILE_INFO as *mut u32).add(t_slot * 4);
                            *info_base.add(0) = tile_m;
                            *info_base.add(1) = packed_col;
                            *info_base.add(2) = 1;
                            mbarrier_arrive(
                                (&raw const TILE_READY as *const Barrier).add(t_slot),
                            );
                            tile_seq += 1;

                            let m_offset = (tile_m * 256 + my_rank * 128) as i32;
                            let b_n_offset = if tile_kind == 1 {
                                (col_bf16_base + my_rank * 96) as i32
                            } else if tile_kind == 2 {
                                (col_bf16_base + my_rank * 112) as i32
                            } else {
                                (col_bf16_base + my_rank * 128) as i32
                            };
                            let tx_bytes = if tile_kind == 1 {
                                (A_TILE_BYTES + B_TILE_BYTES_192) * 2
                            } else {
                                (A_TILE_BYTES + B_TILE_BYTES_256) * 2
                            };

                            let mut k_idx: u32 = 0;
                            #[unroll(4)]
                            while k_idx < k_iters {
                                let smem_a_ptr =
                                    (&raw mut SMEM_BUF as *mut u8).add(stage * 16384);
                                let smem_b_ptr = (&raw mut SMEM_BUF as *mut u8)
                                    .add(b_smem_offset + stage * 16384);
                                let tma_bar_const =
                                    (&raw const TMA_BAR as *const Barrier).add(stage);
                                let aliased_bar = aliased_tma_bar0.add(stage);
                                let mma_bar_const =
                                    (&raw const MMA_BAR as *const Barrier).add(stage);

                                while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}

                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(tma_bar_const, 1, tx_bytes);
                                }
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                if tile_kind == 1 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 2) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                } else if tiles_m == 16 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 2) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 3) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 96,
                                        aliased_bar,
                                        self_mask,
                                    );
                                }
                                if k_idx + 4 == k_iters {
                                    mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                                    if elect_one_cta {
                                        clc_try_cancel_multicast(
                                            resp_ptr as *mut u8,
                                            &raw mut CLC_BAR,
                                        );
                                    }
                                }

                                stage += 1;
                                if stage == num_stages {
                                    stage = 0;
                                    mma_parity ^= 1;
                                }
                                k_idx += 1;
                            }
                        } else {
                            mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                            if elect_one_cta {
                                clc_try_cancel_multicast(
                                    resp_ptr as *mut u8,
                                    &raw mut CLC_BAR,
                                );
                            }
                        }

                        while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}
                        clc_iter += 1;

                        let resp_lo = *resp_ptr;
                        let resp_hi = *resp_ptr.add(1);
                        let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled == 0 {
                            let t_slot = tile_seq & 3;
                            let info_base = (&raw mut TILE_INFO as *mut u32).add(t_slot * 4);
                            *info_base.add(2) = 0;
                            mbarrier_arrive(
                                (&raw const TILE_READY as *const Barrier).add(t_slot),
                            );
                            break;
                        }

                        let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        raw_tile_idx = first_stolen / 2;
                    }
                }
            }

            if warp_id == MMA_WARP {
                if elect_one_cta && lane_id == 0 {
                    let mut tile_iter: u32 = 0;
                    let mut stage: usize = 0;
                    let mut tma_parity: u32 = 0;
                    let smem_a_base0 =
                        cvta_generic_to_shared_offset(&raw const SMEM_BUF as *const u8);
                    let smem_b_base0 = smem_a_base0 + b_smem_offset as u64;
                    let a_desc0 =
                        build_smem_descriptor(smem_a_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);
                    let b_desc0 =
                        build_smem_descriptor(smem_b_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);

                    loop {
                        let t_slot = (tile_iter & 3) as usize;
                        let t_parity = (tile_iter >> 2) & 1;
                        while !mbarrier_try_wait_parity(
                            (&raw const TILE_READY as *const Barrier).add(t_slot),
                            t_parity,
                        ) {}

                        let info_base = (&raw const TILE_INFO as *const u32).add(t_slot * 4);
                        let has_work = *info_base.add(2);
                        if has_work == 0 {
                            break;
                        }
                        let packed_col = *info_base.add(1);
                        let tile_kind = packed_col >> 30;
                        let idesc = if tile_kind == 1 {
                            idesc_192
                        } else if tile_kind == 2 {
                            idesc_224
                        } else {
                            idesc_256
                        };

                        let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                        let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                        if tile_iter >= NUM_ACCUM_STAGES {
                            let empty_parity =
                                ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                            if accum_stage == 0 {
                                while !mbarrier_try_wait_parity(
                                    &raw const ACCUM_EMPTY0,
                                    empty_parity,
                                ) {}
                            } else {
                                while !mbarrier_try_wait_parity(
                                    &raw const ACCUM_EMPTY1,
                                    empty_parity,
                                ) {}
                            }
                        }

                        let mut k_idx: u32 = 0;
                        #[unroll(4)]
                        while k_idx < k_iters {
                            let stage_desc_offset = (stage as u64) << 10;
                            let a_desc_s = a_desc0 + stage_desc_offset;
                            let b_desc_s = b_desc0 + stage_desc_offset;
                            let tma_bar_const =
                                (&raw const TMA_BAR as *const Barrier).add(stage);
                            let mma_bar_mut =
                                (&raw mut MMA_BAR as *mut Barrier).add(stage);

                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}

                            let tmem_dst = tmem_addr + tmem_stage_offset;
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s,
                                b_desc_s,
                                idesc,
                                k_idx > 0,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 2,
                                b_desc_s + 2,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 4,
                                b_desc_s + 4,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 6,
                                b_desc_s + 6,
                                idesc,
                                true,
                            );

                            tcgen05_commit_multicast_cg2(
                                mma_bar_mut as *mut u64,
                                CTA_MASK_PAIR,
                            );

                            stage += 1;
                            if stage == num_stages {
                                stage = 0;
                                tma_parity ^= 1;
                            }
                            k_idx += 1;
                        }

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

                        tile_iter += 1;
                    }
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let mut epi_tile_iter: u32 = 0;

                let leader_accum_empty0_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as *const u8);
                let leader_accum_empty1_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as *const u8);

                let is_odd_lane = (lane_id & 1) != 0;
                let is_bit1_lane = (lane_id & 2) != 0;
                let lane_row_in_16 = ((lane_id / 4) + (lane_id & 1) * 8) as usize;
                let lane_col_u32 = ((lane_id & 2) * 2) as usize;

                loop {
                    let t_slot = (epi_tile_iter & 3) as usize;
                    let t_parity = (epi_tile_iter >> 2) & 1;
                    while !mbarrier_try_wait_parity(
                        (&raw const TILE_READY as *const Barrier).add(t_slot),
                        t_parity,
                    ) {}

                    let info_base = (&raw const TILE_INFO as *const u32).add(t_slot * 4);
                    let has_work = *info_base.add(2);
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = *info_base.add(0);
                    let packed_col = *info_base.add(1);
                    let tile_kind = packed_col >> 30;
                    let col_u32_start = ((packed_col & 0x3FFF_FFFF) * 16) as usize;

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128 + warp_id * 32) as usize;
                    let tile_col_base = col_u32_start + lane_col_u32;
                    let col_chunks: u32 = if tile_kind == 1 { 3 } else { 4 };

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;
                        let global_row =
                            tile_row_base + (tmem_row_block as usize * 16) + lane_row_in_16;
                        let row_out_base = global_row * n_u32 + tile_col_base;

                        let mut col_chunk = 0u32;
                        #[unroll(never)]
                        while col_chunk < col_chunks {
                            let col_offset = col_chunk * 64;

                            let regs = tcgen05_ld_16x256b_x8_pure(
                                tmem_addr + tmem_stage_offset + (tmem_row << 16) + col_offset,
                            );
                            tcgen05_load_wait();

                            if tmem_row_block == 1 && col_chunk + 1 == col_chunks && lane_id == 0 {
                                if elect_one_cta {
                                    if accum_stage == 0 {
                                        mbarrier_arrive(&raw const ACCUM_EMPTY0);
                                    } else {
                                        mbarrier_arrive(&raw const ACCUM_EMPTY1);
                                    }
                                } else if accum_stage == 0 {
                                    mbarrier_arrive_cluster(leader_accum_empty0_addr);
                                } else {
                                    mbarrier_arrive_cluster(leader_accum_empty1_addr);
                                }
                            }

                            let col_u32_base = (col_chunk as usize) * 32;
                            let max_pairs: usize =
                                if tile_kind == 2 && col_chunk == 3 { 2 } else { 4 };
                            let mut pair = 0usize;
                            #[unroll]
                            while pair < 4 {
                                if pair < max_pairs {
                                    let base = pair * 8;
                                    let p0_lo = cvt_bf16x2_f32(regs[base], regs[base + 1]);
                                    let p0_hi = cvt_bf16x2_f32(regs[base + 2], regs[base + 3]);
                                    let send0 = if is_odd_lane { p0_lo } else { p0_hi };
                                    let recv0 = warp::shuffle_xor_sync(u32::MAX, send0, 1);
                                    let lo0 = if is_odd_lane { recv0 } else { p0_lo };
                                    let hi0 = if is_odd_lane { p0_hi } else { recv0 };

                                    let p1_lo = cvt_bf16x2_f32(regs[base + 4], regs[base + 5]);
                                    let p1_hi = cvt_bf16x2_f32(regs[base + 6], regs[base + 7]);
                                    let send1 = if is_odd_lane { p1_lo } else { p1_hi };
                                    let recv1 = warp::shuffle_xor_sync(u32::MAX, send1, 1);
                                    let lo1 = if is_odd_lane { recv1 } else { p1_lo };
                                    let hi1 = if is_odd_lane { p1_hi } else { recv1 };

                                    let send_a = if is_bit1_lane { lo0 } else { lo1 };
                                    let send_b = if is_bit1_lane { hi0 } else { hi1 };
                                    let recv_a = warp::shuffle_xor_sync(u32::MAX, send_a, 2);
                                    let recv_b = warp::shuffle_xor_sync(u32::MAX, send_b, 2);

                                    let w0 = if is_bit1_lane { recv_a } else { lo0 };
                                    let w1 = if is_bit1_lane { recv_b } else { hi0 };
                                    let w2 = if is_bit1_lane { lo1 } else { recv_a };
                                    let w3 = if is_bit1_lane { hi1 } else { recv_b };

                                    #[repr(C, align(16))]
                                    struct U32x4(u32, u32, u32, u32);
                                    let ptr = out.get_unchecked_mut(
                                        row_out_base + col_u32_base + pair * 8,
                                    ) as *mut u32
                                        as *mut U32x4;
                                    *ptr = U32x4(w0, w1, w2, w3);
                                }
                                pair += 1;
                            }

                            col_chunk += 1;
                        }
                        tmem_row_block += 1;
                    }

                    epi_tile_iter += 1;
                }
            }

            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            }
    }

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
                static mut SMEM_BUF: SharedArray<u8, 229376, 128> = SharedArray::UNINIT;
            static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

            static mut TILE_INFO: SharedArray<u32, 16, 4> = SharedArray::UNINIT;

            // Up to 8-stage TMA <-> MMA pipeline
            static mut TMA_BAR: SharedArray<u64, 8, 8> = SharedArray::UNINIT;
            static mut MMA_BAR: SharedArray<u64, 8, 8> = SharedArray::UNINIT;
            static mut ACCUM_FULL0: Barrier = Barrier::UNINIT;
            static mut ACCUM_FULL1: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY0: Barrier = Barrier::UNINIT;
            static mut ACCUM_EMPTY1: Barrier = Barrier::UNINIT;
            static mut TILE_READY: SharedArray<u64, 4, 8> = SharedArray::UNINIT;

            static mut CLC_RESPONSE: SharedArray<u64, 2, 16> = SharedArray::UNINIT;
            static mut CLC_BAR: Barrier = Barrier::UNINIT;

            const A_TILE_BYTES: u32 = 128 * 64 * 2;
            const B_PANEL32_BYTES: u32 = 32 * 64 * 2;
            const B_TILE_BYTES_256: u32 = 128 * 64 * 2;
            const B_TILE_BYTES_192: u32 = 96 * 64 * 2;
            const SBO_BYTES: u32 = 1024;
            const LBO_BYTES: u32 = 16;
            const SWIZZLE_128B: u8 = 2;

            const TMA_WARP: u32 = 4;
            const MMA_WARP: u32 = 5;
            const NUM_ACCUM_STAGES: u32 = 2;
            const ACCUM_STAGE_COLS: u32 = 256;
            const CTA_MASK_PAIR: u16 = 0b11;
            const PEER_BIT_MASK: u32 = 0xFEFFFFF8;
            let num_stages: usize = 7;
            let b_smem_offset: usize = 7 * 16384;
            let swizzle_g: u32 = if tiles_m == 16 { 2 } else { 8 };

            let n = n as u32;
            let k = k as u32;
            let tid = thread::threadIdx_x();
            let _ctaid = thread::blockIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;

            let my_rank = cluster::cluster_ctaidX();
            let self_mask: u16 = 1u16 << (my_rank as u16);
            if tid == 0 {
                let mut s = 0usize;
                while s < 7 {
                    mbarrier_init((&raw mut TMA_BAR as *mut Barrier).add(s), 1);
                    mbarrier_init((&raw mut MMA_BAR as *mut Barrier).add(s), 1);
                    s += 1;
                }
                mbarrier_init(&raw mut ACCUM_FULL0, 1);
                mbarrier_init(&raw mut ACCUM_FULL1, 1);
                mbarrier_init(&raw mut ACCUM_EMPTY0, 8);
                mbarrier_init(&raw mut ACCUM_EMPTY1, 8);
                let mut ts = 0usize;
                while ts < 4 {
                    mbarrier_init((&raw mut TILE_READY as *mut Barrier).add(ts), 1);
                    ts += 1;
                }
                mbarrier_init(&raw mut CLC_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            cluster::cluster_sync();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);
            let elect_one_cta = my_rank == 0;

            let idesc_256 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape::M256_N256)
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_192 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 192 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();
            let idesc_224 = Tcgen05InstructionDescriptor::builder()
                .shape(Tcgen05MmaShape { m: 256, n: 224 })
                .element_type(Tcgen05ElementType::F16)
                .accumulator_type(Tcgen05AccumulatorType::F32)
                .build()
                .raw();

            let k_iters = k / 64;

            if warp_id == TMA_WARP {
                if lane_id == 0 {
                    let mut stage: usize = 0;
                    let mut mma_parity: u32 = 1;
                    let mut tile_seq: usize = 0;
                    let mut clc_iter: u32 = 0;

                    let cluster_base_id = thread::blockIdx_x() - my_rank;
                    let mut raw_tile_idx = cluster_base_id / 2;
                    let wide_tiles_n = _tiles_n / 2;
                    let wide_total = tiles_m * wide_tiles_n;
                    let num_waves = (wide_total + 73) / 74;
                    let work_total = num_waves * 74;
                    let num_convert = work_total - wide_total;
                    let use_n224 = tiles_m >= 32;
                    let full_tiles = if use_n224 {
                        wide_total - 7 * num_convert
                    } else {
                        wide_total - 3 * num_convert
                    };
                    let head_cols = if use_n224 {
                        wide_tiles_n - 7
                    } else if tiles_m == 16 {
                        wide_tiles_n - 9
                    } else {
                        wide_tiles_n - 3
                    };
                    let head_tiles = tiles_m * head_cols;
                    let resp_ptr = &raw mut CLC_RESPONSE as *mut u64;
                    let aliased_tma_bar0 = (((&raw mut TMA_BAR as *mut Barrier) as usize)
                        & (PEER_BIT_MASK as usize))
                        as *mut Barrier;

                    loop {
                        let clc_parity = clc_iter & 1;
                        if raw_tile_idx < work_total {
                            let (tile_m, col_bf16_base, tile_kind) = if raw_tile_idx < head_tiles {
                                let group_tiles = swizzle_g * tiles_m;
                                let group = raw_tile_idx / group_tiles;
                                let in_group = raw_tile_idx % group_tiles;
                                let n_start = group * swizzle_g;
                                let band_w = if swizzle_g < head_cols - n_start {
                                    swizzle_g
                                } else {
                                    head_cols - n_start
                                };
                                let raw_m = in_group / band_w;
                                let tm = if (group & 1) != 0 {
                                    tiles_m - 1 - raw_m
                                } else {
                                    raw_m
                                };
                                let tn = n_start + in_group % band_w;
                                (tm, tn * 256, 0u32)
                                } else if use_n224 {
                                if raw_tile_idx < full_tiles {
                                    let rem = raw_tile_idx - head_tiles;
                                    let t = rem / 7;
                                    let sub = rem % 7;
                                    (t, (head_cols + sub) * 256, 0u32)
                                } else {
                                    let tail = raw_tile_idx - full_tiles;
                                    let t = (full_tiles - head_tiles) / 7 + (tail / 8);
                                    let sub = tail & 7;
                                    (t, head_cols * 256 + sub * 224, 2u32)
                                }
                            } else if raw_tile_idx < full_tiles {
                                let rem = raw_tile_idx - head_tiles;
                                let t = rem / 3;
                                let sub = rem % 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn = head_cols + band * 3 + sub;
                                (tm, tn * 256, 0u32)
                            } else {
                                let tail = raw_tile_idx - full_tiles;
                                let t = (full_tiles - head_tiles) / 3 + (tail / 4);
                                let sub = tail & 3;
                                let band = t / tiles_m;
                                let tm = t % tiles_m;
                                let tn_base = head_cols + band * 3;
                                (tm, tn_base * 256 + sub * 192, 1u32)
                            };

                            let packed_col = (col_bf16_base >> 5) | (tile_kind << 30);

                            let t_slot = tile_seq & 3;
                            let info_base = (&raw mut TILE_INFO as *mut u32).add(t_slot * 4);
                            *info_base.add(0) = tile_m;
                            *info_base.add(1) = packed_col;
                            *info_base.add(2) = 1;
                            mbarrier_arrive(
                                (&raw const TILE_READY as *const Barrier).add(t_slot),
                            );
                            tile_seq += 1;

                            let m_offset = (tile_m * 256 + my_rank * 128) as i32;
                            let b_n_offset = if tile_kind == 1 {
                                (col_bf16_base + my_rank * 96) as i32
                            } else if tile_kind == 2 {
                                (col_bf16_base + my_rank * 112) as i32
                            } else {
                                (col_bf16_base + my_rank * 128) as i32
                            };
                            let tx_bytes = if tile_kind == 1 {
                                (A_TILE_BYTES + B_TILE_BYTES_192) * 2
                            } else {
                                (A_TILE_BYTES + B_TILE_BYTES_256) * 2
                            };

                            let mut k_idx: u32 = 0;
                            #[unroll(4)]
                            while k_idx < k_iters {
                                let smem_a_ptr =
                                    (&raw mut SMEM_BUF as *mut u8).add(stage * 16384);
                                let smem_b_ptr = (&raw mut SMEM_BUF as *mut u8)
                                    .add(b_smem_offset + stage * 16384);
                                let tma_bar_const =
                                    (&raw const TMA_BAR as *const Barrier).add(stage);
                                let aliased_bar = aliased_tma_bar0.add(stage);
                                let mma_bar_const =
                                    (&raw const MMA_BAR as *const Barrier).add(stage);

                                while !mbarrier_try_wait_parity(mma_bar_const, mma_parity) {}

                                if elect_one_cta {
                                    mbarrier_arrive_expect_tx(tma_bar_const, 1, tx_bytes);
                                }
                                let k_base = (k_idx * 64) as i32;
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_a_ptr,
                                    a_tma,
                                    k_base,
                                    m_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                    smem_b_ptr,
                                    b_tma,
                                    k_base,
                                    b_n_offset,
                                    aliased_bar,
                                    self_mask,
                                );
                                if tile_kind == 1 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 2) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                } else if tiles_m == 16 {
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add(B_PANEL32_BYTES as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 32,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 2) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 64,
                                        aliased_bar,
                                        self_mask,
                                    );
                                    cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                        smem_b_ptr.add((B_PANEL32_BYTES * 3) as usize),
                                        b_tma,
                                        k_base,
                                        b_n_offset + 96,
                                        aliased_bar,
                                        self_mask,
                                    );
                                }
                                if k_idx + 4 == k_iters {
                                    mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                                    if elect_one_cta {
                                        clc_try_cancel_multicast(
                                            resp_ptr as *mut u8,
                                            &raw mut CLC_BAR,
                                        );
                                    }
                                }

                                stage += 1;
                                if stage == num_stages {
                                    stage = 0;
                                    mma_parity ^= 1;
                                }
                                k_idx += 1;
                            }
                        } else {
                            mbarrier_arrive_expect_tx(&raw const CLC_BAR, 1, 16);
                            if elect_one_cta {
                                clc_try_cancel_multicast(
                                    resp_ptr as *mut u8,
                                    &raw mut CLC_BAR,
                                );
                            }
                        }

                        while !mbarrier_try_wait_parity(&raw const CLC_BAR, clc_parity) {}
                        clc_iter += 1;

                        let resp_lo = *resp_ptr;
                        let resp_hi = *resp_ptr.add(1);
                        let is_canceled = clc_query_is_canceled(resp_lo, resp_hi);
                        if is_canceled == 0 {
                            let t_slot = tile_seq & 3;
                            let info_base = (&raw mut TILE_INFO as *mut u32).add(t_slot * 4);
                            *info_base.add(2) = 0;
                            mbarrier_arrive(
                                (&raw const TILE_READY as *const Barrier).add(t_slot),
                            );
                            break;
                        }

                        let first_stolen = clc_query_get_first_ctaid_x(resp_lo, resp_hi);
                        raw_tile_idx = first_stolen / 2;
                    }
                }
            }

            if warp_id == MMA_WARP {
                if elect_one_cta && lane_id == 0 {
                    let mut tile_iter: u32 = 0;
                    let mut stage: usize = 0;
                    let mut tma_parity: u32 = 0;
                    let smem_a_base0 =
                        cvta_generic_to_shared_offset(&raw const SMEM_BUF as *const u8);
                    let smem_b_base0 = smem_a_base0 + b_smem_offset as u64;
                    let a_desc0 =
                        build_smem_descriptor(smem_a_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);
                    let b_desc0 =
                        build_smem_descriptor(smem_b_base0, LBO_BYTES, SBO_BYTES, SWIZZLE_128B);

                    loop {
                        let t_slot = (tile_iter & 3) as usize;
                        let t_parity = (tile_iter >> 2) & 1;
                        while !mbarrier_try_wait_parity(
                            (&raw const TILE_READY as *const Barrier).add(t_slot),
                            t_parity,
                        ) {}

                        let info_base = (&raw const TILE_INFO as *const u32).add(t_slot * 4);
                        let has_work = *info_base.add(2);
                        if has_work == 0 {
                            break;
                        }
                        let packed_col = *info_base.add(1);
                        let tile_kind = packed_col >> 30;
                        let idesc = if tile_kind == 1 {
                            idesc_192
                        } else if tile_kind == 2 {
                            idesc_224
                        } else {
                            idesc_256
                        };

                        let accum_stage = tile_iter % NUM_ACCUM_STAGES;
                        let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                        if tile_iter >= NUM_ACCUM_STAGES {
                            let empty_parity =
                                ((tile_iter - NUM_ACCUM_STAGES) / NUM_ACCUM_STAGES) & 1;
                            if accum_stage == 0 {
                                while !mbarrier_try_wait_parity(
                                    &raw const ACCUM_EMPTY0,
                                    empty_parity,
                                ) {}
                            } else {
                                while !mbarrier_try_wait_parity(
                                    &raw const ACCUM_EMPTY1,
                                    empty_parity,
                                ) {}
                            }
                        }

                        let mut k_idx: u32 = 0;
                        #[unroll(4)]
                        while k_idx < k_iters {
                            let stage_desc_offset = (stage as u64) << 10;
                            let a_desc_s = a_desc0 + stage_desc_offset;
                            let b_desc_s = b_desc0 + stage_desc_offset;
                            let tma_bar_const =
                                (&raw const TMA_BAR as *const Barrier).add(stage);
                            let mma_bar_mut =
                                (&raw mut MMA_BAR as *mut Barrier).add(stage);

                            while !mbarrier_try_wait_parity(tma_bar_const, tma_parity) {}

                            let tmem_dst = tmem_addr + tmem_stage_offset;
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s,
                                b_desc_s,
                                idesc,
                                k_idx > 0,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 2,
                                b_desc_s + 2,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 4,
                                b_desc_s + 4,
                                idesc,
                                true,
                            );
                            tcgen05_mma_f16_cg2(
                                tmem_dst,
                                a_desc_s + 6,
                                b_desc_s + 6,
                                idesc,
                                true,
                            );

                            tcgen05_commit_multicast_cg2(
                                mma_bar_mut as *mut u64,
                                CTA_MASK_PAIR,
                            );

                            stage += 1;
                            if stage == num_stages {
                                stage = 0;
                                tma_parity ^= 1;
                            }
                            k_idx += 1;
                        }

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

                        tile_iter += 1;
                    }
                }

                tcgen05_relinquish_alloc_permit_cg2();
            }

            if warp_id < 4 {
                let mut epi_tile_iter: u32 = 0;

                let leader_accum_empty0_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY0, 0) as *const u8);
                let leader_accum_empty1_addr =
                    cvta_generic_to_shared_offset(cluster::map_shared_rank(&raw const ACCUM_EMPTY1, 0) as *const u8);

                let is_odd_lane = (lane_id & 1) != 0;
                let is_bit1_lane = (lane_id & 2) != 0;
                let lane_row_in_16 = ((lane_id / 4) + (lane_id & 1) * 8) as usize;
                let lane_col_u32 = ((lane_id & 2) * 2) as usize;

                loop {
                    let t_slot = (epi_tile_iter & 3) as usize;
                    let t_parity = (epi_tile_iter >> 2) & 1;
                    while !mbarrier_try_wait_parity(
                        (&raw const TILE_READY as *const Barrier).add(t_slot),
                        t_parity,
                    ) {}

                    let info_base = (&raw const TILE_INFO as *const u32).add(t_slot * 4);
                    let has_work = *info_base.add(2);
                    if has_work == 0 {
                        break;
                    }

                    let tile_m = *info_base.add(0);
                    let packed_col = *info_base.add(1);
                    let tile_kind = packed_col >> 30;
                    let col_u32_start = ((packed_col & 0x3FFF_FFFF) * 16) as usize;

                    let accum_stage = epi_tile_iter % NUM_ACCUM_STAGES;
                    let tmem_stage_offset = accum_stage * ACCUM_STAGE_COLS;

                    let full_parity = (epi_tile_iter / NUM_ACCUM_STAGES) & 1;
                    if accum_stage == 0 {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL0, full_parity) {}
                    } else {
                        while !mbarrier_try_wait_parity(&raw const ACCUM_FULL1, full_parity) {}
                    }

                    let n_u32 = (n / 2) as usize;
                    let tile_row_base = (tile_m * 256 + my_rank * 128 + warp_id * 32) as usize;
                    let tile_col_base = col_u32_start + lane_col_u32;
                    let col_chunks: u32 = if tile_kind == 1 { 3 } else { 4 };

                    let mut tmem_row_block = 0u32;
                    while tmem_row_block < 2 {
                        let tmem_row = warp_id * 32 + tmem_row_block * 16;
                        let global_row =
                            tile_row_base + (tmem_row_block as usize * 16) + lane_row_in_16;
                        let row_out_base = global_row * n_u32 + tile_col_base;

                        let mut col_chunk = 0u32;
                        #[unroll(never)]
                        while col_chunk < col_chunks {
                            let col_offset = col_chunk * 64;

                            let regs = tcgen05_ld_16x256b_x8_pure(
                                tmem_addr + tmem_stage_offset + (tmem_row << 16) + col_offset,
                            );
                            tcgen05_load_wait();

                            if tmem_row_block == 1 && col_chunk + 1 == col_chunks && lane_id == 0 {
                                if elect_one_cta {
                                    if accum_stage == 0 {
                                        mbarrier_arrive(&raw const ACCUM_EMPTY0);
                                    } else {
                                        mbarrier_arrive(&raw const ACCUM_EMPTY1);
                                    }
                                } else if accum_stage == 0 {
                                    mbarrier_arrive_cluster(leader_accum_empty0_addr);
                                } else {
                                    mbarrier_arrive_cluster(leader_accum_empty1_addr);
                                }
                            }

                            let col_u32_base = (col_chunk as usize) * 32;
                            let max_pairs: usize =
                                if tile_kind == 2 && col_chunk == 3 { 2 } else { 4 };
                            let mut pair = 0usize;
                            #[unroll]
                            while pair < 4 {
                                if pair < max_pairs {
                                    let base = pair * 8;
                                    let p0_lo = cvt_bf16x2_f32(regs[base], regs[base + 1]);
                                    let p0_hi = cvt_bf16x2_f32(regs[base + 2], regs[base + 3]);
                                    let send0 = if is_odd_lane { p0_lo } else { p0_hi };
                                    let recv0 = warp::shuffle_xor_sync(u32::MAX, send0, 1);
                                    let lo0 = if is_odd_lane { recv0 } else { p0_lo };
                                    let hi0 = if is_odd_lane { p0_hi } else { recv0 };

                                    let p1_lo = cvt_bf16x2_f32(regs[base + 4], regs[base + 5]);
                                    let p1_hi = cvt_bf16x2_f32(regs[base + 6], regs[base + 7]);
                                    let send1 = if is_odd_lane { p1_lo } else { p1_hi };
                                    let recv1 = warp::shuffle_xor_sync(u32::MAX, send1, 1);
                                    let lo1 = if is_odd_lane { recv1 } else { p1_lo };
                                    let hi1 = if is_odd_lane { p1_hi } else { recv1 };

                                    let send_a = if is_bit1_lane { lo0 } else { lo1 };
                                    let send_b = if is_bit1_lane { hi0 } else { hi1 };
                                    let recv_a = warp::shuffle_xor_sync(u32::MAX, send_a, 2);
                                    let recv_b = warp::shuffle_xor_sync(u32::MAX, send_b, 2);

                                    let w0 = if is_bit1_lane { recv_a } else { lo0 };
                                    let w1 = if is_bit1_lane { recv_b } else { hi0 };
                                    let w2 = if is_bit1_lane { lo1 } else { recv_a };
                                    let w3 = if is_bit1_lane { hi1 } else { recv_b };

                                    #[repr(C, align(16))]
                                    struct U32x4(u32, u32, u32, u32);
                                    let ptr = out.get_unchecked_mut(
                                        row_out_base + col_u32_base + pair * 8,
                                    ) as *mut u32
                                        as *mut U32x4;
                                    *ptr = U32x4(w0, w1, w2, w3);
                                }
                                pair += 1;
                            }

                            col_chunk += 1;
                        }
                        tmem_row_block += 1;
                    }

                    epi_tile_iter += 1;
                }
            }

            cluster::cluster_sync();

            if warp_id == 0 {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }
            }
    }
}

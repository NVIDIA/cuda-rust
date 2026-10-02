/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified tcgen05 Example (SM100+ / Blackwell only)
//!
//! Tests tcgen05 (Tensor Core Gen 5) infrastructure:
//!
//! **cta_group::1 (single CTA):**
//! - tcgen05_alloc() / tcgen05_dealloc()
//! - tcgen05_fence_before_thread_sync() / tcgen05_fence_after_thread_sync()
//! - tcgen05_commit()
//! - Tcgen05SmemDescriptor / Tcgen05InstructionDescriptor builders
//! - tcgen05_mma_ws_f16()
//! - tcgen05_cp_smem_to_tmem() / tcgen05_ld_16x256b_pure()
//!
//! **cta_group::2 (CTA pairs):**
//! - tcgen05_alloc_cg2() / tcgen05_dealloc_cg2()
//! - tcgen05_mma_f16_cg2()
//! - tcgen05_commit_cg2()
//!
//! CTA pairs place 2 CTAs on adjacent SMs (a TPC) to cooperate on larger
//! MMA tiles. All tcgen05 instructions in a kernel must use the same
//! cta_group value.
//!
//! NOTE: tcgen05 is Blackwell datacenter-only (sm_100a).
//!
//! Build and run with:
//!   cargo oxide run tcgen05

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use cuda_device::barrier::Barrier;
use cuda_device::shared::{SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    self, Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor,
    Tcgen05MmaShape, Tcgen05SmemDescriptor, Tcgen05SwizzleMode, tcgen05_alloc, tcgen05_alloc_cg2,
    tcgen05_commit, tcgen05_commit_multicast_cg2, tcgen05_cp_smem_to_tmem, tcgen05_dealloc,
    tcgen05_dealloc_cg2, tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2, tcgen05_mma_ws_f16,
};
use cuda_device::{CuSimd, DisjointSlice, cluster, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel for tcgen05 sync primitives (no MMA).
    #[kernel]
    // `unused_unsafe` on host only: fence helpers are shims there; device needs the explicit block.
    #[allow(unused_unsafe)]
    pub unsafe fn tcgen05_fence_test(mut output: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let gid = thread::index_1d();

            // Test SMEM descriptor builder (single-thread)
            let smem_addr = cvta_generic_to_shared_offset(&raw const SMEM as *const u8);
            let desc = Tcgen05SmemDescriptor::builder()
                .address(smem_addr)
                .leading_dim_bytes(128)
                .stride_bytes(128)
                .swizzle(Tcgen05SwizzleMode::Swizzle32B)
                .build()
                .raw();

            // Test fence primitives (single-thread)
            tcgen05_fence_before_thread_sync();
            tcgen05_fence_after_thread_sync();

            if tid == 0
                && let Some(out_elem) = output.get_mut(gid)
            {
                *out_elem = desc;
            }
        }
    }

    /// Test kernel for tcgen05 TMEM allocation.
    #[kernel]
    pub unsafe fn tcgen05_alloc_test(mut output: DisjointSlice<u32>) {
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let gid = thread::index_1d();
            let warp_id = warp::warp_id();

            if tid == 0 {
                *(&raw mut TMEM_ADDR as *mut u32) = 0xDEADBEEF;
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 32);
            }

            thread::sync_threads();

            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            if tid == 0
                && let Some(out_elem) = output.get_mut(gid)
            {
                *out_elem = tmem_addr;
            }

            if warp_id == 0 && tmem_addr != 0xDEADBEEF {
                tcgen05_dealloc(tmem_addr, 32);
            }
        }
    }

    /// Test kernel for tcgen05 commit with mbarrier.
    #[kernel]
    pub unsafe fn tcgen05_commit_test(mut output: DisjointSlice<u64>) {
        static mut MBAR: Barrier = Barrier::UNINIT;
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let gid = thread::index_1d();

            if tid == 0 {
                cuda_device::barrier::mbarrier_init(&raw mut MBAR, 1);
            }
            thread::sync_threads();

            let smem_addr = cvta_generic_to_shared_offset(&raw const SMEM as *const u8);
            let desc = Tcgen05SmemDescriptor::builder()
                .address(smem_addr)
                .leading_dim_bytes(128)
                .stride_bytes(128)
                .swizzle(Tcgen05SwizzleMode::None)
                .build()
                .raw();

            if tid == 0 {
                tcgen05_fence_before_thread_sync();
                tcgen05_commit(&raw mut MBAR as *mut u64);
            }

            cuda_device::barrier::mbarrier_try_wait(&raw const MBAR, 0);

            if tid == 0
                && let Some(out_elem) = output.get_mut(gid)
            {
                *out_elem = desc;
            }

            if tid == 0 {
                cuda_device::barrier::mbarrier_inval(&raw mut MBAR);
            }
        }
    }

    /// Minimal MMA test kernel - full tcgen05 pipeline.
    #[kernel]
    pub unsafe fn tcgen05_mma_minimal(mut output: DisjointSlice<u32>) {
        static mut SMEM_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        static mut SMEM_D: SharedArray<u32, 32, 4> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        static mut MBAR: Barrier = Barrier::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let gid = thread::index_1d();
            let warp_id = warp::warp_id();

            // Step 1: Initialize mbarrier
            if tid == 0 {
                cuda_device::barrier::mbarrier_init(&raw mut MBAR, 1);
            }
            thread::sync_threads();

            // Step 2: Allocate TMEM
            if warp_id == 0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 64);
            }
            thread::sync_threads();

            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            // Step 3: Copy A from SMEM to TMEM
            if tid == 0 {
                let smem_a_addr = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
                let a_desc = Tcgen05SmemDescriptor::builder()
                    .address(smem_a_addr)
                    .leading_dim_bytes(1024)
                    .stride_bytes(128)
                    .swizzle(Tcgen05SwizzleMode::None)
                    .build()
                    .raw();

                tcgen05_cp_smem_to_tmem(tmem_addr, a_desc);
            }
            thread::sync_threads();

            // Step 4: Issue MMA
            if tid == 0 {
                let smem_b_addr = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);
                let b_desc = Tcgen05SmemDescriptor::builder()
                    .address(smem_b_addr)
                    .leading_dim_bytes(1024)
                    .stride_bytes(128)
                    .swizzle(Tcgen05SwizzleMode::None)
                    .build()
                    .raw();

                let idesc = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M64_N64)
                    .element_type(Tcgen05ElementType::F16)
                    .accumulator_type(Tcgen05AccumulatorType::F32)
                    .build()
                    .raw();

                tcgen05_mma_ws_f16(tmem_addr, tmem_addr, 0, b_desc, idesc, false);

                // Step 5: Signal completion
                tcgen05_fence_before_thread_sync();
                tcgen05_commit(&raw mut MBAR as *mut u64);
            }

            // Step 6: Wait for MMA
            cuda_device::barrier::mbarrier_try_wait(&raw const MBAR, 0);

            // Step 7: Read D from TMEM
            if warp_id == 0 {
                let regs = tcgen05_ld_16x256b_pure(tmem_addr);
                tcgen05_load_wait();

                let lane_id = tid & 31;
                let smem_d_ptr = &raw mut SMEM_D as *mut f32;
                *smem_d_ptr.add(lane_id as usize) = regs[0];
            }
            thread::sync_threads();

            // Step 8: Output results
            if tid == 0 {
                let base_idx = gid.get();
                *output.get_unchecked_mut(base_idx) = tmem_addr;

                let smem_d_ptr = &raw const SMEM_D as *const u32;
                *output.get_unchecked_mut(base_idx + 1) = *smem_d_ptr;
            }

            // Step 9: Cleanup
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_dealloc(tmem_addr, 64);
            }

            if tid == 0 {
                cuda_device::barrier::mbarrier_inval(&raw mut MBAR);
            }
        }
    }

    /// Keeps every cta_group::1 tcgen05 copy form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_cp_cg1(tmem_addr: u32, smem_desc: u64) {
        unsafe {
            tcgen05::tcgen05_cp_smem_to_tmem(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x256b_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x256b_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13_b4x16_p64(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13_b6x16_p32(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13(tmem_addr, smem_desc);
        }
    }

    /// Keeps the cta_group::1 tcgen05 control forms in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_control_cg1(tmem_addr: u32, mbar: *mut u64) {
        unsafe {
            if thread::threadIdx_x() == 0 {
                tcgen05::tcgen05_shift_down(tmem_addr);
                tcgen05::tcgen05_commit_multicast(mbar, 1);
            }
        }
    }

    /// Keeps the cta_group::1 base tcgen05 MMA forms in device code.
    ///
    /// This kernel is compile-only and is never launched. The base MMA
    /// coverage is split by CTA group because ptxas enforces one tcgen05
    /// granularity per function ("Function ... uses single CTA
    /// (.cta_group::1) and CTA pair granularity (.cta_group::2) and that is
    /// not allowed"); a mixed kernel makes the whole module unassemblable
    /// and unloadable, taking the runtime test kernels down with it.
    #[kernel]
    pub unsafe fn compile_tcgen05_mma_base_cg1(
        d_tmem: u32,
        a_tmem: u32,
        metadata_tmem: u32,
        a_desc: u64,
        b_desc: u64,
        idesc: u32,
    ) {
        unsafe {
            tcgen05::tcgen05_mma_shared::<0, 1, 0>(d_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_shared::<2, 1, 2>(d_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_tensor::<0, 1, 0>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_sp_shared::<2, 1, 2>(
                d_tmem,
                a_desc,
                b_desc,
                idesc,
                false,
                metadata_tmem,
            );
            tcgen05::tcgen05_mma_sp_tensor_ashift::<0, 1, 0>(
                d_tmem,
                a_tmem,
                b_desc,
                idesc,
                false,
                metadata_tmem,
            );
        }
    }

    /// Keeps the cta_group::2 base tcgen05 MMA forms in device code.
    ///
    /// This kernel is compile-only and is never launched. See
    /// `compile_tcgen05_mma_base_cg1` for why the coverage is split by CTA
    /// group.
    #[kernel]
    pub unsafe fn compile_tcgen05_mma_base_cg2(
        d_tmem: u32,
        a_tmem: u32,
        metadata_tmem: u32,
        a_desc: u64,
        b_desc: u64,
        idesc: u32,
    ) {
        unsafe {
            tcgen05::tcgen05_mma_shared::<1, 2, 1>(d_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_shared::<3, 2, 3>(d_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_tensor_ashift::<1, 2, 1>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_sp_tensor::<3, 2, 3>(
                d_tmem,
                a_tmem,
                b_desc,
                idesc,
                false,
                metadata_tmem,
            );
        }
    }

    /// Keeps every warp-specialized tcgen05 MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_mma_ws(
        d_tmem: u32,
        a_tmem: u32,
        metadata_tmem: u32,
        a_desc: u64,
        b_desc: u64,
        zero_column_mask: u64,
        idesc: u32,
    ) {
        unsafe {
            tcgen05::tcgen05_mma_ws_shared::<0, 0, 0>(d_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_shared_zero_col_mask::<1, 1, 1>(
                d_tmem,
                a_desc,
                b_desc,
                idesc,
                false,
                zero_column_mask,
            );
            tcgen05::tcgen05_mma_ws_sp_shared::<2, 2, 2>(
                d_tmem,
                a_desc,
                b_desc,
                idesc,
                false,
                metadata_tmem,
            );
            tcgen05::tcgen05_mma_ws_sp_shared_zero_col_mask::<3, 3, 3>(
                d_tmem,
                a_desc,
                b_desc,
                idesc,
                false,
                metadata_tmem,
                zero_column_mask,
            );
            tcgen05::tcgen05_mma_ws_sp_tensor::<0, 1, 2>(
                d_tmem,
                a_tmem,
                b_desc,
                idesc,
                false,
                metadata_tmem,
            );
            tcgen05::tcgen05_mma_ws_sp_tensor_zero_col_mask::<1, 2, 3>(
                d_tmem,
                a_tmem,
                b_desc,
                idesc,
                false,
                metadata_tmem,
                zero_column_mask,
            );
            tcgen05::tcgen05_mma_ws_tensor::<0, 0, 0>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_tensor::<1, 1, 1>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_tensor::<2, 2, 2>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_tensor::<3, 3, 3>(d_tmem, a_tmem, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_tensor_zero_col_mask::<2, 3, 0>(
                d_tmem,
                a_tmem,
                b_desc,
                idesc,
                false,
                zero_column_mask,
            );

            tcgen05::tcgen05_mma_ws_e4m3(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_e5m2(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_e2m3(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_e3m2(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
            tcgen05::tcgen05_mma_ws_e2m1(d_tmem, a_tmem, a_desc, b_desc, idesc, false);
        }
    }

    /// Keeps every generated tcgen05 load form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_ld(tmem_addr: u32) {
        unsafe {
            let _ = tcgen05::tcgen05_ld_16x64b_x1_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x1_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x2_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x2_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x4_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x4_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x8_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x8_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x16_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x16_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x32_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x32_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x64_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x64_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x128_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x64b_x128_pack16(tmem_addr);

            let _ = tcgen05::tcgen05_ld_16x128b_x1_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x1_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x2_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x2_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x4_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x4_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x8_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x8_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x16_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x16_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x32_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x32_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x64_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x128b_x64_pack16(tmem_addr);

            let _ = tcgen05::tcgen05_ld_16x256b_x1_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x1_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x2_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x2_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x4_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x4_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x8_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x8_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x16_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x16_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x32_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x256b_x32_pack16(tmem_addr);

            let _ = tcgen05::tcgen05_ld_32x32b_x1_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x1_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x2_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x2_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x4_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x4_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x8_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x8_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x16_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x16_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x32_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x32_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x64_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x64_pack16(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x128_raw(tmem_addr);
            let _ = tcgen05::tcgen05_ld_32x32b_x128_pack16(tmem_addr);
        }
    }

    /// Keeps every generated non-offset tcgen05 store form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    pub unsafe fn compile_tcgen05_st(tmem_addr: u32) {
        unsafe {
            tcgen05::tcgen05_st_16x64b_x1_raw(tmem_addr, 0);
            tcgen05::tcgen05_st_16x64b_x1_unpack16(tmem_addr, 0);
            tcgen05::tcgen05_st_16x64b_x2_raw(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x64b_x2_unpack16(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x64b_x4_raw(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x64b_x4_unpack16(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x64b_x8_raw(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x64b_x8_unpack16(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x64b_x16_raw(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x64b_x16_unpack16(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x64b_x32_raw(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x64b_x32_unpack16(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x64b_x64_raw(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x64b_x64_unpack16(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x64b_x128_raw(tmem_addr, CuSimd::new([0; 128]));
            tcgen05::tcgen05_st_16x64b_x128_unpack16(tmem_addr, CuSimd::new([0; 128]));

            tcgen05::tcgen05_st_16x128b_x1_raw(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x128b_x1_unpack16(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x128b_x2_raw(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x128b_x2_unpack16(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x128b_x4_raw(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x128b_x4_unpack16(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x128b_x8_raw(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x128b_x8_unpack16(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x128b_x16_raw(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x128b_x16_unpack16(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x128b_x32_raw(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x128b_x32_unpack16(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x128b_x64_raw(tmem_addr, CuSimd::new([0; 128]));
            tcgen05::tcgen05_st_16x128b_x64_unpack16(tmem_addr, CuSimd::new([0; 128]));

            tcgen05::tcgen05_st_16x256b_x1_raw(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x256b_x1_unpack16(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x256b_x2_raw(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x256b_x2_unpack16(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x256b_x4_raw(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x256b_x4_unpack16(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x256b_x8_raw(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x256b_x8_unpack16(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x256b_x16_raw(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x256b_x16_unpack16(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x256b_x32_raw(tmem_addr, CuSimd::new([0; 128]));
            tcgen05::tcgen05_st_16x256b_x32_unpack16(tmem_addr, CuSimd::new([0; 128]));

            tcgen05::tcgen05_st_32x32b_x1_raw(tmem_addr, 0);
            tcgen05::tcgen05_st_32x32b_x1_unpack16(tmem_addr, 0);
            tcgen05::tcgen05_st_32x32b_x2_raw(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_32x32b_x2_unpack16(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_32x32b_x4_raw(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_32x32b_x4_unpack16(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_32x32b_x8_raw(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_32x32b_x8_unpack16(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_32x32b_x16_raw(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_32x32b_x16_unpack16(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_32x32b_x32_raw(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_32x32b_x32_unpack16(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_32x32b_x64_raw(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_32x32b_x64_unpack16(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_32x32b_x128_raw(tmem_addr, CuSimd::new([0; 128]));
            tcgen05::tcgen05_st_32x32b_x128_unpack16(tmem_addr, CuSimd::new([0; 128]));
        }
    }

    /// Keeps every half-split-offset tcgen05 load form in device code.
    #[kernel]
    pub unsafe fn compile_tcgen05_ld_offset(tmem_addr: u32) {
        unsafe {
            let _ = tcgen05::tcgen05_ld_16x32bx2_x1_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x1_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x2_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x2_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x4_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x4_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x8_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x8_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x16_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x16_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x32_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x32_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x64_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x64_pack16::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x128_raw::<16>(tmem_addr);
            let _ = tcgen05::tcgen05_ld_16x32bx2_x128_pack16::<16>(tmem_addr);
        }
    }

    /// Keeps every half-split-offset tcgen05 store form in device code.
    #[kernel]
    pub unsafe fn compile_tcgen05_st_offset(tmem_addr: u32) {
        unsafe {
            tcgen05::tcgen05_st_16x32bx2_x1_raw::<16>(tmem_addr, 0);
            tcgen05::tcgen05_st_16x32bx2_x1_unpack16::<16>(tmem_addr, 0);
            tcgen05::tcgen05_st_16x32bx2_x2_raw::<16>(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x32bx2_x2_unpack16::<16>(tmem_addr, CuSimd::new([0; 2]));
            tcgen05::tcgen05_st_16x32bx2_x4_raw::<16>(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x32bx2_x4_unpack16::<16>(tmem_addr, CuSimd::new([0; 4]));
            tcgen05::tcgen05_st_16x32bx2_x8_raw::<16>(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x32bx2_x8_unpack16::<16>(tmem_addr, CuSimd::new([0; 8]));
            tcgen05::tcgen05_st_16x32bx2_x16_raw::<16>(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x32bx2_x16_unpack16::<16>(tmem_addr, CuSimd::new([0; 16]));
            tcgen05::tcgen05_st_16x32bx2_x32_raw::<16>(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x32bx2_x32_unpack16::<16>(tmem_addr, CuSimd::new([0; 32]));
            tcgen05::tcgen05_st_16x32bx2_x64_raw::<16>(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x32bx2_x64_unpack16::<16>(tmem_addr, CuSimd::new([0; 64]));
            tcgen05::tcgen05_st_16x32bx2_x128_raw::<16>(tmem_addr, CuSimd::new([0; 128]));
            tcgen05::tcgen05_st_16x32bx2_x128_unpack16::<16>(tmem_addr, CuSimd::new([0; 128]));
        }
    }

    // =============================================================================
    // CTA Pair (cta_group::2) Test Kernels
    // =============================================================================

    /// Test TMEM alloc/dealloc with cta_group::2 (CTA pairs).
    ///
    /// Launches 2 CTAs as a cluster (= 1 CTA pair on adjacent SMs).
    /// Both CTAs cooperatively allocate TMEM with tcgen05_alloc_cg2,
    /// verify they get a valid address, then deallocate.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn tcgen05_alloc_cg2_test(mut output: DisjointSlice<u32>) {
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let block_rank = cluster::block_rank();

            if tid == 0 {
                *(&raw mut TMEM_ADDR as *mut u32) = 0xDEADBEEF;
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 32);
            }
            thread::sync_threads();

            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            if tid == 0 {
                *output.get_unchecked_mut(block_rank as usize) = tmem_addr;
            }

            if warp_id == 0 && tmem_addr != 0xDEADBEEF {
                tcgen05_dealloc_cg2(tmem_addr, 32);
            }
        }
    }

    /// Test MMA with cta_group::2 (CTA pairs).
    ///
    /// Two CTAs form a CTA pair. Both allocate TMEM with cta_group::2, and
    /// a single issuing thread from the pair launches the cooperative MMA.
    /// The pair cooperates on a larger effective
    /// tile (each SM's tensor core handles its half of the rows).
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn tcgen05_mma_cg2_test(mut output: DisjointSlice<u32>) {
        static mut SMEM_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        static mut MBAR: Barrier = Barrier::UNINIT;

        unsafe {
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let block_rank = cluster::block_rank();

            if tid == 0 {
                cuda_device::barrier::mbarrier_init(&raw mut MBAR, 1);
            }
            thread::sync_threads();

            if warp_id == 0 {
                tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();

            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            if tid == 0 && block_rank == 0 {
                let smem_b_addr = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);
                let b_desc = Tcgen05SmemDescriptor::builder()
                    .address(smem_b_addr)
                    .leading_dim_bytes(1024)
                    .stride_bytes(128)
                    .swizzle(Tcgen05SwizzleMode::None)
                    .build()
                    .raw();

                let idesc = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N128)
                    .element_type(Tcgen05ElementType::F16)
                    .accumulator_type(Tcgen05AccumulatorType::F32)
                    .build()
                    .raw();

                let a_smem_addr = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
                let a_desc = Tcgen05SmemDescriptor::builder()
                    .address(a_smem_addr)
                    .leading_dim_bytes(1024)
                    .stride_bytes(128)
                    .swizzle(Tcgen05SwizzleMode::None)
                    .build()
                    .raw();

                tcgen05_mma_f16_cg2(tmem_addr, a_desc, b_desc, idesc, false);

                tcgen05_fence_before_thread_sync();
                // cta_group::2 commit is issued by one thread in the CTA pair and
                // multicast to both CTAs' barriers (cluster ranks 0 and 1).
                tcgen05_commit_multicast_cg2(&raw mut MBAR as *mut u64, 0b11u16);
            }

            cuda_device::barrier::mbarrier_try_wait(&raw const MBAR, 0);

            if tid == 0 {
                *output.get_unchecked_mut(block_rank as usize) = tmem_addr;
            }

            thread::sync_threads();

            if warp_id == 0 && tmem_addr != 0xDEADBEEF {
                tcgen05_dealloc_cg2(tmem_addr, 512);
            }

            if tid == 0 {
                cuda_device::barrier::mbarrier_inval(&raw mut MBAR);
            }
        }
    }

    /// Keeps every cta_group::2 tcgen05 copy form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn compile_tcgen05_cp_cg2(tmem_addr: u32, smem_desc: u64) {
        unsafe {
            tcgen05::tcgen05_cp_smem_to_tmem_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x128b_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x256b_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_128x256b_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_32x128b_warpx4_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_4x256b_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_01_23_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13_b4x16_p64_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13_b6x16_p32_cg2(tmem_addr, smem_desc);
            tcgen05::tcgen05_cp_64x128b_warpx2_02_13_cg2(tmem_addr, smem_desc);
        }
    }

    /// Keeps the cta_group::2 tcgen05 control form in device code.
    ///
    /// This kernel is compile-only and is never launched.
    #[kernel]
    #[cluster_launch(2, 1, 1)]
    pub unsafe fn compile_tcgen05_control_cg2(tmem_addr: u32, mbar: *mut u64) {
        unsafe {
            if thread::threadIdx_x() == 0 && cluster::block_rank() == 0 {
                tcgen05::tcgen05_shift_down_cg2(tmem_addr);
                tcgen05::tcgen05_commit_cg2(mbar);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified tcgen05 Example ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // Gate on the GPUs that can actually execute this module BEFORE trying
    // to load it (same set as gemm_sol_final; keep in sync with
    // mir-importer's tcgen05 target support). Deciding "wrong GPU" from a
    // module-load failure is not sound: the driver reports arch-incompatible
    // PTX and genuinely malformed PTX with the same CUDA_ERROR_INVALID_PTX,
    // so a load-error fallback silently converts compiler bugs into a
    // PTX-only "pass".
    if !can_execute_tcgen05_ptx(major, minor) {
        println!("\n⚠️  WARNING: tcgen05 requires sm_100 (datacenter Blackwell)!");
        println!("   Your GPU is sm_{}{}", major, minor);
        if major == 9 {
            println!("   Hopper GPUs use WGMMA, not tcgen05.");
        } else if major >= 10 {
            println!("   Consumer Blackwell has no tcgen05.");
        }
        println!("   PTX was generated successfully; run on sm_100 to execute kernels.");
        return verify_ptx_only();
    }

    let module = match kernels::load(&ctx) {
        Ok(m) => m,
        Err(e) => {
            // This GPU passed the capability gate above, so the module must
            // load. CUDA_ERROR_INVALID_PTX here means the driver rejected
            // the generated PTX itself: a compiler bug, never a "wrong GPU"
            // situation. Fail loudly instead of degrading to the PTX-only
            // verification path.
            let driver_status = match &e {
                cuda_host::EmbeddedModuleError::Driver(driver) => Some(driver.0),
                _ => None,
            };
            println!("\n❌ embedded module load failed: {e:?} (driver status = {driver_status:?})");
            if driver_status == Some(sys::cudaError_enum_CUDA_ERROR_INVALID_PTX) {
                return Err(format!(
                    "driver rejected the generated PTX as invalid on sm_{major}{minor}, \
                     which should execute it (CUDA_ERROR_INVALID_PTX): {e:?}"
                )
                .into());
            }
            return Err(e.into());
        }
    };
    println!("✓ PTX loaded successfully\n");

    run_tcgen05_fence_test(&stream, &module)?;
    run_tcgen05_alloc_test(&stream, &module)?;
    run_tcgen05_commit_test(&stream, &module)?;
    run_tcgen05_mma_minimal_test(&stream, &module)?;

    println!("\n--- CTA Pair (cta_group::2) Tests ---\n");
    run_tcgen05_alloc_cg2_test(&stream, &module)?;
    run_tcgen05_mma_cg2_test(&stream, &module)?;

    println!("\n=== tcgen05 Test Complete ===");
    Ok(())
}

// Keep this execution set in sync with mir-importer's tcgen05 target
// support (and with gemm_sol_final's copy). Other GPU generations may
// inspect and assemble the generated artifact, but must not turn a
// module-load failure into an execution pass.
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tcgen05.ptx");

    if !ptx_path.exists() {
        return Err("PTX file not found".into());
    }

    println!("\n📝 PTX Verification:");
    println!("   PTX file generated at: {}", ptx_path.display());

    // Assemble for the arch the module actually targets (read from the PTX
    // `.target` line). The old hardcoded `-arch=sm_120a` could never succeed
    // against this sm_100a-target module and only printed noise.
    let ptx = std::fs::read_to_string(&ptx_path)?;
    let arch = ptx
        .lines()
        .find_map(|line| line.strip_prefix(".target "))
        .map(|rest| rest.split([',', ' ']).next().unwrap_or(rest).to_string())
        .unwrap_or_else(|| "sm_100a".to_string());

    let ptxas_result = std::process::Command::new("ptxas")
        .arg(format!("-arch={arch}"))
        .arg(&ptx_path)
        .arg("-o")
        .arg("/dev/null")
        .output();

    match ptxas_result {
        Ok(output) if output.status.success() => {
            println!("   ✓ PTX validated by ptxas ({arch})");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            // A ptxas that predates this arch (or this PTX ISA) cannot
            // judge the module; only a real assembly rejection is fatal.
            if stderr.contains("is not defined for option 'gpu-name'")
                || stderr.contains("Unsupported .version")
            {
                println!("   ℹ️  installed ptxas is too old for {arch} - cannot validate PTX");
            } else {
                return Err(
                    format!("ptxas -arch={arch} rejected the generated PTX:\n{stderr}").into(),
                );
            }
        }
        Err(_) => {
            println!("   ℹ️  ptxas not found - cannot validate PTX");
        }
    }

    println!("\n📝 To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());
    println!("\n   Look for: tcgen05.fence, tcgen05.alloc, tcgen05.mma.ws instructions");

    Ok(())
}

fn run_tcgen05_fence_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 Fence Primitives ---\n");

    let mut output = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 256,
    };

    println!("Launching tcgen05_fence_test kernel...");
    unsafe { module.tcgen05_fence_test((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("SMEM descriptor: 0x{:016x}", host_output[0]);
    println!("✓ Fence primitives executed successfully\n");

    Ok(())
}

fn run_tcgen05_alloc_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 TMEM Allocation ---\n");

    let mut output = DeviceBuffer::<u32>::zeroed(stream, 1)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 16,
    };

    println!("Launching tcgen05_alloc_test kernel...");
    unsafe { module.tcgen05_alloc_test((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("TMEM address: 0x{:08x}", host_output[0]);

    if host_output[0] == 0xDEADBEEF {
        println!("❌ TMEM address unchanged (tcgen05.alloc did NOT write to shared memory)\n");
    } else {
        println!(
            "✓ TMEM allocation successful! Address: 0x{:08x}\n",
            host_output[0]
        );
    }

    Ok(())
}

fn run_tcgen05_commit_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 Commit with mbarrier ---\n");

    let mut output = DeviceBuffer::<u64>::zeroed(stream, 1)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 256 + 64,
    };

    println!("Launching tcgen05_commit_test kernel...");
    unsafe { module.tcgen05_commit_test((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("SMEM descriptor: 0x{:016x}", host_output[0]);
    println!("✓ Commit with mbarrier executed successfully\n");

    Ok(())
}

fn run_tcgen05_mma_minimal_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 MMA Minimal ---\n");

    let mut output = DeviceBuffer::<u32>::zeroed(stream, 3)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 8192 + 256,
    };

    println!("Launching tcgen05_mma_minimal kernel...");
    unsafe { module.tcgen05_mma_minimal((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("TMEM address: 0x{:08x}", host_output[0]);

    let d0_bits = host_output[1];
    let d0_f32 = f32::from_bits(d0_bits);
    println!("D[0] sample: 0x{:08x} ({:.6e} as f32)", d0_bits, d0_f32);
    println!("  (Expected: garbage since SMEM_A and SMEM_B are uninitialized)");
    println!("✓ MMA minimal test executed successfully\n");

    Ok(())
}

fn run_tcgen05_alloc_cg2_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 TMEM Alloc cta_group::2 ---\n");

    let mut output = DeviceBuffer::<u32>::zeroed(stream, 2)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (2, 1, 1),
        shared_mem_bytes: 16,
    };

    println!("Launching tcgen05_alloc_cg2_test (cluster=2x1x1)...");
    unsafe { module.tcgen05_alloc_cg2_test((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("  CTA rank 0 TMEM addr: 0x{:08x}", host_output[0]);
    println!("  CTA rank 1 TMEM addr: 0x{:08x}", host_output[1]);

    let ok0 = host_output[0] != 0xDEADBEEF;
    let ok1 = host_output[1] != 0xDEADBEEF;
    if ok0 && ok1 {
        println!("✓ CTA pair alloc_cg2 successful (TMEM addr 0 is valid base)\n");
    } else {
        println!("❌ CTA pair alloc_cg2 failed\n");
    }

    Ok(())
}

fn run_tcgen05_mma_cg2_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: tcgen05 MMA cta_group::2 ---\n");

    let mut output = DeviceBuffer::<u32>::zeroed(stream, 2)?;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (2, 1, 1),
        shared_mem_bytes: 8192 + 256,
    };

    println!("Launching tcgen05_mma_cg2_test (cluster=2x1x1)...");
    unsafe { module.tcgen05_mma_cg2_test((stream).as_ref(), cfg, &mut output) }?;

    stream.synchronize()?;

    let host_output = output.to_host_vec(stream)?;
    println!("  CTA rank 0 TMEM addr: 0x{:08x}", host_output[0]);
    println!("  CTA rank 1 TMEM addr: 0x{:08x}", host_output[1]);

    let ok0 = host_output[0] != 0xDEADBEEF;
    let ok1 = host_output[1] != 0xDEADBEEF;
    if ok0 && ok1 {
        println!("✓ CTA pair MMA cg2 successful\n");
    } else {
        println!("❌ CTA pair MMA cg2 failed\n");
    }

    Ok(())
}

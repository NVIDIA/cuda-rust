/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe Stage 6 Blackwell (`sm_100a`) `128 x 128 x 16` Tensor Core matrix
//! multiply (`C = A * B^T`) combining 2D TMA loads, `mbarrier` token chains, PTX
//! `generic -> async` proxy coherence, `tcgen05.mma`, and Tensor Memory (`TMEM`)
//! with zero `unsafe` blocks in the kernel body.
//!
//! ```text
//!   STAGE 6 BLACKWELL `tcgen05` + `TMEM` PIPELINE (`tcgen05_matmul_verified`)
//!   ─────────────────────────────────────────────────────────────────────────────
//!   • Phase 0 (Init, `tid == 0`):
//!       `mbarrier_init` on `TMA_A_BAR`, `TMA_B_BAR`, `MMA_BAR`
//!       + `fence_proxy_async_shared_cta()` + `thread::sync_threads()`.
//!   • Phase 1 (`TMEM` Alloc, `warp_id == 0`):
//!       `tcgen05_alloc(&raw mut TMEM_ADDR[0], 128)` + `thread::sync_threads()`,
//!       then all 128 threads read `let tmem_addr = TMEM_ADDR[0];` (bounds check erased!).
//!   • Phase 2 (2D TMA Global -> Shared, `tid == 0`):
//!       `mbarrier_arrive_expect_tx` (`4,096` bytes each) + `cp_async_bulk_tensor_2d_g2s_cta`
//!       into `SMEM_A` and `SMEM_B`, awaited by `mbarrier_try_wait_parity`.
//!   • Phase 3 (Blackwell Tensor Core MMA, `tid == 0`):
//!       `tcgen05_mma_f16(tmem_addr, a_desc, b_desc, IDESC, false)`
//!       + `tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64)`,
//!       awaited by all 128 threads via `mbarrier_try_wait_parity(&raw const MMA_BAR, 0)`.
//!   • Phase 4 (Warp-Partitioned `TMEM` Epilogue, Warps `0..3`):
//!       Each warp `w` (`0..4`) reads its own 32-lane slice `(w * 32) << 16` via
//!       `tcgen05_ld_32x32b_x1_raw` + `tcgen05_load_wait()` and writes `C[tid, 0..128]`
//!       through `DisjointSlice<f32, LinearTiles<128>>`.
//!   • Phase 5 (`TMEM` Dealloc, `warp_id == 0`):
//!       `thread::sync_threads()` + `tcgen05_dealloc(tmem_addr, 128)` on every exit path.
//! ```

use std::mem::MaybeUninit;

use cuda_core::{
    CudaContext, DeviceBuffer, LaunchConfig1D,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::{
    DisjointSlice, LinearTiles, LocalIndex32, SharedArray,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
        mbarrier_try_wait_parity,
    },
    cuda_module, kernel, launch_bounds, launch_contract,
    shared::cvta_generic_to_shared_offset,
    tcgen05::{
        Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
        Tcgen05SmemDescriptor, Tcgen05SwizzleMode, tcgen05_alloc, tcgen05_commit_shared_cluster,
        tcgen05_dealloc, tcgen05_fence_after_thread_sync, tcgen05_ld_32x32b_x1_raw,
        tcgen05_load_wait, tcgen05_mma_f16,
    },
    thread,
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_cta},
    verify, warp,
};
use cuda_host::tiling::to_k_major_f16;
use half::f16;

/// Number of threads per 1D CTA (`128` = `4` warps, matching the `128` `TMEM` lanes).
/// Each thread `tid` (`0..128`) owns one output row of `C[128, 128]`.
const BLOCK_SIZE: u32 = 128;

/// Output rows `M` of matrices `A` (`[M, K]`) and `C` (`[M, N]`).
/// Matches the `M128_N128` `tcgen05.mma` tile height (`128` lanes).
const TILE_M: usize = 128;

/// Output columns `N` of matrices `B` (`[N, K]` transposed) and `C` (`[M, N]`).
/// Matches the `128`-column `TMEM` allocation and `LinearTiles<128>` output view.
const TILE_N: usize = 128;

/// Reduction width `K` of one dense FP16 `tcgen05.mma` step (`16` `f16` elements = `32` bytes).
/// Both `A` and `B` are stored in K-major pre-tiled layout on the host.
const TILE_K: usize = 16;

/// Byte capacity of one `128 x 16` `f16` operand tile in shared memory (`4,096` bytes).
/// Passed to `mbarrier_arrive_expect_tx` and checked against `SMEM_A` / `SMEM_B` in Lean 4.
const OPERAND_TILE_BYTES: usize = TILE_M * TILE_K * 2;

/// Number of `TMEM` columns dynamically allocated and freed by Warp 0 (`128` columns).
/// Verified by `TmemValidCols 128`, `TmemLifecycle 128 128 true`, and `NoOobTmemAccess`.
const TMEM_COLS: u32 = 128;

/// Compile-time `tcgen05` instruction descriptor for dense `128 x 128` FP16 -> FP32 MMA.
/// Decoded by `lean-exporter` to verify `128` lanes and `128` `TMEM` columns in Lean 4.
const MMA_IDESC: u32 = Tcgen05InstructionDescriptor::builder()
    .shape(Tcgen05MmaShape::M128_N128)
    .element_type(Tcgen05ElementType::F16)
    .accumulator_type(Tcgen05AccumulatorType::F32)
    .build()
    .raw();

/// Device kernel module containing the Lean-verified Stage 6 `tcgen05` + `TMEM` GEMM kernel.
#[cuda_module]
mod kernels {
    use super::*;

    /// Compute `C[128, 128] = A[128, 16] * B[128, 16]^T` on Blackwell `tcgen05` Tensor Cores
    /// with zero `unsafe` blocks in the kernel body.
    ///
    /// Lean 4 + `lean-exporter` verify at compile time:
    /// - PTX `generic -> async` proxy coherence (`fence_proxy_async_shared_cta()`) and
    ///   `mbarrier_init` -> `sync_threads()` ordering on `TMA_A_BAR`, `TMA_B_BAR`, and `MMA_BAR`.
    /// - `no_oob_idx` and `no_data_race` on `SMEM_A`, `SMEM_B`, and `TMEM_ADDR` (erasing the
    ///   `TMEM_ADDR[0]` runtime bounds check).
    /// - `tmem_lifecycle`: warp-collective `tcgen05_alloc` / `tcgen05_dealloc` symmetry (`128` cols),
    ///   `tcgen05_commit` -> `mbarrier_try_wait_parity` -> `tcgen05_ld` -> `tcgen05_load_wait`
    ///   token ordering, `TMEM` column bounds, and warpgroup 32-lane slice bounds.
    #[kernel(launch_context = lc)]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (128, 1, 1),
    )]
    #[verify(
        no_oob_idx(SMEM_A, SMEM_B, TMEM_ADDR),
        no_data_race(SMEM_A, SMEM_B, TMEM_ADDR),
        uniform_barriers,
        tmem_lifecycle
    )]
    pub fn tcgen05_matmul_verified(
        #[grid_constant] a_tmap: &TmaDescriptor,
        #[grid_constant] b_tmap: &TmaDescriptor,
        mut out: DisjointSlice<f32, LinearTiles<TILE_N>>,
        tile_x: i32,
        tile_y: i32,
    ) {
        /// 128-byte aligned `4,096`-byte shared-memory tile holding K-major `A[128, 16]` (`f16`).
        /// Populated by 2D TMA on `TMA_A_BAR` and read asynchronously by `tcgen05_mma_f16`.
        static mut SMEM_A: SharedArray<u8, OPERAND_TILE_BYTES, 128> = SharedArray::UNINIT;

        /// 128-byte aligned `4,096`-byte shared-memory tile holding K-major `B[128, 16]` (`f16`).
        /// Populated by 2D TMA on `TMA_B_BAR` and read asynchronously by `tcgen05_mma_f16`.
        static mut SMEM_B: SharedArray<u8, OPERAND_TILE_BYTES, 128> = SharedArray::UNINIT;

        /// Single `u32` shared-memory slot receiving the dynamic `TMEM` base column address.
        /// Written by Warp 0 via `tcgen05_alloc` and read by all 128 threads after `sync_threads()`.
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

        /// Hardware `mbarrier` tracking completion of the `4,096`-byte `A` tile TMA copy.
        static mut TMA_A_BAR: Barrier = Barrier::UNINIT;

        /// Hardware `mbarrier` tracking completion of the `4,096`-byte `B` tile TMA copy.
        static mut TMA_B_BAR: Barrier = Barrier::UNINIT;

        /// Hardware `mbarrier` signaled by `tcgen05_commit_shared_cluster` when `tcgen05_mma_f16` completes.
        static mut MMA_BAR: Barrier = Barrier::UNINIT;

        let idx32 = thread::index_1d_u32(lc);
        let tid = thread::threadIdx_x() as usize;
        let warp_id = warp::warp_id();

        // Phase 0: Initialize all three mbarriers and flush the generic -> async proxy bridge.
        if tid == 0 {
            mbarrier_init(&raw mut TMA_A_BAR, 1);
            mbarrier_init(&raw mut TMA_B_BAR, 1);
            mbarrier_init(&raw mut MMA_BAR, 1);
            fence_proxy_async_shared_cta();
        }
        thread::sync_threads();

        // Phase 1: Warp 0 collectively allocates 128 TMEM columns; all warps sync before reading.
        if warp_id == 0 {
            tcgen05_alloc(&raw mut TMEM_ADDR[0], TMEM_COLS);
        }
        thread::sync_threads();
        let tmem_addr = TMEM_ADDR[0];

        // Phase 2: Thread 0 issues 2D TMA loads of `A` and `B` into `SMEM_A` and `SMEM_B`.
        if tid == 0 {
            let _ = mbarrier_arrive_expect_tx(&raw const TMA_A_BAR, 1, OPERAND_TILE_BYTES as u32);
            cp_async_bulk_tensor_2d_g2s_cta(
                &raw mut SMEM_A as *mut u8,
                a_tmap,
                tile_x,
                tile_y,
                &raw mut TMA_A_BAR,
            );
            let _ = mbarrier_arrive_expect_tx(&raw const TMA_B_BAR, 1, OPERAND_TILE_BYTES as u32);
            cp_async_bulk_tensor_2d_g2s_cta(
                &raw mut SMEM_B as *mut u8,
                b_tmap,
                tile_x,
                tile_y,
                &raw mut TMA_B_BAR,
            );
        }

        while !mbarrier_try_wait_parity(&raw const TMA_A_BAR, 0) {}
        while !mbarrier_try_wait_parity(&raw const TMA_B_BAR, 0) {}
        tcgen05_fence_after_thread_sync();

        // Phase 3: Thread 0 issues single-thread Blackwell `tcgen05.mma` and commits to `MMA_BAR`.
        if tid == 0 {
            let smem_a_off = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
            let smem_b_off = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);
            let a_desc = Tcgen05SmemDescriptor::for_k_major(
                smem_a_off,
                TILE_M,
                TILE_K,
                2,
                Tcgen05SwizzleMode::None,
            )
            .raw();
            let b_desc = Tcgen05SmemDescriptor::for_k_major(
                smem_b_off,
                TILE_N,
                TILE_K,
                2,
                Tcgen05SwizzleMode::None,
            )
            .raw();

            tcgen05_mma_f16(tmem_addr, a_desc, b_desc, MMA_IDESC, false);
            tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
        }

        while !mbarrier_try_wait_parity(&raw const MMA_BAR, 0) {}
        tcgen05_fence_after_thread_sync();

        // Phase 4: Each warp `0..4` loads its own 32-lane warpgroup slice `(warp_id * 32) << 16`
        // across columns `0..128` and stores its disjoint row into `out`.
        let tmem_row_base = tmem_addr + ((warp_id * 32) << 16);
        let mut out_row = out.tile_thread32(idx32);
        let mut col = 0usize;
        while col < TILE_N {
            let bits = tcgen05_ld_32x32b_x1_raw(tmem_row_base + col as u32);
            tcgen05_load_wait();
            if let Some(ref mut row_view) = out_row
                && let Some(local_col) = LocalIndex32::<TILE_N>::new(col as u32)
            {
                row_view.at(local_col).write(f32::from_bits(bits));
            }
            col += 1;
        }

        // Phase 5: Synchronize all warps before Warp 0 frees all 128 TMEM columns.
        thread::sync_threads();
        if warp_id == 0 {
            tcgen05_dealloc(tmem_addr, TMEM_COLS);
        }
    }
}

/// Return true if `(major, minor)` supports datacenter Blackwell `tcgen05` execution.
/// Consumer Blackwell (`sm_120`) and pre-Blackwell GPUs verify Lean proofs + PTX only.
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

/// Encode a 2D tiled `TmaDescriptor` (`CUtensorMap`) for a `[height, width]` `f16` matrix.
/// Uses contiguous unswizzled K-major layout matching `Tcgen05SwizzleMode::None`.
fn create_tma_descriptor_f16(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * std::mem::size_of::<u16>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if result != cuda_sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {result:?}").into());
    }
    Ok(unsafe { tensor_map.as_ptr().cast::<TmaDescriptor>().read() })
}

/// Deterministic exact-integer test value for `A[row, k]` so every output cell is distinct.
fn a_test_val(row: usize, k: usize) -> f32 {
    ((row % 7) + k + 1) as f32
}

/// Deterministic exact-integer test value for `B[col, k]` (`B` stored transposed as `[N, K]`).
fn b_test_val(col: usize, k: usize) -> f32 {
    ((col % 5) + 2 * k + 1) as f32
}

/// Entry point: loads the Lean-verified `tcgen05` module on a datacenter Blackwell GPU
/// (`sm_100a`) and checks all `16,384` FP32 outputs bit-for-bit against the CPU reference.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== Provably Safe Stage 6 Blackwell tcgen05 & TMEM Matmul (`safe_tcgen05_matmul`) ===\n"
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{major}{minor}");

    if !can_execute_tcgen05_ptx(major, minor) {
        println!(
            "Skipping hardware execution: `tcgen05` + `TMEM` require datacenter Blackwell (`sm_100a`); Lean 4 proofs and PTX verified at compile time."
        );
        return Ok(());
    }

    // SAFETY:Loads the embedded module compiled for this binary.
    let module = unsafe { kernels::load(&ctx) }?;

    let mut host_a_rm = Vec::with_capacity(TILE_M * TILE_K);
    for r in 0..TILE_M {
        for k in 0..TILE_K {
            host_a_rm.push(f16::from_f32(a_test_val(r, k)));
        }
    }
    let mut host_b_rm = Vec::with_capacity(TILE_N * TILE_K);
    for c in 0..TILE_N {
        for k in 0..TILE_K {
            host_b_rm.push(f16::from_f32(b_test_val(c, k)));
        }
    }

    let mut host_a_tiled = vec![f16::ZERO; TILE_M * TILE_K];
    let mut host_b_tiled = vec![f16::ZERO; TILE_N * TILE_K];
    to_k_major_f16(&host_a_rm, &mut host_a_tiled, TILE_M, TILE_K);
    to_k_major_f16(&host_b_rm, &mut host_b_tiled, TILE_N, TILE_K);

    let host_a_bits: Vec<u16> = host_a_tiled.iter().map(|x| x.to_bits()).collect();
    let host_b_bits: Vec<u16> = host_b_tiled.iter().map(|x| x.to_bits()).collect();

    let d_a = DeviceBuffer::from_host(&stream, &host_a_bits)?;
    let d_b = DeviceBuffer::from_host(&stream, &host_b_bits)?;
    let mut d_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; TILE_M * TILE_N])?;

    let a_tmap = create_tma_descriptor_f16(
        d_a.cu_deviceptr() as *mut std::ffi::c_void,
        TILE_K as u64,
        TILE_M as u64,
        TILE_K as u32,
        TILE_M as u32,
    )?;
    let b_tmap = create_tma_descriptor_f16(
        d_b.cu_deviceptr() as *mut std::ffi::c_void,
        TILE_K as u64,
        TILE_N as u64,
        TILE_K as u32,
        TILE_N as u32,
    )?;

    let config = LaunchConfig1D::new(1, BLOCK_SIZE, 0);
    let prep = module.prepare_tcgen05_matmul_verified(config)?;
    // SAFETY: `a_tmap` and `b_tmap` encode live `128 x 16` `f16` `d_a` and `d_b` device allocations.
    unsafe {
        module.tcgen05_matmul_verified(&stream, &prep, a_tmap, b_tmap, &mut d_out, 0, 0)?;
    }
    stream.synchronize()?;

    let got = d_out.to_host_vec(&stream)?;
    for r in 0..TILE_M {
        for c in 0..TILE_N {
            let mut expected = 0.0f32;
            for k in 0..TILE_K {
                expected += a_test_val(r, k) * b_test_val(c, k);
            }
            let actual = got[r * TILE_N + c];
            assert!(
                (actual - expected).abs() < 1e-4,
                "mismatch at C[{r}, {c}]: got {actual}, expected {expected}"
            );
        }
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract:   domain = 1, block = (128, 1, 1)");
    println!(
        "  • Proved in Lean 4:  no_oob_idx(SMEM_A, SMEM_B, TMEM_ADDR), no_data_race(SMEM_A, SMEM_B, TMEM_ADDR),"
    );
    println!(
        "                       uniform_barriers, tmem_lifecycle (TmemLifecycle 128 128 true + NoOobTmemAccess)"
    );
    println!(
        "  • Proxy & mbarrier:  generic -> async proxy fence + mbarrier_init -> sync_threads() verified"
    );
    println!(
        "  • Verified on GPU:   all 16,384 FP32 elements of 128x128x16 tcgen05 matmul matched (`0` unsafe blocks in kernel!)"
    );
    Ok(())
}

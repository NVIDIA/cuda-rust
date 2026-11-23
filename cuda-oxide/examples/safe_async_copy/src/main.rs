/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe 1-stage asynchronous TMA bulk copy (`async_copy_stage`)
//! synchronized via `mbarrier_arrive_expect_tx` and `mbarrier_try_wait_parity`.
//!
//! ```text
//!   STAGE 5 TOKEN-ORDERED ASYNC TMA COPY (`async_copy_stage`)
//!   ─────────────────────────────────────────────────────────────────────────────
//!   • Init (`if tid == 0`):
//!       `mbarrier_init(&raw mut BAR, 1)` + `fence_proxy_async_shared_cta()`
//!       followed by `thread::sync_threads()`.
//!   • Producer (`if tid == 0`):
//!       `mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)`
//!       `cp_async_bulk_tensor_2d_g2s_cta(&raw mut TILE as *mut u8, tmap, 0, 0, &raw mut BAR)`
//!       ──► Mints token `StageProduced(TILE, BAR)` in Phase 1 (`no_data_race_single_leader`).
//!   • Consumer Wait (all 256 threads):
//!       `while !mbarrier_try_wait_parity(&raw const BAR, 0) {}`
//!       ──► Redeems `StageProduced(TILE, BAR)` and unlocks `StageReady(BAR)` (Phase 2).
//!   • Consumer Read (all 256 threads):
//!       `let val = TILE[tid];`
//!       ──► Proved in-bounds (`no_oob_tile_1d_slot`) and separated from the TMA
//!           write by `StageReady(BAR)` (`no_data_race_diff_phase`), erasing the
//!           `TILE[tid]` bounds check with zero `unsafe` blocks in the kernel!
//! ```

use std::mem::MaybeUninit;

use cuda_core::{
    CudaContext, DeviceBuffer, LaunchConfig1D,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::{
    DisjointSlice, SharedArray,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
        mbarrier_try_wait_parity,
    },
    cuda_module, kernel, launch_bounds, launch_contract, thread,
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_cta},
    verify,
};

/// Threads per 1D block (`256`).
const BLOCK_SIZE: u32 = 256;
/// Tile width in `f32` elements (`64`).
const TILE_WIDTH: u32 = 64;
/// Tile height in `f32` elements (`64`).
const TILE_HEIGHT: u32 = 64;
/// Total elements in the 64x64 shared tile (`4,096` floats = `16 KB`).
const TILE_SIZE: usize = (TILE_WIDTH * TILE_HEIGHT) as usize;
/// Byte count of one 64x64 `f32` TMA transfer (`16,384` bytes).
const TILE_BYTES: u32 = (TILE_SIZE * 4) as u32;

/// Device kernel module containing the provably safe 1-stage async TMA copy kernel.
#[cuda_module]
mod kernels {
    use super::*;

    /// 1-stage TMA bulk global-to-shared copy (`cp_async_bulk_tensor_2d_g2s_cta`)
    /// synchronized with `mbarrier_arrive_expect_tx` and `mbarrier_try_wait_parity`:
    /// - Thread 0 initializes `BAR` with arrival count `1` and issues the 16 KB
    ///   2D TMA transfer into `TILE` plus `mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)`.
    /// - All 256 threads wait on `while !mbarrier_try_wait_parity(&raw const BAR, 0) {}`
    ///   before reading `TILE[tid]` and writing `out`.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (256, 1, 1),
    )]
    #[verify(no_oob_idx(TILE), no_data_race(TILE), uniform_barriers)]
    pub fn async_copy_stage(
        #[grid_constant] tensor_map: &TmaDescriptor,
        mut out: DisjointSlice<f32>,
        tile_x: i32,
        tile_y: i32,
    ) {
        /// 128-byte aligned 64x64 shared-memory destination tile (`4,096` floats).
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        /// 64-bit hardware `mbarrier` tracking producer arrival and 16 KB TMA completion.
        static mut BAR: Barrier = Barrier::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let tid = thread::threadIdx_x() as usize;

        if tid == 0 {
            mbarrier_init(&raw mut BAR, 1);
            fence_proxy_async_shared_cta();
        }
        thread::sync_threads();

        // Phase 1 (Producer): Thread 0 arms `BAR` for 16,384 bytes and launches TMA into `TILE`.
        if tid == 0 {
            let _ = mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES);
            cp_async_bulk_tensor_2d_g2s_cta(
                &raw mut TILE as *mut u8,
                tensor_map,
                tile_x,
                tile_y,
                &raw mut BAR,
            );
        }

        // Token handoff (`StageProduced(TILE, BAR)` -> `StageReady(BAR)`):
        while !mbarrier_try_wait_parity(&raw const BAR, 0) {}

        // Phase 2 (Consumer): `StageReady(BAR)` dominates this read (`0` unsafe blocks!).
        let val = TILE[tid];
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = val;
        }
    }
}

/// Build a 2D tiled `TmaDescriptor` (`CUtensorMap`) for a `[height, width]` `f32` device tensor.
fn create_tma_descriptor(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== Provably Safe Stage 5 Async TMA Copy & mbarrier Token Ordering (`safe_async_copy`) ===\n"
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;

    let host_tensor: Vec<f32> = (0..TILE_SIZE).map(|i| (i + 100) as f32).collect();
    let d_tensor = DeviceBuffer::from_host(&stream, &host_tensor)?;
    let mut d_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; BLOCK_SIZE as usize])?;

    let tensor_map = create_tma_descriptor(
        d_tensor.cu_deviceptr() as *mut std::ffi::c_void,
        u64::from(TILE_WIDTH),
        u64::from(TILE_HEIGHT),
        TILE_WIDTH,
        TILE_HEIGHT,
    )?;

    let config = LaunchConfig1D::new(1, BLOCK_SIZE, 0);
    let prep = module.prepare_async_copy_stage(config)?;
    unsafe {
        module.async_copy_stage(&stream, &prep, tensor_map, &mut d_out, 0, 0)?;
    }
    stream.synchronize()?;

    let out = d_out.to_host_vec(&stream)?;
    for (i, &got) in out.iter().enumerate() {
        let expected = (i + 100) as f32;
        assert!(
            (got - expected).abs() < 1e-4,
            "async_copy_stage mismatch at {i}: got {got}, expected {expected}"
        );
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract:   domain = 1, block = (256, 1, 1)");
    println!(
        "  • async_copy_stage:  Proved no_oob_idx(TILE), no_data_race(TILE), uniform_barriers"
    );
    println!(
        "                       Phase 1 (Producer `if tid == 0`): `mbarrier_arrive_expect_tx(&BAR, 16384)`"
    );
    println!(
        "                         + `cp_async_bulk_tensor_2d_g2s_cta(&TILE, ..., &BAR)` -> `StageProduced(TILE, BAR)`"
    );
    println!(
        "                       Wait (`while !mbarrier_try_wait_parity(&BAR, 0)`) -> unlocks `StageReady(BAR)`"
    );
    println!(
        "                       Phase 2 (Consumer): `TILE[tid]` proved in-bounds (`no_oob_tile_1d_slot`)"
    );
    println!("                         and barrier-ordered (`no_data_race_diff_phase`)");
    println!(
        "                       Elided: 1 SharedArray `mir.assert` bounds check (`0` unsafe blocks in kernel!)"
    );
    println!(
        "\nVerified 1-stage async TMA copy and mbarrier token handoff across all {BLOCK_SIZE} threads!"
    );
    Ok(())
}

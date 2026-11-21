/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe Stage 8 warp-specialized 4-stage producer/consumer ring
//! (`warp_ring_stage`) using `SharedArray<Barrier, 4, 8>` and zero `unsafe` blocks
//! inside the kernel body.
//!
//! ```text
//!   STAGE 8 WARP-SPECIALIZED 4-STAGE RING (`warp_ring_stage`, 160 threads = 5 warps)
//!   ---------------------------------------------------------------------------------
//!   • Init (`if tid == 0`):
//!       Counted loop `while s < 4` initializes `FULL_BAR[s]`, `EMPTY_BAR[s]` (count 4),
//!       and `META_BAR[s]`, flushes `fence_proxy_async_shared_cta()`, then `sync_threads()`.
//!   • Warp 4 (`TMA Producer`, `lane_id == 0`):
//!       Streams `12` tiles (`3` full ring laps) through `stage in 0..4`:
//!         1. Waits on `EMPTY_BAR[stage]` (`empty_parity` starts at `1` on lap 0)
//!         2. Publishes tile scale into `TILE_META[stage]` + `mbarrier_arrive(&META_BAR[stage])`
//!         3. Arms `FULL_BAR[stage]` (`4,096` bytes) + `cp_async_bulk_tensor_2d_g2s_cta`
//!            into `RING_BUF[stage * 1024 .. +1024]`
//!         4. Wraps `stage += 1; if stage == 4 { stage = 0; empty_parity ^= 1; }`
//!   • Warps 0..3 (`4 Consumer Warps`, `128` threads):
//!       For each of the `12` tiles across `c_stage in 0..4`:
//!         1. Waits on `META_BAR[c_stage]` and reads `TILE_META[c_stage]`
//!         2. Waits on `FULL_BAR[c_stage]` and accumulates `8` strided elements per thread
//!            from `RING_BUF[c_stage * 1024 + k * 128 + tid]`
//!         3. `warp::sync_mask` + `if lane_id == 0 { mbarrier_arrive(&EMPTY_BAR[c_stage]); }`
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
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
        mbarrier_init, mbarrier_try_wait_parity,
    },
    cuda_module, kernel, launch_bounds, launch_contract, thread,
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_cta},
    verify, warp,
};

/// Total threads per 1D CTA (`160` = `4` consumer warps + `1` producer warp).
const BLOCK_SIZE: u32 = 160;
/// Number of 32-lane consumer warps (`Warps 0..3` = `128` consumer threads).
const CONSUMER_WARPS: u32 = 4;
/// Dedicated TMA + metadata producer warp index (`Warp 4`, threads `128..159`).
const PRODUCER_WARP: u32 = 4;
/// Number of consumer threads (`128`).
const CONSUMER_THREADS: usize = (CONSUMER_WARPS * 32) as usize;
/// Number of pipeline ring stages (`4`).
const NUM_STAGES: usize = 4;
/// Width of each 2D TMA tile in `f32` elements (`32`).
const TILE_WIDTH: u32 = 32;
/// Height of each 2D TMA tile in `f32` elements (`32`).
const TILE_HEIGHT: u32 = 32;
/// Number of `f32` elements in a single stage (`32 * 32 = 1,024` floats).
const STAGE_ELEMS: usize = (TILE_WIDTH * TILE_HEIGHT) as usize;
/// Total `f32` capacity of the 4-stage shared ring buffer (`4,096` floats = `16 KB`).
const RING_ELEMS: usize = NUM_STAGES * STAGE_ELEMS;
/// Byte size of one `32 x 32` `f32` stage transfer (`4,096` bytes).
const STAGE_BYTES: u32 = (STAGE_ELEMS * 4) as u32;
/// Number of tiles streamed through the 4-stage ring (`12` tiles = `3` full ring laps).
const NUM_TILES: u32 = 12;

/// Device kernel module containing the Lean-verified Stage 8 warp-specialized ring kernel.
#[cuda_module]
mod kernels {
    use super::*;

    /// Stream `num_tiles` 2D TMA tiles from `Warp 4` into `Warps 0..3` across a 4-stage
    /// `SharedArray<Barrier, 4, 8>` circular buffer with zero `unsafe` blocks.
    #[kernel(launch_context = lc)]
    #[launch_bounds(160)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (160, 1, 1),
    )]
    #[verify(
        no_oob_idx(RING_BUF, TILE_META, FULL_BAR, EMPTY_BAR, META_BAR),
        no_data_race(RING_BUF, TILE_META),
        uniform_barriers
    )]
    pub fn warp_ring_stage(
        #[grid_constant] tensor_map: &TmaDescriptor,
        mut out: DisjointSlice<f32>,
    ) {
        /// 128-byte aligned 4-stage circular tile buffer (`4 * 1,024 = 4,096` `f32`s).
        static mut RING_BUF: SharedArray<f32, RING_ELEMS, 128> = SharedArray::UNINIT;
        /// 4-slot per-stage scale mailbox written by `Warp 4` and read by `Warps 0..3`.
        static mut TILE_META: SharedArray<u32, NUM_STAGES, 16> = SharedArray::UNINIT;
        /// Per-stage TMA completion `mbarrier` array (`arrival count = 1`).
        static mut FULL_BAR: SharedArray<Barrier, NUM_STAGES, 8> = SharedArray::UNINIT;
        /// Per-stage consumer release `mbarrier` array (`arrival count = 4` consumer warps).
        static mut EMPTY_BAR: SharedArray<Barrier, NUM_STAGES, 8> = SharedArray::UNINIT;
        /// Per-stage metadata mailbox `mbarrier` array (`arrival count = 1`).
        static mut META_BAR: SharedArray<Barrier, NUM_STAGES, 8> = SharedArray::UNINIT;

        let _ = lc;
        let idx = thread::index_1d();
        let tid_u32 = thread::threadIdx_x();
        let tid = tid_u32 as usize;
        let warp_id = warp::warp_id();
        let lane_id = tid_u32 % 32;

        // Phase 0: Thread 0 initializes all 3 per-stage mbarrier arrays and flushes the proxy bridge.
        if tid == 0 {
            let mut s = 0usize;
            while s < NUM_STAGES {
                mbarrier_init(&raw mut FULL_BAR[s], 1);
                mbarrier_init(&raw mut EMPTY_BAR[s], CONSUMER_WARPS);
                mbarrier_init(&raw mut META_BAR[s], 1);
                s += 1;
            }
            fence_proxy_async_shared_cta();
        }
        thread::sync_threads();

        // Producer role (`Warp 4`, `lane 0`): streams tiles into the 4-stage ring.
        if warp_id == PRODUCER_WARP {
            if lane_id == 0 {
                let mut stage: usize = 0;
                let mut empty_parity: u32 = 1;
                let mut tile_idx: u32 = 0;
                while tile_idx < NUM_TILES {
                    while !mbarrier_try_wait_parity(&raw const EMPTY_BAR[stage], empty_parity) {}

                    TILE_META[stage] = tile_idx + 1;
                    let _ = mbarrier_arrive(&raw const META_BAR[stage]);

                    let _ = mbarrier_arrive_expect_tx(&raw const FULL_BAR[stage], 1, STAGE_BYTES);
                    let stage_base = stage * STAGE_ELEMS;
                    cp_async_bulk_tensor_2d_g2s_cta(
                        (&raw mut RING_BUF[stage_base]).cast::<u8>(),
                        tensor_map,
                        0,
                        (tile_idx * TILE_HEIGHT) as i32,
                        &raw mut FULL_BAR[stage],
                    );

                    stage += 1;
                    if stage == NUM_STAGES {
                        stage = 0;
                        empty_parity ^= 1;
                    }
                    tile_idx += 1;
                }
            }
        }

        // Consumer role (`Warps 0..3`, `128` threads): drains all `num_tiles` tiles from the ring.
        if warp_id < CONSUMER_WARPS {
            let mut c_stage: usize = 0;
            let mut full_parity: u32 = 0;
            let mut acc: f32 = 0.0;
            let mut c_tile: u32 = 0;
            while c_tile < NUM_TILES {
                while !mbarrier_try_wait_parity(&raw const META_BAR[c_stage], full_parity) {}
                let scale = TILE_META[c_stage] as f32;

                while !mbarrier_try_wait_parity(&raw const FULL_BAR[c_stage], full_parity) {}
                let c_base = c_stage * STAGE_ELEMS;
                let mut k: usize = 0;
                while k < 8 {
                    let elem = RING_BUF[c_base + k * CONSUMER_THREADS + tid];
                    acc += elem * scale;
                    k += 1;
                }

                warp::sync_mask(u32::MAX);
                if lane_id == 0 {
                    let _ = mbarrier_arrive(&raw const EMPTY_BAR[c_stage]);
                }

                c_stage += 1;
                if c_stage == NUM_STAGES {
                    c_stage = 0;
                    full_parity ^= 1;
                }
                c_tile += 1;
            }

            if let Some(out_slot) = out.get_mut(idx) {
                *out_slot = acc;
            }
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
    println!("=== Provably Safe Stage 8 Warp-Specialized Ring (`safe_warp_ring`) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx) }?;

    let total_height = TILE_HEIGHT * NUM_TILES;
    let total_elems = (TILE_WIDTH * total_height) as usize;
    let host_src: Vec<f32> = (0..total_elems)
        .map(|idx| {
            let tile = idx / STAGE_ELEMS;
            let in_tile = idx % STAGE_ELEMS;
            let lane = in_tile % CONSUMER_THREADS;
            ((tile + 1) * 10 + (lane % 7)) as f32 * 0.25
        })
        .collect();
    let d_src = DeviceBuffer::from_host(&stream, &host_src)?;
    let mut d_out = DeviceBuffer::from_host(&stream, &vec![0.0f32; CONSUMER_THREADS])?;

    let tensor_map = create_tma_descriptor(
        d_src.cu_deviceptr() as *mut std::ffi::c_void,
        u64::from(TILE_WIDTH),
        u64::from(total_height),
        TILE_WIDTH,
        TILE_HEIGHT,
    )?;

    let config = LaunchConfig1D::new(1, BLOCK_SIZE, 0);
    let prep = module.prepare_warp_ring_stage(config)?;
    unsafe {
        module.warp_ring_stage(&stream, &prep, tensor_map, &mut d_out)?;
    }
    stream.synchronize()?;

    let got_out = d_out.to_host_vec(&stream)?;
    for tid in 0..CONSUMER_THREADS {
        let mut expected = 0.0f32;
        for t in 0..(NUM_TILES as usize) {
            let scale = (t + 1) as f32;
            for k in 0..8 {
                expected += host_src[t * STAGE_ELEMS + k * CONSUMER_THREADS + tid] * scale;
            }
        }
        let got = got_out[tid];
        assert!(
            (got - expected).abs() < 1e-3,
            "warp_ring_stage mismatch at consumer tid {tid}: got {got}, expected {expected}"
        );
    }

    println!("Contract & Lean 4 Proof Summary:");
    println!(
        "  • Launch contract:   domain = 1, block = (160, 1, 1) (4 consumer warps + 1 producer warp)"
    );
    println!("  • Verified statics:  RING_BUF, TILE_META, FULL_BAR[4], EMPTY_BAR[4], META_BAR[4]");
    println!(
        "  • Inductive ring:    Proved `stage < 4` across `if stage == 4 {{ stage = 0 }}` wraparound"
    );
    println!(
        "  • Slot ordering:     Proved `symAsyncSlotGuard` (`stage != c_stage` unless barrier-awaited)"
    );
    println!(
        "  • Kernel safety:     0 `unsafe` blocks in kernel; all SharedArray & Barrier bounds elided!"
    );
    println!(
        "\nVerified {NUM_TILES}-tile (3-lap) warp-specialized 4-stage TMA + Barrier ring across all {CONSUMER_THREADS} consumer threads!"
    );
    Ok(())
}

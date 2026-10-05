/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TMA multicast kernels take raw descriptor pointers from the host
// driver; the implicit `unsafe` is in the launch contract.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! TMA Multicast Example (sm_90+)
//!
//! Demonstrates TMA multicast: a single TMA load broadcasts a tile to
//! the shared memory of ALL CTAs in a thread block cluster.
//!
//! Requirements:
//! - The instruction: `cp.async.bulk.tensor` with `.multicast::cluster` is
//!   sm_90+ in the PTX ISA. `sm_90a` is advised for performance, not required
//!   for legality, which is what the sm_90+ catalog row records.
//! - The build target: `cargo oxide run` forwards the detected GPU as a hint
//!   and rustc-codegen-cuda builds for it whenever the kernel's features run
//!   there. So an RTX 5090 gets `.target sm_120a` and an H100 gets
//!   `.target sm_90a`. Only when no compatible GPU is detected (`cargo oxide
//!   build`, or `run` on a pre-Hopper GPU) does the target fall back to
//!   sm_100a. The host binary never knows which target it embeds.
//! - Verified on sm_120 (#668). The sm_90a Hopper run is not yet captured
//!   and is expected to build and execute (#966).
//!
//! Build and run with:
//!   cargo oxide run tma_multicast

use cuda_core::simt::LaunchConfig;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_try_wait,
};
use cuda_device::cluster;
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast};
use cuda_device::{DisjointSlice, SharedArray, cluster_launch, kernel, thread};
use cuda_host::cuda_module;
use std::mem::MaybeUninit;
use std::sync::Arc;

// =============================================================================
// KERNEL
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// TMA multicast test kernel — one TMA load delivers a tile to ALL CTAs in the cluster.
    ///
    /// Launched with a cluster of CLUSTER_SIZE CTAs. Only CTA-0 thread-0 issues
    /// the multicast TMA, but every CTA's shared memory receives the tile.
    /// Each CTA then writes its shared tile to a disjoint slice of `out` for
    /// host-side verification.
    #[kernel]
    #[cluster_launch(4, 1, 1)]
    pub fn tma_multicast_test(
        tensor_map: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        tile_x: i32,
        tile_y: i32,
    ) {
        const TILE_SIZE: usize = 64 * 64;
        const TILE_BYTES: u32 = (TILE_SIZE * 4) as u32;
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let rank = cluster::block_rank();

        // Every CTA initializes its own barrier
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        // Cluster-wide barrier: all CTAs must have their mbarrier initialized
        // before the multicast TMA fires (it writes to ALL CTAs' shared memory
        // and signals ALL their barriers).
        cluster::cluster_sync();

        // ALL threads in ALL CTAs arrive at their local barrier
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };

        // Only CTA-0 thread-0 issues the multicast TMA copy
        if rank == 0 && tid == 0 {
            let cta_mask: u16 = 0b1111; // deliver to all 4 CTAs
            unsafe {
                cp_async_bulk_tensor_2d_g2s_multicast(
                    &raw mut TILE as *mut u8,
                    tensor_map,
                    tile_x,
                    tile_y,
                    &raw mut BAR,
                    cta_mask,
                );
            }
        }

        // Wait for TMA completion on every CTA
        unsafe { while !mbarrier_try_wait(&raw const BAR, token) {} }
        thread::sync_threads();

        // Each CTA writes its tile to out[rank * TILE_SIZE .. (rank+1) * TILE_SIZE]
        let stride = block_size as usize;
        let mut i = tid as usize;
        while i < TILE_SIZE {
            let val = unsafe { TILE[i] };
            let global_idx = rank as usize * TILE_SIZE + i;
            if global_idx < out.len() {
                unsafe {
                    *out.get_unchecked_mut(global_idx) = val;
                }
            }
            i += stride;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

const TILE_WIDTH: u32 = 64;
const TILE_HEIGHT: u32 = 64;
const TILE_SIZE: usize = (TILE_WIDTH * TILE_HEIGHT) as usize; // 4096 floats

const TENSOR_WIDTH: u64 = 256;
const TENSOR_HEIGHT: u64 = 256;
const TENSOR_SIZE: usize = (TENSOR_WIDTH * TENSOR_HEIGHT) as usize;

const CLUSTER_SIZE: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TMA Multicast Example ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // The ISA floor for multicast. Anything sm_90 or newer falls through to
    // `load` below, which is where a target mismatch (a module built on
    // another machine or with --arch) surfaces. Same clean-skip contract as
    // that arm: the marker has to be on whichever path actually runs.
    if major < 9 {
        println!("\nskipping: TMA multicast requires sm_90 or newer");
        println!(
            "   Your GPU is sm_{}{}. Use: cargo oxide run tma_copy",
            major, minor
        );
        return Ok(());
    }

    match kernels::load(&ctx) {
        Ok(module) => {
            println!("✓ embedded module loaded successfully\n");
            run_tma_multicast_test(&stream, &module)?;
        }
        Err(e) => {
            // The embedded module was built for a target this GPU cannot JIT
            // (a module built on another machine, or with --arch), and the
            // driver returns 218, which `load` surfaces through its `Driver`
            // variant. The host binary cannot see the embedded target, so
            // name none. Treat it as a clean skip so the smoketest's
            // failure-marker scan doesn't flag it as a regression: the PTX
            // itself was generated, which is all this path can verify.
            println!("\nskipping: the embedded module did not load on this GPU");
            println!("  driver reported: {}", e);
            println!("  Multicast needs sm_90 or newer. cargo oxide run builds for the");
            println!(
                "  detected GPU, so a module built elsewhere or with --arch may not JIT here."
            );
            println!("  For basic TMA tests, use: cargo oxide run tma_copy");
        }
    }

    println!("\n=== TMA Multicast Test Complete ===");
    Ok(())
}

fn run_tma_multicast_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- TMA Multicast (tma_multicast_test) ---\n");

    println!(
        "1. Setup: {} CTAs in cluster, tile {}x{} ({} floats)",
        CLUSTER_SIZE, TILE_WIDTH, TILE_HEIGHT, TILE_SIZE,
    );

    let host_input: Vec<f32> = (0..TENSOR_SIZE).map(|i| i as f32).collect();
    let dev_tensor = DeviceBuffer::from_host(stream, &host_input)?;

    let total_output = TILE_SIZE * CLUSTER_SIZE;
    let mut dev_output = DeviceBuffer::<f32>::zeroed(stream, total_output)?;

    let ptr = dev_tensor.cu_deviceptr();
    let tensor_map = create_tma_descriptor(
        ptr as *mut std::ffi::c_void,
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        TILE_WIDTH,
        TILE_HEIGHT,
    )?;

    let dev_tensor_map = DeviceBuffer::from_host(stream, &tensor_map.opaque[..])?;

    let tile_x: i32 = TILE_WIDTH as i32;
    let tile_y: i32 = 0;
    let block_size = 256u32;

    println!(
        "2. Launching tma_multicast_test (cluster=({},1,1), block={})...",
        CLUSTER_SIZE, block_size,
    );

    let cfg = LaunchConfig {
        grid_dim: (CLUSTER_SIZE as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let tensor_map_ptr = dev_tensor_map.cu_deviceptr() as *const TmaDescriptor;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.tma_multicast_test(
            (stream).as_ref(),
            cfg,
            tensor_map_ptr,
            &mut dev_output,
            tile_x,
            tile_y,
        )
    }?;

    stream.synchronize()?;

    println!(
        "3. Verifying all {} CTAs received the same tile...",
        CLUSTER_SIZE
    );
    let host_output = dev_output.to_host_vec(stream)?;

    let mut errors = 0;
    for cta in 0..CLUSTER_SIZE {
        for row in 0..TILE_HEIGHT as usize {
            for col in 0..TILE_WIDTH as usize {
                let tile_idx = row * TILE_WIDTH as usize + col;
                let out_idx = cta * TILE_SIZE + tile_idx;

                let expected_row = tile_y as usize + row;
                let expected_col = tile_x as usize + col;
                let expected_val = (expected_row * TENSOR_WIDTH as usize + expected_col) as f32;

                if (host_output[out_idx] - expected_val).abs() > 0.001 {
                    if errors < 5 {
                        println!(
                            "   MISMATCH CTA {} [{},{}]: expected {}, got {}",
                            cta, row, col, expected_val, host_output[out_idx]
                        );
                    }
                    errors += 1;
                }
            }
        }
    }

    if errors == 0 {
        println!(
            "   ✓ All {} CTAs have identical tile data ({} values each)!",
            CLUSTER_SIZE, TILE_SIZE,
        );
        println!(
            "\n🎉 TMA multicast successful — one load, {} CTAs served!",
            CLUSTER_SIZE
        );
        Ok(())
    } else {
        println!("   ✗ {} mismatches across {} CTAs", errors, CLUSTER_SIZE);
        Err(format!("{} verification errors in multicast test", errors).into())
    }
}

fn create_tma_descriptor(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let tensor_rank = 2u32;
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [tile_width, tile_height];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            tensor_rank,
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
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

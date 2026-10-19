/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

//! TMA Copy Example (SM90+ / Hopper+)
//!
//! Demonstrates TMA (Tensor Memory Accelerator) usage:
//! - cp_async_bulk_tensor_2d_g2s: Async 2D tensor copy global → shared
//! - mbarrier: Barrier-based completion tracking
//! - `#[grid_constant]`: the descriptor is passed by value in the launch packet,
//!   without allocating or uploading a separate descriptor buffer.
//!
//! Note: This example requires Hopper (sm_90) or newer GPUs.
//! For TMA multicast (also sm_90+), see the `tma_multicast` example.
//!
//! Build and run with:
//!   cargo oxide run tma_copy

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
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use std::mem::MaybeUninit;
use std::sync::Arc;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// TMA 2D tile copy test kernel.
    ///
    /// Pattern: ALL threads participate in barrier
    /// - Thread 0: arrive_expect_tx (arrive + set expected bytes) + issue TMA
    /// - Other threads: regular arrive
    /// - ALL threads: wait on barrier
    #[kernel]
    pub fn tma_copy_2d_test(
        #[grid_constant] tensor_map: &TmaDescriptor,
        mut out: DisjointSlice<f32>,
        tile_x: i32,
        tile_y: i32,
    ) {
        const TILE_SIZE: usize = 64 * 64;
        const TILE_BYTES: u32 = (TILE_SIZE * 4) as u32;
        // TMA destinations require 128-byte alignment
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        // Barriers use natural alignment (8 bytes for i64)
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // Thread 0 initializes barrier with BLOCK SIZE (all threads will arrive)
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
                // CRITICAL: Fence to make barrier init visible to TMA async proxy!
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        // Thread 0: issue TMA + arrive with expected bytes
        if tid == 0 {
            unsafe {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE as *mut u8,
                    tensor_map,
                    tile_x,
                    tile_y,
                    &raw mut BAR,
                );
            }
        }

        // ALL threads arrive at barrier
        // Thread 0: arrive with expected TX bytes
        // Other threads: regular arrive
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };

        // ALL threads wait for barrier completion (using try_wait for better scheduling)
        unsafe {
            while !mbarrier_try_wait(&raw const BAR, token) {
                // Hardware may briefly suspend thread while waiting
            }
        }

        // Now TMA is complete, shared memory has the data
        thread::sync_threads();

        // Each thread copies one element to output
        let idx = gid.get();
        if idx < TILE_SIZE {
            let val = unsafe { TILE[idx] };
            if let Some(out_elem) = out.get_mut(gid) {
                *out_elem = val;
            }
        }
    }

    /// Simple TMA pipeline test - ALL threads participate in barrier.
    #[kernel]
    pub fn tma_pipeline_test(
        #[grid_constant] tensor_map: &TmaDescriptor,
        mut out: DisjointSlice<u32>,
    ) {
        const TILE_SIZE: usize = 1024;
        const TILE_BYTES: u32 = (TILE_SIZE * 4) as u32;
        // TMA destinations require 128-byte alignment
        static mut TILE: SharedArray<f32, TILE_SIZE, 128> = SharedArray::UNINIT;
        // Barriers use natural alignment (8 bytes for i64)
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // Thread 0 initializes barrier
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
                // CRITICAL: Fence to make barrier init visible to TMA async proxy!
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        // Thread 0 issues TMA
        if tid == 0 {
            unsafe {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE as *mut u8,
                    tensor_map,
                    0,
                    0,
                    &raw mut BAR,
                );
            }
        }

        // ALL threads arrive
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };

        // ALL threads wait (using try_wait for better scheduling)
        unsafe { while !mbarrier_try_wait(&raw const BAR, token) {} }

        thread::sync_threads();

        // Mark success
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1u32;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

/// Tile dimensions for the TMA copy (must match kernel's SharedArray size)
const TILE_WIDTH: u32 = 64;
const TILE_HEIGHT: u32 = 64;
const TILE_SIZE: usize = (TILE_WIDTH * TILE_HEIGHT) as usize; // 4096 floats

/// Total tensor dimensions
const TENSOR_WIDTH: u64 = 256;
const TENSOR_HEIGHT: u64 = 256;
const TENSOR_SIZE: usize = (TENSOR_WIDTH * TENSOR_HEIGHT) as usize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TMA Copy Example ===\n");

    // Initialize CUDA context
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Check compute capability
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    if major < 9 {
        // PTX generation is all this example can verify below sm_90, so mark
        // the run as a clean skip rather than leaving it to be judged on
        // whatever `verify_ptx_only` happens to print.
        println!("\nskipping: TMA requires sm_90+ (Hopper or newer)");
        println!("   Your GPU is sm_{}{}", major, minor);
        println!("   This example will only verify PTX compilation.\n");
        return verify_ptx_only(&ctx);
    }

    // Load the CUDA module embedded in this binary
    println!("Loading embedded CUDA module");
    let module = kernels::load(&ctx)?;
    println!("✓ Module loaded successfully\n");

    // Run tests
    run_tma_copy_test(&stream, &module)?;
    run_tma_pipeline_test(&stream, &module)?;

    println!("\n=== TMA Copy Test Complete ===");
    Ok(())
}

/// Fallback for GPUs that cannot execute the TMA kernels: inspect the loose
/// PTX build artifact beside this crate. The main path loads the module
/// embedded in the binary instead; only this fallback reads the loose file.
fn verify_ptx_only(ctx: &Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tma_copy.ptx");
    println!("Checking loose PTX artifact: {}", ptx_path.display());

    if !ptx_path.exists() {
        return Err(
            "loose PTX artifact not found (build with `cargo oxide build tma_copy`)".into(),
        );
    }

    let ptx_file = ptx_path.to_str().ok_or("PTX path is not valid UTF-8")?;

    // Just verify it loads (may fail on pre-Hopper due to TMA instructions)
    match ctx.load_module_from_file(ptx_file) {
        Ok(_) => println!("✓ PTX module loaded (surprisingly, on pre-Hopper GPU)"),
        Err(e) => println!("ℹ️  PTX load failed (expected on pre-Hopper): {}", e),
    }

    println!("\n📝 To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());
    println!("\n   Look for: cp.async.bulk.tensor.2d instructions");

    Ok(())
}

fn run_tma_copy_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test 1: TMA Copy (tma_copy_2d_test) ---\n");

    println!(
        "1. Allocating {}x{} tensor ({} floats, {} KB)",
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        TENSOR_SIZE,
        TENSOR_SIZE * 4 / 1024
    );

    let host_input: Vec<f32> = (0..TENSOR_SIZE).map(|i| i as f32).collect();
    let dev_tensor = DeviceBuffer::from_host(stream, &host_input)?;

    let mut dev_output = DeviceBuffer::<f32>::zeroed(stream, TILE_SIZE)?;

    println!(
        "2. Creating TMA descriptor (tile: {}x{})",
        TILE_WIDTH, TILE_HEIGHT
    );
    let ptr = dev_tensor.cu_deviceptr();
    let tensor_map = create_tma_descriptor(
        ptr as *mut std::ffi::c_void,
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        TILE_WIDTH,
        TILE_HEIGHT,
    )?;

    println!("3. Launching tma_copy_2d_test kernel...");

    let tile_x: i32 = 0;
    let tile_y: i32 = 0;
    let block_size = 256u32;
    let grid_size = (TILE_SIZE as u32).div_ceil(block_size);

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.tma_copy_2d_test(
            (stream).as_ref(),
            cfg,
            tensor_map,
            &mut dev_output,
            tile_x,
            tile_y,
        )
    }?;

    stream.synchronize()?;

    println!("4. Verifying results...");
    let host_output = dev_output.to_host_vec(stream)?;

    let mut errors = 0;
    for row in 0..TILE_HEIGHT as usize {
        for col in 0..TILE_WIDTH as usize {
            let tile_idx = row * TILE_WIDTH as usize + col;
            let expected = (tile_y as usize * TILE_HEIGHT as usize + row) * TENSOR_WIDTH as usize
                + (tile_x as usize * TILE_WIDTH as usize + col);
            let expected_val = expected as f32;

            if (host_output[tile_idx] - expected_val).abs() > 0.001 {
                if errors < 5 {
                    println!(
                        "   MISMATCH at [{},{}]: expected {}, got {}",
                        row, col, expected_val, host_output[tile_idx]
                    );
                }
                errors += 1;
            }
        }
    }

    if errors == 0 {
        println!("   ✓ All {} values match!", TILE_SIZE);
        println!("\n🎉 TMA copy successful!");
    } else {
        println!("   ✗ {} mismatches out of {}", errors, TILE_SIZE);
        return Err(format!("{} verification errors", errors).into());
    }

    Ok(())
}

fn run_tma_pipeline_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Test 2: TMA Pipeline (tma_pipeline_test) ---\n");

    const PIPELINE_TILE_WIDTH: u32 = 32;
    const PIPELINE_TILE_HEIGHT: u32 = 32;

    let host_input: Vec<f32> = (0..TENSOR_SIZE).map(|i| i as f32).collect();
    let dev_tensor = DeviceBuffer::from_host(stream, &host_input)?;

    let block_size = 256u32;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, block_size as usize)?;

    let ptr = dev_tensor.cu_deviceptr();
    let tensor_map = create_tma_descriptor(
        ptr as *mut std::ffi::c_void,
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        PIPELINE_TILE_WIDTH,
        PIPELINE_TILE_HEIGHT,
    )?;

    println!(
        "1. Launching tma_pipeline_test kernel (tile: {}x{})...",
        PIPELINE_TILE_WIDTH, PIPELINE_TILE_HEIGHT
    );

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.tma_pipeline_test((stream).as_ref(), cfg, tensor_map, &mut dev_output) }?;

    stream.synchronize()?;

    println!("2. Verifying results...");
    let host_output = dev_output.to_host_vec(stream)?;

    let success_count = host_output.iter().filter(|&&x| x == 1).count();
    if success_count == block_size as usize {
        println!("   ✓ All {} threads completed successfully!", block_size);
        println!("\n🎉 TMA pipeline test successful!");
        Ok(())
    } else {
        println!(
            "   ✗ Only {}/{} threads succeeded",
            success_count, block_size
        );
        Err(format!(
            "Pipeline test failed: {}/{} threads",
            success_count, block_size
        )
        .into())
    }
}

/// Create a TMA tensor map descriptor for a 2D f32 tensor
fn create_tma_descriptor(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
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

    const { assert!(size_of::<TmaDescriptor>() == size_of::<CUtensorMap>()) };
    // SAFETY: the successful driver call initialized all 128 descriptor bytes.
    // Both types carry those same opaque bytes, and CUtensorMap's alignment is
    // at least TmaDescriptor's. The returned Copy value goes into launch storage.
    const { assert!(align_of::<CUtensorMap>() >= align_of::<TmaDescriptor>()) };
    Ok(unsafe { tensor_map.as_ptr().cast::<TmaDescriptor>().read() })
}

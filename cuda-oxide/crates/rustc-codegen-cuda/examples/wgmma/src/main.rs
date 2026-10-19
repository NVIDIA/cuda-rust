/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified WGMMA Example (SM90 / Hopper only)
//!
//! Demonstrates WGMMA (Warpgroup Matrix Multiply-Accumulate) infrastructure:
//! - wgmma_fence()
//! - wgmma_commit_group()
//! - wgmma_wait_group::<N>()
//! - make_smem_desc()
//!
//! Note: WGMMA is Hopper-only (sm_90). It does NOT work on Blackwell (sm_120).
//!
//! Build and run with:
//!   cargo oxide run wgmma

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::shared::SharedArray;
use cuda_device::wgmma::{make_smem_desc, wgmma_commit_group, wgmma_fence, wgmma_wait_group};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel for WGMMA sync primitives (no MMA).
    ///
    /// This tests the basic WGMMA infrastructure:
    /// - make_smem_desc(): Create SMEM descriptor with swizzle
    /// - wgmma_fence(): Ensure prior memory operations complete
    /// - wgmma_commit_group(): Commit current instruction group
    /// - wgmma_wait_group::<0>(): Wait for all groups to complete
    #[kernel]
    pub unsafe fn wgmma_sync_test(mut output: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        unsafe {
            let desc = make_smem_desc(&raw const SMEM as *const u8);

            wgmma_fence();
            wgmma_commit_group();
            wgmma_wait_group::<0>();

            if tid == 0
                && let Some(output_elem) = output.get_mut(gid)
            {
                *output_elem = desc;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified WGMMA Example ===\n");

    // Initialize CUDA context
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Check compute capability
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    if major < 9 {
        println!("\n⚠️  WARNING: WGMMA requires sm_90 (Hopper) or newer!");
        println!("   Your GPU is sm_{}{}", major, minor);
        return verify_ptx_only(&ctx);
    }

    if major >= 10 {
        println!("\n⚠️  WGMMA is Hopper-only (sm_90).");
        println!("   Your GPU is sm_{}{} (Blackwell).", major, minor);
        println!("   WGMMA instructions don't exist on this architecture.");
        println!("\n   To test WGMMA, use a Hopper GPU (H100, H200).");
        return verify_ptx_only(&ctx);
    }

    // Load the CUDA module embedded in this binary
    println!("\nLoading embedded CUDA module");
    let module = kernels::load(&ctx)?;
    println!("✓ Module loaded successfully\n");

    // Test: Sync primitives
    run_wgmma_sync_test(&stream, &module)?;

    println!("\n=== WGMMA Test Complete ===");
    Ok(())
}

/// Fallback for GPUs that cannot execute the WGMMA kernels: inspect the loose
/// PTX build artifact beside this crate. The main path loads the module
/// embedded in the binary instead; only this fallback reads the loose file.
fn verify_ptx_only(ctx: &Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wgmma.ptx");
    println!("Checking loose PTX artifact: {}", ptx_path.display());

    if !ptx_path.exists() {
        return Err("loose PTX artifact not found (build with `cargo oxide build wgmma`)".into());
    }

    let ptx_file = ptx_path.to_str().ok_or("PTX path is not valid UTF-8")?;

    // Just verify it loads (may fail due to WGMMA instructions)
    match ctx.load_module_from_file(ptx_file) {
        Ok(_) => println!("✓ PTX module loaded (surprisingly, on non-Hopper GPU)"),
        Err(e) => println!("ℹ️  PTX load failed (expected on non-Hopper): {}", e),
    }

    println!("\n📝 To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());
    println!("\n   Look for: wgmma.fence.sync.aligned instructions");

    Ok(())
}

fn run_wgmma_sync_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Test: WGMMA Sync Primitives ---\n");

    // Allocate output buffer (1 u64 for descriptor)
    let mut dev_output = DeviceBuffer::<u64>::zeroed(stream, 1)?;

    // Launch with 128 threads (1 warpgroup)
    let cfg = LaunchConfig {
        block_dim: (128, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 256,
    };

    println!("Launching wgmma_sync_test kernel...");
    unsafe { module.wgmma_sync_test((stream).as_ref(), cfg, &mut dev_output) }?;

    stream.synchronize()?;

    // Read back result
    let host_output = dev_output.to_host_vec(stream)?;

    println!("SMEM descriptor: 0x{:016x}", host_output[0]);

    // Verify descriptor has expected swizzle bits set (bits 62-63 = 3)
    let swizzle_bits = (host_output[0] >> 62) & 0x3;
    if swizzle_bits == 3 {
        println!("✓ Swizzle mode correct (128B)");
    } else {
        println!("✗ Unexpected swizzle mode: {}", swizzle_bits);
    }

    // Verify leading dimension (bits 16-29)
    let leading_dim = (host_output[0] >> 16) & 0x3FFF;
    println!("  Leading dimension offset: {} (raw bits)", leading_dim);

    // Verify stride (bits 32-45)
    let stride = (host_output[0] >> 32) & 0x3FFF;
    println!("  Stride offset: {} (raw bits)", stride);

    Ok(())
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified tcgen05 Matmul Example (SM100+ / Blackwell only)
//!
//! 128×128×16 matmul using tcgen05 tensor cores with TMA and pre-tiled input.
//!
//! Build and run with:
//!   cargo oxide run tcgen05_matmul

use cuda_core::simt::LaunchConfig;
use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer,
    sys::{
        self as cuda_sys, CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive_expect_tx, mbarrier_init,
    mbarrier_inval, mbarrier_try_wait, mbarrier_try_wait_parity,
};
use cuda_device::convert::{bf16_to_f32, cvt_bf16x2_f32};
use cuda_device::shared::{SharedArray, cvta_generic_to_shared_offset};
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    stmatrix_m8n8_x2, tcgen05_alloc, tcgen05_commit_shared_cluster, tcgen05_dealloc,
    tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use half::f16;
use std::mem::MaybeUninit;
use std::sync::Arc;

// =============================================================================
// KERNEL
// =============================================================================

/// Build a tcgen05 SMEM descriptor from components.
#[inline(always)]
fn build_smem_descriptor(
    smem_addr: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    let swizzle_bits = (swizzle as u64) << 61;

    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit | swizzle_bits
}
#[cuda_module]
mod kernels {
    use super::*;

    /// 128×128×16 matmul kernel with PRE-TILED input data.
    ///
    /// Input matrices A and B must be pre-tiled on the host using
    /// `cuda_host::tiling::to_k_major_f16`:
    /// - A (128×16): K-major tiled
    /// - B (128×16): K-major tiled (B is stored transposed as N×K = 128×16)
    #[kernel]
    pub unsafe fn tcgen05_matmul_128x128_tiled(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        mut out: DisjointSlice<u32>,
        tile_a_x: i32,
        tile_a_y: i32,
        tile_b_x: i32,
        tile_b_y: i32,
    ) {
        // A: 128×16 f16 = 4096 bytes, B: 128×16 f16 = 4096 bytes
        static mut SMEM_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
        // Output: 128×128 bf16 = 16384 elements = 8192 packed u32
        static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
        static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
        static mut TMA_BAR: Barrier = Barrier::UNINIT;
        static mut MMA_BAR: Barrier = Barrier::UNINIT;

        const A_TILE_BYTES: u32 = 128 * 16 * 2; // 4096 bytes
        const B_TILE_BYTES: u32 = 128 * 16 * 2; // 4096 bytes

        unsafe {
            let tid = thread::threadIdx_x();
            let warp_id = warp::warp_id();
            let lane_id = tid % 32;
            let is_warp0 = warp_id == 0;
            let is_thread0 = tid == 0;

            // PHASE 1: Initialize barriers
            if is_thread0 {
                mbarrier_init(&raw mut TMA_BAR, 1);
                mbarrier_init(&raw mut MMA_BAR, 1);
                fence_proxy_async_shared_cta();
            }
            thread::sync_threads();

            // PHASE 2: Allocate TMEM (warp-synchronous)
            if is_warp0 {
                tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 512);
            }
            thread::sync_threads();
            let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

            // PHASE 3: TMA load
            if is_thread0 {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut SMEM_A as *mut u8,
                    a_tma,
                    tile_a_x,
                    tile_a_y,
                    &raw mut TMA_BAR,
                );
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut SMEM_B as *mut u8,
                    b_tma,
                    tile_b_x,
                    tile_b_y,
                    &raw mut TMA_BAR,
                );
                mbarrier_arrive_expect_tx(&raw const TMA_BAR, 1, A_TILE_BYTES + B_TILE_BYTES);
            }

            while !mbarrier_try_wait(&raw const TMA_BAR, 0) {}
            thread::sync_threads();

            // PHASE 4: Build SMEM descriptors and execute MMA
            if is_thread0 {
                let smem_a_addr = cvta_generic_to_shared_offset(&raw const SMEM_A as *const u8);
                let smem_b_addr = cvta_generic_to_shared_offset(&raw const SMEM_B as *const u8);

                const SBO_BYTES: u32 = 128; // 64 elements × 2 bytes
                const LBO_BYTES: u32 = 2048; // 16 tiles × 64 elements × 2 bytes
                const SWIZZLE_NONE: u8 = 0;

                let a_desc = build_smem_descriptor(smem_a_addr, LBO_BYTES, SBO_BYTES, SWIZZLE_NONE);
                let b_desc = build_smem_descriptor(smem_b_addr, LBO_BYTES, SBO_BYTES, SWIZZLE_NONE);

                let idesc = Tcgen05InstructionDescriptor::builder()
                    .shape(Tcgen05MmaShape::M128_N128)
                    .element_type(Tcgen05ElementType::F16)
                    .accumulator_type(Tcgen05AccumulatorType::F32)
                    .build()
                    .raw();

                tcgen05_mma_f16(tmem_addr, a_desc, b_desc, idesc, false);
                tcgen05_commit_shared_cluster(&raw mut MMA_BAR as *mut u64);
            }

            while !mbarrier_try_wait_parity(&raw const MMA_BAR, 0) {}
            thread::sync_threads();

            // PHASE 5: Epilogue - Extract TMEM to shared memory via stmatrix
            const N: usize = 128;
            let warp_row_base = (warp_id * 32) as usize;
            let row_stride_bytes = N * 2;

            let row_within_8 = (lane_id % 8) as usize;
            let is_second_matrix = (8..16).contains(&lane_id);
            let col_offset_for_matrix2 = if is_second_matrix { 16usize } else { 0usize };

            let mut tmem_row_block = 0u32;
            while tmem_row_block < 2 {
                let tmem_row = warp_id * 32 + tmem_row_block * 16;

                let mut col_block = 0u32;
                while col_block < 8 {
                    let col_offset = (col_block * 16) as usize;

                    let regs_a =
                        tcgen05_ld_16x256b_pure(tmem_addr + (tmem_row << 16) + col_offset as u32);
                    tcgen05_load_wait();

                    let regs_b = tcgen05_ld_16x256b_pure(
                        tmem_addr + (tmem_row << 16) + col_offset as u32 + 8,
                    );
                    tcgen05_load_wait();

                    let p0_lo = cvt_bf16x2_f32(regs_a[0], regs_a[1]);
                    let p1_lo = cvt_bf16x2_f32(regs_b[0], regs_b[1]);

                    let out_row_lo = warp_row_base + (tmem_row_block as usize * 16) + row_within_8;
                    let smem_addr_lo = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_lo * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_lo, p0_lo, p1_lo);

                    let p0_hi = cvt_bf16x2_f32(regs_a[2], regs_a[3]);
                    let p1_hi = cvt_bf16x2_f32(regs_b[2], regs_b[3]);

                    let out_row_hi =
                        warp_row_base + (tmem_row_block as usize * 16) + 8 + row_within_8;
                    let smem_addr_hi = (&raw mut SMEM_OUT as *mut u8).add(
                        out_row_hi * row_stride_bytes + col_offset * 2 + col_offset_for_matrix2,
                    );
                    stmatrix_m8n8_x2(smem_addr_hi, p0_hi, p1_hi);

                    col_block += 1;
                }
                tmem_row_block += 1;
            }

            thread::sync_threads();

            // PHASE 6: Copy output to global memory
            let mut idx = tid as usize;
            while idx < 8192 {
                *out.get_unchecked_mut(idx) = SMEM_OUT[idx];
                idx += 128;
            }

            // PHASE 7: Cleanup
            thread::sync_threads();
            if is_warp0 {
                tcgen05_dealloc(tmem_addr, 512);
            }
            if is_thread0 {
                mbarrier_inval(&raw mut TMA_BAR);
                mbarrier_inval(&raw mut MMA_BAR);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified tcgen05 Matmul Example ===\n");

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
        if major >= 10 {
            println!(
                "   Your GPU is sm_{}{} (consumer Blackwell has no tcgen05).",
                major, minor
            );
        } else {
            println!("   Your GPU is sm_{}{} (pre-Blackwell).", major, minor);
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
            if driver_status == Some(cuda_sys::cudaError_enum_CUDA_ERROR_INVALID_PTX) {
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

    run_tiled_kernel_test(&stream, &module)?;

    println!("\n=== tcgen05 Matmul Test Complete ===");
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
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tcgen05_matmul.ptx");

    if !ptx_path.exists() {
        return Err("PTX file not found".into());
    }

    println!("\n📝 PTX Verification:");
    println!("   PTX file generated at: {}", ptx_path.display());
    println!("\n📝 To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());
    println!("\n   Look for: tcgen05.mma instructions");

    Ok(())
}

fn run_tiled_kernel_test(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    use cuda_host::tiling::to_k_major_f16;

    println!("--- Test: tcgen05_matmul_128x128_tiled ---\n");

    const M: usize = 128;
    const N: usize = 128;
    const K: usize = 16;
    const OUTPUT_SIZE: usize = M * N / 2; // 8192 packed bf16 pairs

    println!("Matrix: {}×{}×{}", M, N, K);

    // Generate test data: A[i,k] = k, B_stored[n,k] = k+1
    let mut host_a_rowmajor: Vec<f16> = Vec::with_capacity(M * K);
    for _i in 0..M {
        for k in 0..K {
            host_a_rowmajor.push(f16::from_f32(k as f32));
        }
    }

    let mut host_b_rowmajor: Vec<f16> = Vec::with_capacity(N * K);
    for _n in 0..N {
        for k in 0..K {
            host_b_rowmajor.push(f16::from_f32((k + 1) as f32));
        }
    }

    // Pre-tile the data
    println!("Pre-tiling matrices...");
    let mut host_a_tiled = vec![f16::ZERO; M * K];
    let mut host_b_tiled = vec![f16::ZERO; N * K];

    to_k_major_f16(&host_a_rowmajor, &mut host_a_tiled, M, K);
    to_k_major_f16(&host_b_rowmajor, &mut host_b_tiled, N, K);

    let host_a_u16: Vec<u16> = host_a_tiled.iter().map(|v| v.to_bits()).collect();
    let host_b_u16: Vec<u16> = host_b_tiled.iter().map(|v| v.to_bits()).collect();

    // Expected: C[i,j] = sum_k k*(k+1) = 1360
    const EXPECTED_VAL: f32 = 1360.0;
    let expected_sum: f32 = (M * N) as f32 * EXPECTED_VAL;
    println!(
        "Expected: all elements = {}, sum = {}\n",
        EXPECTED_VAL, expected_sum
    );

    // Upload to GPU
    let dev_a = DeviceBuffer::from_host(stream, &host_a_u16)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b_u16)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    // Create TMA descriptors
    let a_tma = create_tma_descriptor_f16(
        a_ptr as *mut std::ffi::c_void,
        K as u64,
        M as u64,
        K as u32,
        M as u32,
    )?;
    let b_tma = create_tma_descriptor_f16(
        b_ptr as *mut std::ffi::c_void,
        K as u64,
        N as u64,
        K as u32,
        N as u32,
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque[..])?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque[..])?;

    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let tile_a_x = 0i32;
    let tile_a_y = 0i32;
    let tile_b_x = 0i32;
    let tile_b_y = 0i32;

    println!("Launching tcgen05_matmul_128x128_tiled...");
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, OUTPUT_SIZE)?;

    unsafe {
        module.tcgen05_matmul_128x128_tiled(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            tile_a_x,
            tile_a_y,
            tile_b_x,
            tile_b_y,
        )
    }?;

    stream.synchronize()?;

    // Verify results
    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let mut result_f32: Vec<f32> = Vec::with_capacity(M * N);
    for &packed in &host_output {
        let (lo, hi) = unpack_bf16_pair(packed);
        result_f32.push(lo);
        result_f32.push(hi);
    }

    let result_sum: f32 = result_f32.iter().sum();

    println!("\nOUTPUT (first 4×8):");
    for i in 0..4 {
        print!("  r{}: ", i);
        for j in 0..8 {
            print!("{:5.0} ", result_f32[i * N + j]);
        }
        println!();
    }

    println!("\nSUM CHECK:");
    println!("  Expected: {:.0}", expected_sum);
    println!("  Got:      {:.0}", result_sum);

    let mut correct_count = 0;
    for &v in &result_f32 {
        if (v - EXPECTED_VAL).abs() < 1.0 {
            correct_count += 1;
        }
    }

    let sum_ok = (result_sum - expected_sum).abs() < 100.0;
    let values_ok = correct_count == M * N;

    if sum_ok && values_ok {
        println!("\n✅ tcgen05_matmul_128x128_tiled PASSED!");
        Ok(())
    } else {
        println!("\n❌ tcgen05_matmul_128x128_tiled FAILED");
        Err("GPU verification failed".into())
    }
}

fn create_tma_descriptor_f16(
    global_address: *mut std::ffi::c_void,
    width: u64,
    height: u64,
    tile_width: u32,
    tile_height: u32,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [width, height];
    let global_strides: [u64; 1] = [width * 2];
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
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

fn unpack_bf16_pair(packed: u32) -> (f32, f32) {
    let lo = (packed & 0xFFFF) as u16;
    let hi = ((packed >> 16) & 0xFFFF) as u16;
    (bf16_to_f32(lo), bf16_to_f32(hi))
}

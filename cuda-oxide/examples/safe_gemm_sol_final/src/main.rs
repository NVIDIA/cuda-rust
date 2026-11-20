/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lean 4 formally verified Blackwell (`sm_100a`) speed-of-light FP16/BF16 GEMM (`C = A x B^T`):
//! 2-CTA cluster (`#[cluster_launch(2, 1, 1)]`), 7-stage `224 KiB` multicast TMA ring,
//! overlapped `CLC` work-stealing, 4-slot `TILE_INFO` mailbox, and 2-stage `512`-column
//! `TMEM` ping-pong (`tcgen05.mma.cta_group::2`).

use std::mem::MaybeUninit;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig1D};
use cuda_device::{
    DisjointSlice,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
        mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait_parity,
    },
    clc::{clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast},
    cluster, cluster_launch,
    convert::{bf16_to_f32, cvt_bf16x2_f32},
    cuda_module, kernel, launch_bounds, launch_contract,
    shared::{SharedArray, cvta_generic_to_shared_offset},
    tcgen05::{
        Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
        tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2,
        tcgen05_ld_16x256b_x8_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
        tcgen05_relinquish_alloc_permit_cg2,
    },
    thread,
    tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2},
    verify, warp,
};
use half::{bf16, f16};

/// Build a 64-bit `tcgen05` shared-memory matrix descriptor from base address, LBO, SBO, and swizzle mode.
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

include!("kernels.rs");

/// Return `true` when compute capability `(major, minor)` supports datacenter `tcgen05` (`sm_100a`+).
fn can_execute_tcgen05_ptx(major: i32, minor: i32) -> bool {
    matches!((major, minor), (10, 0) | (10, 1) | (10, 3) | (11, 0))
}

/// Verify that a loose PTX artifact targets a datacenter `tcgen05` architecture and exports
/// `gemm_sol_clc_multicast_4_stage_pipeline`.
fn verify_tcgen05_ptx_contract(ptx: &str) -> Result<(), String> {
    const TARGETS: [&str; 8] = [
        "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
    ];
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let has_target = document.directives().iter().any(|directive| {
        directive.name() == ".target"
            && directive
                .arguments()
                .split(',')
                .next()
                .is_some_and(|target| TARGETS.contains(&target.trim()))
    });
    if !has_target {
        return Err("missing a supported datacenter tcgen05 target".to_string());
    }
    if !document
        .callables_named("gemm_sol_clc_multicast_4_stage_pipeline")
        .any(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
    {
        return Err(
            "missing expected kernel entry `gemm_sol_clc_multicast_4_stage_pipeline`".to_string(),
        );
    }
    Ok(())
}

/// Encode a 2D `SWIZZLE_128B` + `L2_128B` `TmaDescriptor` for an `f16` (`u16`) device buffer.
fn encode_tma_2d_f16_swizzled(
    buf: &DeviceBuffer<u16>,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<TmaDescriptor, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [box_k, box_mn];
    let element_strides: [u32; 2] = [1, 1];

    unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            buf.cu_deviceptr() as *mut std::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()?;
        Ok(tensor_map.as_ptr().cast::<TmaDescriptor>().read())
    }
}

/// Validate `gemm_sol_clc_multicast_4_stage_pipeline` bit-for-bit on `4096^3` using 12-bit Walsh codes.
fn validate_4096_walsh(
    stream: &CudaStream,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    const M: usize = 4096;
    let k_code = |kk: usize| -> u32 { ((kk as u32).wrapping_mul(251).wrapping_add(17)) & 0xFFF };
    let r_code = |r: usize| -> u32 { ((r as u32).wrapping_mul(197).wrapping_add(101)) & 0xFFF };
    let c_code = |c: usize| -> u32 { ((c as u32).wrapping_mul(109).wrapping_add(1021)) & 0xFFF };
    let fp = |pos: u32, kk: usize| -> f32 { (13 - 2 * (pos ^ k_code(kk)).count_ones() as i32) as f32 };

    let host_a: Vec<u16> = (0..M)
        .flat_map(|i| (0..M).map(move |kk| f16::from_f32(fp(r_code(i), kk)).to_bits()))
        .collect();
    let host_b: Vec<u16> = (0..M)
        .flat_map(|j| (0..M).map(move |kk| f16::from_f32(fp(c_code(j), kk)).to_bits()))
        .collect();
    let expected_by_xor: Vec<u16> = (0..M)
        .map(|d| {
            let score = 13 - 2 * (d as u32).count_ones() as i64;
            bf16::from_f32((M as i64 * score) as f32).to_bits()
        })
        .collect();

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;
    let a_tma = encode_tma_2d_f16_swizzled(&dev_a, M as u64, M as u64, 64, 128)?;
    let b_tma = encode_tma_2d_f16_swizzled(&dev_b, M as u64, M as u64, 64, 32)?;

    let tiles_m = (M / 256) as u32;
    let tiles_n = (M / 128) as u32;
    let wide_total = tiles_m * (tiles_n / 2);
    let total_tiles = wide_total.div_ceil(B200_CLUSTER_PAIRS) * B200_CLUSTER_PAIRS;

    let mut dev_out = DeviceBuffer::<u32>::zeroed(stream, M * M / 2)?;
    let cfg = LaunchConfig1D::new(total_tiles * 2, 192, 0);
    let prep = module.prepare_gemm_sol_clc_multicast_4_stage_pipeline(cfg)?;
    unsafe {
        module.gemm_sol_clc_multicast_4_stage_pipeline(
            stream,
            &prep,
            a_tma,
            b_tma,
            &mut dev_out,
            M as i32,
            M as i32,
            tiles_m,
            tiles_n,
        )?;
    }
    stream.synchronize()?;

    let host_out = dev_out.to_host_vec(stream)?;
    for row in 0..M {
        for col in 0..M {
            let word = host_out[(row * M + col) / 2];
            let got_bits = if col % 2 == 0 {
                (word & 0xFFFF) as u16
            } else {
                (word >> 16) as u16
            };
            let exp_bits = expected_by_xor[(r_code(row) ^ c_code(col)) as usize];
            if got_bits != exp_bits {
                return Err(format!(
                    "mismatch at C[{row},{col}]: got {} vs expected {}",
                    bf16_to_f32(got_bits),
                    bf16_to_f32(exp_bits)
                )
                .into());
            }
        }
    }
    println!("PASSED: Lean-verified gemm_sol_clc_multicast_4_stage_pipeline 4096x4096x4096 (all 16,777,216 BF16 elements exact!)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provably Safe Stage 10 Blackwell GEMM SoL (`safe_gemm_sol_final`) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{major}{minor}");

    if !can_execute_tcgen05_ptx(major, minor) {
        let ptx_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("safe_gemm_sol_final.ptx");
        if ptx_path.exists() {
            let ptx = std::fs::read_to_string(&ptx_path)?;
            verify_tcgen05_ptx_contract(&ptx)?;
            println!(
                "PASS: Lean 4 verified PTX artifact validated at {} (sm_100a hardware required for execution)",
                ptx_path.display()
            );
            return Ok(());
        }
        println!("Note: sm_{major}{minor} does not have tcgen05/TMEM hardware; build with `--arch sm_100a` to verify PTX.");
        return Ok(());
    }

    let module = unsafe { kernels::load(&ctx) }?;
    validate_4096_walsh(&stream, &module)?;
    Ok(())
}

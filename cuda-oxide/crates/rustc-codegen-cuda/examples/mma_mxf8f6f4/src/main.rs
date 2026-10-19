/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Numeric check for the block-scaled mxf8f6f4 register MMA.
//!
//! Runs `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale`
//! with `.f32.e4m3.e4m3.f32.ue8m0` on one warp and compares the full
//! 16x8 result matrix against a CPU reference:
//!
//! - A (16x32) and B (32x8) hold small integers exactly representable in
//!   e4m3; the f32 accumulation is exact, so the comparison is bitwise.
//! - Fragments are packed on the host per the PTX ISA "Matrix Fragments
//!   for mma.m16n8k32" 8-bit layout; the kernel only loads fragments,
//!   issues the MMA, and stores the D fragments.
//! - Both scale operands are 1.0 (ue8m0 byte 127) in every selectable
//!   byte, so scale selection cannot mask a data-path error, but the
//!   scale *selector* logic is only trivially exercised.
//!
//! The instruction only exists on Blackwell consumer parts (sm_120a/f,
//! sm_121a/f), so the device code is always compiled for sm_120a
//! (smoketest passes `--arch=sm_120a`). On any other GPU the example
//! verifies the generated PTX instead of executing and exits cleanly.
//!
//! Build and run with:
//!   cargo oxide run mma_mxf8f6f4 --arch sm_120a

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, wmma};

const M: usize = 16;
const N: usize = 8;
const K: usize = 32;
/// Every selectable ue8m0 byte = 127 -> scale factor 2^0 = 1.0.
const SCALES_ONE: u32 = 0x7F7F_7F7F;

#[cuda_module]
mod kernels {
    use super::*;

    /// One warp: load this lane's fragments, run the block-scaled MMA,
    /// store this lane's D fragments.
    #[kernel]
    pub fn mma_mxf8f6f4_e4m3(a_frag: &[u32], b_frag: &[u32], mut d_out: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let a = [
            a_frag[lane * 4],
            a_frag[lane * 4 + 1],
            a_frag[lane * 4 + 2],
            a_frag[lane * 4 + 3],
        ];
        let b = [b_frag[lane * 2], b_frag[lane * 2 + 1]];
        let c = [0.0f32; 4];
        // SAFETY: all 32 lanes execute the same MMA; fragments follow the
        // documented PTX layout; byte/thread selectors are in range.
        let d = unsafe {
            wmma::mma_m16n8k32_mxf8f6f4_f32_e4m3_e4m3(c, a, b, SCALES_ONE, 0, 0, SCALES_ONE, 0, 0)
        };
        let base = lane * 4;
        if base + 4 <= d_out.len() {
            for (offset, value) in d.into_iter().enumerate() {
                // SAFETY: the bounds check covers this lane's unique slots.
                unsafe { *d_out.get_unchecked_mut(base + offset) = value };
            }
        }
    }
}

/// Encode a small integer (|v| <= 3) as OCP e4m3 (bias 7).
fn encode_e4m3(v: i32) -> u8 {
    let sign = if v < 0 { 0x80u8 } else { 0 };
    match v.abs() {
        0 => sign, // +/-0
        1 => sign | 0x38,
        2 => sign | 0x40,
        3 => sign | 0x44,
        _ => panic!("encode_e4m3 only handles |v| <= 3"),
    }
}

fn a_value(row: usize, k: usize) -> i32 {
    ((row + k) % 7) as i32 - 3
}

fn b_value(k: usize, col: usize) -> i32 {
    ((k * 2 + col) % 5) as i32 - 2
}

/// Non-Blackwell-consumer fallback: confirm the sm_120a PTX was generated
/// and actually contains the block-scaled MMA instruction.
fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("mma_mxf8f6f4.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)?;
    if !ptx.contains("kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0") {
        return Err("generated PTX lacks the kind::mxf8f6f4 block-scale instruction".into());
    }
    println!(
        "PTX was generated successfully: {} contains the kind::mxf8f6f4 instruction",
        ptx_path.display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mxf8f6f4 block-scale MMA numeric check (e4m3 x e4m3 -> f32) ===");
    let ctx = CudaContext::new(0)?;

    // kind::mxf8f6f4 exists only on sm_120/sm_121 (Blackwell consumer).
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{major}{minor}");
    if major != 12 {
        println!("WARNING: mxf8f6f4 block-scale MMA requires sm_120/sm_121 (Blackwell consumer)");
        println!("         detected sm_{major}{minor}; verifying generated PTX only");
        return verify_ptx_only();
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Pack per-lane fragments per the PTX ISA m16n8k32 8-bit layout.
    let mut a_frag = vec![0u32; 32 * 4];
    let mut b_frag = vec![0u32; 32 * 2];
    for lane in 0..32 {
        let group_id = lane >> 2;
        let tig = lane % 4; // threadID_in_group
        for i in 0..16 {
            let row = if (i / 4) % 2 == 0 {
                group_id
            } else {
                group_id + 8
            };
            let col = tig * 4 + (i % 4) + if i >= 8 { 16 } else { 0 };
            let byte = encode_e4m3(a_value(row, col));
            a_frag[lane * 4 + i / 4] |= (byte as u32) << (8 * (i % 4));
        }
        for i in 0..8 {
            let row = tig * 4 + (i % 4) + if i >= 4 { 16 } else { 0 };
            let col = group_id;
            let byte = encode_e4m3(b_value(row, col));
            b_frag[lane * 2 + i / 4] |= (byte as u32) << (8 * (i % 4));
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a_frag).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_frag).unwrap();
    let mut d_dev = DeviceBuffer::<f32>::zeroed(&stream, 32 * 4).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: one warp, launch shape matches the kernel contract.
    unsafe { module.mma_mxf8f6f4_e4m3(&stream, cfg, &a_dev, &b_dev, &mut d_dev) }
        .expect("mma_mxf8f6f4_e4m3 launch failed");
    let d_frag = d_dev.to_host_vec(&stream).unwrap();

    // Scatter D fragments back into a 16x8 matrix per the C/D layout.
    let mut d_gpu = [[f32::NAN; N]; M];
    for lane in 0..32 {
        let group_id = lane >> 2;
        let tig = lane % 4;
        for i in 0..4 {
            let row = if i < 2 { group_id } else { group_id + 8 };
            let col = tig * 2 + (i % 2);
            d_gpu[row][col] = d_frag[lane * 4 + i];
        }
    }

    // CPU reference (exact integer arithmetic).
    let mut mismatches = 0;
    let mut nonzero = 0;
    for (row, gpu_row) in d_gpu.iter().enumerate() {
        for (col, &gpu_value) in gpu_row.iter().enumerate() {
            let mut acc = 0i64;
            for k in 0..K {
                acc += (a_value(row, k) * b_value(k, col)) as i64;
            }
            let expected = acc as f32;
            if expected != 0.0 {
                nonzero += 1;
            }
            if gpu_value != expected {
                if mismatches < 8 {
                    println!("  D[{row}][{col}] = {gpu_value} (expected {expected})");
                }
                mismatches += 1;
            }
        }
    }

    println!(
        "checked {} elements ({nonzero} nonzero expected), {mismatches} mismatches",
        M * N
    );
    if mismatches == 0 && nonzero > 0 {
        println!("SUCCESS");
        Ok(())
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

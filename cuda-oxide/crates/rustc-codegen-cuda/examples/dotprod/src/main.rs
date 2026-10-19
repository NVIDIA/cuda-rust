/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer dot product intrinsics (`dp4a`, `dp2a`); end-to-end GPU test.
//!
//! `dp4a` treats two `u32` operands as vectors of 4 packed bytes each,
//! multiplies corresponding elements, sums the products, and adds a 32-bit
//! accumulator:
//!
//! ```text
//! d = c + a.byte0*b.byte0 + a.byte1*b.byte1 + a.byte2*b.byte2 + a.byte3*b.byte3
//! ```
//!
//! `dp2a` treats `a` as two packed 16-bit values and uses the lower 2 bytes
//! of `b`:
//!
//! ```text
//! d = c + a.half0*b.byte0 + a.half1*b.byte1
//! ```
//!
//! Both instructions are available on `sm_61+` (Pascal and later).
//!
//! Build and run with:
//!   cargo oxide run dotprod --arch sm_80

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::dotprod::{dp2a_s32, dp2a_u32, dp4a_s32, dp4a_u32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 computes all four dot product variants with known operands and
    /// writes the results to `out[0..4]`:
    ///
    /// ```text
    /// out[0] = dp4a_s32(a4, b4, c_s)
    /// out[1] = dp4a_u32(a4, b4, c_u)
    /// out[2] = dp2a_s32(a2, b2, c_s)
    /// out[3] = dp2a_u32(a2, b2, c_u)
    /// ```
    #[kernel]
    pub fn dotprod_test(
        a4: u32,
        b4: u32,
        c_signed: i32,
        c_unsigned: u32,
        a2: u32,
        b2: u32,
        mut out: DisjointSlice<i32>,
    ) {
        let tid = thread::index_1d();
        if tid.get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = dp4a_s32(a4, b4, c_signed);
                *out.get_unchecked_mut(1) = dp4a_u32(a4, b4, c_unsigned) as i32;
                *out.get_unchecked_mut(2) = dp2a_s32(a2, b2, c_signed);
                *out.get_unchecked_mut(3) = dp2a_u32(a2, b2, c_unsigned) as i32;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== dp4a / dp2a integer dot product (sm_61+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // dp4a / dp2a require sm_61+ (Pascal). In practice we expect sm_80+ runners.
    let sm = major * 10 + minor;
    if sm < 61 {
        println!("\nskipping: dp4a/dp2a require sm_61+ (Pascal)");
        println!("  this GPU is sm_{}", sm);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // ---- dp4a operands ---------------------------------------------------
    // Pack bytes [1, 2, 3, 200] into u32 (little-endian: byte0 = LSB).
    //   a4 = 0xC8_03_02_01
    // 200 (0xC8) is >= 128, so its signed interpretation is -56 (200 - 256).
    // This makes dp4a_s32 and dp4a_u32 produce different results.
    let a4: u32 = pack_bytes(1, 2, 3, 200);
    // Pack bytes [5, 6, 7, 8]:
    //   b4 = 0x08_07_06_05
    let b4: u32 = pack_bytes(5, 6, 7, 8);

    // Signed accumulator
    let c_signed: i32 = 100;
    // Unsigned accumulator
    let c_unsigned: u32 = 100;

    // dp4a_u32: d = c + 1*5 + 2*6 + 3*7 + 200*8
    //             = 100 + 5 + 12 + 21 + 1600 = 1738
    let expect_dp4a_u32: i32 = 1738;

    // dp4a_s32: d = c + 1*5 + 2*6 + 3*7 + (-56)*8
    //             = 100 + 5 + 12 + 21 - 448 = -310
    let expect_dp4a_s32: i32 = -310;

    // ---- dp2a operands ---------------------------------------------------
    // Pack two 16-bit halves [300, 0xFF00] into u32 (half0 = low 16 bits).
    //   a2 = 0xFF00_012C
    // 0xFF00 (65280) is >= 32768, so its signed interpretation is
    // -256 (65280 - 65536).  This makes dp2a_s32 and dp2a_u32 differ.
    let a2: u32 = pack_halves(300, 0xFF00);
    // b2 lower 2 bytes = [3, 4]:
    //   b2 = 0x0000_04_03
    let b2: u32 = pack_bytes(3, 4, 0, 0);

    // dp2a_u32: d = c + 300*3 + 65280*4
    //             = 100 + 900 + 261120 = 262120
    // NOTE: dp2a.lo uses b.byte0 and b.byte1, which are the 2 least-significant
    // bytes of b2 = [3, 4].
    let expect_dp2a_u32: i32 = 262120;

    // dp2a_s32: d = c + 300*3 + (-256)*4
    //             = 100 + 900 - 1024 = -24
    let expect_dp2a_s32: i32 = -24;

    println!("\nOperands:");
    println!("  dp4a:  a=0x{:08x}  b=0x{:08x}", a4, b4);
    println!("         bytes(a)=[1,2,3,200]  bytes(b)=[5,6,7,8]");
    println!(
        "         dp4a_u32 expected = {} + 1*5+2*6+3*7+200*8 = {}",
        c_unsigned, expect_dp4a_u32
    );
    println!(
        "         dp4a_s32 expected = {} + 1*5+2*6+3*7+(-56)*8 = {}",
        c_signed, expect_dp4a_s32
    );
    println!("  dp2a:  a=0x{:08x}  b=0x{:08x}", a2, b2);
    println!("         halves(a)=[300,0xFF00]  bytes_lo(b)=[3,4]");
    println!(
        "         dp2a_u32 expected = {} + 300*3+65280*4 = {}",
        c_unsigned, expect_dp2a_u32
    );
    println!(
        "         dp2a_s32 expected = {} + 300*3+(-256)*4 = {}",
        c_signed, expect_dp2a_s32
    );

    // ---- Launch ----------------------------------------------------------
    let cfg = LaunchConfig {
        block_dim: (1, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, 4).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.dotprod_test(
            (stream).as_ref(),
            cfg,
            a4,
            b4,
            c_signed,
            c_unsigned,
            a2,
            b2,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");

    let results = out_dev.to_host_vec(&stream).unwrap();

    // ---- Verify ----------------------------------------------------------
    println!("\nResults:");
    let labels = ["dp4a_s32", "dp4a_u32", "dp2a_s32", "dp2a_u32"];
    let expected = [
        expect_dp4a_s32,
        expect_dp4a_u32,
        expect_dp2a_s32,
        expect_dp2a_u32,
    ];

    let mut failed = false;
    for i in 0..4 {
        let ok = results[i] == expected[i];
        let mark = if ok { "ok" } else { "FAIL" };
        println!(
            "  {}:  got {} expected {} [{}]",
            labels[i], results[i], expected[i], mark
        );
        if !ok {
            failed = true;
        }
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: all dp4a/dp2a results match expected values");
}

/// Pack four bytes into a `u32` (byte0 = least-significant byte).
fn pack_bytes(b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
    (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24)
}

/// Pack two 16-bit halves into a `u32` (half0 = low 16 bits).
fn pack_halves(h0: u16, h1: u16) -> u32 {
    (h0 as u32) | ((h1 as u32) << 16)
}

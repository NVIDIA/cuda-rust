/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

//! `cvt_f16x2_f32` intrinsic: pack an f32 pair into a u32 holding two f16s.
//!
//! The kernel packs (lo, hi) f32 pairs into u32 f16x2 values via a single
//! `cvt.rn.f16x2.f32` PTX instruction. The host verifies bit-exact
//! agreement with a scalar round-to-nearest-even reference computed via
//! Rust `f16` casts, a device-side scalar kernel cross-check, and a
//! constant-literal kernel that pins the lane order.
//!
//! Run: cargo oxide run cvt_f16x2

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn pack_f16x2(lo_in: &[f32], hi_in: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as usize;
        if let Some(out_elem) = out.get_mut(idx) {
            let lo = lo_in[i];
            let hi = hi_in[i];
            *out_elem = cvt_f16x2_f32(lo, hi);
        }
    }

    /// Edge case: constant literal operands (no locals involved).
    #[kernel]
    pub fn pack_f16x2_consts(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = cvt_f16x2_f32(1.5_f32, -2.25_f32);
        }
    }

    /// Scalar reference path: what users must write without the intrinsic.
    #[kernel]
    pub fn pack_f16x2_scalar(lo_in: &[f32], hi_in: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as usize;
        if let Some(out_elem) = out.get_mut(idx) {
            let lo = lo_in[i];
            let hi = hi_in[i];
            *out_elem = ((lo as f16).to_bits() as u32) | (((hi as f16).to_bits() as u32) << 16);
        }
    }
}

fn scalar_ref(lo: f32, hi: f32) -> u32 {
    ((lo as f16).to_bits() as u32) | (((hi as f16).to_bits() as u32) << 16)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 256;
    let ctx = CudaContext::new(0)?;

    // `cvt.rn.f16x2.f32` is a packed conversion and needs sm_80 (Ampere) or
    // later, the same floor `cvt_packed` gates on.
    let (major, minor) = ctx.compute_capability()?;
    if major < 8 {
        println!("skipping: cvt.rn.f16x2.f32 requires sm_80+ (device is sm_{major}{minor})");
        return Ok(());
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let lo_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.337 - 40.0).collect();
    let hi_host: Vec<f32> = (0..N).map(|i| (i as f32) * -1.113 + 17.5).collect();

    let lo_dev = DeviceBuffer::from_host(&stream, &lo_host)?;
    let hi_dev = DeviceBuffer::from_host(&stream, &hi_host)?;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    let cfg = LaunchConfig::for_num_elems(N as u32);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.pack_f16x2(&stream, cfg, &lo_dev, &hi_dev, &mut out_dev) }?;

    let out_host = out_dev.to_host_vec(&stream)?;
    let mut failures = 0;
    for i in 0..N {
        let expect = scalar_ref(lo_host[i], hi_host[i]);
        if out_host[i] != expect {
            if failures < 5 {
                eprintln!(
                    "MISMATCH i={i}: lo={} hi={} got={:#010x} want={:#010x}",
                    lo_host[i], hi_host[i], out_host[i], expect
                );
            }
            failures += 1;
        }
    }

    // Device-side scalar kernel must agree bit-for-bit with the intrinsic.
    let mut out_scalar_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.pack_f16x2_scalar(&stream, cfg, &lo_dev, &hi_dev, &mut out_scalar_dev) }?;
    let out_scalar = out_scalar_dev.to_host_vec(&stream)?;
    for i in 0..N {
        if out_scalar[i] != out_host[i] {
            if failures < 10 {
                eprintln!(
                    "DEVICE SCALAR/INTRINSIC MISMATCH i={i}: scalar={:#010x} intrinsic={:#010x}",
                    out_scalar[i], out_host[i]
                );
            }
            failures += 1;
        }
    }

    // Constant-literal kernel: lane order with known bit patterns.
    // f16(1.5) = 0x3E00 (low half), f16(-2.25) = 0xC080 (high half).
    let mut out_const_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let cfg1 = LaunchConfig::for_num_elems(1);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.pack_f16x2_consts(&stream, cfg1, &mut out_const_dev) }?;
    let got_const = out_const_dev.to_host_vec(&stream)?[0];
    if got_const != 0xC080_3E00 {
        eprintln!("CONST MISMATCH: got={got_const:#010x} want=0xc0803e00");
        failures += 1;
    }

    if failures == 0 {
        println!(
            "SUCCESS: all {N} packed f16x2 values match scalar reference (+ const lane-order check)"
        );
        Ok(())
    } else {
        Err(format!("{failures} mismatches").into())
    }
}

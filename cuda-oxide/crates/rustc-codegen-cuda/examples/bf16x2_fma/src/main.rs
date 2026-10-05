// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavior example for the packed bf16x2 FMA intrinsic.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::bf16x2::fma_bf16x2;
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn run_fma(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        let a = cvt_bf16x2_f32(2.0, 4.0);
        let b = cvt_bf16x2_f32(3.0, 5.0);
        let c = cvt_bf16x2_f32(7.0, 11.0);
        let value = fma_bf16x2(a, b, c);

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn pack_bf16x2(lo: f32, hi: f32) -> u32 {
    u32::from(bf16_bits(lo)) | (u32::from(bf16_bits(hi)) << 16)
}

fn main() {
    println!("=== bf16x2_fma ===");
    let expected = pack_bf16x2(13.0, 31.0);

    let ctx = CudaContext::new(0).expect("CUDA init");

    // `fma.rn.bf16x2` needs sm_80+. Skip cleanly on older devices so the
    // example does not report a hardware-capability failure as a bug.
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 8 {
        println!("skipping: fma.rn.bf16x2 requires sm_80+ (device is sm_{major}{minor})");
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.run_fma(&stream, LaunchConfig::for_num_elems(1), &mut out) }
        .expect("launch run_fma");

    let got = out.to_host_vec(&stream).unwrap()[0];
    println!("expected: 0x{expected:08x}");
    println!("got:      0x{got:08x}");

    if got != expected {
        println!("bf16x2_fma: FAIL");
        std::process::exit(1);
    }
    println!("bf16x2_fma: PASS (lane0 = 2*3+7 = 13, lane1 = 4*5+11 = 31)");
}

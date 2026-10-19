/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual low-level launch API regression test for NVVM IR/libdevice artifacts.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

#[kernel]
pub fn sqrt_bits(seed: f32, mut out: DisjointSlice<u32>) {
    if thread::index_1d().get() == 0 {
        unsafe {
            *out.get_unchecked_mut(0) = seed.sqrt().to_bits();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Manual Libdevice Launch API Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "manual_launch_libdevice")?;

    let seed = 4.0_f32;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    // SAFETY: args mirror `sqrt_bits`'s signature (f32 scalar, then the
    // (ptr, len) pair for its slice parameter); `out` is a live DeviceBuffer.
    unsafe {
        cuda_launch! {
            kernel: sqrt_bits,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [seed, slice_mut(out)]
        }
    }?;

    let got = out.to_host_vec(&stream)?[0];
    assert_eq!(got, seed.sqrt().to_bits());
    println!("PASS: manual libdevice launch matched");
    Ok(())
}

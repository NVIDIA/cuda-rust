/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal `#[constant]` example: one scalar, one kernel, one launch.
//!
//! Build and run with:
//!   cargo oxide run constant_memory_simple

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[constant]
    static SCALE: ConstantMemory<f32> = ConstantMemory::UNINIT;

    #[constant]
    static UNUSED: ConstantMemory<f32> = ConstantMemory::UNINIT;

    #[kernel]
    pub fn multiply(mut out: DisjointSlice<f32>) {
        let s = SCALE.get();
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = (i as f32) * s;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    module.set_scale(&stream, &3.0)?;

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 8)?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.multiply(&stream, LaunchConfig::for_num_elems(8), &mut out) }?;

    let result = out.to_host_vec(&stream)?;
    println!("{:?}", result);

    let expected = vec![0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
    assert_eq!(
        result, expected,
        "constant_memory_simple: kernel output mismatch"
    );
    println!(
        "✓ SUCCESS: constant-memory scale applied correctly ({} elements)",
        result.len()
    );
    Ok(())
}

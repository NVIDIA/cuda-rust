/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused cuda-gdb regression fixture for raw-pointer local values.
//!
//! Run under cuda-gdb with:
//! `CUDA_OXIDE_DEBUG=full cargo oxide debug debug_pointer_locals`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 8;
const INT_SENTINEL: i32 = 41;
const FLOAT_SENTINEL: f32 = 13.0;
const EXPECTED_SUM: i32 = INT_SENTINEL + FLOAT_SENTINEL as i32;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn debug_pointer_locals(
        input: &[i32],
        fdata: &[f32],
        null_address: usize,
        mut out: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();

        if let Some(slot) = out.get_mut(idx) {
            let ptr: *const i32 = &input[tid] as *const i32;
            let fptr: *const f32 = &fdata[tid] as *const f32;
            // Take this through a kernel argument so the optimizer cannot
            // replace the control with an unrelated compile-time constant.
            let null_ptr: *const i32 = null_address as *const i32;

            // All three pointer locals are live here. CUDA_OXIDE_DEBUG_POINTER_BREAKPOINT
            let sum = unsafe { *ptr } + unsafe { *fptr } as i32;
            *slot = if null_ptr.is_null() { sum } else { i32::MIN };
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let input = vec![INT_SENTINEL; N];
    let fdata = vec![FLOAT_SENTINEL; N];
    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let fdata_dev = DeviceBuffer::from_host(&stream, &fdata)?;
    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;

    // SAFETY: all buffers contain N elements, and out.get_mut gates lanes outside 0..N.
    unsafe {
        module.debug_pointer_locals(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &fdata_dev,
            0,
            &mut out_dev,
        )
    }?;
    stream.synchronize()?;

    let output = out_dev.to_host_vec(&stream)?;
    assert_eq!(output, vec![EXPECTED_SUM; N]);
    println!("PASS: raw-pointer locals dereferenced correctly: {output:?}");
    Ok(())
}

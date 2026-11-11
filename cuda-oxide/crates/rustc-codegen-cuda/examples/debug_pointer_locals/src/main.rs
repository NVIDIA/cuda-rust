/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused cuda-gdb regression fixture for raw-pointer local values.
//!
//! Run under cuda-gdb with:
//! `CUDA_OXIDE_DEBUG=full cargo oxide debug debug_pointer_locals`

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 8;
const INT_SENTINEL: i32 = 41;
const FLOAT_SENTINEL: f32 = 13.0;
const PARAM_SCALE: f32 = 2.0;
const PARAM_OFFSET: f32 = 3.0;
const DEEP_LEAF: i32 = 21;
const EXPECTED_SUM: i32 =
    INT_SENTINEL + FLOAT_SENTINEL as i32 + (PARAM_SCALE as i32) + (PARAM_OFFSET as i32) + DEEP_LEAF;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Params {
    pub scale: f32,
    pub offset: f32,
}

unsafe impl cuda_core::DeviceCopy for Params {}

#[derive(Clone, Copy)]
pub struct L1(pub i32);
#[derive(Clone, Copy)]
pub struct L2(pub L1);
#[derive(Clone, Copy)]
pub struct L3(pub L2);
#[derive(Clone, Copy)]
pub struct L4(pub L3);
#[derive(Clone, Copy)]
pub struct L5(pub L4);
#[derive(Clone, Copy)]
pub struct L6(pub L5);
#[derive(Clone, Copy)]
pub struct L7(pub L6);
#[derive(Clone, Copy)]
pub struct L8(pub L7);
#[derive(Clone, Copy)]
pub struct Deep8 {
    pub top: i32,
    pub nested: L8,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn debug_pointer_locals(
        input: &[i32],
        fdata: &[f32],
        ps: &[Params],
        null_address: usize,
        mut out: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let tid: usize = idx.get();

        if let Some(slot) = out.get_mut(idx) {
            let ptr: *const i32 = &input[tid] as *const i32;
            let fptr: *const f32 = &fdata[tid] as *const f32;
            let param_ref: &Params = &ps[tid];
            let param_ptr: *const Params = ps.as_ptr();
            let b8 = Deep8 {
                top: 7,
                nested: L8(L7(L6(L5(L4(L3(L2(
                    L1(DEEP_LEAF + (null_address as i32)),
                ))))))),
            };
            // Take this through a kernel argument so the optimizer cannot
            // replace the control with an unrelated compile-time constant.
            let null_ptr: *const i32 = null_address as *const i32;

            // All pointer and nested locals are live here. CUDA_OXIDE_DEBUG_POINTER_BREAKPOINT
            let sum = unsafe { *ptr }
                + unsafe { *fptr } as i32
                + (param_ref.scale as i32)
                + (unsafe { (*param_ptr).offset } as i32)
                + b8.nested.0.0.0.0.0.0.0.0;
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
    let params = vec![
        Params {
            scale: PARAM_SCALE,
            offset: PARAM_OFFSET,
        };
        N
    ];
    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let fdata_dev = DeviceBuffer::from_host(&stream, &fdata)?;
    let params_dev = DeviceBuffer::from_host(&stream, &params)?;
    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;

    // SAFETY: all buffers contain N elements, and out.get_mut gates lanes outside 0..N.
    unsafe {
        module.debug_pointer_locals(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &fdata_dev,
            &params_dev,
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

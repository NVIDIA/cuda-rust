/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Link two CUDA C++ device libraries during compilation and load the embedded cubin.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[device]
unsafe extern "C" {
    fn external_scale_add(value: i32) -> i32;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn scale_add(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let offset = index.get();
        if let Some(value) = output.get_mut(index) {
            // SAFETY: the C++ definition takes and returns an int32 scalar;
            // it accesses no pointers and has no participation requirement.
            *value = unsafe { external_scale_add(input[offset]) };
        }
    }
}

fn verify(expected_addend: i32) -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let input: Vec<i32> = (-128..129).collect();
    let device_input = DeviceBuffer::from_host(&stream, &input)?;
    let mut device_output = DeviceBuffer::<i32>::zeroed(&stream, input.len())?;
    let module = kernels::load(&context)?;
    // SAFETY: the equal-length buffers cover every access, the bounds check
    // handles the partial last block, and the kernel needs no shared memory.
    unsafe {
        module.scale_add(
            &stream,
            LaunchConfig::for_num_elems(input.len() as u32),
            &device_input,
            &mut device_output,
        )?;
    }
    let actual = device_output.to_host_vec(&stream)?;
    let expected: Vec<i32> = input
        .iter()
        .map(|value| value * 2 + expected_addend)
        .collect();
    assert_eq!(actual, expected);
    println!("PASS external LTO-IR: 257 values, addend={expected_addend}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expected_addend = std::env::args()
        .nth(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3);
    verify(expected_addend)
}

#[test]
fn linked_device_libraries_execute() {
    verify(3).unwrap();
}

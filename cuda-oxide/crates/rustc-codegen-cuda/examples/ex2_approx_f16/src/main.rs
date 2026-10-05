/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::float::ex2_approx_f16;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const INPUTS: [u16; 6] = [0x0000, 0x3c00, 0xbc00, 0x4000, 0x3800, 0xcb80];
const EXPECTED: [u16; 6] = [0x3c00, 0x4000, 0x3800, 0x4400, 0x3da8, 0x0200];
const MAX_RELATIVE_ERROR: f32 = 0.001_046_654_7; // 2^-9.9

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let converted = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 5;
            let normalized = (fraction as u32) << shift;
            sign | ((113 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(converted)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn evaluate(inputs: &[u16], mut outputs: DisjointSlice<u16>) {
        let thread_index = thread::index_1d();
        let index = thread_index.get();
        if index < inputs.len()
            && let Some(output) = outputs.get_mut(thread_index)
        {
            *output = ex2_approx_f16(inputs[index]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ex2.approx.f16 GPU validation ===");
    assert_eq!(f16_bits_to_f32(0x0200), 2.0_f32.powi(-15));
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    if major * 10 + minor < 75 {
        println!("skipping: ex2.approx.f16 requires sm_75+ (device is sm_{major}{minor})");
        return Ok(());
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let inputs = DeviceBuffer::from_host(&stream, &INPUTS)?;
    let mut outputs = DeviceBuffer::<u16>::zeroed(&stream, INPUTS.len())?;

    // SAFETY: the launch covers exactly the allocated input and output lengths.
    unsafe {
        module.evaluate(
            &stream,
            LaunchConfig::for_num_elems(INPUTS.len() as u32),
            &inputs,
            &mut outputs,
        )
    }?;

    let measured = outputs.to_host_vec(&stream)?;
    println!("device: sm_{major}{minor}");
    for ((input, output), expected) in INPUTS.iter().zip(&measured).zip(EXPECTED) {
        println!("input=0x{input:04x} output=0x{output:04x} expected=0x{expected:04x}");
    }
    for (&output, expected) in measured.iter().zip(EXPECTED) {
        let actual = f16_bits_to_f32(output);
        let reference = f16_bits_to_f32(expected);
        let relative_error = (actual - reference).abs() / reference.abs();
        assert!(
            relative_error <= MAX_RELATIVE_ERROR,
            "output 0x{output:04x} differs from reference 0x{expected:04x} by {relative_error:e}, exceeding 2^-9.9"
        );
    }
    println!("PASS: ex2.approx.f16 results are within the PTX 2^-9.9 relative-error bound");
    Ok(())
}

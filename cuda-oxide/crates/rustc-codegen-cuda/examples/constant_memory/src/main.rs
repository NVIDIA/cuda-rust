/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[constant]` static end-to-end test.
//!
//! Demonstrates that:
//! 1. A `#[constant]` static lowers to PTX `.const` (address space 4).
//! 2. The macro-generated `module.set_coeffs(&value)` populates it from
//!    the host via `cuModuleGetGlobal` + `cuMemcpyHtoD`.
//! 3. Re-setting the constant between launches is observable by the kernel.
//!
//! Build and run with:
//!   cargo oxide run constant_memory

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[constant]
    static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;

    #[kernel]
    pub fn apply(mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = output.get_mut(idx) {
            let [c0, c1, c2, c3] = COEFFS.get();
            let x = i as f32;
            // c0 + c1 * x + c2 * x^2 + c3 * x^3
            *out_elem = c0 + c1 * x + c2 * x * x + c3 * x * x * x;
        }
    }
}

fn poly(c: &[f32; 4], x: f32) -> f32 {
    c[0] + c[1] * x + c[2] * x * x + c[3] * x * x * x
}

fn verify(label: &str, got: &[f32], coeffs: &[f32; 4]) {
    let mut errs = 0usize;
    for (i, &v) in got.iter().enumerate() {
        let want = poly(coeffs, i as f32);
        if (v - want).abs() > 1e-2 {
            errs += 1;
            if errs < 4 {
                eprintln!("{label} i={i}: got {v}, want {want}");
            }
        }
    }
    assert_eq!(errs, 0, "{label} produced {errs} mismatches");
    println!("{label}: OK ({} elements match polynomial)", got.len());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Constant Memory End-to-End Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = 256;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    for (label, coeffs) in [
        ("Launch A", [1.0f32, 2.0, 0.5, -0.25]),
        ("Launch B", [-3.0f32, 0.0, 1.0, 0.0]),
    ] {
        // set + launch on the same stream → naturally ordered. The second
        // launch demonstrates that re-setting between launches is observed.
        module.set_coeffs(&stream, &coeffs)?;
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.apply(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &mut output_dev,
            )
        }?;
        verify(label, &output_dev.to_host_vec(&stream)?, &coeffs);
    }

    println!("\nSUCCESS: #[constant] static populated from host, reflected on device.");
    Ok(())
}

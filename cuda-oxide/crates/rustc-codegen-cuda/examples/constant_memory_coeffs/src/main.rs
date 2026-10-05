/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct translation of the canonical CUDA C++ constant-memory example:
//!
//! ```cpp
//! __constant__ float coeffs[4];
//!
//! __global__ void compute(float *out) {
//!     int idx = threadIdx.x;
//!     out[idx] = coeffs[0] * idx + coeffs[1];
//! }
//!
//! float h_coeffs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
//! cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
//! compute<<<1, 10>>>(device_out);
//! ```
//!
//! Build and run with:
//!   cargo oxide run constant_memory_coeffs

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[constant]
    static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;

    #[kernel]
    pub fn compute(mut out: DisjointSlice<f32>) {
        let c = COEFFS.get();
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = c[0] * (i as f32) + c[1];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let h_coeffs: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    module.set_coeffs(&stream, &h_coeffs)?;

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 10)?;
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.compute(&stream, LaunchConfig::for_num_elems(10), &mut out) }?;

    let result = out.to_host_vec(&stream)?;
    println!("{:?}", result);
    // c[0] * idx + c[1] = 1.0 * idx + 2.0
    let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    assert_eq!(
        result, expected,
        "constant_memory_coeffs: kernel output mismatch"
    );
    println!(
        "✓ SUCCESS: constant-memory coefficients applied correctly ({} elements)",
        result.len()
    );
    Ok(())
}

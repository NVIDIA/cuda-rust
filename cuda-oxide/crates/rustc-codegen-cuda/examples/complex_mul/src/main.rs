/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Complex multiply/add regression test (issue #35).
//!
//! A `#[kernel]` that evaluates `z = z*z + c` on a small aggregate
//! `Complex32 { re: f32, im: f32 }` implementing `core::ops::Mul` and
//! `core::ops::Add` with `type Output = Self`.
//!
//! Before the fix, the device codegen backend rejected this with
//! `Alias type not yet supported: Mul::Output`, because the importer's type
//! translator only resolved arithmetic-trait associated outputs when the
//! operand was a primitive. `Complex32` is an ADT, so the projection
//! `<Complex32 as Mul>::Output` fell through to the unsupported-type arm.
//!
//! Run: cargo oxide run complex_mul

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// A minimal complex number. Mirrors the shape of `num_complex::Complex32`
/// (issue #35) without pulling in the dependency.
#[derive(Clone, Copy)]
struct Complex32 {
    re: f32,
    im: f32,
}

impl core::ops::Mul for Complex32 {
    type Output = Complex32;

    // `#[inline(never)]` keeps the call alive through MIR inlining so that the
    // `<Complex32 as Mul>::Output` projection actually reaches the type
    // translator. That is exactly the path that regressed in issue #35.
    #[inline(never)]
    fn mul(self, rhs: Complex32) -> Complex32 {
        Complex32 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl core::ops::Add for Complex32 {
    type Output = Complex32;

    #[inline(never)]
    fn add(self, rhs: Complex32) -> Complex32 {
        Complex32 {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Two iterations of `z = z*z + c`, then write `z.re + z.im`.
    #[kernel]
    pub fn complex_square_add(c_re: f32, c_im: f32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let c = Complex32 { re: c_re, im: c_im };
            let mut z = Complex32 { re: 0.0, im: 0.0 };
            z = z * z + c;
            z = z * z + c;
            *out_elem = z.re + z.im;
        }
    }
}

fn main() {
    println!("=== Complex Multiply/Add Test (issue #35) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();
    run_complex_square_add(&module, &stream);

    println!("\n=== Test Complete ===");
}

fn run_complex_square_add(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) {
    let mut d_out = DeviceBuffer::<f32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // c = (0.5, 0.0): z0 = (0,0) -> z1 = (0.5,0) -> z2 = (0.75,0); re+im = 0.75.
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.complex_square_add(stream.as_ref(), config, 0.5_f32, 0.0_f32, &mut d_out) }
        .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 0.75_f32;
    if (result - expected).abs() < 1e-5 {
        println!("complex_square_add: PASS (result = {})", result);
    } else {
        println!(
            "complex_square_add: FAIL (expected {}, got {})",
            expected, result
        );
    }
}

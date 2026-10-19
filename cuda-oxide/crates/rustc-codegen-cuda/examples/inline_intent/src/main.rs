/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline intent reaches the GPU.
//!
//! Each `#[inline]` form on a helper reaches LLVM as the matching function
//! attribute: `#[inline]` as `inlinehint`, `#[inline(always)]` as
//! `alwaysinline` and `#[inline(never)]` as `noinline`. The one exception is
//! the `#[inline(never)]` that `#[device]` adds to a helper with its own
//! generics. That attribute only keeps each copy visible to the device
//! collector on the host side, so it is not carried to device code and
//! const-generic arguments keep folding into the kernel.
//!
//! The kernel computes the same arithmetic as the host reference through one
//! helper of each kind. `verify-code-shape.sh` checks the attribute on each
//! definition in the pre-opt LLVM IR and what opt made of it in the PTX: real
//! `call.uni` instructions for the `noinline` helpers, no function left for
//! the `alwaysinline` helper, and the const-generic factor folded to an
//! immediate.
//!
//! MIR inlining is off for this crate (see Cargo.toml) so that every helper
//! reaches LLVM as its own definition. Otherwise rustc folds the small ones
//! into the kernel before any attribute could be observed.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, device, kernel};

/// `#[inline(never)]` on a plain function: a real call boundary in the PTX.
#[inline(never)]
pub fn never_add(a: f32, b: f32) -> f32 {
    a + b
}

/// `#[inline]` on a plain function: an `inlinehint` for LLVM.
#[inline]
pub fn hint_mul(a: f32, b: f32) -> f32 {
    a * b
}

/// `#[inline(always)]` on a plain function: `alwaysinline`, so opt leaves no
/// function behind. A plain function is used because a non-generic `#[device]`
/// function is an exported symbol, and rustc ignores `#[inline(always)]` on
/// exported functions.
#[inline(always)]
pub fn always_sub(a: f32, b: f32) -> f32 {
    a - b
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `#[inline(never)]` on a non-generic `#[device]` function: also a real
    /// call boundary. The macro adds no inline attribute of its own here.
    #[inline(never)]
    #[device]
    pub fn never_dev(a: f32, b: f32) -> f32 {
        a * 2.0 + b
    }

    /// A const-generic `#[device]` function. The macro's own `#[inline(never)]`
    /// is a host-side collector boundary and is not carried to device code,
    /// so `N` folds into the kernel as an immediate.
    #[device]
    pub fn scale_by<const N: u32>(x: f32) -> f32 {
        x * (N as f32)
    }

    #[kernel]
    pub fn intent(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((slot, index)) = output.get_mut_indexed() {
            *slot = reference(input, index.get());
        }
    }
}

/// The same arithmetic on host and device. `alwaysinline` folds it into the
/// kernel entry, so the entry itself carries the calls to the `noinline`
/// helpers.
#[inline(always)]
pub fn reference(input: &[f32], i: usize) -> f32 {
    let x = input[i];
    let mut v = never_add(x, 1.0);
    v = hint_mul(v, 2.0);
    v = always_sub(v, 1.0);
    v += kernels::never_dev(x, 1.0);
    v + kernels::scale_by::<4>(x)
}

fn main() {
    const N: usize = 256;
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.25).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).expect("Failed to upload input");
    let mut output_dev =
        DeviceBuffer::<f32>::zeroed(&stream, N).expect("Failed to allocate output");

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: the launch covers N elements and both buffers hold N floats.
    unsafe {
        module.intent(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &mut output_dev,
        )
    }
    .expect("Kernel launch failed");

    let output = output_dev
        .to_host_vec(&stream)
        .expect("Failed to read output");
    let mismatches = output
        .iter()
        .enumerate()
        .filter(|&(i, &got)| (got - reference(&input, i)).abs() > 1e-4)
        .inspect(|&(i, &got)| {
            eprintln!(
                "  mismatch at [{i}]: expected {}, got {got}",
                reference(&input, i)
            );
        })
        .count();
    println!("output[0..4] = {:?}", &output[0..4]);
    if mismatches == 0 {
        println!("SUCCESS: all {N} elements match the host reference");
    } else {
        println!("FAILED: {mismatches} mismatches");
        std::process::exit(1);
    }
}

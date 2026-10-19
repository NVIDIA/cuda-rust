/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for issue #79.
//!
//! A `#[kernel]` calls an `#[inline(never)]` device-reachable function
//! that returns `(f32, f32)` by value, and uses the destructured result.
//! With inlining suppressed, the tuple-returning `mir.call` survives into
//! the `dialect-mir` -> LLVM dialect lowering. The buggy `is_unit` check
//! at `crates/mir-lower/src/convert/ops/call.rs` matches every
//! `MirTupleType` (not just the empty unit tuple), forces the LLVM call
//! result to `void`, then falls through to `erase_operation` on a MIR op
//! that still has uses (the destructured `(a, b)`), and pliron panics.
//!
//! Expected (post-fix): prints `[0.0, 2.0, 4.0, ..., 30.0]` since
//! `(i+1) + (i-1) == 2i` for each output element.
//!
//! Build and run with:
//!   cargo oxide run tuple_return

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel};
use cuda_host::cuda_module;

// Plain `fn` with `#[inline(never)]` is the simplest way to keep the
// tuple-returning call alive through MIR optimization so it actually
// reaches lowering. `#[device]` is not required to trigger the bug.
#[inline(never)]
fn split(x: f32) -> (f32, f32) {
    (x + 1.0, x - 1.0)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<f32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let (a, b) = split(idx.get() as f32);
            *slot = a + b;
        }
    }
}

fn main() {
    const N: usize = 16;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let mut dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.run(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            &mut dev,
        )
    }
    .expect("Kernel launch failed");

    let host = dev.to_host_vec(&stream).unwrap();
    println!("output = {:?}", host);

    let mut errors = 0;
    for (i, &v) in host.iter().enumerate() {
        let expected = (2 * i) as f32;
        if (v - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!("  Error at [{}]: expected {}, got {}", i, expected, v);
            }
            errors += 1;
        }
    }
    if errors == 0 {
        println!("\nSUCCESS: tuple-returning device function lowered correctly");
    } else {
        println!("\nFAILED: {} mismatches", errors);
        std::process::exit(1);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `cuda_device::math::sin_cos` against libdevice, bit for bit.
//!
//! The kernel evaluates `x.sin()`, `x.cos()` (libdevice `__nv_sinf` / `__nv_cosf`)
//! and `sin_cos(x)` for 2^24 arguments: log-spaced magnitudes from 2^-40 up to
//! just below `SIN_COS_MAX_ARG`, both signs, plus the zeros. Every result must
//! carry the same bits, including the sign of zero, and the host's own
//! `sin_cos` (plain Rust, no device intrinsic) must agree with the device's.
//!
//! Run with:
//!   cargo oxide run math_sin_cos

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::math::{SIN_COS_MAX_ARG, sin_cos};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = [sin, cos, sin_cos.0, sin_cos.1]` of `x[i]`, as raw bits.
    #[kernel]
    pub fn compare(x: &[f32], mut out: DisjointSlice<[u32; 4]>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let v = x[i];
            let (s, c) = sin_cos(v);
            *slot = [
                v.sin().to_bits(),
                v.cos().to_bits(),
                s.to_bits(),
                c.to_bits(),
            ];
        }
    }
}

const N: usize = 1 << 24;

/// Log-spaced positive magnitudes (consecutive bit patterns are monotonic),
/// alternating sign, with both zeros at the front.
fn arguments() -> Vec<f32> {
    let lo = 2.0f32.powi(-40).to_bits() as u64;
    let hi = (SIN_COS_MAX_ARG - 0.5).to_bits() as u64;
    let mut x: Vec<f32> = (0..N as u64)
        .map(|i| {
            let magnitude = f32::from_bits((lo + i * (hi - lo) / N as u64) as u32);
            if i % 2 == 1 { -magnitude } else { magnitude }
        })
        .collect();
    x[0] = 0.0;
    x[1] = -0.0;
    x
}

fn main() {
    println!("=== cuda_device::math::sin_cos vs libdevice ===\n");
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let x = arguments();
    let x_dev = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut out_dev = DeviceBuffer::<[u32; 4]>::zeroed(&stream, N).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one thread per argument; both buffers hold N elements.
    unsafe {
        module.compare(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x_dev,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");
    let out = out_dev.to_host_vec(&stream).unwrap();

    let mut mismatches = 0usize;
    for (v, [sin, cos, s, c]) in x.iter().zip(&out) {
        let (hs, hc) = sin_cos(*v);
        if sin != s || cos != c || hs.to_bits() != *s || hc.to_bits() != *c {
            if mismatches < 8 {
                eprintln!(
                    "  x = {v:e} ({:#010x}): sin {sin:#010x} vs {s:#010x} (host {:#010x}), cos {cos:#010x} vs {c:#010x} (host {:#010x})",
                    v.to_bits(),
                    hs.to_bits(),
                    hc.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!(
            "SUCCESS: {N} arguments, sin_cos matches libdevice bit for bit on device and host"
        );
    } else {
        println!("FAILED: {mismatches} of {N} arguments differ");
        std::process::exit(1);
    }
}

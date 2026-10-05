// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Regression test: a fully-uninitialized constant allocation
//! (`MaybeUninit::uninit()` of a non-ZST type, every byte uninit, no
//! provenance) must translate as `undef` instead of failing with
//! "Unsupported constant type in translate_constant".
//!
//! Under -O, rustc const-promotes the `MaybeUninit::uninit()` initializer
//! into an `Allocated` constant whose init mask is empty. Aggregate types
//! take the struct/union dispatch, which has no handler for a constant
//! with no defined bytes. Aligned wrapper structs over vector types (the
//! glam `Align16<Vec3>` pattern) hit this in real shader crates.
//!
//! Run: cargo oxide run uninit_const

use core::mem::MaybeUninit;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// 16-byte aligned aggregate, mirroring glam-style `Align16<Vec3>`.
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct Align16([f32; 3]);

    #[kernel]
    pub fn write_through_uninit(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let t = tid.get() as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut slot: MaybeUninit<Align16> = MaybeUninit::uninit();
            slot.write(Align16([t, t + 1.0, t + 2.0]));
            // SAFETY: written just above.
            let v = unsafe { slot.assume_init() };
            *out_elem = v.0[0] + v.0[1] + v.0[2];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();
    const N: usize = 32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: 32-thread 1D block matches the 32-element output allocation.
    unsafe { module.write_through_uninit(stream.as_ref(), cfg, &mut out) }.expect("launch");
    let got = out.to_host_vec(&stream).unwrap();
    let mut failures = 0;
    for (tid, &v) in got.iter().enumerate() {
        let want = 3.0 * tid as f32 + 3.0;
        if (v - want).abs() > 1e-6 {
            println!("FAIL tid={tid}: got {v} want {want}");
            failures += 1;
        }
    }
    if failures == 0 {
        println!("uninit_const: PASS ({N} threads)");
    } else {
        std::process::exit(1);
    }
}

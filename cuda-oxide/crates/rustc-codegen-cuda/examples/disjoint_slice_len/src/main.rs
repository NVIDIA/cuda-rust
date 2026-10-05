/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #343: calling `DisjointSlice::len()` inside a
//! kernel must compile and return the launch-time length.
//!
//! Before the fix, reading `self.len` through `&self` emitted
//! `mir.extract_field` directly on the `mir.ptr<mir.disjoint_slice<T>>`
//! operand, which failed dialect verification:
//!
//! ```text
//! MirExtractFieldOp operand must be tuple, slice, struct, array, or scalar (newtype)
//! ```
//!
//! Run: cargo oxide run disjoint_slice_len
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Every in-bounds thread writes the slice length to its slot.
    #[kernel]
    pub fn write_len(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let n = out.len() as u32;
        if let Some(slot) = out.get_mut(idx) {
            *slot = n;
        }
    }
}

fn main() {
    // Deliberately not a block-size multiple: a lowering that substitutes
    // launch geometry for the real slice length must not accidentally pass.
    const N: usize = 257;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: 1-D launch over N elements matches the kernel's index space
    unsafe { module.write_len(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev) }
        .expect("write_len launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, N as u32, "out[{i}]: got {v}, want {N}");
    }

    println!("SUCCESS: DisjointSlice::len returns the launch-time length");
}

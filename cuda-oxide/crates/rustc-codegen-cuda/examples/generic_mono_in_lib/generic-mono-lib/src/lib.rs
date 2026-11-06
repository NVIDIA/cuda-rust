/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library crate for the generic_mono_in_lib regression (issue #1365).
//!
//! The `#[cuda_module]` is all-generic, but this library itself monomorphizes
//! `fill::<f64>` inside `fill_f64` and then calls `kernels::load`. Before the
//! fix, the macro skipped the #72 artifact-anchor keep-alive for all-generic
//! modules, so the library's `.oxart` archive member was dead-stripped and
//! `load_all_ptx_bundles_merged` returned `NoModules`.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn fill<T: Copy>(mut out: DisjointSlice<T>, value: T) {
        if let Some(x) = out.get_mut(thread::index_1d()) {
            *x = value;
        }
    }
}

/// Non-generic entry point: `fill::<f64>` is monomorphized in this crate.
pub fn fill_f64(n: usize, value: f64) -> Vec<f64> {
    let ctx = CudaContext::new(0).expect("no CUDA device");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, n).expect("alloc");
    unsafe {
        module.fill::<f64>(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            &mut out,
            value,
        )
    }
    .expect("launch fill::<f64>");
    out.to_host_vec(&stream).expect("copy back")
}

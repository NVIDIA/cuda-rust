/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A crate that only compiles kernels, built without the `host` feature.
//!
//! `scripts/check-device-only-build.sh` type-checks this with no CUDA toolkit
//! present and asserts that `cuda-host`, `cuda-core` and `cuda-bindings` are
//! absent from its resolved graph. Both macro entry points that emit host items
//! are exercised: a bare `#[kernel]`, which without the feature must not emit
//! its `cuda_host::CudaKernel` impl, and a `#[cuda_module]`, which must not
//! emit its `LoadedModule` loader or launchers.

#![no_std]

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// A bare `#[kernel]`, outside any module.
#[kernel]
pub fn scale_bare(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(slot) = out.get_mut(idx) {
        *slot = input[i] * 2.0;
    }
}

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    /// The same shape inside a `#[cuda_module]`, which is where the loader and
    /// launch methods would be emitted if the `host` feature were on.
    #[kernel]
    pub fn scale_in_module(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = input[i] * 3.0;
        }
    }
}

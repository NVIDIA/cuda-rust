/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library crate for the cross_crate_embedded regression test (issue #222).
//!
//! The generic kernel here is the key: because `scale<T>` is generic,
//! `#[cuda_module]` generates `load_all_ptx_bundles_merged` instead of
//! `load_embedded_module`. Without that, the monomorphized PTX (e.g.
//! `scale::<f32>`) would only be present in the binary crate's artifact
//! bundle, not in this library's bundle, causing a "named symbol not found"
//! panic at runtime.

use core::ops::Mul;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = input[i] * factor;
        }
    }
}

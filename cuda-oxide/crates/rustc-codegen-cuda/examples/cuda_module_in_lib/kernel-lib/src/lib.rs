/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel library whose `#[cuda_module]` lives in a *library* crate.
//!
//! This is the regression shape for issue #72: the kernels here are
//! concrete (non-generic), so their PTX is compiled and embedded while
//! *this* crate is built, and the resulting artifact object becomes a
//! member of this crate's `.rlib`. The application crate then loads the
//! module by bundle name with `kernels::load(&ctx)`, which looks up this
//! crate's `CARGO_PKG_NAME` ("module-kernels") in the final binary's
//! `.oxart` section.
//!
//! Before the fix, the artifact archive member defined no symbols, so the
//! linker never extracted it and `load()` failed at runtime with
//! `ModuleNotFound { name: "module-kernels" }`. The fix adds a link-anchor
//! symbol to the artifact object and a matching reference in the generated
//! `load_named()`, which forces the extraction.
//!
//! The package name deliberately contains a hyphen so the symbol-name
//! sanitization (hyphen to underscore) is exercised too.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    /// The root intentionally owns no direct kernels. Keeping the artifact
    /// must depend on `kernels::load()`, not on a child view being referenced.
    pub mod ops {
        use cuda_device::{DisjointSlice, kernel, thread};

        /// Multiply every input element by a constant factor.
        #[kernel]
        pub fn scale_f32(factor: f32, input: &[f32], mut out: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(out_elem) = out.get_mut(idx) {
                *out_elem = input[idx_raw] * factor;
            }
        }

        /// Element-wise addition of two input slices.
        #[kernel]
        pub fn add_f32(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(c_elem) = c.get_mut(idx) {
                *c_elem = a[idx_raw] + b[idx_raw];
            }
        }
    }
}

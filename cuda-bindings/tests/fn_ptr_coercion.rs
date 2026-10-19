// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The generated per-symbol wrappers must be `unsafe extern "C" fn` items so
//! they coerce to C function pointers — cuda-oxide's original bindings
//! surface, and how downstream code stores driver entry points in tables.
//! `extern "C"` exactly (not "C-unwind"): the unwind variant is a different
//! fn-pointer type and this coercion is what would break.

use cuda_bindings::CUresult;

#[test]
fn generated_wrappers_coerce_to_c_fn_pointers() {
    let _: unsafe extern "C" fn(u32) -> CUresult = cuda_bindings::cuInit;
}

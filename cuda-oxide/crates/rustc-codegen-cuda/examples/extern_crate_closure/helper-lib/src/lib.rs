/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helper crate for the extern_crate_closure regression test.
//!
//! The closures below are the regression triggers: they live in a
//! **non-local crate** (from the kernel crate's perspective) and are reached
//! from device code as callable trait receivers: `gated_load` passes a
//! closure directly to `apply`, and `double_even` passes one to `bool::then`.
//! The device collector's cross-crate kernel check used `TyCtxt::item_name`
//! on every candidate DefId, which ICEs rustc for unnamed items such as
//! closures ("item_name: no name for DefPath").
//! Closures in the *local* crate never hit that path (early `LOCAL_CRATE`
//! return), so only an external crate like this one reproduces the ICE.
//!
//! The two helpers deliberately use different source shapes. The exact
//! combinator chain is not important to the regression; what matters is that
//! the resulting callable receiver belongs to this external crate.
#![no_std]

/// Calls a callable trait receiver. `inline(never)` keeps this frame (and
/// with it the `FnOnce::call_once` receiver call) out of MIR inlining, so the
/// collector actually walks it and reaches the closure DefId — mirroring how
/// deeper library call chains behave even without the attribute.
#[inline(never)]
fn apply<F: FnOnce() -> i32>(f: F) -> i32 {
    f()
}

/// Returns `Some(values[index])` iff `keep` and `index` is in bounds.
/// The capturing closure below lives in this (non-local) crate.
#[inline]
pub fn gated_load(values: &[i32], index: usize, keep: bool) -> Option<i32> {
    if keep && index < values.len() {
        Some(apply(|| values[index]))
    } else {
        None
    }
}

/// Doubles the value when it is even, via a capturing closure.
#[inline]
pub fn double_even(value: i32) -> Option<i32> {
    let doubled = value.wrapping_mul(2);
    (value % 2 == 0).then(move || apply(move || doubled))
}

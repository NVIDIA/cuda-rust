/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-pass probe: type-check descriptor prefetch for arbitrary element and layout markers.
use cute_rs::TmaDesc;

// Prefetching metadata preserves the existing element and layout types and
// requires no extra layout trait or conversion at the call site.
/// Type-check descriptor prefetch for arbitrary element and layout markers. This device-only probe is never executed.
unsafe fn prefetch<T, Layout>(descriptor: &TmaDesc<T, Layout>) {
    unsafe { descriptor.prefetch() };
}

/// Type-check descriptor prefetch for arbitrary element and layout markers.

fn main() {
    /// Layout marker with no extra traits, proving prefetch adds no layout bounds.
    struct Layout;
    let _: unsafe fn(&TmaDesc<u16, Layout>) = prefetch;
    let _: unsafe fn(&TmaDesc<f32, Layout>) = prefetch;
}

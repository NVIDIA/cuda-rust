// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a cluster TMA copy whose descriptor and destination shapes differ.
#![feature(f16)]
use cute_rs::{
    ComposedLayout, OperandA, RowMajor, Sm100ClusterTmaCopy, Sm100SharedTile, Swizzle, TensorMap,
};
/// B128-swizzled 128-by-64 descriptor layout.
type A = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
/// B128-swizzled 96-by-64 destination layout.
type B = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
/// Reject a cluster TMA copy whose descriptor and destination shapes differ.
fn main() {
    let desc: *const TensorMap<f16, A> = core::ptr::null();
    let mut dst = unsafe { Sm100SharedTile::<B, OperandA>::from_raw(core::ptr::null_mut()) };
    unsafe {
        Sm100ClusterTmaCopy::<A>::copy(desc, &mut dst, 0, 0, core::ptr::null_mut(), 0);
    }
}

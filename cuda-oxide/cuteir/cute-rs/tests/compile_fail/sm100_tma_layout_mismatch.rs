// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::{
    Composed, OperandA, RowMajor, Sm100ClusterTmaCopy, Sm100SharedTile, Swizzle, TmaDesc,
};
type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
fn main() {
    let desc: *const TmaDesc<f16, A> = core::ptr::null();
    let mut dst = unsafe { Sm100SharedTile::<B, OperandA>::from_raw(core::ptr::null_mut()) };
    unsafe {
        Sm100ClusterTmaCopy::<A>::copy(desc, &mut dst, 0, 0, core::ptr::null_mut(), 0);
    }
}

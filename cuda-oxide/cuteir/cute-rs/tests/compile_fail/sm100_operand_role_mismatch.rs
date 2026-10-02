// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::{Composed, OperandB, RowMajor, Sm100SharedTile, Sm100TiledMma, Swizzle};
type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
fn main() {
    let a = unsafe { Sm100SharedTile::<A, OperandB>::from_raw(core::ptr::null_mut()) };
    let b0 = unsafe { Sm100SharedTile::<B0, OperandB>::from_raw(core::ptr::null_mut()) };
    let b1 = unsafe { Sm100SharedTile::<B1, OperandB>::from_raw(core::ptr::null_mut()) };
    let mma = Sm100TiledMma::<A, B0, B1, 192, 160>::new();
    unsafe {
        mma.gemm(0, &a, &b0, &b1, false);
    }
}

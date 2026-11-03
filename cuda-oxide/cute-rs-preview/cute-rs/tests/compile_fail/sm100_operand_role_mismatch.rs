// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a B-role tile supplied as the paired SM100 MMA A operand.
#![feature(f16)]
use cute_rs::{ComposedLayout, OperandB, RowMajor, Sm100SharedTile, Sm100TiledMma, Swizzle};
/// Local 128-by-64 FP16 A tile with the B128 swizzle.
type A = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
/// Local 96-by-64 B tile belonging to the collective N192 partition.
type B0 = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
/// Local 80-by-64 B tile belonging to the collective N160 partition.
type B1 = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
/// Reject a B-role tile supplied as the paired SM100 MMA A operand.
fn main() {
    let a = unsafe { Sm100SharedTile::<A, OperandB>::from_raw(core::ptr::null_mut()) };
    let b0 = unsafe { Sm100SharedTile::<B0, OperandB>::from_raw(core::ptr::null_mut()) };
    let b1 = unsafe { Sm100SharedTile::<B1, OperandB>::from_raw(core::ptr::null_mut()) };
    let mma = Sm100TiledMma::<A, B0, B1, 192, 160>::new();
    unsafe {
        mma.gemm(0, &a, &b0, &b1, false);
    }
}

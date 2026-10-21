// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject swapped N192 and N160 shared B partitions.
#![feature(f16)]
use cute_rs::{Composed, RowMajor, Sm100TiledMma, Swizzle};
/// Local 128-by-64 FP16 A tile with the B128 swizzle.
type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
/// Local 96-by-64 B tile belonging to the collective N192 partition.
type B0 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
/// Local 80-by-64 B tile belonging to the collective N160 partition.
type B1 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
/// Reject swapped N192 and N160 shared B partitions.
const _: Sm100TiledMma<A, B1, B0, 192, 160> = Sm100TiledMma::new();
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}

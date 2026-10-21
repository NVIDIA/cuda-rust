/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks the zero-storage paired SM100 MMA configuration."]
use super::*;
use crate::{Composed, RowMajor, Swizzle};
/// Local 128-by-64 FP16 A tile with the B128 swizzle.
type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
/// Local 96-by-64 B tile belonging to the collective N192 partition.
type B0 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
/// Local 80-by-64 B tile belonging to the collective N160 partition.
type B1 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
/// The paired SM100 MMA configuration must occupy zero runtime bytes.
#[test]
fn typed_split_plan_has_no_runtime_storage() {
    let mma = Sm100TiledMma::<A, B0, B1, 192, 160>::new();
    assert_eq!(core::mem::size_of_val(&mma), 0);
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::{Composed, RowMajor, Sm100TiledMma, Swizzle};
type A = Composed<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>;
type B0 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<96, 64>>;
type B1 = Composed<Swizzle<3, 3, 3>, 0, RowMajor<80, 64>>;
const _: Sm100TiledMma<A, B1, B0, 192, 160> = Sm100TiledMma::new();
fn main() {}

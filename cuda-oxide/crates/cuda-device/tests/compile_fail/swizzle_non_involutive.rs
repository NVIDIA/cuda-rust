/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Swizzle<B, M, S>` requires `M + B <= S`. Without it the XOR target bits
//! overlap the bits being read, so applying the swizzle twice does not restore
//! the original offset and a store/load pair through it would disagree.

use cuda_device::swizzle::Swizzle;

// M + B = 5, S = 4: the target bits overlap the source bits, so applying the
// swizzle twice would not restore the offset. A const context forces the
// involution check to be evaluated.
const BAD: usize = Swizzle::<5, 0, 4>::apply(0);

fn main() {
    let _ = BAD;
}

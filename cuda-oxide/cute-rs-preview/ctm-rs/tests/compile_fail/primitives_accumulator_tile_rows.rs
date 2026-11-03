// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject the two-row MMA method on a one-row accumulator tile.
use ctm_rs::mma::{AccumulatorTile16x8, Mxf4A, Mxf4B, ScalePairA, ScalePairB};

/// Reject the two-row MMA method on a one-row accumulator tile.

fn main() {
    let mut tile = AccumulatorTile16x8::<1, 8>::zero();
    unsafe {
        tile.mma_mxf4_pair::<0>(
            [Mxf4A::from_registers([0; 4]); 2],
            [Mxf4B::from_registers([0; 2]); 2],
            [ScalePairA::from_bits(0); 2],
            [ScalePairB::from_bits(0); 2],
        );
    }
}

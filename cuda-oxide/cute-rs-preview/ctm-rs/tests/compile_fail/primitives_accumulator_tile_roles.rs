// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject swapped A/B fragment and scale roles in tiled MMA.
use ctm_rs::mma::{AccumulatorTile16x8, Mxf4A, Mxf4B, ScalePairA, ScalePairB};

/// Reject swapped A/B fragment and scale roles in tiled MMA.

fn main() {
    let mut tile = AccumulatorTile16x8::<2, 8>::zero();
    unsafe {
        tile.mma_mxf4_pair::<0>(
            [Mxf4B::from_registers([0; 2]); 2],
            [Mxf4A::from_registers([0; 4]); 2],
            [ScalePairB::from_bits(0); 2],
            [ScalePairA::from_bits(0); 2],
        );
    }
}

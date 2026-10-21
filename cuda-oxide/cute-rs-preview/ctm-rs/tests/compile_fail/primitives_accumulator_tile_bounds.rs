// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a fragment row outside the accumulator tile during const evaluation.
use ctm_rs::mma::AccumulatorTile16x8;

/// Valid two-row, eight-column accumulator tile used by the bounds probe.

const TILE: AccumulatorTile16x8<2, 8> = AccumulatorTile16x8::zero();
/// Force compile-time rejection of a fragment beyond the tile’s last row.
const OUTSIDE: ctm_rs::mma::Accumulator16x8 = TILE.fragment::<2, 0>();

/// Reject a fragment row outside the accumulator tile during const evaluation.

fn main() {
    let _ = OUTSIDE;
}
